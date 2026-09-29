"""Coordinate repeatable learning rounds with replaceable external validation."""
import re
import shutil
from pathlib import Path
import torch
from .dataset import CandidateRegistry, DatasetManager
from .generator import Generator
from .molecule import SpeciesInput, ReferenceSpecies
from .config import Settings
from .checkpoint import Checkpoint
from .model import TrainingDevice
from .reporter import Reporter, TrainingLogger
from .storage import Storage
from .trainer import Trainer, InitialTrainer
from .validators.base import Validator, ValidationCache, ValidationResult


class SelfUpdatingPipeline:
    """Generate, validate, label, fine-tune and select; resume completed phases.

    No training jobs run in __init__. run(resume=True) verifies saved settings and
    completed artifacts, and collects pending validation jobs without regenerating
    molecules. Interrupted training continues after the last fully saved epoch,
    including optimizer state, route sampling and model-selection history.
    """
    VERSION=2

    def __init__(self, positive_path, negative_path, checkpoint_path, validator,
                 output_folder='output', generation_config=None, training_config=None,
                 labeling_config=None, initial_training_config=None, hard_negative_config=None):
        if not isinstance(validator,Validator):
            raise TypeError('validator must implement the Validator interface')
        self.paths={'positive':Path(positive_path) if positive_path is not None else None,
                    'negative':Path(negative_path) if negative_path is not None else None,
                    'checkpoint':Path(checkpoint_path) if checkpoint_path is not None else None}
        if positive_path is None and checkpoint_path is None:
            raise ValueError("Provide given species or a checkpoint containing its datasets")
        self.validator=validator;self.output=Path(output_folder).resolve()
        self.generator=Generator(generation_config)
        if set(self.generator.config['methods']) != {'puct','direct'}:
            raise ValueError('Self-updating rounds require both PUCT and direct generation')
        self.training=Settings.merge(Settings.TRAINING,training_config)
        self.labeling=Settings.merge(Settings.LABELING,labeling_config)
        if set(self.labeling)!={'unmatched_as_negative'} or type(self.labeling['unmatched_as_negative']) is not bool:
            raise ValueError('Labeling supports the boolean unmatched_as_negative setting')
        self.hard_negative=Settings.merge(Settings.HARD_NEGATIVE,hard_negative_config)
        if set(self.hard_negative)!={'percent','max_count','puct_weight'}:
            raise ValueError('Hard-negative selection supports percent, max_count, and puct_weight')
        # Validate this before output replacement. DatasetManager validates again
        # when it performs the actual ranking after validation is complete.
        DatasetManager.validate_hardness_options(self.hard_negative['percent'],
                                                 self.hard_negative['max_count'],
                                                 self.hard_negative['puct_weight'])
        self.initial_training=Settings.merge(Settings.INITIAL,{
            "seed": self.training["seed"], "threads": self.training["threads"],
            "device": self.training["device"], "gpu_ids": self.training["gpu_ids"],
            "log_level": self.training["log_level"],
            **(initial_training_config or {})})
        levels = {self.generator.config["log_level"], self.training["log_level"],
                  self.initial_training["log_level"]}
        if len(levels) != 1:
            raise ValueError("Use one shared log_level for generation, initial training, and fine-tuning")
        self.log_level = levels.pop()
        # Validate numerical/device settings before creating or clearing outputs.
        Settings.training(self.training)
        Settings.training(self.initial_training,initial=True)
        TrainingDevice.resolve(self.training['device'], self.training.get('gpu_ids'))
        if self.paths["checkpoint"] is None:
            TrainingDevice.resolve(self.initial_training["device"], self.initial_training.get('gpu_ids'))
        self.signature=None;self.state=None;self.log=None

    def configuration(self):
        inputs={}
        for key,path in self.paths.items():
            if path is None:inputs[key]=None;continue
            resolved=path.resolve()
            if resolved==self.output or self.output in resolved.parents:
                raise ValueError('Input files must be outside the pipeline output folder')
            inputs[key]={'path':str(resolved),'sha256':Storage.digest(resolved)}
        validator_config=self.validator.configuration()
        # Reference files must not be placed under a folder that fresh runs replace.
        if 'reference_path' in validator_config:
            ref=Path(validator_config['reference_path']).resolve()
            if ref==self.output or self.output in ref.parents:
                raise ValueError('Validation reference must be outside output')
        implementation={p.name+str(p.parent.name):Storage.digest(p)
                        for p in Path(__file__).parent.rglob('*.py')}
        return {'version':self.VERSION,'inputs':inputs,'generation':self.generator.config,
                'training':self.training,'labeling':self.labeling,
                'hard_negative':self.hard_negative,
                'initial_training':self.initial_training,
                'validator':validator_config,'validator_fingerprint':self.validator.fingerprint(),
                'implementation_sha256':Storage.fingerprint(implementation)}

    def prepare(self, resume, rounds):
        self.resume=resume
        settings=self.configuration();self.signature=Storage.fingerprint(settings)
        marker=self.output/'run_state.json'
        if resume:
            if not marker.is_file():raise ValueError('No saved run to resume')
            self.state=Storage.read(marker)
            if self.state['signature']!=self.signature:
                raise ValueError('Inputs, code or settings changed; use a separate output or start a fresh run')
            started_rounds=[int(match.group(1)) for key in self.state['steps']
                            if (match:=re.match(r'round_(\d+)/',key))]
            if started_rounds and rounds<max(started_rounds):
                raise ValueError('Cannot resume with fewer rounds than already started')
            # Verify every completed phase before accepting its saved result.
            for key in self.state['steps']:self.done(key)
        else:
            if self.output.exists() and any(self.output.iterdir()) and not marker.is_file():
                raise ValueError('Output is not an existing pipeline run; choose an empty directory')
            # Delete only pipeline-owned output directories/files, never arbitrary inputs.
            self.output.mkdir(parents=True,exist_ok=True)
            for child in self.output.iterdir():
                if re.fullmatch(r'round_\d+',child.name) or child.name in ('initial_training','validation_jobs','checkpoints','inputs','route_cache'):
                    if child.is_symlink():child.unlink()
                    elif child.is_dir():shutil.rmtree(child)
                elif child.name in ('run_state.json','config.json','summary.json','validation_cache.json','pipeline.log'):
                    child.unlink()
            self.state={'signature':self.signature,'steps':{},'rounds':[],'status':'running'}
            Storage.write(marker,self.state)
        Storage.write(self.output/'config.json',settings)
        self.log=TrainingLogger(self.output/'pipeline.log',append=resume,level=self.log_level)

    def done(self, key):
        step=self.state['steps'].get(key)
        if step is None:return False
        for path,expected in step['artifacts'].items():
            file=self.output/path
            if not file.is_file() or Storage.digest(file)!=expected:
                raise ValueError(f'Saved artifact changed or missing: {file}')
        return True

    def complete(self, key, paths):
        self.state['steps'][key]={'artifacts':{str(Path(p).relative_to(self.output)):Storage.digest(p) for p in paths}}
        Storage.write(self.output/'run_state.json',self.state)

    def save_checkpoint(self, name, checkpoint, round_number=None):
        """Copy an immutable, portable checkpoint into the run's checkpoint folder."""
        checkpoint=Path(checkpoint)
        folder=self.output/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'{name}.pt'
        source_hash=Storage.digest(checkpoint)
        if target.exists():
            if Storage.digest(target)!=source_hash:
                raise ValueError(f'Checkpoint already exists with different contents: {target}')
        else:
            Checkpoint.copy(checkpoint,target)
        manifest_path=folder/'manifest.json'
        manifest=Storage.read(manifest_path) if manifest_path.exists() else {}
        manifest[name]={'path':str(target),'sha256':source_hash,'round':round_number,
                        'source_checkpoint':str(checkpoint)}
        Storage.write(manifest_path,manifest)
        return target

    def method_validation_reports(self, groups, folder):
        """Add the post-validation three-category plot to each generator folder."""
        if not self.generator.config['save_media']:
            return
        for method in self.generator.config['methods']:
            Reporter.method_validation(groups, folder / 'generation' / method, method, log=self.log.info)

    def log_path(self, path):
        """Use short output-relative paths where possible, without rejecting parents."""
        path = Path(path)
        try:
            return path.relative_to(self.output)
        except ValueError:
            return path

    def initial_model(self):
        if self.paths['checkpoint'] is not None:return self.paths['checkpoint'].resolve()
        folder=self.output/'initial_training';checkpoint=folder/'trained_growth_gnn.pt'
        if not self.done('initial_training'):
            self.log.stage('STAGE I | Initial supervised imitation training', 'START',
                           epochs=self.initial_training['epochs'],
                           device=self.initial_training['device'])
            cfg={**self.initial_training,
                 'reference_path':str(self.output/'inputs/given_species.json'),
                 'route_cache_folder':str(self.output/'route_cache'),
                 'output_folder':str(folder)}
            InitialTrainer(cfg).run(resume=self.resume)
            self.complete('initial_training',[checkpoint])
            self.log.stage('STAGE I | Initial supervised imitation training', 'COMPLETE',
                           checkpoint=self.log_path(checkpoint))
        else:
            self.log.stage('STAGE I | Initial supervised imitation training', 'REUSED',
                           checkpoint=self.log_path(checkpoint))
        return checkpoint

    def run(self, rounds=1, resume=False):
        if type(rounds) is not int or rounds<1 or type(resume) is not bool:
            raise ValueError('rounds must be positive; resume must be boolean')
        records=self.read_input_records()
        self.prepare(resume, rounds)
        try:
            self.log.stage('RUN', 'RESUME' if resume else 'START', rounds=rounds,
                           device=self.training['device'], gpu_ids=self.training.get('gpu_ids'))
            self.normalize_inputs(records)
            return self.cycle(rounds)
        except BaseException:
            self.log.exception('PIPELINE INTERRUPTED; completed phases and validation results are retained for resume.')
            raise
        finally:self.log.close()

    def read_input_records(self):
        """Validate inputs before a fresh run replaces any prior output."""
        parent = (torch.load(self.paths["checkpoint"],map_location="cpu",weights_only=True)
                  if self.paths["checkpoint"] is not None else None)
        trained = parent["config"] if parent else self.initial_training
        if self.paths["positive"] is None:
            if not parent.get("datasets"):
                raise ValueError("This checkpoint has no bundled datasets; supply positive_path")
            positive = parent["datasets"]["positive"]
            negative = parent["datasets"].get("negative", [])
        else:
            positive = SpeciesInput.read(self.paths["positive"])
            negative = []
        if self.paths["negative"] is not None:
            negative = SpeciesInput.read_negative(self.paths["negative"])
        for records, excluded in ((positive, trained["excluded_species"]),
                                   ({r["structure_id"]: r["graph"] for r in negative}, [])):
            if records:
                reference = ReferenceSpecies(None,excluded,trained["atom_symbols"],
                    trained["max_valency"],trained["max_atoms"],records=records)
                if any(g.number_of_edges()+2 > trained["max_steps"] for g in reference.graphs.values()):
                    raise ValueError("Input species needs more actions than max_steps")
        return {"positive": positive, "negative": negative}

    def normalize_inputs(self, records=None):
        """Save and hash complete input snapshots as one resumable phase."""
        if self.done("inputs"):
            return
        records = self.read_input_records() if records is None else records
        folder=self.output/'inputs'
        given=folder/'given_species.json'
        negative=folder/'negative_species.json'
        Storage.write(given,records["positive"])
        Storage.write(negative,records["negative"])
        paths=[given,negative]
        if hasattr(self.validator,'normalized_records'):
            reference=folder/'reference_species.json'
            Storage.write(reference,self.validator.normalized_records())
            paths.append(reference)
        self.complete("inputs",paths)

    def cycle(self, rounds):
        checkpoint=self.initial_model()
        checkpoint=self.save_checkpoint('initial_training' if self.paths['checkpoint'] is None else 'starting_checkpoint',
                                       checkpoint,round_number=0)
        positive=Storage.read(self.output/'inputs/given_species.json')
        negative=Storage.read(self.output/'inputs/negative_species.json')
        # Excluded seed species must not appear in reports as active training members.
        parent=torch.load(checkpoint,map_location='cpu',weights_only=True)
        positive={k:v for k,v in positive.items() if k not in parent['config']['excluded_species']}
        datasets=DatasetManager(positive,negative)
        registry=CandidateRegistry()
        for record in datasets.negative:
            observations=record.get("observations") or [{
                'method':'imported','source_structure_id':record['structure_id'],'round':0,
                'occurrences':record.get('occurrences',0)}]
            for observation in observations:
                registry.add(record['graph'],observation)
        cache=ValidationCache(self.output/'validation_cache.json',self.validator)
        for number in range(1,rounds+1):
            folder=self.output/f'round_{number:03d}';folder.mkdir(parents=True,exist_ok=True)
            key=f'round_{number:03d}'
            self.log.stage(f'ROUND {number}/{rounds}', 'START',
                           parent=self.log_path(checkpoint))
            if self.done(key+'/complete'):
                summary=Storage.read(folder/'summary.json')
                self.state['rounds']=[r for r in self.state['rounds'] if r['round']!=number]+[summary]
                saved=Storage.read(folder/'datasets/snapshot.json')
                datasets=DatasetManager(saved['positive'],saved['negative'],saved['unresolved'])
                registry=CandidateRegistry(Storage.read(folder/'generation/candidates.json'))
                checkpoint=self.save_checkpoint(f'round_{number:03d}',folder/'training/trained_growth_gnn.pt',number)
                self.log.stage(f'ROUND {number}/{rounds}', 'REUSED',
                               checkpoint=self.log_path(checkpoint))
                continue
            inputs=folder/'inputs'
            positive_path=inputs/'positive.json';negative_path=inputs/'negative.json'
            if not self.done(key+'/inputs'):
                Storage.write(positive_path,datasets.positive);Storage.write(negative_path,datasets.negative)
                Storage.write(inputs/'parent.json',{'path':str(checkpoint),'sha256':Storage.digest(checkpoint)})
                self.complete(key+'/inputs',[positive_path,negative_path,inputs/'parent.json'])
            parent_signature=Storage.read(inputs/'parent.json')
            if Storage.digest(checkpoint)!=parent_signature['sha256']:
                raise ValueError('Round parent checkpoint changed')
            for method in self.generator.config['methods']:
                out=folder/'generation'/method
                stage = 'II-A | PUCT exploration' if method == 'puct' else 'II-B | Direct sampling exploration'
                if not self.done(key+'/generation/'+method):
                    self.log.stage(f'ROUND {number}/{rounds} | STAGE {stage}', 'START',
                                   attempts=self.generator.config['runs'] * self.generator.config['attempts_per_run'])
                    self.generator.run(method,checkpoint,positive_path,out,number,rounds)
                    self.complete(key+'/generation/'+method,[out/name for name in
                                  ('config.json','summary.json','unique_structures.json','attempts.json')])
                generation_summary=Storage.read(out/'summary.json')
                self.log.stage(f'ROUND {number}/{rounds} | STAGE {stage}', 'COMPLETE',
                               attempts=generation_summary['total_attempts'],
                               references=generation_summary['unique_reference_structures'],
                               candidates=generation_summary['unique_new_candidates'])
                registry.merge_generation(Storage.read(out/'unique_structures.json'),method,number,out/'attempts.json')
            candidates=folder/'generation/candidates.json'
            if not self.done(key+'/merge'):
                Storage.write(candidates,registry.records);self.complete(key+'/merge',[candidates])
            else:registry=CandidateRegistry(Storage.read(candidates))
            validation=folder/'validation';validation.mkdir(exist_ok=True)
            if not self.done(key+'/validation'):
                self.log.stage(f'ROUND {number}/{rounds} | STAGE III | Validation and dataset update', 'START')
                results={};pending=0
                for record in registry.records:
                    given=datasets.positive_name(record['graph'])
                    if given:
                        result=ValidationResult(record['structure_id'],'completed','accepted',
                            'Retained given species; external validation not repeated','given',{'matched_species':[given]})
                    else:result=cache.validate(record,self.output/'validation_jobs')
                    results[record['structure_id']]=result.to_dict()
                    pending+=result.status=='pending'
                Storage.write(validation/'results.json',results)
                if pending:
                    self.state.update(status='waiting_for_validation',pending_round=number,pending_count=pending)
                    Storage.write(self.output/'run_state.json',self.state)
                    self.log.stage(f'ROUND {number}/{rounds} | STAGE III | Validation and dataset update',
                                   'WAITING', pending=pending)
                    return {'status':'waiting_for_validation','round':number,'pending':pending}
                self.complete(key+'/validation',[validation/'results.json'])
            else:results=Storage.read(validation/'results.json')
            datafolder=folder/'datasets'
            if not self.done(key+'/datasets'):
                groups=datasets.update(registry.records,results,**self.labeling)
                Storage.write(datafolder/'snapshot.json',datasets.snapshot())
                Storage.write(datafolder/'positive_species.json',datasets.positive)
                Storage.write(datafolder/'negative_species.json',datasets.negative)
                Storage.write(datafolder/'unresolved_species.json',datasets.unresolved)
                hard_ids,hard_summary=datasets.select_hard_negatives(
                    self.hard_negative['percent'],self.hard_negative['max_count'],
                    self.hard_negative['puct_weight'])
                Storage.write(datafolder/'hard_negative_selection.json',hard_summary)
                Reporter.validation(groups,validation,plots=self.generator.config['save_media'],log=self.log.info)
                self.method_validation_reports(groups,folder)
                self.complete(key+'/datasets',[datafolder/name for name in
                    ('snapshot.json','positive_species.json','negative_species.json','unresolved_species.json',
                     'hard_negative_selection.json')]
                    +[validation/'species_groups.json',validation/'summary.json'])
            else:
                saved=Storage.read(datafolder/'snapshot.json')
                datasets=DatasetManager(saved['positive'],saved['negative'],saved['unresolved'])
                hard_summary=Storage.read(datafolder/'hard_negative_selection.json')
                hard_ids=[entry['structure_id'] for entry in hard_summary['ranking'] if entry['selected']]
            validation_summary=Storage.read(validation/'summary.json')
            self.log.stage(f'ROUND {number}/{rounds} | STAGE III | Validation and dataset update',
                           'COMPLETE', accepted=validation_summary['display_group_counts']['accepted'],
                           unvalidated=validation_summary['display_group_counts']['unvalidated'],
                           given=validation_summary['display_group_counts']['given'],
                           hard_negatives=len(hard_ids))
            training=folder/'training'
            if not self.done(key+'/training'):
                self.log.stage(f'ROUND {number}/{rounds} | STAGE IV | Fine-tuning and checkpoint selection',
                               'START', epochs=self.training['epochs'], hard_negatives=len(hard_ids))
                options={**self.training,'seed':self.training['seed']+number-1,
                         'positive_path':str(datafolder/'positive_species.json'),
                         'negative_path':str(datafolder/'negative_species.json'),
                         'hard_negative_ids':hard_ids,
                         'checkpoint_path':str(checkpoint),'output_folder':str(training),
                         'route_cache_folder':str(self.output/'route_cache'),
                         'round_number':number, 'round_total':rounds}
                Trainer(options).run(resume=self.resume)
                self.complete(key+'/training',[training/'trained_growth_gnn.pt',training/'evaluation.json'])
            selected=training/'trained_growth_gnn.pt';evaluation=Storage.read(training/'evaluation.json')
            self.log.stage(f'ROUND {number}/{rounds} | STAGE IV | Fine-tuning and checkpoint selection',
                           'COMPLETE', selected_epoch=evaluation['selected_epoch'],
                           selection=evaluation['selection'])
            saved_checkpoint=self.save_checkpoint(f'round_{number:03d}',selected,number)
            summary={'round':number,'datasets':datasets.snapshot()['counts'],
                     'hard_negative_selection':hard_summary,
                     'generation':{m:Storage.read(folder/'generation'/m/'summary.json') for m in self.generator.config['methods']},
                     'selected_checkpoint':str(saved_checkpoint),'selection':evaluation['selection'],
                     'selected_epoch':evaluation['selected_epoch']}
            Storage.write(folder/'summary.json',summary)
            self.complete(key+'/complete',[folder/'summary.json'])
            self.state['rounds']=[r for r in self.state['rounds'] if r['round']!=number]+[summary]
            Storage.write(self.output/'run_state.json',self.state)
            self.log.stage(f'ROUND {number}/{rounds} | CHECKPOINT', 'SAVED',
                           checkpoint=self.log_path(saved_checkpoint))
            self.log.stage(f'ROUND {number}/{rounds}', 'COMPLETE')
            checkpoint=saved_checkpoint
        self.state.update(status='complete',pending_count=0)
        self.state.pop('pending_round',None)
        Storage.write(self.output/'run_state.json',self.state)
        result={'status':'complete','rounds':self.state['rounds'],'selected_checkpoint':str(checkpoint)}
        Storage.write(self.output/'summary.json',result)
        self.log.stage('RUN', 'COMPLETE', rounds=len(self.state['rounds']),
                       checkpoint=self.log_path(checkpoint))
        return result
