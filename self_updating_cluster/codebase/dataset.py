"""Persistent graph identities, source observations and training-set updates."""
import copy
import math
from collections import Counter
from .molecule import Molecule


class CandidateRegistry:
    """Deduplicate across methods/rounds while preserving trace provenance."""

    def __init__(self, records=None):
        self.records=copy.deepcopy(records or [])
        self.buckets={}
        for record in self.records:
            self.buckets.setdefault(Molecule.key(record['graph']),[]).append(record)
        ids=[r['structure_id'] for r in self.records]
        if len(ids)!=len(set(ids)):
            raise ValueError('Duplicate registry IDs')

    def add(self, graph, observation=None):
        key=Molecule.key(graph)
        found=next((r for r in self.buckets.get(key,[]) if Molecule.same(graph,r['graph'])),None)
        if found is None:
            used={r['structure_id'] for r in self.records}
            number=len(self.records)+1
            while f'G{number:06d}' in used:number+=1
            found={'structure_id':f'G{number:06d}','graph':copy.deepcopy(graph),'observations':[]}
            self.records.append(found);self.buckets.setdefault(key,[]).append(found)
        if observation is not None and observation not in found['observations']:
            found['observations'].append(copy.deepcopy(observation))
        return found

    def merge_generation(self, records, method, round_number, attempts_path):
        for record in records:
            self.add(record['graph'],{
                'method':method,'round':round_number,'source_structure_id':record['structure_id'],
                'occurrences':record['occurrences'],'attempts_path':str(attempts_path),
                # All traces remain in attempts.json. Retain one representative
                # route per method/round here for easy validation/training analysis.
                'actions':record['actions']})


class DatasetManager:
    """Retain positives, remove conflicts, and apply explicit labeling policy."""

    def __init__(self, positive, negative, unresolved=None):
        if not isinstance(positive,dict) or not positive or not isinstance(negative,list):
            raise ValueError('Expected positive graph dictionary and negative record list')
        self.positive={}
        for name,graph in positive.items():
            Molecule.require_supported_training(graph)
            if not self.positive_name(graph):self.positive[name]=copy.deepcopy(graph)
        self.negative=[]
        for record in negative:
            Molecule.require_supported_training(record['graph'])
            if not self.positive_name(record['graph']) and not any(Molecule.same(record['graph'],r['graph']) for r in self.negative):
                self.negative.append(copy.deepcopy(record))
        self.unresolved=copy.deepcopy(unresolved or [])

    def positive_name(self, graph):
        return next((name for name,other in self.positive.items() if Molecule.same(graph,other)),None)

    @staticmethod
    def validate_hardness_options(percent, max_count, puct_weight):
        if isinstance(percent,bool) or not isinstance(percent,(int,float)) or not math.isfinite(percent) or not 0 < percent <= 1:
            raise ValueError('hard_negative_percent must be finite and in (0, 1]')
        if isinstance(puct_weight,bool) or not isinstance(puct_weight,(int,float)) or not math.isfinite(puct_weight) or puct_weight < 0:
            raise ValueError('puct_hardness_weight must be finite and nonnegative')
        if type(max_count) is not int or max_count < 1:
            raise ValueError('hard_negative_max_count must be a positive integer')

    @staticmethod
    def negative_hardness(record, puct_weight):
        """Summarize how often one graph appeared under each generator.

        Direct sampling is the primary signal because it reflects the frozen
        policy without PUCT's exploration bonus. PUCT still contributes a smaller,
        explicit amount. Imported old negatives have no method-specific history;
        their saved occurrence count is retained separately as a fallback signal.
        """
        counts=Counter()
        for observation in record.get('observations',[]):
            method=observation.get('method')
            occurrence=observation.get('occurrences',0)
            if method in ('direct','puct','imported') and type(occurrence) is int and occurrence >= 0:
                counts[method]+=occurrence
        if not counts['imported'] and type(record.get('occurrences')) is int and record['occurrences'] >= 0:
            counts['imported']=record['occurrences']
        # Older Stage III negatives do not identify their generator. Conservatively
        # weight their count like PUCT, rather than treating it as a direct-policy
        # measurement.
        score=counts['direct']+puct_weight*(counts['puct']+counts['imported'])
        return {'direct_occurrences':counts['direct'],'puct_occurrences':counts['puct'],
                'imported_occurrences':counts['imported'],'puct_weight':puct_weight,
                'score':score}

    def select_hard_negatives(self, percent, max_count, puct_weight):
        """Return the top percentage of negatives, limited to ``max_count``."""
        self.validate_hardness_options(percent,max_count,puct_weight)
        ranked=[]
        for record in self.negative:
            hardness=self.negative_hardness(record,puct_weight)
            ranked.append((record,hardness))
        ranked.sort(key=lambda item:(-item[1]['score'],-item[1]['direct_occurrences'],
                                     -item[1]['puct_occurrences'],item[0]['structure_id']))
        percent_count=math.ceil(percent*len(ranked)) if ranked else 0
        selected_count=min(percent_count,max_count)
        selected=[]
        for rank,(record,hardness) in enumerate(ranked,1):
            selected.append({'rank':rank,'structure_id':record['structure_id'],**hardness,
                             'selected':rank<=selected_count})
        return [entry['structure_id'] for entry in selected if entry['selected']],{
            'strategy':'top_percent_by_direct_plus_weighted_puct_appearances',
            'hard_negative_percent':percent,'hard_negative_max_count':max_count,
            'puct_hardness_weight':puct_weight,'eligible_negative_species':len(ranked),
            'percent_selected_negative_species':percent_count,
            'selected_negative_species':selected_count,
            'ranking':selected,
        }

    def update(self, records, results, unmatched_as_negative=False):
        if type(unmatched_as_negative) is not bool:
            raise ValueError('unmatched_as_negative must be boolean')
        groups={'accepted':[],'rejected':[],'unknown':[],'given':[]}
        unresolved=[]
        for record in records:
            result=results[record['structure_id']]
            graph=record['graph'];given=self.positive_name(graph)
            entry={**copy.deepcopy(record),'validation':copy.deepcopy(result),
                   'matched_species':result['details'].get('matched_species',[])}
            if given:
                entry.update(species_key=given,training_label='positive',label_basis='retained_positive')
                groups['given'].append(entry)
                continue
            # Current evidence replaces previous negative labeling for this graph.
            self.negative=[r for r in self.negative if not Molecule.same(graph,r['graph'])]
            status,decision=result['status'],result['decision']
            reference_negative=(status=='completed' and decision=='unknown' and unmatched_as_negative
                                and result['details'].get('unmatched_reference') is True)
            accepted=status=='completed' and decision=='accepted'
            rejected=status=='completed' and decision=='rejected'
            group='accepted' if accepted else 'rejected' if rejected else 'unknown'
            entry['training_label']='positive' if accepted else 'negative' if rejected or reference_negative else 'unresolved'
            entry['label_basis']='reference_absence_policy' if reference_negative else 'validator_decision'
            if accepted or rejected or reference_negative:
                try:Molecule.require_supported_training(graph)
                except ValueError as error:
                    entry['training_label']='unresolved';entry['training_exclusion_reason']=str(error)
            if entry['training_label']=='positive':
                name=next(iter(entry['matched_species']),record['structure_id'])
                if name in self.positive:name=f"{name}__{record['structure_id']}"
                self.positive[name]=copy.deepcopy(graph);entry['species_key']=name
            elif entry['training_label']=='negative':
                self.negative.append({'structure_id':record['structure_id'],'graph':copy.deepcopy(graph),
                                      'label_basis':entry['label_basis'],'validation':copy.deepcopy(result),
                                      'observations':copy.deepcopy(entry['observations'])})
            else:unresolved.append(entry)
            groups[group].append(entry)
        # A later accepted record must not leave an identical graph in negatives.
        self.negative=[r for r in self.negative if not self.positive_name(r['graph'])]
        self.unresolved=unresolved
        return groups

    def snapshot(self):
        return {'positive':self.positive,'negative':self.negative,'unresolved':self.unresolved,
                'counts':{'positive':len(self.positive),'negative':len(self.negative),
                          'unresolved':len(self.unresolved)}}
