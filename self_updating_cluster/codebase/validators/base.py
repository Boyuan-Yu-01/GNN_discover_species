"""Stable validation contract: evidence is separate from training-label policy."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from ..storage import Storage


@dataclass
class ValidationResult:
    structure_id: str
    status: str
    decision: str
    reason: str
    validator: str
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ('completed','pending','failed'):
            raise ValueError('Invalid validation status')
        if self.decision not in ('accepted','rejected','unknown'):
            raise ValueError('Invalid validation decision')
        if self.status != 'completed' and self.decision != 'unknown':
            raise ValueError('Pending/failed validation cannot accept or reject a molecule')
        if not self.structure_id or not self.validator or not isinstance(self.details,dict):
            raise ValueError('Validation needs a structure ID, validator name and details dictionary')
        Storage.fingerprint(asdict(self))  # Reject non-JSON or non-finite result values.

    def to_dict(self):
        return asdict(self)


class Validator(ABC):
    """Implement submit/collect without coupling generation or training to an engine.

    A submitter must use the stable workdir/job ID to avoid submitting a duplicate
    external job if the process stops between submission and saving its result.
    Result details should include a recoverable job ID for pending work.
    """
    name = 'base'
    version = 1

    @abstractmethod
    def configuration(self):
        """Return JSON settings including reference/engine version and decision rules."""

    def fingerprint(self):
        return Storage.fingerprint({'name':self.name,'version':self.version,
                                    'settings':self.configuration()})

    def cache_payload(self, record):
        # General validators may depend on geometry; do not discard it here.
        return record['graph']

    @abstractmethod
    def submit(self, record, workdir):
        """Return ValidationResult, potentially with status='pending'."""

    def collect(self, record, previous, workdir):
        if previous.status == 'pending':
            raise NotImplementedError('This validator must implement collection of pending jobs')
        return previous


class ValidationCache:
    """Persist every result immediately; reuse only matching validator/input keys."""

    def __init__(self, path, validator):
        self.path=Path(path);self.validator=validator
        self.data=Storage.read(path) if self.path.exists() else {}

    def validate(self, record, workdir):
        signature=self.validator.fingerprint()
        key=Storage.fingerprint({'validator':signature,'input':self.validator.cache_payload(record)})
        folder=Path(workdir)/key
        folder.mkdir(parents=True,exist_ok=True)
        if key in self.data:
            previous=ValidationResult(**{**self.data[key], 'structure_id':record['structure_id']})
            result=self.validator.collect(record,previous,folder) if previous.status=='pending' else previous
        else:
            result=self.validator.submit(record,folder)
        if not isinstance(result,ValidationResult) or result.structure_id != record['structure_id']:
            raise ValueError('Validator returned the wrong result type or structure ID')
        if self.validator.fingerprint()!=signature:
            raise RuntimeError('Validator settings/reference changed during validation')
        self.data[key]=result.to_dict();Storage.write(self.path,self.data)
        return result
