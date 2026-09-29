"""Future DFT extension point. No DFT calculations or decisions are implemented.

Choose an engine, geometry preparation, charge/spin handling, submission method
and scientific acceptance criteria before implementing submit() and collect().
Store the original and optimized structures separately in result details/artifacts.
A failed calculation must return failed/unknown, never rejected automatically.
"""
from .base import Validator


class DFTValidator(Validator):
    name='dft'

    def __init__(self, **settings):
        raise NotImplementedError('DFT validation is not implemented; engine and acceptance criteria must be specified')

    def configuration(self):
        raise NotImplementedError('DFT validation is not implemented')

    def submit(self, record, workdir):
        raise NotImplementedError('DFT validation is not implemented')
