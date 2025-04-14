import os
import torch

import __main__
from dataclasses import dataclass
from colbert.utils.utils import timestamp

from .core_config import DefaultVal  # Ensure DefaultVal is defined in core_config.py or replace with its correct definition
from dataclasses import field


@dataclass
class RunSettings:
    """
        The defaults here have a special status in Run(), which initially calls assign_defaults(),
        so these aren't soft defaults in that specific context.
    """

    overwrite: bool = field(default_factory=lambda: False)

    root: str = field(default_factory=lambda: os.path.join(os.getcwd(), 'experiments'))
    experiment: str = field(default_factory=lambda: 'default')

    index_root: str = field(default_factory=lambda: None)
    name: str = field(default_factory=lambda: timestamp(daydir=True))

    rank: int = field(default_factory=lambda: 1)
    nranks: int = field(default_factory=lambda: 1)
    amp: bool = field(default_factory=lambda: True)

    total_visible_gpus = torch.cuda.device_count()
    # total_visible_gpus = 0
    gpus: int = field(default_factory=lambda: torch.cuda.device_count())

    @property
    def gpus_(self):
        value = self.gpus

        if isinstance(value, int):
            value = list(range(value))

        if isinstance(value, str):
            value = value.split(',')

        value = list(map(int, value))
        value = sorted(list(set(value)))

        assert all(device_idx in range(0, self.total_visible_gpus) for device_idx in value), value

        return value

    @property
    def index_root_(self):
        return self.index_root or os.path.join(self.root, self.experiment, 'indexes/')

    @property
    def script_name_(self):
        if '__file__' in dir(__main__):
            cwd = os.path.abspath(os.getcwd())
            script_path = os.path.abspath(__main__.__file__)
            root_path = os.path.abspath(self.root)

            if script_path.startswith(cwd):
                script_path = script_path[len(cwd):]

            else:
                try:
                    commonpath = os.path.commonpath([script_path, root_path])
                    script_path = script_path[len(commonpath):]
                except:
                    pass


            assert script_path.endswith('.py')
            script_name = script_path.replace('/', '.').strip('.')[:-3]

            assert len(script_name) > 0, (script_name, script_path, cwd)

            return script_name

        return 'none'

    @property
    def path_(self):
        return os.path.join(self.root, self.experiment, self.script_name_, self.name)

    @property
    def device_(self):
        return self.gpus_[self.rank % self.nranks]


@dataclass
class TokenizerSettings:
    query_token_id: str = field(default_factory=lambda: "[unused0]")
    doc_token_id: str = field(default_factory=lambda: "[unused1]")
    query_token: str = field(default_factory=lambda: "[Q]")
    doc_token: str = field(default_factory=lambda: "[D]")


@dataclass
class ResourceSettings:
    checkpoint: str = field(default_factory=lambda: None)
    triples: str = field(default_factory=lambda: None)
    collection: str = field(default_factory=lambda: None)
    queries: str = field(default_factory=lambda: None)
    index_name: str = field(default_factory=lambda: None)


@dataclass
class DocSettings:
    dim: int = field(default_factory=lambda: 128)
    doc_maxlen: int = field(default_factory=lambda: 220)
    mask_punctuation: bool = field(default_factory=lambda: True)


@dataclass
class QuerySettings:
    query_maxlen: int = field(default_factory=lambda: 32)
    attend_to_mask_tokens : bool = field(default_factory=lambda: False)
    interaction: str = field(default_factory=lambda: 'colbert')


@dataclass
class TrainingSettings:
    similarity: str = field(default_factory=lambda: 'cosine')

    bsize: int = field(default_factory=lambda: 32)

    accumsteps: int = field(default_factory=lambda: 1)

    lr: float = field(default_factory=lambda: 3e-06)

    maxsteps: int = field(default_factory=lambda: 500_000)

    save_every: int = field(default_factory=lambda: None)

    resume: bool = field(default_factory=lambda: False)

    ## NEW:
    warmup: int = field(default_factory=lambda: None)

    warmup_bert: int = field(default_factory=lambda: None)

    relu: bool = field(default_factory=lambda: False)

    nway: int = field(default_factory=lambda: 2)

    use_ib_negatives: bool = field(default_factory=lambda: False)

    reranker: bool = field(default_factory=lambda: False)

    distillation_alpha: float = field(default_factory=lambda: 1.0)

    ignore_scores: bool = field(default_factory=lambda: False)

    model_name: str = field(default_factory=lambda: "bert-base-uncased")

@dataclass
class IndexingSettings:
    index_path: str = field(default_factory=lambda: None)

    nbits: int = field(default_factory=lambda: 1)

    kmeans_niters: int = field(default_factory=lambda: 4)

    resume: bool = field(default_factory=lambda: False)

    @property
    def index_path_(self):
        return self.index_path or os.path.join(self.index_root_, self.index_name)

@dataclass
class SearchSettings:
    ncells: int = field(default_factory=lambda: None)
    centroid_score_threshold: float = field(default_factory=lambda: None)
    ndocs: int = field(default_factory=lambda: None)
