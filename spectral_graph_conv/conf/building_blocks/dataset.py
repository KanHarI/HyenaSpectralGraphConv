import dataclasses


@dataclasses.dataclass
class DatasetConf:
    n_nodes: int
    vocab_size: int
