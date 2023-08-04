import dataclasses


@dataclasses.dataclass
class ResnetConf:
    n_layers: int
    n_embed: int
    filter_approximation_rank: int
    linear_size_multiplier: int
    dropout: float
    ln_eps: float
    nll_epsilon: float
