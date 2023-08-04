import dataclasses


@dataclasses.dataclass
class ExperimentConf:
    wandb_log: bool
    project_name: str
    run_name: str
    log_interval: int
    eval_interval: int
    eval_iters: int
