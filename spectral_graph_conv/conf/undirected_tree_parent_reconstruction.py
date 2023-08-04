import dataclasses

from spectral_graph_conv.conf.building_blocks.activation import ActivationConf
from spectral_graph_conv.conf.building_blocks.dataset import DatasetConf
from spectral_graph_conv.conf.building_blocks.dtype import DtypeConf
from spectral_graph_conv.conf.building_blocks.experiment import ExperimentConf
from spectral_graph_conv.conf.building_blocks.optimizer import OptimizerConf
from spectral_graph_conv.conf.building_blocks.resnet import ResnetConf


@dataclasses.dataclass
class UndirectedTreeParentReconstructionConf:
    experiment: ExperimentConf
    resnet: ResnetConf
    dataset: DatasetConf
    activation_conf: ActivationConf
    optimizer: OptimizerConf
    dtype_conf: DtypeConf
    device: str
    embedder_sigma: float
