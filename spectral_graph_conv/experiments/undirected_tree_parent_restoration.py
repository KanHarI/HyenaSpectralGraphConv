from typing import Any

import dacite
import hydra

from spectral_graph_conv.conf.undirected_tree_parent_reconstruction import (
    UndirectedTreeParentReconstructionConf,
)


@hydra.main(
    config_path="../conf",
    config_name="undirected_tree_parent_reconstruction",
    version_base=None,
)
def main(hydra_cfg: dict[Any, Any]) -> int:
    config: UndirectedTreeParentReconstructionConf = dacite.from_dict(
        data_class=UndirectedTreeParentReconstructionConf, data=hydra_cfg
    )
    x = 1
    return 0


if __name__ == "__main__":
    main()
