import dataclasses

import torch


@dataclasses.dataclass
class DtypeConf:
    _dtype: str

    @property
    def dtype(self) -> torch.dtype:
        match self._dtype:
            case "bfloat16":
                return torch.bfloat16
            case "float16":
                return torch.float16
            case "float32":
                return torch.float32
            case "float64":
                return torch.float64
            case "complex32":
                return torch.complex32
            case "complex64":
                return torch.complex64
            case "complex128":
                return torch.complex128
            case _:
                raise ValueError(f"Unknown dtype: {self._dtype}")
