# ponytail: single-GPU stub; real megatron-core does not install on Windows.
import attrs
from . import parallel_state  # noqa: F401


@attrs.define
class ModelParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
