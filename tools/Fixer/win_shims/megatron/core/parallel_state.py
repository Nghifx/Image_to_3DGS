def is_initialized():
    return False


def _zero(*a, **k):
    return 0


def _one(*a, **k):
    return 1


def _none(*a, **k):
    return None


get_data_parallel_rank = get_context_parallel_rank = get_pipeline_model_parallel_rank = get_tensor_model_parallel_rank = _zero
get_data_parallel_world_size = get_context_parallel_world_size = get_tensor_model_parallel_world_size = _one
get_data_parallel_group = get_context_parallel_group = _none
