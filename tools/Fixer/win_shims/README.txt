Pure-PyTorch stand-ins so cosmos_predict2 imports on Windows without
transformer-engine / megatron-core (neither builds on Windows).
Only what Fixer's single-GPU inference touches is implemented. Put this dir on sys.path first.
