import torch
from . import attention, distributed, module  # noqa: F401


class RMSNorm(torch.nn.RMSNorm):
    # TE RMSNorm keeps a single "weight" param, same key as torch.nn.RMSNorm.
    def __init__(self, hidden_size, eps=1e-5, **kw):
        super().__init__(hidden_size, eps=eps)

    def forward(self, x):
        return torch.nn.functional.rms_norm(x.float(), self.normalized_shape, self.weight.float(), self.eps).to(x.dtype)
