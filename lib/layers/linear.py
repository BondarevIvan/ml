import torch
from torchtyping import TensorType as TT


class Linear:
    def __init__(self, inp_size: int, out_size: int, bias: bool = True):
        self._weights = torch.randn((inp_size, out_size)) / (inp_size**0.5)
        self._bias = torch.zeros(out_size) if bias else None
        # без requires_grad оптимизатор не получит градиентов
        self._weights.requires_grad_()
        if self._bias is not None:
            self._bias.requires_grad_()

    def forward(self, x: TT["B", "INP"]) -> TT["B", "OUT"]:
        return x @ self._weights + (
            self._bias if self._bias is not None else 0
        )

    def parameters(self):
        return [p for p in (self._weights, self._bias) if p is not None]

    def __call__(self, x: TT["B", "T"]) -> TT["B", "T", "C"]:
        return self.forward(x)
