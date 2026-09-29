import torch
from torchtyping import TensorType as TT


class Embedding:
    def __init__(self, vocab_size: int, emb_size: int):
        self._weights = torch.randn((vocab_size, emb_size)) / vocab_size**0.5
        self._weights.requires_grad_()

    def parameters(self):
        return [self._weights]

    def forward(self, x: TT["B", "T"]) -> TT["B", "T", "C"]:
        return self._weights[x]

    def __call__(self, x: TT["B", "T"]) -> TT["B", "T", "C"]:
        return self.forward(x)
