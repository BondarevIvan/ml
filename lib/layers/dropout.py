import torch
import torch.nn as nn
from torchtyping import TensorType as TT


class Dropout:
    def __init__(self, probability: float = 0.0):
        assert probability >= 0.0
        self._probability = probability

    def forward(self, x: TT) -> TT:
        capacity = x.view(-1).shape[0]
        return (x.view(-1) * (torch.rand(capacity) > self._probability)).view(x.shape) / (1 - self._probability)

    def parameters(self) -> list:
        return []

    def __call__(self, x: TT) -> TT:
        return self.forward(x)
