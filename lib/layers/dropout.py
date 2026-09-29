import torch
import torch.nn as nn
from torchtyping import TensorType as TT


class Dropout:
    def __init__(self, probability: float = 0.0):
        assert probability >= 0.0
        self._probability = probability

    def forward(self, x: TT) -> TT:
        if self._probability == 0.0:
            return x
        # маска создаётся на том же устройстве, что и x (cpu / cuda / mps)
        mask = torch.rand(x.shape, device=x.device) > self._probability
        return x * mask / (1 - self._probability)

    def parameters(self) -> list:
        return []

    def __call__(self, x: TT) -> TT:
        return self.forward(x)
