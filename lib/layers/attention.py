from lib.layers import Linear, Dropout
from torchtyping import TensorType as TT
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope(x: TT["B", "T", "D"], base: float = 10000.0, offset: int = 0) -> TT["B", "T", "D"]:
    *_, T, D = x.shape
    assert D % 2 == 0, "RoPE требует чётную размерность головы"

    # θ_i = base^(-2i/D), i = 0..D/2-1
    inv_freq = base ** (-torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)  # (D/2,)
    pos = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)            # (T,)
    angles = pos[:, None] * inv_freq[None, :]                                               # (T, D/2)
    cos, sin = angles.cos(), angles.sin()

    x1 = x[..., 0::2].float()  # чётные координаты пары
    x2 = x[..., 1::2].float()  # нечётные

    # поворот каждой пары (x1, x2) на свой угол
    out = torch.stack((x1 * cos - x2 * sin,
                       x1 * sin + x2 * cos), dim=-1).flatten(-2)  # (..., T, D)
    return out.type_as(x)


class Head:
    def __init__(self, emb_dim: int, head_size: int, dropout: float = 0.0):
        self._q = Linear(emb_dim, head_size, bias=False)
        self._k = Linear(emb_dim, head_size, bias=False)
        self._v = Linear(emb_dim, head_size, bias=False)
        self._scale = 1.0 / head_size**0.5
        self._dropout = Dropout(dropout)

    def forward(self, x: TT["B", "T", "C"]) -> TT["B", "T", "C"]:
        q = _rope(self._q(x))
        k = _rope(self._k(x))
        v = self._v(x)

        scores = q @ k.transpose(-2, -1) * self._scale              # (B, T, T)

        # Causal mask: позиция t видит только токены 0..t.
        # Без неё при teacher forcing модель подсмотрит следующий токен (= ответ).
        T = x.shape[-2]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return self._dropout(weights @ v)

    def parameters(self) -> list:
        return self._q.parameters() + self._k.parameters() + self._v.parameters()

    def __call__(self, x: TT["B", "T", "C"]) -> TT["B", "T", "C"]:
        return self.forward(x)


class Multihead:
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        assert embed_dim % num_heads == 0
        self._head_size = embed_dim // num_heads
        self._heads = [Head(embed_dim, self._head_size, dropout) for _ in range(num_heads)]
        self._proj = Linear(embed_dim, embed_dim)
        self._dropout = Dropout(dropout)

    def forward(self, x: TT["B", "T", "C"]) -> TT["B", "T", "C"]:
        concats = torch.cat([
            head(x) for head in self._heads
        ], dim=-1)
        return self._dropout(self._proj(concats))

    def parameters(self) -> list:
        return sum([
            head.parameters()
            for head in self._heads
        ], start=[]) + self._proj.parameters()

    def __call__(self, x: TT["B", "T", "C"]) -> TT["B", "T", "C"]:
        return self.forward(x)
