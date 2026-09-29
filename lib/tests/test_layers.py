from lib.layers import (
    Embedding,
    Linear,
    Dropout,
    Multihead
)
import torch.nn as nn
import torch


def test_embedding():
    class TestEmbedding(Embedding):
        def __init__(self, vocab_size, emb_size):
            super().__init__(vocab_size, emb_size)
            # Для примера (3, 6)
            #  0  1  2  3  4  5
            #  6  7  8  9 10 11
            # 12 13 14 15 16 17
            self._weights = torch.arange(vocab_size * emb_size).view(vocab_size, emb_size)

    embedder = TestEmbedding(3, 6)
    inp = torch.tensor([[0, 1, 2]])
    out = embedder(inp)
    assert (out[0][0] == torch.arange(6)).all()
    assert (out[0][1] == 6 + torch.arange(6)).all()
    assert (out[0][2] == 12 + torch.arange(6)).all()


def test_linear():
    class TestLinear(Linear):
        def __init__(self, inp_size, out_size, bias = True):
            super().__init__(inp_size, out_size, bias)
            # Для примера (3, 6)
            #  0  1  2  3  4  5
            #  6  7  8  9 10 11
            # 12 13 14 15 16 17
            self._weights = torch.arange(inp_size * out_size).view(inp_size, out_size)

    linear = TestLinear(3, 6, bias=False)
    out = linear(torch.tensor([
        [1, 1, 2]
    ]))
    assert (out[0] == torch.tensor([30, 34, 38, 42, 46, 50])).all()


def test_dropout():
    layer = Dropout()
    inp = torch.arange(3)
    assert (inp == layer(inp)).all()

    layer = Dropout(probability=0.5)
    inp = torch.ones(size=(1024, ))
    out = layer(inp)
    # -5sigma; 5sigma. 1 фейл на 3'500'000 запусков
    assert (out.sum() // 2 in range(512-80, 512+81))


def test_attention():
    vocab_size = 100
    emb_size = 24
    T = 40
    embedder = Embedding(vocab_size=vocab_size, emb_size=emb_size)
    multihead = Multihead(embed_dim=emb_size, num_heads=4, dropout=0.0)
    inp1 = [(i * i) % vocab_size for i in range(T)]
    inp2 = [(j + 1) * j % vocab_size for j in range(T)]

    inp = torch.tensor([inp1, inp2])
    out = multihead(embedder(inp))
    out1 = multihead(embedder(torch.tensor([inp1])))
    out2 = multihead(embedder(torch.tensor([inp2])))

    assert out1.allclose(out[0])
    assert out2.allclose(out[1])


def test_attention_is_causal():
    # изменение будущих токенов не должно влиять на выходы в прошлых позициях
    torch.manual_seed(0)
    embedder = Embedding(vocab_size=50, emb_size=16)
    multihead = Multihead(embed_dim=16, num_heads=4)
    a = torch.randint(0, 50, (1, 10))
    b = a.clone()
    b[0, 6:] = (b[0, 6:] + 1) % 50

    with torch.no_grad():
        out_a = multihead(embedder(a))
        out_b = multihead(embedder(b))

    assert out_a[0, :6].allclose(out_b[0, :6])
    assert not out_a[0, 6:].allclose(out_b[0, 6:])
