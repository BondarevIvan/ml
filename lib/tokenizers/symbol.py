class Tokenizer:

    def __init__(self, text: str):
        raise NotImplementedError()

    def encode(self, text: str):
        raise NotImplementedError()

    def decode(self, encoded: list[int]) -> str:
        raise NotImplementedError()


class SymbolTokenizer(Tokenizer):
    def __init__(self, text: str, pad: str = "<PAD>", unk: str = "<UNK>"):
        unique_symbols = sorted(list(set(text)))
        self._unk = len(unique_symbols)
        self._pad = len(unique_symbols) + 1
        self._itos = {
            ind: sym
            for ind, sym in enumerate(unique_symbols)
        } | {
            self._unk : "<UNK>",
            self._pad : "<PAD>"
        }
        self._stoi = {
            sym: ind
            for ind, sym in enumerate(unique_symbols)
        }

    @property
    def vocab_size(self) -> int:
        return len(self._itos)

    def encode(self, text: str) -> list[int]:
        return [self._stoi.get(el, self._unk) for el in text]

    def decode(self, encoded: list[int]) -> str:
        return "".join([self._itos.get(e, self._unk) for e in encoded])

    @property
    def pad_id(self) -> int:
        return self._pad

    @property
    def unk_id(self) -> int:
        return self._unk
