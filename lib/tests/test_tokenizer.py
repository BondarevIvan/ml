from lib.tokenizers import SymbolTokenizer


def test_symbol_tokenizer():
    tokenizer = SymbolTokenizer("hello world")

    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"
    assert tokenizer.decode(tokenizer.encode("a")) == "<UNK>"
