"""
tokenizer.py -- Character-level tokenizer.

The simplest possible tokenizer: every unique character in the training
text becomes one token (e.g. "hi" -> [1, 2]). Real production LLMs use
subword tokenizers (BPE / SentencePiece) because they're far more
efficient, but character-level keeps this fully transparent, needs zero
extra dependencies, and is easy to verify is bug-free -- which matters
when you're building the whole stack yourself.

See the README for how to swap in a BPE tokenizer later.
"""


class CharTokenizer:
    def __init__(self, text=None, stoi=None, itos=None):
        """
        Build a fresh vocabulary from `text`, OR restore one from a saved
        `stoi`/`itos` mapping (used when loading a checkpoint for generation).
        """
        if text is not None:
            chars = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
        elif stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = itos
        else:
            raise ValueError("Provide either `text` or both `stoi` and `itos`.")

        self.vocab_size = len(self.stoi)

    def encode(self, s):
        """String -> list of integer token ids."""
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        """List of integer token ids -> string."""
        return "".join(self.itos[i] for i in ids)
