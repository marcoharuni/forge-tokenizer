"""High-level tokenizer built on the pure Python BPE implementation."""

from __future__ import annotations

from collections import Counter
import base64
import json
from pathlib import Path

from forge_tokenizer.bpe import build_vocab_from_merges, decode_tokens, encode_word, train_bpe
from forge_tokenizer.normalizer import normalize_text
from forge_tokenizer.pretokenizer import pretokenize


class ForgeTokenizer:
    """Readable educational byte-level BPE tokenizer."""

    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, byte_level: bool = True, lowercase: bool = False, normalization: str = "NFC"):
        self.byte_level = byte_level
        self.lowercase = lowercase
        self.normalization = normalization
        self.merges: list[tuple[bytes, bytes]] = []
        self.vocab, self.id_to_token_map = build_vocab_from_merges(self.merges, self.SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the current vocabulary."""

        return len(self.vocab)

    def _prepare(self, text: str) -> str:
        text = normalize_text(text, self.normalization)
        return text.lower() if self.lowercase else text

    def train(self, corpus_text: str, num_merges: int = 100) -> "ForgeTokenizer":
        """Train BPE merges from raw corpus text and return ``self``."""

        prepared = self._prepare(corpus_text)
        freqs = Counter(pretokenize(prepared))
        self.merges, self.training_stats = train_bpe(dict(freqs), num_merges, byte_level=self.byte_level)
        self.vocab, self.id_to_token_map = build_vocab_from_merges(self.merges, self.SPECIAL_TOKENS)
        return self

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        dropout: float = 0.0,
        seed: int | None = None,
    ) -> list[int]:
        """Encode text into token IDs."""

        prepared = self._prepare(text)
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.vocab["<bos>"])
        for index, piece in enumerate(pretokenize(prepared)):
            piece_seed = None if seed is None else seed + index
            for token in encode_word(piece, self.merges, self.byte_level, dropout=dropout, seed=piece_seed):
                ids.append(self.vocab.get(token, self.vocab["<unk>"]))
        if add_special_tokens:
            ids.append(self.vocab["<eos>"])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Decode token IDs back into text."""

        byte_tokens: list[bytes] = []
        pieces: list[str] = []
        for idx in ids:
            token = self.id_to_token_map.get(int(idx), "<unk>")
            if isinstance(token, str):
                if token in self.SPECIAL_TOKENS and skip_special_tokens:
                    continue
                if byte_tokens:
                    pieces.append(decode_tokens(byte_tokens))
                    byte_tokens = []
                if not (token in self.SPECIAL_TOKENS and skip_special_tokens):
                    pieces.append(token)
            else:
                byte_tokens.append(token)
        if byte_tokens:
            pieces.append(decode_tokens(byte_tokens))
        return "".join(pieces)

    def token_to_id(self, token) -> int | None:
        """Return an ID for a raw token object, or ``None`` if absent."""

        return self.vocab.get(token)

    def id_to_token(self, idx: int):
        """Return the token object for an ID, or ``None`` if absent."""

        return self.id_to_token_map.get(idx)

    def explain(self, text: str) -> list[dict]:
        """Return token IDs and decoded token pieces for a text string."""

        rows = []
        for token_id in self.encode(text):
            token = self.id_to_token(token_id)
            decoded = token if isinstance(token, str) else decode_tokens([token])
            display = token if isinstance(token, str) else base64.b16encode(token).decode("ascii")
            rows.append({"id": token_id, "token": display, "decoded": decoded})
        return rows

    def save(self, path: str | Path) -> None:
        """Save tokenizer configuration and merge rules to JSON."""

        payload = {
            "byte_level": self.byte_level,
            "lowercase": self.lowercase,
            "normalization": self.normalization,
            "special_tokens": self.SPECIAL_TOKENS,
            "merges": [
                [base64.b64encode(left).decode("ascii"), base64.b64encode(right).decode("ascii")]
                for left, right in self.merges
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ForgeTokenizer":
        """Load a tokenizer saved with :meth:`save`."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(
            byte_level=payload["byte_level"],
            lowercase=payload["lowercase"],
            normalization=payload["normalization"],
        )
        tokenizer.merges = [
            (base64.b64decode(left), base64.b64decode(right)) for left, right in payload["merges"]
        ]
        tokenizer.vocab, tokenizer.id_to_token_map = build_vocab_from_merges(
            tokenizer.merges, tokenizer.SPECIAL_TOKENS
        )
        return tokenizer
