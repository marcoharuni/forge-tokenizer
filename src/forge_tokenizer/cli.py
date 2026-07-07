"""Command line interface for the educational tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge_tokenizer.metrics import compression_ratio, fertility, roundtrip_accuracy
from forge_tokenizer.tokenizer import ForgeTokenizer


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def cmd_train(args) -> None:
    tokenizer = ForgeTokenizer().train(_read(args.input), num_merges=args.merges)
    tokenizer.save(args.output)
    print(f"saved tokenizer to {args.output}")
    print(f"vocab_size={tokenizer.vocab_size}")


def cmd_encode(args) -> None:
    tokenizer = ForgeTokenizer.load(args.tokenizer)
    ids = tokenizer.encode(args.text, add_special_tokens=args.special_tokens)
    print(",".join(str(idx) for idx in ids))


def cmd_decode(args) -> None:
    tokenizer = ForgeTokenizer.load(args.tokenizer)
    ids = [int(part.strip()) for part in args.ids.split(",") if part.strip()]
    print(tokenizer.decode(ids, skip_special_tokens=args.skip_special_tokens))


def cmd_metrics(args) -> None:
    tokenizer = ForgeTokenizer.load(args.tokenizer)
    text = _read(args.input)
    print(f"fertility={fertility(tokenizer, text):.4f}")
    print(f"compression_ratio={compression_ratio(tokenizer, text):.4f}")
    print(f"roundtrip_accuracy={roundtrip_accuracy(tokenizer, [text]):.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forge-tokenizer")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--merges", type=int, default=80)
    train.set_defaults(func=cmd_train)

    encode = sub.add_parser("encode")
    encode.add_argument("--tokenizer", required=True)
    encode.add_argument("--text", required=True)
    encode.add_argument("--special-tokens", action="store_true")
    encode.set_defaults(func=cmd_encode)

    decode = sub.add_parser("decode")
    decode.add_argument("--tokenizer", required=True)
    decode.add_argument("--ids", required=True)
    decode.add_argument("--skip-special-tokens", action="store_true")
    decode.set_defaults(func=cmd_decode)

    metrics = sub.add_parser("metrics")
    metrics.add_argument("--tokenizer", required=True)
    metrics.add_argument("--input", required=True)
    metrics.set_defaults(func=cmd_metrics)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
