"""Explicit commands keep data audit separate from model selection and final test."""
import argparse

from .data import build_spot, download_spot


def main():
    parser = argparse.ArgumentParser(description="Binance crypto regime research")
    parser.add_argument("command", choices=["data"])
    args = parser.parse_args()
    if args.command == "data":
        _, errors = download_spot()
        build_spot()
        if errors:
            raise RuntimeError(f"{len(errors)} archive downloads failed; see manifest")


if __name__ == "__main__":
    main()
