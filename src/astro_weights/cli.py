"""Command-line entry point."""

import argparse

from astro_weights import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="astro-weights")
    parser.add_argument(
        "--version", action="version", version=f"astro-weights {__version__}"
    )
    parser.parse_args(argv)
    return 0
