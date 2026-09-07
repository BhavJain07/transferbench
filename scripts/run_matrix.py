"""Compatibility entry point for the matrix CLI."""

import sys

from transferbench.cli import app

if __name__ == "__main__":
    app(["matrix", *sys.argv[1:]])
