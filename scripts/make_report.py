"""Compatibility entry point for self-contained HTML reporting."""

import sys

from transferbench.cli import app

if __name__ == "__main__":
    app(["report", *sys.argv[1:]])
