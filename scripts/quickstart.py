"""Run the bundled offline quickstart with an HTML report."""

from transferbench.cli import app, bundled_config

if __name__ == "__main__":
    app(["matrix", str(bundled_config("quickstart.yaml")), "--report"])
