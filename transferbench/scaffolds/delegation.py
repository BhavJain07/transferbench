"""Ready-to-run delegation adapter with the common name/run/as_solver interface."""

from transferbench.scaffolds import get_scaffold

delegate_agent = get_scaffold("delegate_agent")

__all__ = ["delegate_agent"]
