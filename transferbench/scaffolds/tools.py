"""Ready-to-run tool adapter with the common name/run/as_solver interface."""

from transferbench.scaffolds import get_scaffold

tool_agent = get_scaffold("tool_agent")

__all__ = ["tool_agent"]
