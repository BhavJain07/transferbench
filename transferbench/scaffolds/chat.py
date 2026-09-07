"""Ready-to-run chat adapters with the common name/run/as_solver interface."""

from transferbench.scaffolds import get_scaffold

chat_single = get_scaffold("chat_single")
chat_multi = get_scaffold("chat_multi")

__all__ = ["chat_single", "chat_multi"]
