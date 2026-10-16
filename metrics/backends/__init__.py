"""Selectable graph backends for node centrality computation."""

from importlib import import_module

from .base import GraphBackend, NodeCentralities

__all__ = ["BACKEND_NAMES", "GraphBackend", "NodeCentralities", "get_backend"]

BACKEND_NAMES: tuple[str, ...] = ("networkx", "igraph", "graph_tool")

# name -> (module path, class name). The module is imported only on demand, so a
# missing optional library (graph_tool) never breaks the rest of the registry.
_BACKENDS: dict[str, tuple[str, str]] = {
    "networkx": ("metrics.backends.networkx_backend", "NetworkXBackend"),
    "igraph": ("metrics.backends.igraph_backend", "IGraphBackend"),
    "graph_tool": ("metrics.backends.graph_tool_backend", "GraphToolBackend"),
}

_INSTALL_HINTS: dict[str, str] = {
    "networkx": "install it with 'pip install networkx'",
    "igraph": "install it with 'pip install igraph'",
    "graph_tool": (
        "install it from conda-forge (e.g. 'pixi add graph-tool'); "
        "graph_tool is not available on PyPI"
    ),
}


def get_backend(name: str) -> GraphBackend:
    """Instantiate the centrality backend registered under ``name``.

    Args:
        name: One of :data:`BACKEND_NAMES`.

    Returns:
        A ready-to-use :class:`~metrics.backends.base.GraphBackend` instance.

    Raises:
        ValueError: If ``name`` is not a known backend.
        ImportError: If the backend's library is not installed.
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown metrics backend '{name}'. "
            f"Available backends: {', '.join(BACKEND_NAMES)}"
        )

    module_path, class_name = _BACKENDS[name]
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"The '{name}' metrics backend is unavailable ({exc}). "
            f"To use it, {_INSTALL_HINTS[name]}."
        ) from exc

    backend_class: type[GraphBackend] = getattr(module, class_name)
    return backend_class()
