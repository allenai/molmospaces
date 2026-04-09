"""Arena Object wrapper — uses stock Object (collision_group=0) so pick objects and scene background share the same group and PhysX contacts resolve correctly."""

from __future__ import annotations

try:
    from isaaclab_arena.assets.object import Object

    _ARENA_AVAILABLE = True
except ImportError:
    _ARENA_AVAILABLE = False
    Object = None  # type: ignore[misc, assignment]


def get_arena_object_class():
    """Return the stock Arena ``Object`` class (collision_group=0)."""
    if not _ARENA_AVAILABLE or Object is None:
        raise ImportError("isaaclab_arena / isaaclab is required.")
    return Object
