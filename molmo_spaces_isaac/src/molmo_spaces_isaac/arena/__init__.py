"""Isaac Lab Arena integration for MolmoSpaces (proof of concept).

Compose Arena environments using THOR USD assets and MolmoSpaces benchmark tasks.
Requires isaaclab_arena to be installed (e.g. from source in Docker).

Episode-to-spec conversion is imported at load time (no Isaac/Omniverse deps).
Task and THOR asset helpers are lazy-loaded so --list_only works without SimulationApp.
"""

from molmo_spaces_isaac.arena.episode_to_arena import (
    ArenaEpisodeSpec,
    episode_dict_to_arena_spec,
)

__all__ = [
    "ArenaEpisodeSpec",
    "MolmoSpacesPickTask",
    "create_thor_object_for_arena",
    "episode_dict_to_arena_spec",
]


def __getattr__(name: str):
    """Lazy-load task and thor_asset so they are not imported until needed (they pull in isaaclab/omni)."""
    if name == "MolmoSpacesPickTask":
        from molmo_spaces_isaac.arena.molmospaces_pick_task import MolmoSpacesPickTask
        return MolmoSpacesPickTask
    if name == "create_thor_object_for_arena":
        from molmo_spaces_isaac.arena.thor_asset import create_thor_object_for_arena
        return create_thor_object_for_arena
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
