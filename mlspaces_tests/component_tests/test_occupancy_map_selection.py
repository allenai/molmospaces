"""Selecting between the two occupancy-map implementations.

`CPUMujocoEnv.get_occupancy_map` serves either ProcTHORMap/iTHORMap ("thor",
the default) or AABBMap ("aabb", from the FetchMan repo). They grid the same
scene differently, so a task or task sampler must be able to hold both --
and several agent radii of each -- with neither disturbing the other.

These exercise the selection/caching contract on a bare CPUMujocoEnv instance
with the two builders stubbed; building real maps needs a scene and a
renderer, and is covered by scripts/g1_molmo_port_comparison/.
"""

from collections import OrderedDict

import pytest

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.utils.scene_maps import (
    DEFAULT_OCCUPANCY_MAP_IMPL,
    OCCUPANCY_MAP_CACHE_SIZE,
    OCCUPANCY_MAP_IMPLS,
)


class _FakeMap:
    """Stands in for a built map; identity is what the tests compare."""

    def __init__(self, impl, agent_radius):
        self.impl = impl
        self.agent_radius = agent_radius


@pytest.fixture
def env(monkeypatch):
    """A CPUMujocoEnv with only the state get_occupancy_map touches, and both
    map builders replaced by counters."""
    env = object.__new__(CPUMujocoEnv)
    env._occupancy_maps = OrderedDict()
    env._mj_base_scene_path = "/scenes/house_0.xml"
    env.occupancy_map_impl = DEFAULT_OCCUPANCY_MAP_IMPL
    env.builds = []
    # __del__ -> close() runs on collection even for a bare instance.
    env._executor = None
    env._renderer = None
    env.object_managers = []

    def fake_get_thormap(agent_radius=0.35, px_per_m=200) -> _FakeMap:
        env.builds.append(("thor", agent_radius))
        return _FakeMap("thor", agent_radius)

    monkeypatch.setattr(CPUMujocoEnv, "get_thormap", staticmethod(fake_get_thormap))

    class FakeAABBMap:
        @staticmethod
        def from_model_path(xml_path, agent_radius=0.15, px_per_m=200) -> _FakeMap:
            env.builds.append(("aabb", agent_radius))
            return _FakeMap("aabb", agent_radius)

    import molmo_spaces.utils.aabb_map as aabb_module

    monkeypatch.setattr(aabb_module, "AABBMap", FakeAABBMap)
    return env


def test_defaults_to_thor(env):
    assert env.get_occupancy_map().impl == "thor"
    assert DEFAULT_OCCUPANCY_MAP_IMPL == "thor"


def test_env_impl_is_honoured(env):
    env.occupancy_map_impl = "aabb"
    assert env.get_occupancy_map().impl == "aabb"


def test_per_call_impl_overrides_the_env(env):
    env.occupancy_map_impl = "aabb"
    assert env.get_occupancy_map(impl="thor").impl == "thor"


def test_unknown_impl_raises(env):
    with pytest.raises(ValueError, match="unknown occupancy map impl"):
        env.get_occupancy_map(impl="quadtree")


def test_both_impls_coexist_without_evicting_each_other(env):
    thor = env.get_occupancy_map(agent_radius=0.35, impl="thor")
    aabb = env.get_occupancy_map(agent_radius=0.2, impl="aabb")

    # Both are served from cache on re-request -- no rebuild, no eviction.
    assert env.get_occupancy_map(agent_radius=0.35, impl="thor") is thor
    assert env.get_occupancy_map(agent_radius=0.2, impl="aabb") is aabb
    assert env.builds == [("thor", 0.35), ("aabb", 0.2)]

    # And they really are different objects with different grids.
    assert thor is not aabb
    assert (thor.impl, aabb.impl) == ("thor", "aabb")


def test_radii_of_one_impl_do_not_evict_each_other(env):
    """The single-slot get_thormap cache used to re-render the scene whenever
    a second caller asked for a different radius (see G1PickPlannerPolicy.
    _nav_maps' warning)."""
    wide = env.get_occupancy_map(agent_radius=0.35, impl="thor")
    tight = env.get_occupancy_map(agent_radius=0.15, impl="thor")

    assert env.get_occupancy_map(agent_radius=0.35, impl="thor") is wide
    assert env.get_occupancy_map(agent_radius=0.15, impl="thor") is tight
    assert env.builds == [("thor", 0.35), ("thor", 0.15)]


def test_cache_is_bounded_oldest_first(env):
    first = env.get_occupancy_map(agent_radius=0.1, impl="thor")
    for i in range(OCCUPANCY_MAP_CACHE_SIZE):
        env.get_occupancy_map(agent_radius=0.2 + 0.1 * i, impl="thor")

    assert len(env._occupancy_maps) == OCCUPANCY_MAP_CACHE_SIZE
    # The oldest entry was dropped, so asking again rebuilds it.
    again = env.get_occupancy_map(agent_radius=0.1, impl="thor")
    assert again is not first


def test_scene_change_drops_every_cached_map(env):
    env.get_occupancy_map(impl="thor")
    env.get_occupancy_map(impl="aabb")
    assert len(env._occupancy_maps) == 2

    env._occupancy_maps = OrderedDict()  # what _initialize_with_model does
    env._mj_base_scene_path = "/scenes/house_1.xml"
    env.get_occupancy_map(impl="thor")
    assert len(env._occupancy_maps) == 1


def test_impls_advertised():
    assert set(OCCUPANCY_MAP_IMPLS) == {"thor", "aabb"}
