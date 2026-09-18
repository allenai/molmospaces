"""Pre-compile MjSpec transforms shared by the native scene build and the
FetchMan port (`fetchman/scene_g1ms.py`).

These are the five things gold's `Scene` does to an MJCF between loading it and
compiling it. They lived only in the port, which is why a natively-built model
of the same house was a structurally different system (nq 1642 vs gold's 68).
Each is exposed as a standalone function here so both stacks apply the same
bytes, and as a config flag on `BaseMujocoTaskSamplerConfig` so a scene can opt
out.

`freeze_non_mobile_bodies` is the load-bearing one: everything else is a small
constant-size edit, but stripping joints turns a whole house of dynamic props
into static geometry and is what makes the two state vectors comparable.
"""

import re

import mujoco
import numpy as np

from molmo_spaces.utils.constants.object_constants import is_pickup_type

# Anti-tumble damping ceiling, per DOF. A flat 1.0 NaN-explodes gram-scale
# objects, so the cap is scaled by the DOF's own inertia (dof_M0).
FREEJOINT_DAMPING_LAMBDA_MAX = 20.0


def freeze_non_mobile_bodies(
    spec,
    metadata: dict,
    mobile_regex: str | None,
    articulated_regex: str | None = None,
    robot_prefix: str = "",
) -> int:
    """Strip joints + collisions from bodies that are not sampling candidates,
    turning them into inert static geometry. Returns the number of bodies frozen.

    `mobile_regex=None` keeps every body mobile -- the no-op default, and what
    every scene that has not opted in gets.

    A body survives if its name or THOR category matches `mobile_regex` AND it
    is either a THOR pickup type or explicitly marked non-static in metadata
    (objaverse assets carry `is_static: False`). Subtrees rooted at a body
    matching `articulated_regex` are left completely intact, so drawers and
    doors keep the hinge/slide joints the open/close tasks actuate.

    Robot subtrees (`robot_prefix`) are always descended into but never frozen.
    """
    if mobile_regex is None:
        return 0

    pattern = re.compile(mobile_regex, re.IGNORECASE)
    art_pattern = re.compile(articulated_regex, re.IGNORECASE) if articulated_regex else None
    frozen = 0

    def _is_articulated_root(child) -> bool:
        if art_pattern is None or not child.name:
            return False
        meta = metadata.get(child.name, {})
        cat = meta.get("category", child.name.split("_")[0])
        if not (art_pattern.search(child.name) or art_pattern.search(cat)):
            return False
        # Confirm there are actual non-free joints in the subtree per metadata.
        jmap = (meta.get("name_map") or {}).get("joints") or {}
        return any("free" not in thor_jname.lower() for thor_jname in jmap.values())

    def _strip(parent, in_articulated: bool = False) -> None:
        nonlocal frozen
        for child in parent.bodies:
            if robot_prefix and child.name and child.name.startswith(robot_prefix):
                _strip(child)
                continue

            if in_articulated or _is_articulated_root(child):
                _strip(child, in_articulated=True)
                continue

            keep = False
            meta = metadata.get(child.name, {})
            cat = meta.get("category", child.name.split("_")[0] if child.name else "")
            name_match = bool(pattern.search(child.name)) if child.name else False
            is_static_meta = meta.get("is_static", None)
            if is_static_meta is False and name_match:
                keep = True
            elif is_pickup_type(cat) and name_match:
                keep = True

            if not keep:
                joints = list(child.joints)
                for jnt in joints:
                    spec.delete(jnt)
                if joints:
                    frozen += 1
                    for geom in child.geoms:
                        geom.contype = 0
                        geom.conaffinity = 0

            _strip(child)

    _strip(spec.worldbody)
    return frozen


# The parametric "jaw" probe's dimensions, gold's own (fetchman/scene_g1ms.py).
# Only `length` differs from ObjectManipulationPlannerPolicyConfig's default
# (0.03 vs 0.05).
GOLD_PROBE_WIDTH = 0.08
GOLD_PROBE_LENGTH = 0.03
GOLD_PROBE_HEIGHT = 0.01
GOLD_PROBE_BASE_POS = (0.0, 0.0, -0.04)

# The two probe shapes. "jaw" is the parametric three-cylinder open jaw, built
# here and scattered `count` at a time over candidate grasps. "gripper_xml" is a
# robot's own gripper model, attached from the MJCF that ships beside its
# robot XML -- one body, articulated fingers, the real gripper envelope.
PROBE_SHAPE_JAW = "jaw"
PROBE_SHAPE_GRIPPER_XML = "gripper_xml"

# The gripper_xml probe's names are fixed by the MJCF it is attached from, so
# they are constants rather than functions of an index -- there is only ever one.
GRIPPER_PROBE_BODY_NAME = "gripper_probe"
GRIPPER_PROBE_JOINT_NAME = "gripper_probe_joint"
GRIPPER_PROBE_FINGER_JOINT_NAMES = ("gripper_probe_joint_a", "gripper_probe_joint_b")
# Finger slide targets that hold the attached gripper open, per its "open" key.
GRIPPER_PROBE_FINGER_OPEN_QPOS = (0.04, -0.04)


def grasp_probe_body_name(i: int) -> str:
    """The one place jaw probe bodies are named. Every producer and consumer --
    the scene build, `get_noncolliding_grasp_mask`, the FetchMan port -- goes
    through this, so there is exactly one naming scheme and no second set of
    probes under a parallel name."""
    return f"grasp_probe_{i}"


def grasp_probe_joint_name(i: int) -> str:
    return f"grasp_probe_joint_{i}"


def is_grasp_probe_body_name(name: str) -> bool:
    """True for any probe body of either shape -- what scans over the scene use
    to skip probes without spelling the names out again."""
    return name == GRIPPER_PROBE_BODY_NAME or bool(re.fullmatch(r"grasp_probe_\d+", name))


def add_grasp_probes(
    spec,
    count: int = 1,
    *,
    shape: str = PROBE_SHAPE_JAW,
    width: float = GOLD_PROBE_WIDTH,
    length: float = GOLD_PROBE_LENGTH,
    height: float = GOLD_PROBE_HEIGHT,
    base_pos=GOLD_PROBE_BASE_POS,
    rgba=(1, 0, 0, 0.6),
    group: int = 0,
    gripper_xml=None,
) -> list:
    """Add `count` gripper stand-ins on freejoints, parked at z=10 and
    gravity-compensated so they stay put. Returns the probe bodies.

    This is the ONLY place probe bodies are created, for either shape. One jaw
    probe is gold's; a batch is what `get_noncolliding_grasp_mask` scatters over
    candidate grasps to test many per collision pass. They are the same body
    either way -- see ObjectManipulationPlannerPolicyConfig.grasp_collision_batch_size
    for the speed trade-off that sets `count`.

    `shape` picks the geometry:

    * `PROBE_SHAPE_JAW` -- the parametric three-cylinder open jaw built from
      `width`/`length`/`height`/`base_pos`. Cheap, robot-agnostic, `count` of them.
    * `PROBE_SHAPE_GRIPPER_XML` -- the robot's own gripper model, attached from
      `gripper_xml`. G1 ships one; it is a different envelope (box pads plus
      sliding fingers) and is what pick_planner_policy_g1 drives for the final
      pre-grasp clearance check. Its names are fixed by the MJCF, so only
      `count=1` is possible; a missing file is a no-op, which is how robots
      without a gripper model skip it.

    `contype=0` with `conaffinity=0b1111` makes a probe a pure sensor: scene
    geometry registers contact against it, but it never pushes anything. The
    gripper_xml probe sets the same flags in its MJCF.
    """
    if shape == PROBE_SHAPE_GRIPPER_XML:
        if gripper_xml is None or not gripper_xml.exists():
            return []
        if count != 1:
            raise ValueError(
                f"{PROBE_SHAPE_GRIPPER_XML} probes are named by their MJCF, so only "
                f"one can be attached (got count={count})"
            )
        gprobe_spec = mujoco.MjSpec.from_file(str(gripper_xml))
        frame = spec.worldbody.add_frame()
        return [frame.attach_body(gprobe_spec.worldbody.first_body(), "", "")]

    if shape != PROBE_SHAPE_JAW:
        raise ValueError(f"unknown grasp probe shape {shape!r}")

    bodies = []
    for i in range(count):
        probe = spec.worldbody.add_body(name=grasp_probe_body_name(i), pos=[0, 0, 10], gravcomp=1)
        probe.add_freejoint(name=grasp_probe_joint_name(i))

        _kw = dict(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            rgba=list(rgba),
            group=group,
            contype=0,
            conaffinity=0b1111,
        )
        hw = width / 2
        for fromto in [
            ([0, -hw, 0], [0, hw, 0]),
            ([0, -hw, 0], [0, -hw, length]),
            ([0, hw, 0], [0, hw, length]),
        ]:
            g = probe.add_geom(**_kw)
            g.size[0] = height / 2
            g.fromto[:3] = np.array(fromto[0]) + np.asarray(base_pos, dtype=np.float64)
            g.fromto[3:] = np.array(fromto[1]) + np.asarray(base_pos, dtype=np.float64)
        bodies.append(probe)

    return bodies


def add_base_weld(spec, robot_prefix: str = "", body: str = "pelvis") -> bool:
    """Add the inactive `pelvis_weld` equality that locks the robot base to the
    world. Returns False (a no-op) when the robot has no such body, which is
    how every non-humanoid robot skips it.

    Left inactive: the sampler activates it only while posing the robot.
    """
    target = f"{robot_prefix}{body}"
    if not any(b.name == target for b in spec.bodies):
        return False
    weld = spec.add_equality()
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.name = f"{body}_weld"
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1 = target
    weld.name2 = ""
    weld.active = False
    weld.solref = [0.0002, 1.0]
    weld.solimp = [0.999, 0.9999, 0.0001, 0.5, 2.0]
    return True


def enable_sleep(spec) -> None:
    """Turn on MuJoCo's island sleep flag. With most of a house frozen this is
    most of the speedup; on a fully dynamic scene it does much less."""
    spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)


def cap_freejoint_damping(model, lambda_max: float = FREEJOINT_DAMPING_LAMBDA_MAX) -> None:
    """Post-compile: damp every free joint's 6 DOFs, capped by the DOF's own
    inertia. Keeps loose props from tumbling forever without exploding the
    light ones. Mutates `model` in place."""
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            d0 = int(model.jnt_dofadr[jid])
            for k in range(6):
                model.dof_damping[d0 + k] = min(1.0, lambda_max * float(model.dof_M0[d0 + k]))
