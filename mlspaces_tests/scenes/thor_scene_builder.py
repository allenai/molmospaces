"""MjSpec scene building for the iTHOR grasp/articulation scripts.

Replaces `molmo_spaces.editor.thor_model_editor.ThorMjModelEditor`, which these
scripts were written against and which has never existed in this repo. Same
call sequence, same resulting XML, but built with `mujoco.MjSpec` the way the
rest of the repo constructs scenes -- `spec.worldbody.add_frame(pos, quat)` then
`frame.attach_body(root, prefix, "")`, as in
`molmo_spaces/robots/abstract.py::add_robot_to_scene` and
`fetchman/scene_g1ms.py`.

The compiler/option/size defaults below are not invented: they are what the
scenes the resource manager ships already carry, so `set_compiler`,
`set_options` and `set_size(5000)` reproduce a scene's own header rather than
changing it. See `assets/scenes/ithor/FloorPlan8_physics.xml`.
"""

from pathlib import Path

import mujoco

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

ROBOT_PREFIX = "robot_0/"

# Mirrors `<option impratio="10" gravity="0 0 -9.8" integrator="implicitfast"
# cone="elliptic" jacobian="sparse" noslip_iterations="4">` plus its
# `<flag energy="enable" multiccd="enable" contact="enable" warmstart="enable"/>`.
DEFAULT_GRAVITY = (0.0, 0.0, -9.8)
DEFAULT_IMPRATIO = 10.0
DEFAULT_NOSLIP_ITERATIONS = 4


class ThorSceneBuilder:
    """Load a THOR scene, insert a robot and a mocap target, write it back out.

    Used as a fixed sequence by every caller:

        builder = ThorSceneBuilder.from_xml_path(scene)
        builder.set_options()
        builder.set_size(size=5000)
        builder.set_compiler()
        builder.add_robot(xml_path=robot_path, pos=pos, quat=[1, 0, 0, 0])
        builder.add_mocap_body(name="target_ee_pose", gripper_weld=True,
                               gripper_name="robot_0/", pos=pos, quat=[1, 0, 0, 0])
        builder.save_xml(save_path=out)
    """

    def __init__(self, spec: mujoco.MjSpec, source_dir: Path | None = None):
        self.spec = spec
        # Where the scene's meshes and textures actually live. MjSpec resolves
        # them against the file it was loaded from, but `to_xml` emits the
        # paths as they were written -- relative -- so a scene saved anywhere
        # but beside its source would lose them.
        self.source_dir = source_dir

    @classmethod
    def from_xml_path(cls, xml_path) -> "ThorSceneBuilder":
        xml_path = Path(xml_path)
        spec = mujoco.MjSpec.from_file(str(xml_path))
        _absolutise_assets(spec, xml_path.parent.resolve())
        return cls(spec, source_dir=xml_path.parent.resolve())

    def set_compiler(self) -> "ThorSceneBuilder":
        """`<compiler angle="radian" autolimits="true" boundmass="0"
        balanceinertia="true"/>`. `degree=False` is MuJoCo's spelling of
        `angle="radian"`."""
        compiler = self.spec.compiler
        compiler.degree = False
        compiler.autolimits = True
        compiler.boundmass = 0.0
        compiler.balanceinertia = True
        return self

    def set_options(self) -> "ThorSceneBuilder":
        """The scenes' own `<option>`: elliptic cone on an implicitfast
        integrator with a sparse jacobian, plus the energy flag the scripts
        read back as a settling signal."""
        option = self.spec.option
        option.gravity = list(DEFAULT_GRAVITY)
        option.impratio = DEFAULT_IMPRATIO
        option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
        option.noslip_iterations = DEFAULT_NOSLIP_ITERATIONS
        option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
        # contact and warmstart are on unless disabled, so clear rather than set.
        option.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        option.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        return self

    def set_size(self, size: int = 5000) -> "ThorSceneBuilder":
        """`<size njmax=... nconmax=.../>`. A THOR house with a gripper in it
        overruns MuJoCo's defaults, which is what this guards against."""
        self.spec.njmax = int(size)
        self.spec.nconmax = int(size)
        return self

    def add_robot(
        self, xml_path, pos, quat, prefix: str = ROBOT_PREFIX, float_base: bool = True
    ) -> "ThorSceneBuilder":
        """Attach a robot/gripper MJCF at `pos`/`quat` under `prefix`.

        A frame carries the pose and the prefix carries the namespace, as in
        `molmo_spaces/robots/abstract.py::add_robot_to_scene` -- but this
        attaches the whole spec rather than one body subtree. The floating
        grippers these scripts use ship their own `target_ee_pose` mocap and the
        weld that drives it, and `attach_body` would take only the first body
        subtree and leave both behind. `MjSpec.attach` brings the equality
        constraints and actuators across too.
        """
        xml_path = resolve_robot_xml(xml_path)
        robot_spec = mujoco.MjSpec.from_file(str(xml_path))
        # The robot's meshes live under its own directory, not the scene's, and
        # a saved scene has only one meshdir -- so each side's asset paths are
        # made absolute before the two specs are merged.
        _absolutise_assets(robot_spec, xml_path.parent.resolve())
        root_name = robot_spec.worldbody.first_body().name
        attach_frame = self.spec.worldbody.add_frame(pos=list(pos), quat=list(quat))
        self.spec.attach(robot_spec, prefix=prefix, frame=attach_frame)

        if float_base:
            self._ensure_floating_base(prefix)
        return self

    def _ensure_floating_base(self, prefix: str) -> None:
        """Give the attached gripper a free joint if it has none.

        `add_robot` here means "drop this gripper into the world and fly it",
        and a mocap weld can only move a body with DOFs. The `floating_*`
        grippers already carry one; the arm-mounted MJCFs
        (`franka_droid/robotiq_2f85_v4/2f85.xml`) do not, because they are
        normally bolted to a wrist.
        """
        if self._floating_base(prefix) is not None:
            return
        for body in self.spec.bodies:
            if body.name and body.name.startswith(prefix) and not list(body.joints):
                body.add_freejoint(name=f"{body.name}free_joint")
                return

    def has_body(self, name: str) -> bool:
        return any(b.name == name for b in self.spec.bodies)

    def add_mocap_body(
        self,
        name: str,
        pos,
        quat,
        gripper_weld: bool = False,
        gripper_name: str = ROBOT_PREFIX,
    ) -> "ThorSceneBuilder":
        """Add a mocap body the scripts drive through `data.mocap_pos/quat`, and
        weld the gripper's floating base to it.

        Both halves are conditional, because the floating grippers differ:
        `floating_rum` ships `target_ee_pose` *and* its weld, `floating_robotiq`
        ships the mocap but no weld. Adding a second mocap would leave
        `data.mocap_pos[0]` pointing at whichever MuJoCo ordered first, and a
        second weld would fight the first -- so each is added only if missing.
        """
        prefixed = f"{gripper_name}{name}"
        mocap_name = prefixed if self.has_body(prefixed) else name
        if mocap_name == name:
            self.spec.worldbody.add_body(name=name, mocap=True, pos=list(pos), quat=list(quat))

        if gripper_weld:
            target = self._floating_base(gripper_name)
            if target is None:
                raise ValueError(f"No floating base under prefix {gripper_name!r} to weld to")
            if not self._has_weld(mocap_name, target):
                weld = self.spec.add_equality()
                weld.type = mujoco.mjtEq.mjEQ_WELD
                weld.name = f"{name}_weld"
                weld.objtype = mujoco.mjtObj.mjOBJ_BODY
                weld.name1 = mocap_name
                weld.name2 = target
                weld.active = True
                # anchor(3), relpose pos(3), relpose quat(4), torquescale(1).
                # `add_equality` does not hand back a zeroed anchor -- it comes
                # out as [0, 1, 0], which offsets the weld by a metre and lets
                # the gripper sag away from the mocap instead of tracking it.
                weld.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                weld.solref = [0.02, 1.0]
        return self

    def _floating_base(self, prefix: str) -> str | None:
        """The gripper's free-floating root: the body under `prefix` carrying a
        free joint. That -- not simply the first body -- is what the mocap has
        to be welded to, since a body without a freejoint cannot be flown."""
        for body in self.spec.bodies:
            if not (body.name and body.name.startswith(prefix)):
                continue
            if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints):
                return body.name
        return None

    def _has_weld(self, body_a: str, body_b: str) -> bool:
        pair = {body_a, body_b}
        for eq in self.spec.equalities:
            if eq.type == mujoco.mjtEq.mjEQ_WELD and {eq.name1, eq.name2} == pair:
                return True
        return False

    def save_xml(self, save_path) -> str:
        """Compile (so the written XML is known-loadable) and write it out.

        Asset paths were made absolute on the way in, so the result loads from
        any directory -- not only from beside the scene it was built off.
        """
        save_path = Path(save_path)
        self.spec.compile()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(self.spec.to_xml())
        return str(save_path)


# Gripper directories that were renamed between when these scripts were written
# and the current asset tree. Only mappings confirmed against what the resource
# manager actually ships are listed; anything else resolves unchanged under
# ASSETS_DIR and will fail loudly if it is gone.
LEGACY_ROBOT_DIRS = {
    "rum_gripper": "floating_rum",
    "robotiq_2f85": "floating_robotiq",
}


def resolve_robot_xml(path) -> Path:
    """Turn a script's `assets/robots/<dir>/<file>.xml` into a real path.

    There is no repo-relative `assets/` in a checkout -- the robots live in the
    resource manager's tree, under `$MLSPACES_ASSETS_DIR` when set.
    """
    parts = Path(path).parts
    if "robots" in parts:
        parts = parts[parts.index("robots") :]
    rest = list(parts[1:])
    if rest and rest[0] in LEGACY_ROBOT_DIRS:
        renamed = LEGACY_ROBOT_DIRS[rest[0]]
        # floating_rum/floating_robotiq ship a single `model.xml`.
        rest = [renamed, "model.xml"]
    return ASSETS_DIR.joinpath("robots", *rest)


def _absolutise_assets(spec: mujoco.MjSpec, base_dir: Path) -> None:
    """Rewrite every mesh/texture path in `spec` to an absolute one under
    `base_dir`, honouring any meshdir/texturedir already set, then clear those
    directories.

    A scene and the robot attached into it come from different trees, and the
    saved XML carries a single meshdir/texturedir pair -- so relative paths from
    one side or the other would break. Absolute paths sidestep that.
    """
    meshdir = spec.compiler.meshdir or ""
    texturedir = spec.compiler.texturedir or ""

    def _resolve(file_name: str, subdir: str) -> str:
        path = Path(file_name)
        if path.is_absolute():
            return str(path)
        return str((base_dir / subdir / path).resolve())

    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = _resolve(mesh.file, meshdir)
    for texture in spec.textures:
        if texture.file:
            texture.file = _resolve(texture.file, texturedir)

    spec.compiler.meshdir = ""
    spec.compiler.texturedir = ""


# The scripts were written against this name.
ThorMjModelEditor = ThorSceneBuilder
