import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import msgspec
import mujoco as mj
import numpy as np
import tyro
import usdex.core
from p_tqdm import p_uimap
from pxr import Gf, Kind, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from molmo_spaces_isaac import MOLMO_SPACES_ISAAC_BASE_DIR
from molmo_spaces_isaac.assets.utils.body import (
    convert_bodies_flatten,
    convert_bodies_flatten_collapsed,
)
from molmo_spaces_isaac.assets.utils.contacts import convert_contact_excludes
from molmo_spaces_isaac.assets.utils.data import (
    AssetGenMetadata,
    AssetParameters,
    BaseConversionData,
    Tokens,
)
from molmo_spaces_isaac.assets.utils.flatten import export_flatten
from molmo_spaces_isaac.assets.utils.material import convert_materials
from molmo_spaces_isaac.assets.utils.mesh import convert_meshes
from molmo_spaces_isaac.utils.prims import compute_bbox_size

ERRORS_FILENAME = "mjcf_usd_assets_conversion_errors.txt"
METADATA_FILENAME = "usd_assets_metadata.json"

PARAMETERS_FILE = MOLMO_SPACES_ISAAC_BASE_DIR / "resources" / "usd_assets_parameters.yaml"
ASSET_ID_TO_CATEGORY_FILE = (
    MOLMO_SPACES_ISAAC_BASE_DIR / "resources" / "asset_id_to_object_type.json"
)

FIX_ROTATION = R.from_rotvec([90, 0, 0], degrees=True).as_matrix()

SUFFIXES_TO_AVOID = ("_old", "_fix", "_upt", "_orig", "_sc", "_alt")

THOR_ASSETS_IDS_TO_SKIP = (
    "Light_Switch",
    "RoboTHOR_dresser_aneboda",
    "Laptop_20",
    "bin_6",
    "bin_11",
)


def generate_object_hash(asset_id: str) -> str:
    hasher = hashlib.md5()
    hasher.update(asset_id.encode())
    return hasher.hexdigest()


@dataclass
class ConversionResult:
    success: bool
    articulated: bool = False
    error_msg: str = ""
    out_usdpath: Path | None = None
    metadata: AssetGenMetadata | None = None


@dataclass
class Args:
    mode: Literal["convert-single", "convert-all"]
    model_path: Path | None = None
    folder_path: Path | None = None
    is_objaverse: bool = False
    output_dir: Path | None = None

    export_scene: bool = True
    export_sites: bool = False

    fix_rotation: bool = False

    fix_base: bool = False

    max_workers: int = 1

    normalize_mesh_scale: bool = True

    use_physx: bool = False

    use_newton: bool = False

    verbose: bool = False


G_ARGS: Args | None = None

G_PARAMETERS: dict[str, AssetParameters] = {}
G_ASSET_ID_TO_CATEGORY: dict[str, str] = {}


def create_physics_scene(data: BaseConversionData) -> None:
    asset_stage: Usd.Stage = data.content[Tokens.ASSET]
    content_stage: Usd.Stage = data.content[Tokens.CONTENTS]
    physics_stage: Usd.Stage = data.content[Tokens.PHYSICS]

    # ensure the name is valid across all layers
    safe_name = data.name_cache.getPrimName(asset_stage.GetPseudoRoot(), "PhysicsScene")

    # author the scene in the physics layer
    scene: UsdPhysics.Scene = UsdPhysics.Scene.Define(
        physics_stage, asset_stage.GetPseudoRoot().GetPath().AppendChild(safe_name)
    )

    # reference the scene in the asset layer, but from the content layer
    content_scene: Usd.Prim = content_stage.GetPseudoRoot().GetChild(safe_name)
    usdex.core.definePayload(asset_stage.GetPseudoRoot(), content_scene, safe_name)

    gravity_vector: Gf.Vec3d = Gf.Vec3d(data.spec.option.gravity.astype(np.float64).tolist())
    scene.CreateGravityDirectionAttr().Set(gravity_vector.GetNormalized())
    scene.CreateGravityMagnitudeAttr().Set(gravity_vector.GetLength())


def convert(model_path: Path) -> ConversionResult:  # noqa: PLR0915
    global G_ARGS, G_PARAMETERS, G_ASSET_ID_TO_CATEGORY  # noqa: PLW0602

    assert G_ARGS is not None, "Arguments struct must be defined by now"

    if not model_path.is_file():
        error_msg = f"Given mjcf file '{model_path}' doesn't exist or is not a valid file"
        return ConversionResult(success=False, error_msg=error_msg)

    assert G_ARGS.output_dir is not None, "Must provide an output directory via --output-dir"

    if not G_ARGS.output_dir.is_dir():
        error_msg = (
            f"Given output dir '{G_ARGS.output_dir}' doesn't exist or is not a valid directory"
        )
        return ConversionResult(success=False, error_msg=error_msg)

    prefix = "obja_" if G_ARGS.is_objaverse else ""

    output_asset_dir = G_ARGS.output_dir / f"{prefix}{model_path.stem}"
    output_asset_dir.mkdir(exist_ok=True)

    asset_id = model_path.stem.replace("_prim", "").replace("_mesh", "")
    hash_id = generate_object_hash(asset_id)

    success = True
    error_msg = ""
    usd_path = output_asset_dir / f"{prefix}{model_path.stem}.usda"
    metadata: AssetGenMetadata | None = None

    is_articulated = False
    # try:
    if True:
        spec = mj.MjSpec.from_file(model_path.as_posix())

        modelname = spec.modelname
        asset_name = usdex.core.getValidPrimName(f"{prefix}{modelname}")
        asset_stage = usdex.core.createStage(
            usd_path.absolute().as_posix(),
            defaultPrimName=asset_name,
            upAxis=UsdGeom.Tokens.z,
            linearUnits=UsdGeom.LinearUnits.meters,
            authoringMetadata="Ai2-THOR MJCF-USD converter",
        )
        assert asset_stage is not None, (
            f"Couldn't create stage for asset '{model_path.stem}', something went wrong T_T"
        )
        _ = usdex.core.defineXform(asset_stage, asset_stage.GetDefaultPrim().GetPath()).GetPrim()

        data = BaseConversionData(
            spec=spec,
            stage=asset_stage,
            usd_path=usd_path,
            export_scene=G_ARGS.export_scene,
            export_sites=G_ARGS.export_sites,
            root_rotation=FIX_ROTATION if G_ARGS.fix_rotation else None,
            thor_parameters=G_PARAMETERS,
            thor_id_to_category=G_ASSET_ID_TO_CATEGORY,
            use_physx=G_ARGS.use_physx,
            use_newton=G_ARGS.use_newton,
        )

        is_articulated = any(joint.type != mj.mjtJoint.mjJNT_FREE for joint in spec.joints)

        data.content[Tokens.ASSET] = asset_stage
        data.content[Tokens.ASSET].SetMetadata(UsdPhysics.Tokens.kilogramsPerUnit, 1)

        asset_payload = usdex.core.createAssetPayload(asset_stage)
        assert asset_payload is not None, (
            "Must be able to create an asset payload for the asset stage"
        )
        data.content[Tokens.CONTENTS] = asset_payload

        convert_meshes(data, prefix=prefix, normalize_mesh_scale=G_ARGS.normalize_mesh_scale)

        asset_content_geo = usdex.core.addAssetContent(
            data.content[Tokens.CONTENTS], Tokens.GEOMETRY.value, format="usda"
        )
        assert asset_content_geo is not None, (
            f"Couldn't create asset content for geometries for model '{model_path.stem}'"
        )
        data.content[Tokens.GEOMETRY] = asset_content_geo

        convert_materials(data, prefix=prefix)

        asset_content_physics = usdex.core.addAssetContent(
            data.content[Tokens.CONTENTS], Tokens.PHYSICS.value, format="usda"
        )
        assert asset_content_physics is not None, (
            f"Couldn't create asset content for physics for model '{model_path.stem}'"
        )
        data.content[Tokens.PHYSICS] = asset_content_physics
        data.content[Tokens.PHYSICS].SetMetadata(UsdPhysics.Tokens.kilogramsPerUnit, 1)
        data.references[Tokens.PHYSICS] = {}

        if data.export_scene:
            create_physics_scene(data)

        root_body_prim: Usd.Prim | None = None
        if is_articulated:
            # TODO(wilbert): change asset_id with parameters dict for joint parameters
            root_body_prim = convert_bodies_flatten(
                data,
                is_articulated,
                prefix=prefix,
                normalize_mesh_scale=G_ARGS.normalize_mesh_scale,
                asset_id=asset_id,
            )
        else:
            root_body_prim = convert_bodies_flatten_collapsed(
                data, prefix=prefix, normalize_mesh_scale=G_ARGS.normalize_mesh_scale
            )

        # if G_ARGS.fix_base:
        #     fixed_joint = UsdPhysics.FixedJoint.Define(
        #         data.content[Tokens.GEOMETRY],
        #         root_body_prim.GetPath().AppendChild("FixedJointRoot"),
        #     )
        #     fixed_joint.GetBody1Rel().SetTargets([root_body_prim.GetPath()])
        #     fixed_joint.GetBody0Rel().SetTargets(
        #         [root_body_prim.GetStage().GetDefaultPrim().GetPath()]
        #     )
        #     fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3d(0, 0, 0))
        #     fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf.GetIdentity())
        #     fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3d(0, 0, 0))
        #     fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf.GetIdentity())

        Usd.ModelAPI(root_body_prim).SetKind(Kind.Tokens.component)

        convert_contact_excludes(data)

        usdex.core.addAssetInterface(asset_stage, source=data.content[Tokens.CONTENTS])

        export_flatten(data, usd_path, "usda")

        if not G_ARGS.is_objaverse:
            metadata = AssetGenMetadata(
                asset_id=asset_id,
                hash_id=hash_id,
                articulated=is_articulated,
            )
            if (bbox_size := compute_bbox_size(root_body_prim)) is not None:
                metadata.bbox_size = bbox_size.tolist()

    # except Exception as e:
    #     success = False
    #     error_msg = f"Couldn't convert mjcf file '{model_path.stem}', error: {e}"
    #     print(f"[ERROR]: {error_msg}")

    return ConversionResult(
        success=success,
        error_msg=error_msg,
        articulated=is_articulated,
        out_usdpath=usd_path,
        metadata=metadata,
    )


def main() -> int:  # noqa: PLR0915
    global G_ARGS, G_PARAMETERS, G_ASSET_ID_TO_CATEGORY  # noqa: PLW0603

    G_ARGS = tyro.cli(Args)

    if PARAMETERS_FILE.is_file():
        with open(PARAMETERS_FILE, "rb") as fhandle:
            G_PARAMETERS = msgspec.yaml.decode(fhandle.read(), type=dict[str, AssetParameters])

    if ASSET_ID_TO_CATEGORY_FILE.is_file():
        with open(ASSET_ID_TO_CATEGORY_FILE, "rb") as fhandle:
            G_ASSET_ID_TO_CATEGORY = msgspec.json.decode(fhandle.read(), type=dict[str, str])

    usdex.core.activateDiagnosticsDelegate()
    usdex.core.setDiagnosticsLevel(
        usdex.core.DiagnosticsLevel.eStatus
        if G_ARGS.verbose
        else usdex.core.DiagnosticsLevel.eWarning
    )

    if G_ARGS.output_dir is None:
        print(
            "[ERROR]: must provide an output directory where to save the generated assets via --output-dir"
        )
        return 1

    if not G_ARGS.output_dir.is_dir():
        print(f"[ERROR]: output-dir @ '{G_ARGS.output_dir}' is not a valid directory")
        return 1

    match G_ARGS.mode:
        case "convert-single":
            if G_ARGS.model_path is None:
                print("[ERROR]: must provide a file for the model to convert via --model_path")
                return 1
            result = convert(G_ARGS.model_path)
            if not result.success:
                print(f"[ERROR]: {result.error_msg}")
            else:
                print(f"[INFO]: successfully converted asset '{G_ARGS.model_path.stem}'")

        case "convert-all":
            if G_ARGS.folder_path is None:
                print(
                    "[ERROR]: must provide a folder to the assets to be converted via --folder_path"
                )
                return 1
            if not G_ARGS.folder_path.is_dir():
                print(
                    f"[ERROR]: given assets folder path '{G_ARGS.folder_path}' is not a valid directory"
                )
                return 1

            use_metadata_file = Path(METADATA_FILENAME)
            usd_metadata = {}
            if use_metadata_file.is_file():
                with open(use_metadata_file, "r") as fhandle:
                    usd_metadata = json.load(fhandle)

            def grab_thor_assets_xmls(folder_path: Path) -> list[Path]:
                def is_valid_path(path: Path) -> bool:
                    return all(
                        substr not in path.stem
                        for substr in SUFFIXES_TO_AVOID + THOR_ASSETS_IDS_TO_SKIP
                    )

                return [path for path in folder_path.rglob("*.xml") if is_valid_path(path)]

            def grab_objaverse_assets_xmls(folder_path: Path) -> list[Path]:
                return [
                    path / f"{path.stem}.xml" for path in folder_path.iterdir() if path.is_dir()
                ]

            models_xmls = (
                grab_thor_assets_xmls(G_ARGS.folder_path)
                if not G_ARGS.is_objaverse
                else grab_objaverse_assets_xmls(G_ARGS.folder_path)
            )

            error_messages: list[str] = []
            if G_ARGS.max_workers > 1:
                results = p_uimap(convert, models_xmls, num_cpus=G_ARGS.max_workers)
                for result in results:
                    if not result.success:
                        error_messages.append(result.error_msg)
                    elif result.metadata is not None:
                        usd_metadata[result.metadata.asset_id] = asdict(result.metadata)
            else:
                for model_xml in tqdm(models_xmls):
                    result = convert(model_xml)
                    if not result.success:
                        error_messages.append(result.error_msg)
                    elif result.metadata is not None:
                        usd_metadata[result.metadata.asset_id] = asdict(result.metadata)

            if len(error_messages) > 0:
                with open(ERRORS_FILENAME, "w") as fhandle:
                    for msg in error_messages:
                        fhandle.write(f"[ERROR]: {msg}\n")

            with open(use_metadata_file, "w") as fhandle:
                json.dump(usd_metadata, fhandle, indent=4)

            dst_path = G_ARGS.output_dir / use_metadata_file.name
            shutil.copyfile(use_metadata_file, dst_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
