import json
from dataclasses import dataclass
from pathlib import Path

# Repo root (utils/common.py -> utils -> molmo_spaces_isaac -> repo)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class AssetGenMetadata:
    asset_id: str
    hash_id: str
    articulated: bool
    bbox_size: list[float]


def load_thor_assets_metadata(filepath: Path) -> dict[str, AssetGenMetadata]:
    if not filepath.is_file():
        raise RuntimeError(f"The given file '{filepath.as_posix()}' is not a valid file")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: dict[str, AssetGenMetadata] = {}
    for key, val in data.items():
        if isinstance(val, dict):
            result[key] = AssetGenMetadata(
                asset_id=val.get("asset_id", key),
                hash_id=val.get("hash_id", ""),
                articulated=bool(val.get("articulated", False)),
                bbox_size=list(val.get("bbox_size", [])),
            )
        else:
            result[key] = AssetGenMetadata(asset_id=key, hash_id="", articulated=False, bbox_size=[])
    return result
