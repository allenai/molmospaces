"""
Interactive grasp classification tool.

Displays all grasp TCP positions on the object mesh. Use click-and-drag
rectangle selection to mark grasps as semantically good.

Usage:
    # Single asset
    python scripts/grasps/classify_grasps.py --asset_id Pan_22
    python scripts/grasps/classify_grasps.py --asset_id Pan_22 --resume
    python scripts/grasps/classify_grasps.py --asset_id Mug_1 --output_dir /tmp/classifications

    # Loop through every instance of one or more categories
    python scripts/grasps/classify_grasps.py --categories Pan,Mug,Bowl
    python scripts/grasps/classify_grasps.py --categories Pan --skip_existing

    # Loop through the full default set of pickupable iTHOR categories
    python scripts/grasps/classify_grasps.py --all_graspable --skip_existing
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("macosx")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from molmo_spaces.grasp_generation.pipeline.extract_leaf_meshes import (
    load_and_transform_mesh,
    parse_mujoco_xml,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, DATA_CACHE_DIR
from molmo_spaces.utils.grasps import load_pickup_grasps

# Default set of categories worth annotating for semantic grasping.
# These are the small handheld objects with clear semantic grasp affordances
# (handles, etc.) — comments show the number of instances available on disk.
DEFAULT_GRASPABLE_CATEGORIES = [
    "ButterKnife",  # 1
    "Cup",          # 30
    "Fork",         # 1
    "Kettle",       # 1
    "Knife",        # 7
    "Ladle",        # 3
    "Mug",          # 3
    "Pan",          # 30
    "Spatula",      # 1
    "Spoon",        # 1
]

# Gripper dimensions (matching view_grasps.py)
GRASP_WIDTH = 0.088
GRASP_LENGTH = 0.12
GRASP_BASE_POS = np.array([0.0, 0.0, -0.06])


def list_assets_for_categories(categories: list[str]) -> list[str]:
    """Return all asset_ids on disk matching `Category_<digits>` for the given categories.

    Looks under ASSETS_DIR/grasps/droid/. Sorted by category then numeric suffix.
    """
    grasps_root = ASSETS_DIR / "grasps" / "droid"
    if not grasps_root.exists():
        return []

    cat_set = set(categories)
    found: list[tuple[str, int, str]] = []
    pattern = re.compile(r"^(?P<cat>.+)_(?P<num>\d+)$")
    for entry in grasps_root.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if not m:
            continue
        cat = m.group("cat")
        if cat in cat_set:
            found.append((cat, int(m.group("num")), entry.name))
    found.sort()
    return [name for _, _, name in found]

# Colors
COLOR_UNSELECTED = np.array([0.8, 0.2, 0.2, 0.6])  # red, semi-transparent
COLOR_SELECTED = np.array([0.1, 0.85, 0.1, 0.9])    # green
RECT_EDGE_COLOR = "dodgerblue"
RECT_FACE_COLOR = "dodgerblue"


def find_object_mjcf(asset_id: str) -> Path:
    """Locate the per-asset MuJoCo XML file for the given asset_id.

    Each iTHOR asset has a directory like
        .../Prefab/{asset_id}/{asset_id}.xml
    next to its raw .obj. This XML carries the per-mesh `scale=` attribute
    and the body pos/quat that bring the raw .obj from its sub-mm Unity frame
    into the same frame the grasp NPZ files were generated in.
    """
    objects_root = DATA_CACHE_DIR / "objects" / "thor"

    # The {asset_id}/{asset_id}.xml pattern avoids matching variants like
    # {asset_id}_old.xml, {asset_id}_mesh.xml, {asset_id}_prim.xml.
    candidates = glob.glob(
        str(objects_root / "**" / asset_id / f"{asset_id}.xml"),
        recursive=True,
    )
    if candidates:
        return Path(candidates[0])

    raise FileNotFoundError(
        f"Could not find MJCF for {asset_id} under {objects_root}"
    )


def load_asset_mesh(asset_id: str) -> trimesh.Trimesh:
    """Load the asset's visual mesh in the same frame as its grasp NPZ.

    Uses the per-asset MJCF (not the raw .obj) so the mesh ends up with the
    correct scale (`<mesh scale=...>`) and orientation/position (`<body pos=...
    quat=...>`) baked in. Skips collision/collider geoms.
    """
    xml_path = find_object_mjcf(asset_id)
    mesh_instances, xml_dir = parse_mujoco_xml(xml_path)
    if not mesh_instances:
        raise ValueError(f"No mesh geoms found in MJCF: {xml_path}")

    transformed: list[trimesh.Trimesh] = []
    for mesh_info in mesh_instances:
        # Drop collision geoms — they're typically primitive boxes/spheres
        # rather than the visual mesh, and clutter the visualization. The
        # naming convention used by extract_leaf_meshes is "Collider" in the
        # geom name; iTHOR also uses "PrimitiveCollider" in some cases.
        geom_name = mesh_info.get("geom_name", "")
        if "Collider" in geom_name or "collider" in geom_name:
            continue
        m = load_and_transform_mesh(mesh_info, xml_dir)
        if m is not None:
            transformed.append(m)

    if not transformed:
        raise ValueError(
            f"No visual mesh geoms could be loaded for {asset_id} from {xml_path}"
        )

    if len(transformed) == 1:
        return transformed[0]
    return trimesh.util.concatenate(transformed)


def compute_tcp_points(grasp_transforms: np.ndarray) -> np.ndarray:
    """Compute the tool center point (midpoint between fingertips) for each grasp.

    In the local TCP frame the fingertip midpoint is at
    GRASP_BASE_POS + [0, 0, GRASP_LENGTH/2].  We transform that by each
    grasp pose to get world coordinates.
    """
    local_tcp = GRASP_BASE_POS + np.array([0.0, 0.0, GRASP_LENGTH / 2])
    local_h = np.array([local_tcp[0], local_tcp[1], local_tcp[2], 1.0])
    # (N,4,4) @ (4,) -> (N,4)
    world_h = (grasp_transforms @ local_h)
    return world_h[:, :3]


class GraspClassifier:
    def __init__(
        self,
        asset_id: str,
        mesh: trimesh.Trimesh,
        grasp_transforms: np.ndarray,
        output_path: Path,
        grasp_source: str,
        resume_data: dict | None = None,
    ):
        self.asset_id = asset_id
        self.mesh = mesh
        self.grasp_transforms = grasp_transforms
        self.output_path = output_path
        self.grasp_source = grasp_source
        self.total = len(grasp_transforms)

        # TCP positions in world frame (N, 3)
        self.tcp_points = compute_tcp_points(grasp_transforms)

        # Boolean mask: True = good grasp
        self.selected = np.zeros(self.total, dtype=bool)
        if resume_data:
            for k, v in resume_data.get("semantically_good_grasp", {}).items():
                idx = int(k)
                if idx < self.total:
                    self.selected[idx] = bool(v)

        # Matplotlib state
        self.fig = None
        self.ax = None
        self.scatter = None
        self.title_text = None

        # Rectangle-drag state (right-click drag)
        self.drag_start = None  # (x, y) display coords
        self.rect_patch = None  # preview rectangle on figure

    def _project_to_screen(self) -> np.ndarray:
        """Project all TCP 3D points to 2D display (pixel) coordinates."""
        M = self.ax.get_proj()
        x2, y2, _ = proj3d.proj_transform(
            self.tcp_points[:, 0],
            self.tcp_points[:, 1],
            self.tcp_points[:, 2],
            M,
        )
        return self.ax.transData.transform(np.column_stack([x2, y2]))

    def _update_colors(self):
        """Update scatter point colors from the selected mask."""
        colors = np.where(
            self.selected[:, None],
            COLOR_SELECTED[None, :],
            COLOR_UNSELECTED[None, :],
        )
        self.scatter.set_facecolors(colors)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        n_good = int(self.selected.sum())
        n_bad = self.total - n_good
        self.title_text.set_text(
            f"{self.asset_id}  —  good={n_good}  bad={n_bad}  /  {self.total}\n"
            f"Shift+drag=select   C=clear   U=undo   Q=save & quit"
        )

    def _save(self):
        data = {
            "asset_id": self.asset_id,
            "grasp_source": self.grasp_source,
            "total_grasps": self.total,
            "semantically_good_grasp": {
                str(i): bool(self.selected[i]) for i in range(self.total)
            },
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)
        n_good = int(self.selected.sum())
        print(
            f"\nSaved {self.total} grasps "
            f"(good={n_good} bad={self.total - n_good}) -> {self.output_path}"
        )

    # --- Rectangle selection helpers ---

    def _start_rect(self, x, y):
        self.drag_start = (x, y)

    def _clear_rect_patch(self):
        if self.rect_patch is not None:
            if self.rect_patch in self.fig.patches:
                self.fig.patches.remove(self.rect_patch)
            self.rect_patch = None

    def _update_rect_preview(self, x, y):
        if self.drag_start is None:
            return
        # Remove old rectangle
        self._clear_rect_patch()

        x0, y0 = self.drag_start
        # Convert display coords to figure-fraction coords
        inv = self.fig.transFigure.inverted()
        p0 = inv.transform((min(x0, x), min(y0, y)))
        p1 = inv.transform((max(x0, x), max(y0, y)))

        self.rect_patch = mpatches.Rectangle(
            p0, p1[0] - p0[0], p1[1] - p0[1],
            linewidth=2,
            edgecolor=RECT_EDGE_COLOR,
            facecolor=RECT_FACE_COLOR,
            alpha=0.15,
            transform=self.fig.transFigure,
        )
        self.fig.patches.append(self.rect_patch)
        self.fig.canvas.draw_idle()

    def _finish_rect(self, x, y):
        # Remove preview rectangle
        self._clear_rect_patch()

        if self.drag_start is None:
            return

        x0, y0 = self.drag_start
        self.drag_start = None

        xmin, xmax = min(x0, x), max(x0, x)
        ymin, ymax = min(y0, y), max(y0, y)

        # Ignore tiny drags (accidental clicks)
        if (xmax - xmin) < 5 and (ymax - ymin) < 5:
            self.fig.canvas.draw_idle()
            return

        # Save state for undo
        self._push_undo()

        # Project TCP points to screen and find those inside the rectangle
        screen_pts = self._project_to_screen()
        inside = (
            (screen_pts[:, 0] >= xmin) & (screen_pts[:, 0] <= xmax)
            & (screen_pts[:, 1] >= ymin) & (screen_pts[:, 1] <= ymax)
        )
        n_newly_selected = int(inside.sum()) - int((self.selected & inside).sum())
        self.selected |= inside

        print(f"\r  Selected {int(inside.sum())} grasps "
              f"({n_newly_selected} new)          ", end="", flush=True)
        self._update_colors()

    # --- Undo ---

    def _push_undo(self):
        if not hasattr(self, "_undo_stack"):
            self._undo_stack = []
        self._undo_stack.append(self.selected.copy())
        # Keep bounded
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _pop_undo(self):
        if hasattr(self, "_undo_stack") and self._undo_stack:
            self.selected = self._undo_stack.pop()
            self._update_colors()
            print("\r  Undid last selection          ", end="", flush=True)

    # --- Event handlers ---

    def _on_key(self, event):
        if event.key is None:
            return
        k = event.key.lower()

        if k == "c":
            self._push_undo()
            self.selected[:] = False
            self._update_colors()
            print("\r  Cleared all selections          ", end="", flush=True)

        elif k == "u":
            self._pop_undo()

        elif k == "q":
            self._save()
            plt.close(self.fig)

    def _on_press(self, event):
        # Shift + left-click starts rectangle selection
        if event.inaxes != self.ax or event.button != 1:
            return
        if event.key != "shift":
            return
        # Suppress matplotlib's built-in rotation for this drag
        self.ax.button_pressed = -1
        self._start_rect(event.x, event.y)

    def _on_motion(self, event):
        if self.drag_start is None:
            return
        self._update_rect_preview(event.x, event.y)

    def _on_release(self, event):
        if self.drag_start is None:
            return
        self._finish_rect(event.x, event.y)

    def run(self):
        verts = np.array(self.mesh.vertices)
        faces = np.array(self.mesh.faces)

        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Draw the object mesh
        face_verts = verts[faces]
        mesh_coll = Poly3DCollection(
            face_verts,
            alpha=0.55,
            facecolor="lightsteelblue",
            edgecolor="slategray",
            linewidth=0.2,
        )
        self.ax.add_collection3d(mesh_coll)

        # Set axis limits from the union of mesh and TCP bounds. They should
        # already share a frame (mesh loaded via per-asset MJCF, grasps from
        # the matching NPZ), so the union is just to give a bit of padding
        # around any grasps that sit slightly outside the mesh hull.
        all_pts = np.vstack([verts, self.tcp_points])
        margin = 0.02
        mins = all_pts.min(axis=0) - margin
        maxs = all_pts.max(axis=0) + margin
        self.ax.set_xlim(mins[0], maxs[0])
        self.ax.set_ylim(mins[1], maxs[1])
        self.ax.set_zlim(mins[2], maxs[2])
        self.ax.set_box_aspect(maxs - mins)

        # Scatter plot of TCP points
        colors = np.where(
            self.selected[:, None],
            COLOR_SELECTED[None, :],
            COLOR_UNSELECTED[None, :],
        )
        self.scatter = self.ax.scatter(
            self.tcp_points[:, 0],
            self.tcp_points[:, 1],
            self.tcp_points[:, 2],
            c=colors,
            s=12,
            depthshade=False,
        )

        self.title_text = self.ax.set_title("")
        self._undo_stack = []
        self._update_title()

        # Connect events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        print("\n--- Grasp Classification ---")
        print("  Left-click drag        = rotate view")
        print("  Shift + left-drag      = rectangle select (marks grasps as good)")
        print("  U = undo last selection   C = clear all   Q = save & quit")
        print("----------------------------")

        plt.show()

        # Auto-save on window close
        self._save()


def classify_one_asset(
    asset_id: str,
    output_dir_arg: str | None,
    resume: bool,
    num_grasps: int,
) -> bool:
    """Run the interactive classifier for a single asset.

    Returns True on success, False if the asset had to be skipped due to
    missing mesh / grasps.
    """
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
    else:
        output_dir = ASSETS_DIR / "grasps" / "droid" / asset_id
    output_path = output_dir / f"{asset_id}_grasp_classifications.json"

    print(f"\nLoading mesh for {asset_id} via per-asset MJCF ...")
    try:
        mesh = load_asset_mesh(asset_id)
    except (FileNotFoundError, ValueError) as e:
        print(f"  Skipping {asset_id}: {e}")
        return False
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    print(f"Loading grasps for {asset_id} ...")
    try:
        grasp_transforms = load_pickup_grasps(asset_id, num_grasps=num_grasps)
        print(f"Loaded {len(grasp_transforms)} grasps")
    except ValueError as e:
        print(f"  Skipping {asset_id}: error loading grasps: {e}")
        return False

    if len(grasp_transforms) == 0:
        print(f"  Skipping {asset_id}: no grasps available")
        return False

    grasp_source = str(
        ASSETS_DIR / "grasps" / "droid" / asset_id / f"{asset_id}_grasps_filtered.npz"
    )

    resume_data = None
    if resume and output_path.exists():
        with open(output_path) as f:
            resume_data = json.load(f)
        n = sum(1 for v in resume_data.get("semantically_good_grasp", {}).values() if v)
        print(f"Resuming from {output_path} ({n} good grasps)")

    classifier = GraspClassifier(
        asset_id=asset_id,
        mesh=mesh,
        grasp_transforms=grasp_transforms,
        output_path=output_path,
        grasp_source=grasp_source,
        resume_data=resume_data,
    )
    classifier.run()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Interactively classify grasps as good/bad for one or more objects."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--asset_id", type=str,
        help="Asset ID of a single object (e.g. Pan_22, Mug_1)",
    )
    group.add_argument(
        "--categories", type=str,
        help="Comma-separated list of categories — loops through every instance "
             "of each category found on disk (e.g. 'Pan,Mug,Bowl').",
    )
    group.add_argument(
        "--all_graspable", action="store_true",
        help=f"Loop through every instance of the default category set: "
             f"{', '.join(DEFAULT_GRASPABLE_CATEGORIES)}",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save classifications. Default: assets/grasps/droid/{asset_id}/. "
             "Only meaningful for single --asset_id; ignored when looping.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing classification file if one exists",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip assets that already have a classification file (useful when looping).",
    )
    parser.add_argument(
        "--num_grasps", type=int, default=10000,
        help="Maximum number of grasps to load (default: 10000)",
    )
    args = parser.parse_args()

    # Build the list of asset_ids to process.
    if args.asset_id:
        asset_ids = [args.asset_id]
        output_dir_arg = args.output_dir
    else:
        if args.all_graspable:
            categories = DEFAULT_GRASPABLE_CATEGORIES
        else:
            categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        asset_ids = list_assets_for_categories(categories)
        if not asset_ids:
            print(f"No assets found on disk for categories: {categories}")
            sys.exit(1)
        if args.output_dir:
            print("Warning: --output_dir is ignored when looping through categories.")
        output_dir_arg = None
        print(f"Found {len(asset_ids)} asset(s) across {len(categories)} categor(ies):")
        for aid in asset_ids:
            print(f"  - {aid}")

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for i, asset_id in enumerate(asset_ids):
        print(f"\n=== [{i + 1}/{len(asset_ids)}] {asset_id} ===")

        if args.skip_existing:
            existing = (
                ASSETS_DIR / "grasps" / "droid" / asset_id
                / f"{asset_id}_grasp_classifications.json"
            )
            if existing.exists():
                print(f"  Skipping (classification already exists at {existing})")
                n_skipped += 1
                continue

        try:
            ok = classify_one_asset(
                asset_id=asset_id,
                output_dir_arg=output_dir_arg,
                resume=args.resume,
                num_grasps=args.num_grasps,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting loop.")
            break

        if ok:
            n_done += 1
        else:
            n_failed += 1

    print(
        f"\nDone. classified={n_done}  skipped={n_skipped}  "
        f"failed/missing={n_failed}  total={len(asset_ids)}"
    )


if __name__ == "__main__":
    main()
