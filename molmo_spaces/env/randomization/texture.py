import json

import mujoco
import numpy as np
from mujoco import MjData, MjModel

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_resource_manager

# ensure assets are ready by instantiating the resource manager
get_resource_manager()

# load json file that describes materials used
TEXTURE_FOLDER_PATH = ASSETS_DIR / "objects" / "thor" / "Textures"


def create_placeholder_texture_file(texture_size: int, log) -> str | None:
    """Create a temporary placeholder texture file for MuJoCo 2D textures."""
    import os
    import tempfile

    from PIL import Image

    try:
        temp_dir = tempfile.mkdtemp(prefix="mujoco_texture_placeholder_")
        placeholder_path = os.path.join(temp_dir, "__TEXTURE_RANDOMIZER_PLACEHOLDER__.png")

        img = Image.new("RGB", (texture_size, texture_size), color=(255, 255, 255))
        img.save(placeholder_path, "PNG")

        if os.path.exists(placeholder_path) and os.path.getsize(placeholder_path) > 0:
            log.debug(f"Created placeholder texture file: {placeholder_path}")
            return placeholder_path
        else:
            log.error(f"Placeholder texture file was not created or is empty: {placeholder_path}")
            return None
    except Exception as e:
        log.error(f"Failed to create placeholder texture file: {e}")
        return None


def create_empty_texture(spec: mujoco.MjSpec, tex_name: str, placeholder_file: str, log) -> bool:
    """Create an empty texture in the spec using the placeholder file."""
    try:
        spec.add_texture(name=tex_name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=placeholder_file)
        return True
    except Exception as e:
        log.warning(f"Failed to create texture {tex_name}: {e}")
        return False


def assign_texture_to_material(material, tex_name: str, mat_name: str, log) -> None:
    """Assign a texture to a material's RGB role."""
    try:
        rgb_role = mujoco.mjtTextureRole.mjTEXROLE_RGB.value
        material.textures[rgb_role] = tex_name
        log.debug(f"Assigned texture {tex_name} to material {mat_name}")
    except Exception as e:
        log.warning(f"Failed to assign texture {tex_name} to material {mat_name}: {e}")


def setup_empty_materials(spec: mujoco.MjSpec | None = None, num_materials: int = 200) -> None:
    """
    Create a pool of empty materials and textures in the MjSpec that can be assigned to geoms at runtime.
    This allows texture randomization to modify materials and textures without affecting other geoms.

    Args:
        spec: MjSpec to modify
        num_materials: Maximum number of empty materials/textures to create. Actual number is based on
                      visual geom count with a safety buffer.
    """
    import logging

    log = logging.getLogger(__name__)

    if spec is None:
        raise ValueError("spec cannot be None")

    # Check if empty materials already exist to avoid duplicates
    if spec.material("__TEXTURE_RANDOMIZER_MAT_0__") is not None:
        log.debug("Empty materials already exist in spec, skipping creation")
        return

    # Count visual geoms to estimate how many materials we might need
    visual_geom_count = sum(
        1
        for geom in spec.geoms
        if (
            hasattr(geom, "contype")
            and hasattr(geom, "conaffinity")
            and geom.contype == 0
            and geom.conaffinity == 0
        )
        or (
            hasattr(geom, "classname")
            and ("__VISUAL_MJT__" in str(geom.classname) or "visual" in str(geom.classname).lower())
        )
    )

    # Calculate number to create: use visual_geom_count with safety buffer, cap at num_materials
    safety_multiplier = 1.0
    num_to_create = min(int(visual_geom_count * safety_multiplier), num_materials)
    num_to_create = max(num_to_create, min(50, num_materials))

    # Use 512x512 as default texture size (can be overridden by caller if needed)
    texture_size = 512
    memory_per_texture_mb = texture_size * texture_size * 3 / (1024**2)
    total_memory_mb = num_to_create * memory_per_texture_mb

    log.info(
        f"Creating {num_to_create} empty materials and textures "
        f"(visual geoms: {visual_geom_count}, texture size: {texture_size}x{texture_size}, "
        f"estimated memory: ~{total_memory_mb:.1f} MB per model)"
    )

    # Create placeholder texture file (MuJoCo requires a file path for 2D textures)
    placeholder_file = create_placeholder_texture_file(texture_size, log)
    if placeholder_file is None:
        log.warning(
            "Failed to create placeholder texture file. Materials will be created without textures."
        )

    # Create empty materials and textures
    created_count = 0
    for i in range(num_to_create):
        mat_name = f"__TEXTURE_RANDOMIZER_MAT_{i}__"
        tex_name = f"__TEXTURE_RANDOMIZER_TEX_{i}__"

        # Create texture if placeholder file is available
        texture_exists = False
        if placeholder_file is not None and spec.texture(tex_name) is None:
            texture_exists = create_empty_texture(spec, tex_name, placeholder_file, log)

        # Create material
        empty_mat = spec.add_material(name=mat_name)
        empty_mat.rgba = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        empty_mat.specular = 0.5
        empty_mat.shininess = 0.5

        # Assign texture to material if it was created
        if texture_exists or (placeholder_file is not None and spec.texture(tex_name) is not None):
            assign_texture_to_material(empty_mat, tex_name, mat_name, log)
        else:
            log.debug(f"Material {mat_name} created without texture")

        created_count += 1

    log.info(f"Created {created_count} empty materials and textures for texture randomization")


class TextureRandomizer:
    """
    Randomizer for geom colors, material properties, and textures in MuJoCo simulations.

    Can randomize:
    - Geom RGBA colors
    - Material properties (RGBA, specular, shininess)
    - Texture bitmaps from loaded texture files

    Args:
        model (MjModel): MuJoCo model
        random_state (np.random.RandomState | None): Random state for reproducibility.
            If None, uses global numpy random state.
        geom_names (list[str] | None): List of geom names to randomize.
            If None, randomizes all geoms in the model.
        randomize_geom_rgba (bool): If True, randomizes geom RGBA colors
        randomize_material_rgba (bool): If True, randomizes material RGBA colors
        randomize_material_specular (bool): If True, randomizes material specular
        randomize_material_shininess (bool): If True, randomizes material shininess
        randomize_texture (bool): If True, randomizes texture bitmaps from loaded textures.
            Default behavior uses textures already loaded in the model XML.
        texture_paths (list[str] | None): Optional list of paths to external texture image files (PNG, etc.).
            If None (default), uses textures already loaded in the model XML.
            If provided, loads textures from these external files instead.
        scene_metadata (dict | None): Optional scene metadata for category-based texture randomization
        rgba_perturbation_size (float): Magnitude of RGBA color randomization
        specular_perturbation_size (float): Magnitude of specular randomization
        shininess_perturbation_size (float): Magnitude of shininess randomization

    Note:
        MjData should be passed to the randomize() method, not to __init__.
    """

    material_database_filename = ASSETS_DIR / "objects" / "thor" / "material-database.json"
    with open(material_database_filename) as f:
        MAT_PER_CATEGORY = json.load(f)

    materials_to_texture_filename = ASSETS_DIR / "objects" / "thor" / "material_to_textures.json"
    with open(materials_to_texture_filename) as f:
        MAT_TO_TEXTURE = json.load(f)

    # Extract texture paths from materials for each category
    # MAT_TO_TEXTURE[mat] is a dict with "_MainTex" key containing the texture path
    # Normalize category keys to lowercase for consistent matching
    CAT_TO_TEXTURE = {}
    for cat in MAT_PER_CATEGORY:
        texture_paths = []
        for mat in MAT_PER_CATEGORY[cat]:
            if mat in MAT_TO_TEXTURE:
                mat_dict = MAT_TO_TEXTURE[mat]
                if "_MainTex" in mat_dict and mat_dict["_MainTex"] is not None:
                    texture_paths.append(mat_dict["_MainTex"])
        CAT_TO_TEXTURE[cat.lower()] = texture_paths

    def __init__(
        self,
        model: MjModel,
        random_state: np.random.RandomState | None = None,
        geom_names: list[str] | None = None,
        randomize_geom_rgba: bool = True,
        randomize_material_rgba: bool = True,
        randomize_material_specular: bool = True,
        randomize_material_shininess: bool = True,
        randomize_texture: bool = True,
        texture_paths: list[str] | None = None,
        scene_metadata: dict | None = None,
        rgba_perturbation_size: float = 0.1,
        specular_perturbation_size: float = 0.1,
        shininess_perturbation_size: float = 0.1,
    ):
        self.model = model
        self.scene_metadata = scene_metadata
        self._empty_material_names: list[str] = []
        self._empty_material_ids: list[int] = []
        self._empty_texture_names: list[str] = []
        self._empty_texture_ids: list[int] = []
        self._empty_material_to_texture: dict[
            int, int
        ] = {}  # Map empty material ID -> empty texture ID
        self._next_empty_material_index = 0
        self._geom_to_empty_material: dict[int, int] = {}  # Map geom_id -> empty material ID
        self._geom_to_original_material: dict[
            int, int
        ] = {}  # Map geom_id -> original material ID (before assignment)
        self._geom_to_original_texture: dict[
            int, int
        ] = {}  # Map geom_id -> original texture ID (before assignment)

        # Find all empty materials and textures created for texture randomization
        self._find_empty_materials()

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        # Build mapping from body names to categories using scene_metadata
        self._body_name_to_category: dict[str, str] = {}
        if scene_metadata:
            objects = scene_metadata.get("objects", {})
            for obj_key, obj_data in objects.items():
                category = obj_data.get("category", "")
                name_map = obj_data.get("name_map", {})
                bodies = name_map.get("bodies", {})
                # Map both hash names and actual names to category
                for hash_name, actual_name in bodies.items():
                    self._body_name_to_category[hash_name] = category
                    self._body_name_to_category[actual_name] = category
                # Also map the object key itself
                self._body_name_to_category[obj_key] = category

        # Get geom names from model (only visual geoms: contype == 0)
        if geom_names is None:
            geom_names = []
            for i in range(model.ngeom):
                # Only include visual geoms (contype == 0 means no collision, visual only)
                if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
                    name_adr = model.name_geomadr[i]
                    if name_adr >= 0:
                        name_bytes = model.names[name_adr:]
                        name = name_bytes.split(b"\x00")[0].decode("utf-8")
                        if name:
                            geom_names.append(name)

        self.geom_names = geom_names

        self.randomize_geom_rgba = randomize_geom_rgba
        self.randomize_material_rgba = randomize_material_rgba
        self.randomize_material_specular = randomize_material_specular
        self.randomize_material_shininess = randomize_material_shininess
        self.randomize_texture = randomize_texture

        self.rgba_perturbation_size = rgba_perturbation_size
        self.specular_perturbation_size = specular_perturbation_size
        self.shininess_perturbation_size = shininess_perturbation_size

        # Load texture files if provided, or extract existing textures from model (default)
        self.texture_bitmaps: list[np.ndarray] = []
        self.texture_ids: list[int] = []  # Texture IDs for on-demand extraction
        self.texture_paths = texture_paths  # Store for later use
        self._texture_cache: dict[
            int, np.ndarray
        ] = {}  # Cache for extracted textures (tex_id -> bitmap)
        self._texture_id_to_category: dict[
            int, str
        ] = {}  # Cache: texture_id -> category (for model textures)
        self._cat_texture_cache: dict[
            str, np.ndarray
        ] = {}  # Cache for category textures loaded from CAT_TO_TEXTURE
        if randomize_texture:
            if texture_paths:
                # Load textures from external files (user explicitly provided paths)
                self.texture_bitmaps = []
                for texture_path in texture_paths:
                    bitmap = self._load_texture_from_path(texture_path, resolve_path=False)
                    if bitmap is not None:
                        self.texture_bitmaps.append(bitmap)
            else:
                # Default: Extract existing textures from the model XML (on-demand)
                self._extract_model_textures()

        self.save_defaults()

    def _load_single_texture_bitmap(
        self, texture_path: str, resolve_path: bool = False
    ) -> np.ndarray | None:
        """
        Load a single texture image file and convert it to a bitmap.

        Args:
            texture_path: Path to texture image file
            resolve_path: If True, resolve relative paths relative to TEXTURE_FOLDER_PATH.
                         If False, use the path as-is.

        Returns:
            Texture bitmap as numpy array, or None if loading fails
        """
        import os
        from pathlib import Path

        try:
            from PIL import Image
        except ImportError as err:
            raise ImportError(
                "PIL (Pillow) is required for texture loading. Install with: pip install Pillow"
            ) from err

        # Resolve path if needed
        if resolve_path:
            if not os.path.isabs(texture_path):
                resolved_path = TEXTURE_FOLDER_PATH / Path(texture_path).name
            else:
                resolved_path = Path(texture_path)
        else:
            resolved_path = Path(texture_path)

        if not resolved_path.exists():
            return None

        try:
            # Load image
            img = Image.open(str(resolved_path))
            # Convert to RGB if needed (remove alpha channel for now)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Convert to numpy array
            img_array = np.array(img, dtype=np.uint8)
            return img_array
        except Exception as e:
            print(f"Warning: Failed to load texture {resolved_path}: {e}")
            return None

    def _find_empty_materials(self) -> None:
        """Find all empty materials and textures created for texture randomization in the model."""
        import logging

        log = logging.getLogger(__name__)

        self._empty_material_names = []
        self._empty_material_ids: list[int] = []
        self._empty_texture_names = []
        self._empty_texture_ids: list[int] = []

        # Search for materials with the naming pattern
        for mat_id in range(self.model.nmat):
            name_adr = self.model.name_matadr[mat_id]
            if name_adr >= 0:
                name_bytes = self.model.names[name_adr:]
                mat_name = name_bytes.split(b"\x00")[0].decode("utf-8")
                if mat_name.startswith("__TEXTURE_RANDOMIZER_MAT_") and mat_name.endswith("__"):
                    self._empty_material_names.append(mat_name)
                    self._empty_material_ids.append(mat_id)

        # Search for textures with the naming pattern and create mapping to materials
        for tex_id in range(self.model.ntex):
            name_adr = self.model.name_texadr[tex_id]
            if name_adr >= 0:
                name_bytes = self.model.names[name_adr:]
                tex_name = name_bytes.split(b"\x00")[0].decode("utf-8")
                if tex_name.startswith("__TEXTURE_RANDOMIZER_TEX_") and tex_name.endswith("__"):
                    self._empty_texture_names.append(tex_name)
                    self._empty_texture_ids.append(tex_id)
                    # Extract index from texture name and match to material
                    try:
                        tex_index_str = tex_name[len("__TEXTURE_RANDOMIZER_TEX_") : -len("__")]
                        tex_index = int(tex_index_str)
                        # Find the material with the same index
                        for mat_idx, mat_id in enumerate(self._empty_material_ids):
                            mat_name = self._empty_material_names[mat_idx]
                            if mat_name.startswith(
                                "__TEXTURE_RANDOMIZER_MAT_"
                            ) and mat_name.endswith("__"):
                                mat_index_str = mat_name[
                                    len("__TEXTURE_RANDOMIZER_MAT_") : -len("__")
                                ]
                                mat_index = int(mat_index_str)
                                if mat_index == tex_index:
                                    # Match found - map material ID to texture ID
                                    self._empty_material_to_texture[mat_id] = tex_id

                                    # CRITICAL: Verify texture is actually assigned to material in compiled model
                                    # Even though we assigned it in MjSpec, we need to check mat_texid
                                    if hasattr(self.model, "mat_texid"):
                                        try:
                                            assigned_tex_id = -1
                                            if isinstance(self.model.mat_texid, np.ndarray):
                                                if (
                                                    self.model.mat_texid.ndim == 2
                                                    and mat_id < self.model.mat_texid.shape[0]
                                                ):
                                                    assigned_tex_id = int(
                                                        self.model.mat_texid[mat_id, 0]
                                                    )
                                                elif (
                                                    self.model.mat_texid.ndim == 1
                                                    and mat_id < len(self.model.mat_texid)
                                                ):
                                                    assigned_tex_id = int(
                                                        self.model.mat_texid[mat_id]
                                                    )

                                            if assigned_tex_id != tex_id:
                                                # Texture not assigned - assign it now
                                                if isinstance(self.model.mat_texid, np.ndarray):
                                                    if (
                                                        self.model.mat_texid.ndim == 2
                                                        and mat_id < self.model.mat_texid.shape[0]
                                                    ):
                                                        self.model.mat_texid[mat_id, 0] = tex_id
                                                    elif (
                                                        self.model.mat_texid.ndim == 1
                                                        and mat_id < len(self.model.mat_texid)
                                                    ):
                                                        self.model.mat_texid[mat_id] = tex_id
                                                log.debug(
                                                    f"✓ Assigned texture {tex_id} to empty material {mat_id} "
                                                    f"({mat_name}) in compiled model"
                                                )
                                            else:
                                                log.debug(
                                                    f"✓ Verified texture {tex_id} is assigned to empty material {mat_id} "
                                                    f"({mat_name}) in compiled model"
                                                )
                                        except (IndexError, TypeError, ValueError) as e:
                                            log.warning(
                                                f"Could not verify/assign texture {tex_id} to material {mat_id}: {e}"
                                            )
                                    break
                    except (ValueError, IndexError):
                        pass

        log.info(
            f"Found {len(self._empty_material_ids)} empty materials and {len(self._empty_texture_ids)} empty textures "
            f"for texture randomization. Mapped {len(self._empty_material_to_texture)} material->texture pairs."
        )

        # Final verification: check how many materials actually have textures assigned
        materials_with_textures = 0
        if hasattr(self.model, "mat_texid"):
            for mat_id in self._empty_material_ids:
                try:
                    tex_id = -1
                    if isinstance(self.model.mat_texid, np.ndarray):
                        if (
                            self.model.mat_texid.ndim == 2
                            and mat_id < self.model.mat_texid.shape[0]
                        ):
                            tex_id = int(self.model.mat_texid[mat_id, 0])
                        elif self.model.mat_texid.ndim == 1 and mat_id < len(self.model.mat_texid):
                            tex_id = int(self.model.mat_texid[mat_id])
                    if tex_id >= 0 and tex_id < self.model.ntex:
                        materials_with_textures += 1
                except (IndexError, TypeError, ValueError):
                    pass

        if materials_with_textures < len(self._empty_material_ids):
            log.warning(
                f"Only {materials_with_textures}/{len(self._empty_material_ids)} empty materials have textures assigned "
                f"in compiled model. This may cause texture randomization to fail."
            )
        else:
            log.debug(
                f"✓ All {materials_with_textures} empty materials have textures assigned in compiled model"
            )

    def _load_texture_from_path(
        self, texture_path: str, resolve_path: bool = True
    ) -> np.ndarray | None:
        """
        Load a single texture image file and convert it to a bitmap.

        Args:
            texture_path: Path to texture image file (can be absolute or relative)
            resolve_path: If True, resolve relative paths relative to TEXTURE_FOLDER_PATH.
                         If False, use the path as-is.

        Returns:
            Texture bitmap as numpy array, or None if loading fails
        """
        return self._load_single_texture_bitmap(texture_path, resolve_path=resolve_path)

    def _extract_model_textures(self) -> None:
        """
        Extract existing textures from the model for randomization.
        This allows randomizing between textures already loaded in the scene.
        Optimized: Only stores texture IDs, extracts bitmaps on-demand during randomization.
        """
        # Instead of extracting all textures upfront (slow and memory-intensive),
        # just store the texture IDs and extract bitmaps on-demand
        self.texture_ids: list[int] = []

        # Collect all 2D texture IDs from the model
        for tex_id in range(self.model.ntex):
            tex_type = int(self.model.tex_type[tex_id])
            if tex_type == 0:  # mjTEXTURE_2D
                self.texture_ids.append(tex_id)

        # Pre-extract only a small sample to verify it works (for debugging)
        # Full extraction happens on-demand during randomization
        self.texture_bitmaps: list[np.ndarray] = []
        if len(self.texture_ids) > 0:
            # Extract first texture as a test
            try:
                test_bitmap = self._get_texture_bitmap(self.texture_ids[0])
                if test_bitmap is not None and test_bitmap.size > 0:
                    print(
                        f"   Found {len(self.texture_ids)} 2D textures in model (extracting on-demand)"
                    )
            except Exception as e:
                print(f"   Warning: Could not extract test texture: {e}")

        if len(self.texture_ids) == 0:
            print(
                f"   Warning: No 2D textures found in model (model has {self.model.ntex} textures total)"
            )
            if self.model.ntex > 0:
                tex_types = [int(self.model.tex_type[i]) for i in range(self.model.ntex)]
                type_counts = {}
                for t in tex_types:
                    type_counts[t] = type_counts.get(t, 0) + 1
                print(f"   Texture types in model: {type_counts} (0=2D, 1=cube, 2=skybox)")

    def save_defaults(self):
        """
        Save default geom and material parameter values from the current model state.
        Optimized for large scenes by only processing visual geoms (contype == 0).
        """
        # Build name-to-geom_id mapping once, only for visual geoms (O(n) instead of O(n*m))
        name_to_geom_id = {}
        for i in range(self.model.ngeom):
            # Only process visual geoms (contype == 0 means no collision, visual only)
            if self.model.geom_contype[i] == 0:
                name_adr = self.model.name_geomadr[i]
                if name_adr >= 0:
                    name_bytes = self.model.names[name_adr:]
                    name = name_bytes.split(b"\x00")[0].decode("utf-8")
                    if name:
                        name_to_geom_id[name] = i

        self._defaults = {}
        self._geom_id_to_defaults: dict[int, dict] = {}  # Fast lookup by geom_id
        # Only process geoms that are in our list
        for name in self.geom_names:
            geom_id = name_to_geom_id.get(name, -1)
            if geom_id < 0:
                continue

            defaults = {}
            defaults["geom_rgba"] = np.array(self.model.geom_rgba[geom_id])

            # Check if geom has a material with texture
            mat_id = int(self.model.geom_matid[geom_id])
            if mat_id >= 0:
                defaults["mat_rgba"] = np.array(self.model.mat_rgba[mat_id])
                defaults["mat_specular"] = float(self.model.mat_specular[mat_id])
                defaults["mat_shininess"] = float(self.model.mat_shininess[mat_id])
                defaults["mat_id"] = mat_id

                # Save texture ID instead of bitmap (much faster, less memory)
                if self.randomize_texture:
                    tex_id = self._get_texture_id_for_geom(geom_id)
                    defaults["texture_id"] = tex_id  # Store ID, not bitmap
                else:
                    defaults["texture_id"] = -1
            else:
                defaults["mat_rgba"] = None
                defaults["mat_specular"] = None
                defaults["mat_shininess"] = None
                defaults["mat_id"] = -1
                defaults["texture_id"] = -1

            self._defaults[name] = defaults
            self._geom_id_to_defaults[geom_id] = defaults  # Fast lookup by geom_id

    def _get_geom_category(self, geom_id: int) -> str | None:
        """
        Get the category of a geom by looking up its body in scene_metadata.

        Args:
            geom_id: Geom ID

        Returns:
            Category string if found, None otherwise
        """
        if not self.scene_metadata:
            return None

        # Get the body ID for this geom
        body_id = int(self.model.geom_bodyid[geom_id])
        if body_id < 0:
            return None

        # Get body name
        body_name_adr = self.model.name_bodyadr[body_id]
        if body_name_adr < 0:
            return None

        body_name_bytes = self.model.names[body_name_adr:]
        body_name = body_name_bytes.split(b"\x00")[0].decode("utf-8")

        if not body_name:
            return None

        # Look up category from our mapping
        return self._body_name_to_category.get(body_name)

    @staticmethod
    def _get_target_keywords() -> list[str]:
        """
        Get the list of keywords that identify target categories for randomization.

        Returns:
            List of keyword strings for floors, countertops, tabletops, doors, walls, etc.
        """
        return [
            "wall",
            "backsplash",
            "quad",
            "mesh",  # walls in iTHOR
            # receptacles
            "island",
            "counter",
            "table",
            "desk",
            "plane",  # countertops in iTHOR
            # floors
            "room",  # floors in ProcTHOR
            "floor",
            "ceiling",
            # doors
            "drawer",
            "cabinet",
            "doorway",
            "door",
            "handle",
        ]

    def _should_randomize_texture(self, name: str, geom_id: int) -> bool:
        """
        Check if a geom should have its texture randomized.
        Uses scene_metadata category if available, otherwise falls back to name matching.
        Only randomize textures for: walls, floors, countertops, tabletops, and doors.

        Args:
            name: Geom name
            geom_id: Geom ID (for category lookup)

        Returns:
            True if texture should be randomized, False otherwise
        """
        # First try to use category from scene_metadata
        if self.scene_metadata:
            category = self._get_geom_category(geom_id)
            if not category:
                category = name

                # Always convert category to lowercase for comparison
                category_lower = category.lower()
            # Check if any keyword is in the category
            keywords = self._get_target_keywords()
            if any(keyword in category_lower for keyword in keywords):
                return True

        return False

    def _build_name_to_geom_id(self) -> dict[str, int]:
        """Build name-to-geom_id mapping for visual geoms."""
        name_to_geom_id = {}
        for i in range(self.model.ngeom):
            if self.model.geom_contype[i] == 0:  # Only visual geoms
                name_adr = self.model.name_geomadr[i]
                if name_adr >= 0:
                    name_bytes = self.model.names[name_adr:]
                    name = name_bytes.split(b"\x00")[0].decode("utf-8")
                    if name:
                        name_to_geom_id[name] = i
        return name_to_geom_id

    def _randomize_material_attributes(self, geom_id: int, mat_id: int) -> None:
        """Randomize material attributes (rgba, specular, shininess) if enabled."""
        if mat_id >= 0:
            if self.randomize_material_rgba:
                self._randomize_material_rgba_direct(geom_id, mat_id)
            if self.randomize_material_specular:
                self._randomize_material_specular_direct(geom_id, mat_id)
            if self.randomize_material_shininess:
                self._randomize_material_shininess_direct(geom_id, mat_id)

    def randomize_object(self, thor_object: "MlSpacesObject", data: MjData | None = None) -> None:
        """
        Randomize colors and material attributes for a single MlSpacesObject.
        Only randomizes geoms that don't have textures.

        Args:
            thor_object: MlSpacesObject instance to randomize
            data: MuJoCo data for forward pass. If None, forward pass is skipped.
        """
        from molmo_spaces.env.data_views import MlSpacesObject

        if not isinstance(thor_object, MlSpacesObject):
            raise TypeError(f"Expected MlSpacesObject, got {type(thor_object)}")

        # Get all geoms for this object
        geom_infos = thor_object.get_geom_infos(include_descendants=True)
        if not geom_infos:
            return  # No geoms to randomize

        colors_randomized = 0

        # Process each geom
        for geom_info in geom_infos:
            geom_id = geom_info["id"]
            if geom_id < 0 or geom_id >= self.model.ngeom:
                continue

            # Check if geom has texture
            tex_id = self._get_texture_id_for_geom(geom_id)
            has_texture = tex_id >= 0

            # Only randomize if geom doesn't have texture
            if not has_texture:
                mat_id = int(self.model.geom_matid[geom_id])

                # Randomize colors and material attributes
                if self.randomize_geom_rgba:
                    self._randomize_geom_rgba_direct(geom_id)
                    colors_randomized += 1
                self._randomize_material_attributes(geom_id, mat_id)

        # Forward pass to propagate changes
        if data is not None:
            mujoco.mj_forward(self.model, data)

        # Debug: Print randomization stats (only occasionally to avoid spam)
        if colors_randomized > 0:
            import random

            if random.random() < 0.01:  # 1% chance to print
                print(
                    f"   Debug: Randomized {colors_randomized} geoms (colors/material) "
                    f"for object {thor_object.name}"
                )

    def _is_target_category(self, category: str | None, name: str) -> bool:
        """
        Check if category or name matches target categories for randomization.

        Uses the same keywords as _should_randomize_texture for consistency.
        Target categories: floors, countertops, tabletops, doors, walls, etc.
        Checks if any keyword is a substring of the category or name.

        Args:
            category: Category from scene_metadata (can be None)
            name: Geom name (used as fallback)

        Returns:
            True if any keyword is a substring of category or name, False otherwise
        """
        # Get keywords to check
        keywords = self._get_target_keywords()

        # Check if any keyword is a substring of the category (if available)
        if category:
            category_lower = category.lower()
            if any(keyword in category_lower for keyword in keywords):
                return True

        # Check if any keyword is a substring of the name
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in keywords)

    def _apply_material_to_geom(
        self, material_name: str, geom_id: int, mat_id: int
    ) -> tuple[bool, bool]:
        """
        Apply a material's texture or color to a geom.
        Randomly chooses between texture and color (if both are available).

        Args:
            material_name: Name of the material from MAT_TO_TEXTURE
            geom_id: Geom ID
            mat_id: Material ID

        Returns:
            Tuple of (texture_applied, color_applied) booleans
        """
        if material_name not in self.MAT_TO_TEXTURE:
            return False, False

        mat_dict = self.MAT_TO_TEXTURE[material_name]
        texture_applied = False
        color_applied = False

        # Check what's available
        texture_path = mat_dict.get("_MainTex")
        has_texture = texture_path and texture_path is not None and self.randomize_texture
        albedo_rgba_str = mat_dict.get("albedo_rgba")
        has_color = albedo_rgba_str is not None and albedo_rgba_str.strip() != ""

        # Prefer to choose texture if both color and texture are available
        if has_texture:
            use_texture = True
        elif has_color:
            use_texture = False
        else:
            # Neither texture nor color available
            return False, False

        # Apply texture if chosen
        if use_texture:
            # Load and apply texture
            selected_texture = None
            if texture_path in self._cat_texture_cache:
                selected_texture = self._cat_texture_cache[texture_path]
            else:
                selected_texture = self._load_texture_from_path(texture_path)
                if selected_texture is not None:
                    self._cat_texture_cache[texture_path] = selected_texture

            if selected_texture is not None:
                # Verify texture bitmap is valid before applying
                if selected_texture.size > 0 and len(selected_texture.shape) == 3:
                    import logging

                    log = logging.getLogger(__name__)
                    log.debug(
                        f"Applying texture to geom {geom_id}, material {mat_id}, "
                        f"texture_path={texture_path}, texture shape: {selected_texture.shape}"
                    )
                    # Apply texture first (this assigns empty material and updates texture data)
                    texture_set_success = self._set_texture_bitmap_by_id(geom_id, selected_texture)

                    if not texture_set_success:
                        import logging

                        log = logging.getLogger(__name__)
                        log.warning(
                            f"Failed to set texture bitmap for geom {geom_id}, material {mat_id}, "
                            f"texture_path={texture_path}. Falling back to color if available."
                        )
                        # Fall back to color if available
                        if has_color:
                            use_texture = False
                        else:
                            return False, False

                    # Verify texture was actually applied by checking if we got a valid texture ID
                    # Get the actual material ID after assignment (might be an empty material)
                    actual_mat_id = int(self.model.geom_matid[geom_id])
                    if actual_mat_id >= 0:
                        # Check if material has a texture assigned
                        tex_assigned = False
                        if hasattr(self.model, "mat_texid"):
                            try:
                                if isinstance(self.model.mat_texid, np.ndarray):
                                    if (
                                        self.model.mat_texid.ndim == 2
                                        and actual_mat_id < self.model.mat_texid.shape[0]
                                    ):
                                        assigned_tex_id = int(
                                            self.model.mat_texid[actual_mat_id, 0]
                                        )
                                        if assigned_tex_id >= 0:
                                            tex_assigned = True
                                    elif self.model.mat_texid.ndim == 1 and actual_mat_id < len(
                                        self.model.mat_texid
                                    ):
                                        assigned_tex_id = int(self.model.mat_texid[actual_mat_id])
                                        if assigned_tex_id >= 0:
                                            tex_assigned = True
                            except (IndexError, TypeError, ValueError):
                                pass

                        if tex_assigned:
                            texture_applied = True
                            # Set RGB to white [1, 1, 1] but preserve original alpha when using textures
                            # original_geom_alpha = self.model.geom_rgba[geom_id][3]
                            # self.model.geom_rgba[geom_id] = np.array([1.0, 1.0, 1.0, original_geom_alpha])
                            original_mat_alpha = self.model.mat_rgba[actual_mat_id][3]
                            self.model.mat_rgba[actual_mat_id] = np.array(
                                [1.0, 1.0, 1.0, original_mat_alpha]
                            )
                        else:
                            import logging

                            log = logging.getLogger(__name__)
                            log.warning(
                                f"Texture bitmap applied but material {actual_mat_id} has no texture assigned "
                                f"for geom {geom_id}, material {material_name}"
                            )
                            texture_applied = False
                    else:
                        import logging

                        log = logging.getLogger(__name__)
                        log.warning(
                            f"Geom {geom_id} has invalid material ID after texture application"
                        )
                        texture_applied = False
                else:
                    import logging

                    log = logging.getLogger(__name__)
                    log.warning(
                        f"Texture bitmap is invalid for geom {geom_id}, material {material_name}: "
                        f"size={selected_texture.size if selected_texture is not None else 0}, "
                        f"shape={selected_texture.shape if selected_texture is not None else None}"
                    )
                    texture_applied = False
            else:
                # Texture loading failed
                import logging

                log = logging.getLogger(__name__)
                log.warning(
                    f"Failed to load texture from path: {texture_path} for geom {geom_id}, material {material_name}"
                )
                # Fall back to color if available
                if has_color:
                    use_texture = False
                else:
                    return False, False

        # Apply color if chosen (or if texture failed and color is available)
        if not use_texture and has_color:
            try:
                # Parse "r g b a" string to float array
                rgba_values = [float(x) for x in albedo_rgba_str.split()]
                if len(rgba_values) >= 4:
                    rgba = np.array(rgba_values[:4], dtype=np.float32)
                    # Apply to geom
                    self.model.geom_rgba[geom_id] = rgba
                    # Apply to material if available
                    if mat_id >= 0:
                        self.model.mat_rgba[mat_id] = rgba
                    # Clear texture reference from material so color shows through
                    # This is important when using empty materials which have placeholder textures
                    if hasattr(self.model, "mat_texid"):
                        try:
                            if isinstance(self.model.mat_texid, np.ndarray):
                                if (
                                    self.model.mat_texid.ndim == 2
                                    and mat_id < self.model.mat_texid.shape[0]
                                ):
                                    # Clear all texture roles
                                    for role_idx in range(self.model.mat_texid.shape[1]):
                                        self.model.mat_texid[mat_id, role_idx] = -1
                                elif self.model.mat_texid.ndim == 1 and mat_id < len(
                                    self.model.mat_texid
                                ):
                                    self.model.mat_texid[mat_id] = -1
                            else:
                                self.model.mat_texid[mat_id] = -1
                        except (IndexError, TypeError, ValueError):
                            pass
                    color_applied = True
            except (ValueError, AttributeError):
                pass

        return texture_applied, color_applied

    def randomize_by_category(self, data: MjData | None = None):
        """
        Randomize textures and colors by category.
        For each geom, randomly picks a material from the appropriate category in MAT_PER_CATEGORY
        and applies its texture (if available) or color. Each geom gets a different random material.

        Targets: floors, countertops, tabletops, doors (including door handles, drawers, cabinets)
        """
        name_to_geom_id = self._build_name_to_geom_id()

        # Track randomization stats for debugging
        textures_randomized = 0
        colors_randomized = 0

        # Group geoms by category for batch processing
        geoms_by_category: dict[str, list[tuple[str, int, int]]] = {}
        # Format: {category_key: [(name, geom_id, mat_id), ...]}

        # Process only geoms in our list (which are already filtered to visual geoms)
        for name in self.geom_names:
            geom_id = name_to_geom_id.get(name, -1)
            if geom_id < 0:
                continue

            # Get category from scene_metadata, fallback to name
            category = self._get_geom_category(geom_id)
            if not category:
                category = name

            # Check if this geom is in target categories
            if not self._is_target_category(category, name):
                continue  # Skip geoms not in target categories

            # Find matching category key from MAT_PER_CATEGORY
            category_lower = category.lower() if category else ""
            name_lower = name.lower()

            # Hard-coded category mapping for ProcTHOR and iTHOR
            # Maps scene category keywords to MAT_PER_CATEGORY keys
            type_map = {
                "room": "floor",
                "doorway": "doorway",
                "door": "doorway",  # Map "door" to "Doorway" in MAT_PER_CATEGORY
                "handle": "doorway",
                "drawer": "doorway",
                "cabinet": "doorway",
                "counter": "countertop",  # Map "counter" to "CounterTop" in MAT_PER_CATEGORY
                "island": "table",  # Map "island" to "CounterTop" in MAT_PER_CATEGORY
                "plane": "table",
                "mesh": "wall",
                "backsplash": "wall",
                "quad": "wall",
            }

            # First, try to map using type_map (check both category and name)
            # Check name first since it's more specific, then category
            mapped_category = None
            for type_key, type_value in type_map.items():
                # Check if keyword is in name (more reliable)
                if type_key.lower() in name_lower:
                    mapped_category = type_value
                    break
                # Also check category if available
                if category and type_key.lower() in category_lower:
                    mapped_category = type_value
                    break

            # If we found a mapping, use it; otherwise use the original category or name
            if mapped_category:
                search_category = mapped_category
            elif category:
                search_category = category_lower
            else:
                search_category = name_lower

            # Find matching category key from MAT_PER_CATEGORY
            matching_category_key = None
            search_category_lower = search_category.lower()

            # First try exact case-insensitive match
            for cat_key in self.MAT_PER_CATEGORY:
                if cat_key.lower() == search_category_lower:
                    matching_category_key = cat_key
                    break

            # If no exact match, try substring matching (bidirectional)
            if matching_category_key is None:
                for cat_key in self.MAT_PER_CATEGORY:
                    cat_key_lower = cat_key.lower()
                    # Check if search_category matches cat_key (bidirectional substring match)
                    if (
                        cat_key_lower in search_category_lower
                        or search_category_lower in cat_key_lower
                    ):
                        matching_category_key = cat_key
                        break

            if matching_category_key is None:
                continue  # No matching category in MAT_PER_CATEGORY

            mat_id = int(self.model.geom_matid[geom_id])

            # Group by category key
            if matching_category_key not in geoms_by_category:
                geoms_by_category[matching_category_key] = []
            geoms_by_category[matching_category_key].append((name, geom_id, mat_id))

        # For each category, randomly pick a different material for each geom
        for category_key, geoms in geoms_by_category.items():
            if not geoms:
                continue

            # Get materials for this category from MAT_PER_CATEGORY
            category_materials = self.MAT_PER_CATEGORY.get(category_key, [])
            if not category_materials:
                continue

            # Apply a randomly selected material to each geom (different for each)
            # IMPORTANT: Each geom gets its own independent random material selection
            # Only uses materials from MAT_PER_CATEGORY for this specific category
            selected_materials = []  # Track selected materials for debugging
            for name, geom_id, _ in geoms:
                # Assign an empty material to this geom first (ensures isolation from other geoms)
                empty_mat_id = self._assign_empty_material_to_geom(geom_id)
                # Get the new material ID after assignment
                new_mat_id = int(self.model.geom_matid[geom_id])

                # Verify the geom actually got the empty material assigned
                if new_mat_id != empty_mat_id:
                    import logging

                    log = logging.getLogger(__name__)
                    log.warning(
                        f"Geom '{name}' (id={geom_id}) expected empty material {empty_mat_id} "
                        f"but got material {new_mat_id}"
                    )

                # Randomly pick one material from this category's valid materials
                # Each geom gets its own random selection - this ensures variety
                # Only selects from materials in MAT_PER_CATEGORY for this category
                material_idx = self.random_state.randint(len(category_materials))
                selected_material = category_materials[material_idx]
                selected_materials.append(selected_material)

                texture_applied, color_applied = self._apply_material_to_geom(
                    selected_material, geom_id, new_mat_id
                )
                if texture_applied:
                    textures_randomized += 1
                    # Verify texture is actually assigned to material
                    actual_mat_id = int(self.model.geom_matid[geom_id])
                    if actual_mat_id >= 0 and hasattr(self.model, "mat_texid"):
                        try:
                            if isinstance(self.model.mat_texid, np.ndarray):
                                if (
                                    self.model.mat_texid.ndim == 2
                                    and actual_mat_id < self.model.mat_texid.shape[0]
                                ):
                                    assigned_tex_id = int(self.model.mat_texid[actual_mat_id, 0])
                                    if assigned_tex_id >= 0:
                                        # Verify texture data is non-zero
                                        tex_adr = int(self.model.tex_adr[assigned_tex_id])
                                        tex_size = int(
                                            self.model.tex_height[assigned_tex_id]
                                            * self.model.tex_width[assigned_tex_id]
                                            * self.model.tex_nchannel[assigned_tex_id]
                                        )
                                        if tex_adr + tex_size <= len(self.model.tex_data):
                                            tex_data_sum = np.sum(
                                                np.abs(
                                                    self.model.tex_data[
                                                        tex_adr : tex_adr + tex_size
                                                    ]
                                                )
                                            )
                                            if tex_data_sum == 0:
                                                import logging

                                                log = logging.getLogger(__name__)
                                                log.warning(
                                                    f"Geom {geom_id} has texture {assigned_tex_id} assigned to material {actual_mat_id}, "
                                                    f"but texture data is all zeros!"
                                                )
                        except (IndexError, TypeError, ValueError):
                            pass
                elif color_applied:
                    colors_randomized += 1
                # Note: If material has texture, color is set to white [1,1,1] to show texture
                # If material has no texture, color from albedo_rgba is applied

        # Forward pass to propagate changes
        if data is not None:
            mujoco.mj_forward(self.model, data)

        # Debug: Print randomization stats (only occasionally to avoid spam)
        if textures_randomized > 0 or colors_randomized > 0:
            import random

            if random.random() < 0.01:  # 1% chance to print
                print(
                    f"   Debug: Randomized {textures_randomized} textures, "
                    f"{colors_randomized} colors by category (material-based)"
                )

    def _randomize_geom_rgba_direct(self, geom_id: int):
        """Randomize geom RGBA color using direct geom_id (faster). Preserves alpha channel."""
        defaults = self._geom_id_to_defaults.get(geom_id)
        if defaults:
            delta = self.random_state.uniform(
                low=-self.rgba_perturbation_size,
                high=self.rgba_perturbation_size,
                size=3,  # Only randomize RGB, not alpha
            )
            original_alpha = defaults["geom_rgba"][3]
            new_rgb = np.clip(defaults["geom_rgba"][:3] + delta, 0.0, 1.0)
            new_rgba = np.array([new_rgb[0], new_rgb[1], new_rgb[2], original_alpha])
            self.model.geom_rgba[geom_id] = new_rgba

    def _randomize_material_rgba_direct(self, geom_id: int, mat_id: int):
        """Randomize material RGBA using direct IDs (faster). Preserves alpha channel."""
        defaults = self._geom_id_to_defaults.get(geom_id)
        if defaults and defaults.get("mat_rgba") is not None:
            delta = self.random_state.uniform(
                low=-self.rgba_perturbation_size,
                high=self.rgba_perturbation_size,
                size=3,  # Only randomize RGB, not alpha
            )
            original_alpha = defaults["mat_rgba"][3]
            new_rgb = np.clip(defaults["mat_rgba"][:3] + delta, 0.0, 1.0)
            new_rgba = np.array([new_rgb[0], new_rgb[1], new_rgb[2], original_alpha])
            self.model.mat_rgba[mat_id] = new_rgba

    def _randomize_material_specular_direct(self, geom_id: int, mat_id: int):
        """Randomize material specular using direct IDs (faster)."""
        defaults = self._geom_id_to_defaults.get(geom_id)
        if defaults and defaults.get("mat_specular") is not None:
            delta = self.random_state.uniform(
                low=-self.specular_perturbation_size,
                high=self.specular_perturbation_size,
            )
            new_specular = np.clip(defaults["mat_specular"] + delta, 0.0, 1.0)
            self.model.mat_specular[mat_id] = new_specular

    def _randomize_material_shininess_direct(self, geom_id: int, mat_id: int):
        """Randomize material shininess using direct IDs (faster)."""
        defaults = self._geom_id_to_defaults.get(geom_id)
        if defaults and defaults.get("mat_shininess") is not None:
            delta = self.random_state.uniform(
                low=-self.shininess_perturbation_size,
                high=self.shininess_perturbation_size,
            )
            new_shininess = np.clip(defaults["mat_shininess"] + delta, 0.0, 1.0)
            self.model.mat_shininess[mat_id] = new_shininess

    def _get_texture_id_for_geom(self, geom_id: int) -> int:
        """
        Get texture ID for a geom if it has a material with a 2D texture.
        Checks all texture roles, not just RGB role.

        Only returns 2D textures (mjTEXTURE_2D = 0) since we can only randomize
        those. Cube textures (mjTEXTURE_CUBE) and skybox textures (mjTEXTURE_SKYBOX)
        are not supported for randomization.

        Args:
            geom_id: Geom ID

        Returns:
            Texture ID for a 2D texture, or -1 if geom has no 2D texture
        """
        mat_id = int(self.model.geom_matid[geom_id])
        if mat_id < 0 or mat_id >= self.model.nmat:
            return -1

        # Check if material has a texture assigned
        # In MuJoCo, materials can reference textures via mat_texid
        # mat_texid is a 2D array (nmat, ntexrole) where ntexrole is number of texture roles
        # mjTEXROLE_RGB = 0 is the main texture role, but textures can be in other roles too
        if hasattr(self.model, "mat_texid"):
            try:
                if isinstance(self.model.mat_texid, np.ndarray):
                    # Check if it's 2D array - check all texture roles
                    if self.model.mat_texid.ndim == 2 and mat_id < self.model.mat_texid.shape[0]:
                        # Check all texture roles (columns) for this material
                        for role_idx in range(self.model.mat_texid.shape[1]):
                            tex_id = int(self.model.mat_texid[mat_id, role_idx])
                            # In MuJoCo, -1 means no texture, valid texture IDs are >= 0
                            if tex_id >= 0 and tex_id < self.model.ntex:
                                # Only return 2D textures (mjTEXTURE_2D = 0)
                                # Cube textures (mjTEXTURE_CUBE) and skybox (mjTEXTURE_SKYBOX)
                                # are not supported for randomization
                                tex_type = int(self.model.tex_type[tex_id])
                                if tex_type == 0:  # mjTEXTURE_2D
                                    return tex_id
                    elif self.model.mat_texid.ndim == 1 and mat_id < len(self.model.mat_texid):
                        tex_id = int(self.model.mat_texid[mat_id])
                        if tex_id >= 0 and tex_id < self.model.ntex:
                            tex_type = int(self.model.tex_type[tex_id])
                            if tex_type == 0:  # mjTEXTURE_2D
                                return tex_id
                else:
                    # Try direct indexing (1D array)
                    tex_id = int(self.model.mat_texid[mat_id])
                    if tex_id >= 0 and tex_id < self.model.ntex:
                        tex_type = int(self.model.tex_type[tex_id])
                        if tex_type == 0:  # mjTEXTURE_2D
                            return tex_id
            except (IndexError, TypeError, ValueError):
                # mat_texid might not be accessible or structured differently
                return -1
        return -1

    def _get_texture_bitmap(self, tex_id: int, use_cache: bool = True) -> np.ndarray:
        """
        Get texture bitmap for a texture ID.
        Uses caching to avoid re-extracting the same texture multiple times.

        Args:
            tex_id: Texture ID
            use_cache: If True, use cached version if available

        Returns:
            Texture bitmap as (height, width, nchannel) array
        """
        # Check cache first
        if use_cache and tex_id in self._texture_cache:
            return self._texture_cache[tex_id].copy()  # Return copy to avoid mutations

        # Extract texture
        height = int(self.model.tex_height[tex_id])
        width = int(self.model.tex_width[tex_id])
        nchannel = int(self.model.tex_nchannel[tex_id])
        tex_adr = int(self.model.tex_adr[tex_id])
        size = height * width * nchannel
        data = self.model.tex_data[tex_adr : tex_adr + size]
        bitmap = data.reshape((height, width, nchannel))

        # Cache it
        if use_cache:
            self._texture_cache[tex_id] = bitmap.copy()

        return bitmap

    def _assign_empty_material_to_geom(self, geom_id: int) -> int:
        """
        Assign an empty material to a geom if it doesn't already have one.
        Returns the material ID assigned to the geom.

        Args:
            geom_id: Geom ID

        Returns:
            Material ID assigned to the geom
        """
        # Check if this geom already has an empty material assigned
        if geom_id in self._geom_to_empty_material:
            return self._geom_to_empty_material[geom_id]

        # Check if we have any empty materials available
        if not self._empty_material_ids:
            # No empty materials available, return the current material
            current_mat_id = int(self.model.geom_matid[geom_id])
            return current_mat_id

        # Get the next available empty material
        if self._next_empty_material_index >= len(self._empty_material_ids):
            # All materials used, cycle back (reuse materials)
            self._next_empty_material_index = 0

        empty_mat_id = self._empty_material_ids[self._next_empty_material_index]
        self._next_empty_material_index += 1

        # Get the original material to copy its properties
        original_mat_id = int(self.model.geom_matid[geom_id])

        # Store original material and texture IDs for later use
        self._geom_to_original_material[geom_id] = original_mat_id
        original_tex_id = self._get_texture_id_for_geom(geom_id) if original_mat_id >= 0 else -1
        self._geom_to_original_texture[geom_id] = original_tex_id

        if original_mat_id >= 0:
            # Copy properties from original material to empty material
            self.model.mat_rgba[empty_mat_id] = self.model.mat_rgba[original_mat_id].copy()
            self.model.mat_specular[empty_mat_id] = self.model.mat_specular[original_mat_id]
            self.model.mat_shininess[empty_mat_id] = self.model.mat_shininess[original_mat_id]
            if hasattr(self.model, "mat_emission"):
                self.model.mat_emission[empty_mat_id] = self.model.mat_emission[
                    original_mat_id
                ].copy()
            if hasattr(self.model, "mat_reflectance"):
                self.model.mat_reflectance[empty_mat_id] = self.model.mat_reflectance[
                    original_mat_id
                ]

            # DO NOT copy the original texture reference - use the empty material's own texture slot instead
            # The empty material has its own texture slot (from MjSpec setup) that we can use for new textures
            # This ensures we don't modify shared textures that other geoms might be using
            # If we need to preserve the original texture, we'll handle that when applying the new texture/color

        # Assign the empty material to this geom
        self.model.geom_matid[geom_id] = empty_mat_id

        # CRITICAL: Ensure the empty material has its texture assigned in mat_texid
        # This is necessary even if we're not applying a texture bitmap yet,
        # because the material should have a texture slot ready for when we do apply one
        if empty_mat_id in self._empty_material_to_texture:
            tex_id = self._empty_material_to_texture[empty_mat_id]
            if tex_id >= 0 and hasattr(self.model, "mat_texid"):
                try:
                    if isinstance(self.model.mat_texid, np.ndarray):
                        if (
                            self.model.mat_texid.ndim == 2
                            and empty_mat_id < self.model.mat_texid.shape[0]
                        ):
                            # Check if texture is already assigned
                            current_tex_id = int(self.model.mat_texid[empty_mat_id, 0])
                            if current_tex_id != tex_id:
                                self.model.mat_texid[empty_mat_id, 0] = tex_id
                                import logging

                                log = logging.getLogger(__name__)
                                log.debug(
                                    f"✓ Assigned texture {tex_id} to empty material {empty_mat_id} "
                                    f"for geom {geom_id} during material assignment"
                                )
                        elif self.model.mat_texid.ndim == 1 and empty_mat_id < len(
                            self.model.mat_texid
                        ):
                            # Check if texture is already assigned
                            current_tex_id = int(self.model.mat_texid[empty_mat_id])
                            if current_tex_id != tex_id:
                                self.model.mat_texid[empty_mat_id] = tex_id
                                import logging

                                log = logging.getLogger(__name__)
                                log.debug(
                                    f"✓ Assigned texture {tex_id} to empty material {empty_mat_id} "
                                    f"for geom {geom_id} during material assignment"
                                )
                except (IndexError, TypeError, ValueError) as e:
                    import logging

                    log = logging.getLogger(__name__)
                    log.warning(
                        f"Failed to assign texture {tex_id} to empty material {empty_mat_id} "
                        f"for geom {geom_id}: {e}"
                    )

        # Store the mapping
        self._geom_to_empty_material[geom_id] = empty_mat_id
        return empty_mat_id

    def _set_texture_bitmap_by_id(self, geom_id: int, bitmap: np.ndarray) -> bool:
        """
        Set texture bitmap for a geom by geom_id (faster than by name).
        Assigns an empty material to the geom first to avoid affecting other geoms.

        Args:
            geom_id: Geom ID
            bitmap: Texture bitmap as (height, width, nchannel) array

        Returns:
            True if texture bitmap was successfully set, False otherwise
        """
        if geom_id < 0:
            return False

        # Assign an empty material to this geom first (to avoid affecting other geoms)
        empty_mat_id = self._assign_empty_material_to_geom(geom_id)

        # Get the empty material's own texture slot (the placeholder texture from MjSpec setup)
        # This ensures we're updating a unique texture slot, not a shared one
        tex_id = -1
        if empty_mat_id >= 0:
            # Use the pre-computed mapping from material ID to texture ID
            if empty_mat_id in self._empty_material_to_texture:
                tex_id = self._empty_material_to_texture[empty_mat_id]
            else:
                # Fallback: try to find texture by matching names
                mat_name_adr = self.model.name_matadr[empty_mat_id]
                if mat_name_adr >= 0:
                    name_bytes = self.model.names[mat_name_adr:]
                    mat_name = name_bytes.split(b"\x00")[0].decode("utf-8")
                    # Extract the index from the material name (e.g., "__TEXTURE_RANDOMIZER_MAT_5__" -> 5)
                    if mat_name.startswith("__TEXTURE_RANDOMIZER_MAT_") and mat_name.endswith("__"):
                        try:
                            mat_index_str = mat_name[len("__TEXTURE_RANDOMIZER_MAT_") : -len("__")]
                            mat_index = int(mat_index_str)
                            # Find texture with the same index
                            for tex_idx, tex_name in enumerate(self._empty_texture_names):
                                if tex_name.startswith(
                                    "__TEXTURE_RANDOMIZER_TEX_"
                                ) and tex_name.endswith("__"):
                                    tex_index_str = tex_name[
                                        len("__TEXTURE_RANDOMIZER_TEX_") : -len("__")
                                    ]
                                    tex_index = int(tex_index_str)
                                    if tex_index == mat_index:
                                        tex_id = self._empty_texture_ids[tex_idx]
                                        # Cache the mapping for future use
                                        self._empty_material_to_texture[empty_mat_id] = tex_id
                                        break
                        except (ValueError, IndexError):
                            pass

            # CRITICAL: Ensure this texture is assigned to the material BEFORE writing texture data
            # This must happen before we write to model.tex_data, otherwise the texture won't be visible
            if tex_id >= 0 and hasattr(self.model, "mat_texid"):
                try:
                    if isinstance(self.model.mat_texid, np.ndarray):
                        if (
                            self.model.mat_texid.ndim == 2
                            and empty_mat_id < self.model.mat_texid.shape[0]
                        ):
                            # Always assign to RGB role (role 0) to ensure it's set
                            self.model.mat_texid[empty_mat_id, 0] = tex_id
                            import logging

                            log = logging.getLogger(__name__)
                            log.debug(
                                f"✓ Assigned texture {tex_id} to empty material {empty_mat_id} "
                                f"(role 0) for geom {geom_id}"
                            )
                        elif self.model.mat_texid.ndim == 1 and empty_mat_id < len(
                            self.model.mat_texid
                        ):
                            self.model.mat_texid[empty_mat_id] = tex_id
                            import logging

                            log = logging.getLogger(__name__)
                            log.debug(
                                f"✓ Assigned texture {tex_id} to empty material {empty_mat_id} "
                                f"for geom {geom_id}"
                            )
                        else:
                            import logging

                            log = logging.getLogger(__name__)
                            log.warning(
                                f"Failed to assign texture {tex_id} to empty material {empty_mat_id}: "
                                f"mat_texid shape mismatch (ndim={self.model.mat_texid.ndim}, "
                                f"shape={self.model.mat_texid.shape}, empty_mat_id={empty_mat_id})"
                            )
                except (IndexError, TypeError, ValueError) as e:
                    import logging

                    log = logging.getLogger(__name__)
                    log.warning(
                        f"Failed to assign texture {tex_id} to empty material {empty_mat_id}: {e}"
                    )
                    import traceback

                    log.debug(traceback.format_exc())

        if tex_id < 0:
            # No empty texture found - this means the empty material doesn't have a texture slot
            # This can happen if texture creation failed during setup_empty_materials
            import logging

            log = logging.getLogger(__name__)
            log.warning(
                f"Geom {geom_id} with empty material {empty_mat_id} has no texture slot to update. "
                f"Empty materials: {len(self._empty_material_ids)}, Empty textures: {len(self._empty_texture_ids)}, "
                f"Mapped pairs: {len(self._empty_material_to_texture)}"
            )
            return False  # No texture to set

        height = int(self.model.tex_height[tex_id])
        width = int(self.model.tex_width[tex_id])
        nchannel = int(self.model.tex_nchannel[tex_id])
        tex_adr = int(self.model.tex_adr[tex_id])

        # Ensure bitmap has correct number of channels
        if bitmap.shape[2] != nchannel:
            # Convert to correct number of channels
            if nchannel == 3 and bitmap.shape[2] == 4:
                # RGBA to RGB
                bitmap = bitmap[:, :, :3]
            elif nchannel == 4 and bitmap.shape[2] == 3:
                # RGB to RGBA (add alpha channel)
                alpha = np.ones((bitmap.shape[0], bitmap.shape[1], 1), dtype=bitmap.dtype) * 255
                bitmap = np.concatenate([bitmap, alpha], axis=2)
            elif bitmap.shape[2] > nchannel:
                bitmap = bitmap[:, :, :nchannel]
            else:
                # Pad with channels
                padding = (
                    np.ones(
                        (bitmap.shape[0], bitmap.shape[1], nchannel - bitmap.shape[2]),
                        dtype=bitmap.dtype,
                    )
                    * 255
                )
                bitmap = np.concatenate([bitmap, padding], axis=2)

        # Resize bitmap if needed
        if bitmap.shape[:2] != (height, width):
            try:
                from PIL import Image
            except ImportError as err:
                raise ImportError(
                    "PIL (Pillow) is required for texture resizing. Install with: pip install Pillow"
                ) from err
            img = Image.fromarray(bitmap.astype(np.uint8))
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            bitmap = np.array(img, dtype=np.uint8)

        # Ensure bitmap is in [0, 255] range
        bitmap = np.clip(bitmap, 0, 255).astype(np.uint8)

        # Copy to model texture data
        size = height * width * nchannel
        if tex_adr + size > len(self.model.tex_data):
            import logging

            log = logging.getLogger(__name__)
            log.error(
                f"Texture data overflow: tex_id={tex_id}, tex_adr={tex_adr}, size={size}, "
                f"tex_data_len={len(self.model.tex_data)}, height={height}, width={width}, nchannel={nchannel}"
            )
            return False

        # Flatten bitmap and ensure it's the right size
        bitmap_flat = bitmap.flatten()
        if len(bitmap_flat) != size:
            import logging

            log = logging.getLogger(__name__)
            log.error(
                f"Texture bitmap size mismatch: expected {size} elements, got {len(bitmap_flat)}. "
                f"Bitmap shape: {bitmap.shape}, texture size: {width}x{height}x{nchannel}"
            )
            return False

        # Write texture data
        self.model.tex_data[tex_adr : tex_adr + size] = bitmap_flat

        # CRITICAL: Verify and RE-ASSIGN texture to material AFTER writing texture data
        # This ensures the texture assignment is correct even if something cleared it
        if hasattr(self.model, "mat_texid"):
            try:
                if isinstance(self.model.mat_texid, np.ndarray):
                    if (
                        self.model.mat_texid.ndim == 2
                        and empty_mat_id < self.model.mat_texid.shape[0]
                    ):
                        assigned_tex_id = int(self.model.mat_texid[empty_mat_id, 0])
                        if assigned_tex_id != tex_id:
                            import logging

                            log = logging.getLogger(__name__)
                            log.warning(
                                f"⚠ Texture {tex_id} updated but material {empty_mat_id} has texture {assigned_tex_id} assigned. "
                                f"Re-assigning texture to material."
                            )
                            self.model.mat_texid[empty_mat_id, 0] = tex_id
                        else:
                            import logging

                            log = logging.getLogger(__name__)
                            log.debug(
                                f"✓ Verified texture {tex_id} is assigned to material {empty_mat_id} "
                                f"(role 0) for geom {geom_id}"
                            )
                    elif self.model.mat_texid.ndim == 1 and empty_mat_id < len(
                        self.model.mat_texid
                    ):
                        assigned_tex_id = int(self.model.mat_texid[empty_mat_id])
                        if assigned_tex_id != tex_id:
                            import logging

                            log = logging.getLogger(__name__)
                            log.warning(
                                f"⚠ Texture {tex_id} updated but material {empty_mat_id} has texture {assigned_tex_id} assigned. "
                                f"Re-assigning texture to material."
                            )
                            self.model.mat_texid[empty_mat_id] = tex_id
                        else:
                            import logging

                            log = logging.getLogger(__name__)
                            log.debug(
                                f"✓ Verified texture {tex_id} is assigned to material {empty_mat_id} "
                                f"for geom {geom_id}"
                            )
                    else:
                        # mat_texid shape doesn't match - try to assign anyway
                        import logging

                        log = logging.getLogger(__name__)
                        log.warning(
                            f"⚠ mat_texid shape mismatch for material {empty_mat_id}. "
                            f"Attempting to assign texture {tex_id} anyway."
                        )
                        try:
                            if self.model.mat_texid.ndim == 2:
                                self.model.mat_texid[empty_mat_id, 0] = tex_id
                            else:
                                self.model.mat_texid[empty_mat_id] = tex_id
                        except Exception:
                            pass
            except (IndexError, TypeError, ValueError) as e:
                import logging

                log = logging.getLogger(__name__)
                log.warning(f"Could not verify texture assignment to material {empty_mat_id}: {e}")
                import traceback

                log.debug(traceback.format_exc())

        # Debug logging
        import logging

        log = logging.getLogger(__name__)
        log.debug(
            f"✓ Successfully updated texture {tex_id} for geom {geom_id} (empty_mat={empty_mat_id}): "
            f"size={width}x{height}x{nchannel}, tex_adr={tex_adr}, data_size={size}, "
            f"bitmap_shape={bitmap.shape}, bitmap_dtype={bitmap.dtype}"
        )

        # Verify texture data was actually written (not all zeros)
        written_data_sum = np.sum(np.abs(self.model.tex_data[tex_adr : tex_adr + size]))
        bitmap_sum = np.sum(np.abs(bitmap_flat))
        if written_data_sum == 0:
            log.error(
                f"✗ Texture {tex_id} data write FAILED - all zeros after write! "
                f"Expected sum={bitmap_sum}, got sum={written_data_sum}"
            )
            return False
        elif written_data_sum != bitmap_sum:
            log.warning(
                f"⚠ Texture {tex_id} data write mismatch: expected sum={bitmap_sum}, "
                f"got sum={written_data_sum}"
            )
            # Still return True as some data was written
        else:
            log.debug(
                f"✓ Texture {tex_id} data verified: sum={written_data_sum}, "
                f"size={size}, tex_adr={tex_adr}"
            )

        return True  # Success

    def _randomize_texture_all(self, name: str, geom_id: int, mat_id: int) -> None:
        """
        Randomize texture by applying a randomly selected texture from already loaded textures.
        This method ignores category filtering and randomizes all textures using already loaded textures
        (from texture_bitmaps, _cat_texture_cache, or model textures).

        Args:
            name: Geom name
            geom_id: Geom ID
            mat_id: Material ID
        """
        # Get the target texture ID for this geom
        target_tex_id = self._get_texture_id_for_geom(geom_id)
        if target_tex_id < 0:
            return  # No texture to randomize

        # Collect all available textures from different sources
        available_textures = []

        # 1. Add textures from texture_bitmaps (if external textures were provided)
        if self.texture_paths and self.texture_bitmaps:
            available_textures.extend(self.texture_bitmaps)

        # 2. Add textures from _cat_texture_cache (loaded from CAT_TO_TEXTURE paths)
        if self._cat_texture_cache:
            available_textures.extend(self._cat_texture_cache.values())

        # 3. If no loaded textures available, use model textures
        if not available_textures:
            if not self.texture_ids:
                return  # No textures available

            # Randomly select a source texture ID from model
            if len(self.texture_ids) == 1:
                return  # Only one texture available, can't randomize

            available_ids = [tid for tid in self.texture_ids if tid != target_tex_id]
            if not available_ids:
                available_ids = self.texture_ids

            source_tex_id = available_ids[self.random_state.randint(len(available_ids))]

            # Extract bitmap on-demand with caching
            try:
                selected_texture = self._get_texture_bitmap(source_tex_id, use_cache=True)
            except Exception:
                return  # Skip if extraction fails
        else:
            # Randomly select from already loaded textures
            selected_texture = available_textures[
                self.random_state.randint(len(available_textures))
            ]

        # Apply the selected texture to the target
        success = self._set_texture_bitmap_by_id(geom_id, selected_texture)
        if not success:
            import logging

            log = logging.getLogger(__name__)
            log.warning(
                f"Failed to set texture bitmap for geom {geom_id} in _randomize_texture_all"
            )

    def randomize(self, data: MjData | None = None) -> None:
        """
        Randomize all textures, colors, and material attributes for all geoms, regardless of category.
        This method bypasses category filtering and applies full randomization to all geoms.

        Args:
            data (MjData | None): MuJoCo data for forward pass. If None, forward pass is skipped.
        """
        name_to_geom_id = self._build_name_to_geom_id()

        # Track randomization stats for debugging
        textures_randomized = 0
        colors_randomized = 0

        # Process all geoms
        for name in self.geom_names:
            geom_id = name_to_geom_id.get(name, -1)
            if geom_id < 0:
                continue

            mat_id = int(self.model.geom_matid[geom_id])

            # Randomize texture if available
            tex_id = self._get_texture_id_for_geom(geom_id)
            if tex_id >= 0:
                # Check if we have textures available (either from files or model)
                has_textures = (self.texture_paths and len(self.texture_bitmaps) > 0) or len(
                    self.texture_ids
                ) > 0
                if has_textures and self.randomize_texture:
                    # Randomize texture without category filtering
                    self._randomize_texture_all(name, geom_id, mat_id)
                    textures_randomized += 1

                    # Always set RGB to white [1, 1, 1] but preserve original alpha when using textures
                    # so texture renders at full intensity while maintaining transparency
                    original_geom_alpha = self.model.geom_rgba[geom_id][3]
                    self.model.geom_rgba[geom_id] = np.array([1.0, 1.0, 1.0, original_geom_alpha])
                    if mat_id >= 0:
                        original_mat_alpha = self.model.mat_rgba[mat_id][3]
                        self.model.mat_rgba[mat_id] = np.array([1.0, 1.0, 1.0, original_mat_alpha])
                else:
                    # No texture or texture randomization disabled, randomize colors and materials
                    if self.randomize_geom_rgba:
                        self._randomize_geom_rgba_direct(geom_id)
                        colors_randomized += 1
                    self._randomize_material_attributes(geom_id, mat_id)
            else:
                # No texture, randomize colors and materials
                if self.randomize_geom_rgba:
                    self._randomize_geom_rgba_direct(geom_id)
                    colors_randomized += 1
                self._randomize_material_attributes(geom_id, mat_id)

        # Forward pass to propagate changes
        # IMPORTANT: mj_forward must be called after texture changes for them to take effect
        if data is not None:
            mujoco.mj_forward(self.model, data)

        # Debug: Print randomization stats (only occasionally to avoid spam)
        if textures_randomized > 0 or colors_randomized > 0:
            import random

            if random.random() < 0.01:  # 1% chance to print
                print(
                    f"   Debug: Randomized {textures_randomized} textures, "
                    f"{colors_randomized} colors (all, no category filter)"
                )
