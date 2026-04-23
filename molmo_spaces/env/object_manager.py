import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Collection
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mujoco
import numpy as np
import stringcase

from molmo_spaces.env.data_views import (
    Door,
    MlSpacesArticulationObject,
    MlSpacesFreeJointBody,
    MlSpacesObject,
    create_mlspaces_body,
)
from molmo_spaces.utils.constants.object_constants import (
    AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA,
)
from molmo_spaces.utils.lemma_utils import normalize_expression
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms, geom_aabb
from molmo_spaces.utils.object_metadata import ObjectMeta, clip_sim, compute_text_clip
from molmo_spaces.utils.synset_utils import (
    filter_synsets_to_remove_hyponyms,
    generate_all_hypernyms_with_exclusions,
    is_hypernym_of,
    wn,
)

if TYPE_CHECKING:
    from molmo_spaces.env.env import BaseMujocoEnv

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
"""
- Object can be a MlSpacesObject or instance of some subclass
- Name is the object's name
- Id is the object's top-level body id
"""
ObjectOrNameOrIdType = MlSpacesObject | str | int


class Context(Enum):
    SCENE = "scene"
    # ROOM = "room"
    BENCH = "bench"
    # VISIBLE = "visible"
    OBJECT = "object"


class ObjectManager:
    STRUCTURAL_TYPES = {
        "world",
        "room",
        "floor",
        "wall",
        "window",
        # "door",
        "doorframe",
        "doorway",
        "ceiling",
    }

    def __init__(
        self,
        env: "BaseMujocoEnv",
        batch_idx: int,
        caching_enabled: bool = True,
        name_caching_enabled: bool = True,
    ):
        self._env = env
        self._batch_idx = batch_idx
        self.data = env.mj_datas[batch_idx]

        # Caches for natural names and possible types
        self._name_caching_enabled = name_caching_enabled
        self._object_name_and_context_to_source_to_natural_names = {}
        self._object_name_and_context_to_natural_names = {}
        self._object_name_to_possible_type_names = {}

        # More caches
        self._caching_enabled = caching_enabled
        self._model_cache: dict[str, Any] = defaultdict(dict)
        self._data_cache: dict[str, Any] = defaultdict(dict)

    @cached_property
    def scene_metadata(self) -> dict | None:
        return self._env.current_scene_metadata

    def invalidate_data_cache(self) -> None:
        self._data_cache.clear()

    def invalidate_model_cache(self) -> None:
        self._model_cache.clear()

    def invalidate_all_caches(self) -> None:
        self._model_cache.clear()
        self._data_cache.clear()
        self._object_name_to_possible_type_names.clear()
        self._object_name_and_context_to_natural_names.clear()
        self._object_name_and_context_to_source_to_natural_names.clear()

        if "scene_metadata" in self.__dict__:
            del self.__dict__["scene_metadata"]
        if "_mesh_name_to_file" in self.__dict__:
            del self.__dict__["_mesh_name_to_file"]

    def object_metadata(self, object_or_name_or_id: ObjectOrNameOrIdType) -> dict:
        return (
            (self.scene_metadata or {})
            .get("objects", {})
            .get(self.get_object_name(object_or_name_or_id), {})
        )

    @property
    def model(self) -> mujoco.MjModel:
        return self._env.current_model

    @property
    def model_path(self) -> Path:
        return Path(self._env.current_model_path)

    @cached_property
    def _mesh_name_to_file(self) -> dict[str, str]:
        """Lazily parsed map from mesh name to file path from the scene XML <asset> section."""
        tree = ET.parse(self.model_path)
        asset_elem = tree.getroot().find("asset")
        if asset_elem is None:
            return {}
        result: dict[str, str] = {}
        for mesh_elem in asset_elem.findall("mesh"):
            name = mesh_elem.get("name")
            file = mesh_elem.get("file")
            if name and file:
                result[name] = file
        return result

    def get_collision_obj_files(
        self, object_or_name_or_id: ObjectOrNameOrIdType, xml_path: Path | None = None
    ) -> tuple[list[Path], np.ndarray]:
        """Return collision mesh file paths and the world pose of the body holding them.

        Parses an XML file to find all descendant collision geoms of the
        object's body, resolves their mesh references to file paths, and
        returns the list of absolute .obj file paths together with the 4x4
        world pose of the child body that contains the geoms.

        Args:
            object_or_name_or_id: The object to look up.
            xml_path: Optional XML file to parse instead of the scene model path.
                      Useful for dynamically added objects whose meshes are defined
                      in a standalone XML rather than the scene XML.

        Returns:
            A tuple of (obj_files, pose_4x4) where obj_files is a list of
            absolute .obj file paths and pose_4x4 is the 4x4 world pose of
            the child body holding the collision geoms.
        """
        from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

        object_name = self.get_object_name(object_or_name_or_id)
        source_path = xml_path if xml_path is not None else self.model_path
        tree = ET.parse(source_path)
        root = tree.getroot()

        identity_pose = np.eye(4)

        # Build mesh name -> file map from the source XML's <asset> section
        if xml_path is not None:
            asset_elem = root.find("asset")
            mesh_name_to_file: dict[str, str] = {}
            if asset_elem is not None:
                for mesh_elem in asset_elem.findall("mesh"):
                    name = mesh_elem.get("name")
                    file = mesh_elem.get("file")
                    if name and file:
                        mesh_name_to_file[name] = file
        else:
            mesh_name_to_file = self._mesh_name_to_file

        # Find the body element matching the object name
        if xml_path is not None:
            base_name = object_name.split("/")[-1]
            body_elem = root.find(f".//body[@name='{base_name}']")
            if body_elem is None:
                # Fallback: standalone XMLs typically have a single top-level body
                worldbody = root.find("worldbody")
                if worldbody is not None:
                    bodies = worldbody.findall("body")
                    if len(bodies) == 1:
                        body_elem = bodies[0]
        else:
            body_elem = root.find(f".//body[@name='{object_name}']")

        if body_elem is None:
            return [], identity_pose

        # Collect collision geoms (descendants with "collision" in their name and type="mesh")
        obj_files: list[Path] = []
        for geom in body_elem.iter("geom"):
            if geom.get("class") == "__VISUAL_MJT__":
                continue
            if geom.get("type") != "mesh":
                continue
            mesh_name = geom.get("mesh")
            if mesh_name and mesh_name in mesh_name_to_file:
                rel_path = mesh_name_to_file[mesh_name]
                if rel_path.startswith("../../"):
                    abs_path = ASSETS_DIR / rel_path[len("../../") :]
                else:
                    abs_path = source_path.parent / rel_path
                obj_files.append(abs_path.resolve())

        # Find the child body that holds the collision geoms and get its world pose
        parent_body_id = self.get_object_body_id(object_or_name_or_id)
        child_ids = np.where(self.model.body_parentid == parent_body_id)[0]
        geom_body_id = parent_body_id
        for child_id in child_ids:
            if self.model.body_geomnum[child_id] > 0:
                geom_body_id = int(child_id)
                break

        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[geom_body_id].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[geom_body_id]

        return obj_files, pose

    def get_object_name(self, object_or_name_or_id: ObjectOrNameOrIdType) -> str:
        if isinstance(object_or_name_or_id, str):
            object_name = object_or_name_or_id
        elif isinstance(object_or_name_or_id, MlSpacesObject):
            object_name = object_or_name_or_id.name
        else:
            object_name = self.model.body(object_or_name_or_id).name
            assert object_name is not None, (
                f"Object with ID {object_or_name_or_id} does not have a name"
            )

        return object_name

    def get_object(self, object_or_name_or_id: ObjectOrNameOrIdType) -> MlSpacesObject:
        if isinstance(object_or_name_or_id, MlSpacesObject):
            return object_or_name_or_id

        return self.get_object_by_name(self.get_object_name(object_or_name_or_id))

    def get_object_body_id(self, object_or_name_or_id: ObjectOrNameOrIdType) -> int:
        if isinstance(object_or_name_or_id, str):
            object_body_id = self.get_object_by_name(
                self.get_object_name(object_or_name_or_id)
            ).body_id
        elif isinstance(object_or_name_or_id, MlSpacesObject):
            object_body_id = object_or_name_or_id.body_id
        else:
            object_body_id = object_or_name_or_id

        return object_body_id

    def get_context_objects(
        self,
        object_or_name_or_id: ObjectOrNameOrIdType,
        context_type: Context = Context.BENCH,
        bench_geom_ids: list[int] = None,
        cameras: list[str] = None,
        room_ids: list[str] = None,
        disable_caching: bool = True,
        **kwargs,
    ) -> list[MlSpacesObject]:
        # Util for task sampling, so model-scope
        scope_cache = self._model_cache

        oname = self.get_object_name(object_or_name_or_id)

        if self._caching_enabled and not disable_caching:
            plain_key = (
                f"{context_type.name}"
                f"_{sorted(bench_geom_ids or [])}"
                f"_{sorted(cameras or [])}"
                f"_{sorted(room_ids or [])}"
            )
            key = hashlib.md5(plain_key.encode()).hexdigest()
        else:
            # Make sure we can't use cache
            if not disable_caching:
                assert not scope_cache
            key = "DUMMY_KEY"
            scope_cache.pop(key, None)

        if key not in scope_cache[oname]:
            target = self.get_object(object_or_name_or_id)

            if context_type == Context.BENCH:
                assert bench_geom_ids, f"When using {context_type}, `bench_geom_ids` must be given"

                # Note: We actually take any object on top of the bench object
                # (so not only on the given geom ids surfaces)
                context_objects = self.objects_on_bench(bench_geom_ids)

            # elif context_type == Context.VISIBLE:
            #     assert cameras, f"When using {context_type}, `cameras` must be given"
            #     raise NotImplementedError

            elif context_type == Context.SCENE:
                name_to_obj = {obj.name: obj for obj in self.list_top_level_objects()}
                if target.name not in name_to_obj:
                    log.warning(f"{target.name} was not in `list_top_level_objects`")
                    name_to_obj[target.name] = target

                context_objects = [name_to_obj[name] for name in sorted(name_to_obj.keys())]

            # elif context_type == Context.ROOM:
            #     assert room_ids, f"When using {context_type}, `room_ids` must be given"
            #     raise NotImplementedError

            else:  # if context_type == Context.OBJECT:
                context_objects = [target]

            if self._caching_enabled and not disable_caching:
                scope_cache[oname][key] = context_objects
            else:
                scope_cache.pop(oname, None)
                return context_objects

        return scope_cache[oname][key]

    def get_context_synsets(
        self,
        context_objects: Collection[ObjectOrNameOrIdType],
    ) -> list[str]:
        annotated_synsets = sorted(
            set(
                self.get_annotation_synset(object_or_name_or_id)
                for object_or_name_or_id in context_objects
            )
            - {None}
        )

        if not annotated_synsets:
            return []

        valid_synsets = filter_synsets_to_remove_hyponyms(annotated_synsets)

        extended_synsets = set()
        for synset in valid_synsets:
            all_hypernyms = generate_all_hypernyms_with_exclusions(synset)
            if all_hypernyms:
                extended_synsets |= {syn.name() for syn in all_hypernyms}

        return sorted(extended_synsets)

    def default_object_context_synsets(self, target: ObjectOrNameOrIdType) -> set[str]:
        all_hypernyms = generate_all_hypernyms_with_exclusions(self.get_annotation_synset(target))
        if all_hypernyms:
            object_hypernyms = cast(
                set[str],
                {hyp.name() for hyp in all_hypernyms},
            )
            return object_hypernyms

        return set()

    @staticmethod
    def most_concrete_synset(all_hypernyms) -> Any:
        for current in all_hypernyms:
            if not any(is_hypernym_of(other, current) for other in all_hypernyms - {current}):
                return current
        raise ValueError(f"No most concrete element among {all_hypernyms}?!")

    def get_object_hypernyms(
        self, target: ObjectOrNameOrIdType, context_synsets: Collection[str]
    ) -> list[str]:
        object_hypernyms = self.default_object_context_synsets(target)
        return sorted(object_hypernyms & set(context_synsets))

    @staticmethod
    def _remove_obja(category: str) -> str:
        for prefix in ["Obja", "obja"]:
            if category.startswith(prefix):
                return category[len(prefix) :].strip()
        return category.strip()

    @staticmethod
    def _name_versions_from_category(category: str) -> list[str]:
        res = []
        seen_versions = set()

        camel = stringcase.capitalcase(stringcase.camelcase(ObjectManager._remove_obja(category)))

        # lower = camel.lower()
        snake = stringcase.snakecase(camel).strip()  # always lower case
        space = normalize_expression(snake).strip()

        # for version in [lower, snake, space]:
        for version in [space]:
            if version not in seen_versions:
                res.append(version)
                seen_versions.add(version)

        return res

    def fallback_expression(self, object_name: str) -> str:
        # skip 3 digits at end of name and a hexadecimal code, keep everything before (in lower case)
        name = " ".join(object_name.split("_")[:-4]).lower()

        # remove digits
        name = re.sub(r"\d+", "", name).strip()

        # remove "obja" prefix, if any
        if name.startswith("obja"):
            name = name[4:]  # len("obja") = 4

        # no valid identifier
        if len(name.strip()) == 0:
            metadata = self.object_metadata(object_name)
            asset_id = metadata["asset_id"]
            name = ObjectMeta.annotation(asset_id)["category"].lower()

            # remove digits
            name = re.sub(r"\d+", "", name).strip()

            # remove "obja" prefix, if any
            if name.startswith("obja"):
                name = name[4:]  # len("obja") = 4

        # collapse multiple whitespace into a single space and strip ends
        return re.sub(r"\s+", " ", name).strip()

    @staticmethod
    def _name_versions_from_synset(synset: str) -> list[str]:
        res = []
        seen_versions = set()

        for lemma in wn.synset(synset).lemma_names():
            space = normalize_expression(lemma).strip()
            # lower = space.replace(" ", "")
            # snake = space.replace(" ", "_")

            # for version in [lower, snake, space]:
            for version in [space]:
                if version not in seen_versions:
                    res.append(version)
                    seen_versions.add(version)

        return res

    @staticmethod
    def _name_versions_from_short_descriptions(descriptions: Collection[str]) -> list[str]:
        res = []
        seen_versions = set()

        for description in descriptions:
            space = normalize_expression(description).strip()
            # lower = space.replace(" ", "")
            # snake = space.replace(" ", "_")

            # for version in [lower, snake, space]:
            for version in [space]:
                if version not in seen_versions:
                    res.append(version)
                    seen_versions.add(version)

        return res

    def get_cache_key(
        self, object_or_name_or_id: ObjectOrNameOrIdType, context_synsets: Collection[str] = None
    ):
        plain_key = "__".join(
            [self.get_object_name(object_or_name_or_id)] + sorted(context_synsets or [])
        )
        return hashlib.md5(plain_key.encode()).hexdigest()

    def _extract_names_from_context(
        self, object_or_name_or_id: ObjectOrNameOrIdType, context_synsets: Collection[str] = None
    ) -> dict[str, list[str]]:
        cache_key = self.get_cache_key(object_or_name_or_id, context_synsets=context_synsets)

        if cache_key in self._object_name_and_context_to_source_to_natural_names:
            return self._object_name_and_context_to_source_to_natural_names[cache_key]

        source_to_names = {}

        category = self.get_annotation_category(object_or_name_or_id)
        source_to_names[category] = ObjectManager._name_versions_from_category(category)
        if category in AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA:
            thor_space = normalize_expression(
                AI2THOR_OBJECT_TYPE_TO_MOST_SPECIFIC_WORDNET_LEMMA[category]
            ).strip()
            if thor_space not in source_to_names[category]:
                source_to_names[category].append(thor_space)

        category_from_name = self.category_from_name(object_or_name_or_id)
        if category_from_name != category:
            source_to_names[category_from_name] = ObjectManager._name_versions_from_category(
                category_from_name
            )

        hypernyms = self.get_object_hypernyms(
            object_or_name_or_id,
            context_synsets or self.default_object_context_synsets(object_or_name_or_id),
        )

        for synset in hypernyms:
            source_to_names[synset] = ObjectManager._name_versions_from_synset(synset)

        maybe_asset_id = self.object_metadata(object_or_name_or_id).get("asset_id")
        if maybe_asset_id:
            from_short_descriptions = ObjectManager._name_versions_from_short_descriptions(
                ObjectMeta.short_descriptions(maybe_asset_id)
            )
            if from_short_descriptions:
                source_to_names[maybe_asset_id] = from_short_descriptions

        if self._name_caching_enabled:
            self._object_name_and_context_to_source_to_natural_names[cache_key] = source_to_names

        return source_to_names

    def get_natural_object_names(
        self, object_or_name_or_id: ObjectOrNameOrIdType, context_synsets: list[str] = None
    ):
        cache_key = self.get_cache_key(object_or_name_or_id, context_synsets=context_synsets)

        if cache_key not in self._object_name_and_context_to_natural_names:
            all_names = set(
                sum(
                    self._extract_names_from_context(
                        object_or_name_or_id, context_synsets
                    ).values(),
                    [],
                )
            )
            res = sorted(all_names, key=lambda x: (len(x), x))
            if self._name_caching_enabled:
                self._object_name_and_context_to_natural_names[cache_key] = res
            else:
                return res

        return self._object_name_and_context_to_natural_names[cache_key]

    def _augment_natural_name(self, space: str) -> list[str]:
        return [space, space.replace(" ", ""), space.replace(" ", "_")]

    def get_possible_object_types(self, object_or_name_or_id: ObjectOrNameOrIdType) -> list[str]:
        object_name = self.get_object_name(object_or_name_or_id)

        if object_name not in self._object_name_to_possible_type_names:
            default_context = self.default_object_context_synsets(object_or_name_or_id)
            keys = set(
                self._extract_names_from_context(object_or_name_or_id, default_context).keys()
            )
            natural_names = set(
                sum(
                    self._extract_names_from_context(
                        object_or_name_or_id, default_context
                    ).values(),
                    [],
                )
            )
            names = set(
                sum(
                    [self._augment_natural_name(natural_name) for natural_name in natural_names],
                    [],
                )
            )
            res = sorted(keys | names, key=lambda x: (len(x), x))
            if self._name_caching_enabled:
                self._object_name_to_possible_type_names[object_name] = res
            else:
                return res

        return self._object_name_to_possible_type_names[object_name]

    def category_from_name(self, object_or_name_or_id: ObjectOrNameOrIdType):
        name = self.get_object_name(object_or_name_or_id)
        try:
            return re.compile(r"^(.*?)(?=[0-9a-fA-F]{32})").match(name).group(1).strip("_").lower()
        except AttributeError:
            return re.compile(r"^([A-Za-z_]+)").match(name).group(1).strip("_").lower()

    def get_annotation_category(self, object_or_name_or_id: ObjectOrNameOrIdType) -> str:
        object_name = self.get_object_name(object_or_name_or_id)

        object_meta = self.object_metadata(object_name)
        # Might contain Obja/obja, be lowercase, snake case, camel case
        return object_meta.get("category", self.category_from_name(object_name).strip())

    def get_annotation_synset(self, object_or_name_or_id: ObjectOrNameOrIdType) -> str | None:
        object_meta = self.object_metadata(object_or_name_or_id)

        return (ObjectMeta.annotation(object_meta.get("asset_id", "DUMMY")) or {}).get("synset")

    def has_some_valid_identifier(
        self, object_or_name_or_id: ObjectOrNameOrIdType, valid_identifiers: Collection[str]
    ):
        """
        If empty valid_identifiers, equivalent to accept any.
        If None, equivalent to accept none
        """

        if valid_identifiers is None:
            return False
        elif len(valid_identifiers) == 0:
            return True

        # Ensure at least one identifier within the given valid identifiers
        for identifier in self.get_possible_object_types(object_or_name_or_id):
            if identifier in valid_identifiers:
                return True

        return False

    def has_receptacle_site(self, object_or_name_or_id: ObjectOrNameOrIdType) -> bool:
        # Ensure receptacles have some site
        return bool(self.object_metadata(object_or_name_or_id).get("name_map", {}).get("sites", {}))

    def is_receptacle(
        self, object_or_name_or_id: ObjectOrNameOrIdType, receptacle_types: Collection[str]
    ) -> bool:
        """
        If receptacle_types is None, match None
        If empty list, match any with receptacle site
        """
        return self.has_receptacle_site(object_or_name_or_id) and self.has_some_valid_identifier(
            object_or_name_or_id, receptacle_types
        )

    def is_structural(self, object_or_name_or_id: ObjectOrNameOrIdType) -> bool:
        cache_in_use = self._model_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "structural" not in cache_in_use[oname]:
            is_structural = self.has_some_valid_identifier(
                object_or_name_or_id, self.STRUCTURAL_TYPES
            )
            if self._caching_enabled:
                cache_in_use[oname]["structural"] = is_structural
            else:
                cache_in_use.pop(oname, None)
                return is_structural

        return cache_in_use[oname]["structural"]

    def is_excluded(self, object_or_name_or_id: ObjectOrNameOrIdType) -> bool:
        cache_in_use = self._model_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "excluded" not in cache_in_use[oname]:
            is_excluded = (
                self._env.config.robot_config.robot_namespace in oname
            ) or not descendant_geoms(
                self.model, self.get_object(object_or_name_or_id).body_id, True
            )
            if self._caching_enabled:
                cache_in_use[oname]["excluded"] = is_excluded
            else:
                cache_in_use.pop(oname, None)
                return is_excluded

        return cache_in_use[oname]["excluded"]

    def has_free_joint(self, object_or_name_or_id: ObjectOrNameOrIdType) -> bool:
        # Ensure the object has a free joint
        try:
            MlSpacesFreeJointBody(self.data, self.get_object_name(object_or_name_or_id))
        except AssertionError:
            return False

        return True

    def is_pickup_candidate(
        self, object_or_name_or_id: ObjectOrNameOrIdType, pickup_types: Collection[str]
    ) -> bool:
        """
        If pickup_types is None, match None
        If empty list, match any with free joint
        """
        return self.has_free_joint(object_or_name_or_id) and self.has_some_valid_identifier(
            object_or_name_or_id, pickup_types
        )

    def top_level_bodies(self) -> list[int]:
        """Return bodies whose parent is the world body."""
        cache_in_use = self._model_cache
        cache_key = "__scene__top_level_bodies__"

        if cache_key not in cache_in_use:
            bodies = MlSpacesObject.get_top_level_bodies(self.model)
            if self._caching_enabled:
                cache_in_use[cache_key] = bodies
            else:
                cache_in_use.pop(cache_key, None)
                return bodies

        return cast(list[int], cache_in_use[cache_key])

    def get_objects_of_type(self, object_types: Collection[str]) -> list[MlSpacesObject]:
        """Return top-level scene objects for which at least one valid identifier matches any in object_types.
        If pickup is True, objects need to have a free joint to be returned.
        Returns MlSpacesObject instances (not MujocoBody) built from top-level bodies.
        If empty list, return any, if None, accept None
        """
        results: list[MlSpacesObject] = []

        for b in self.top_level_bodies():
            name = self.get_object_name(b)
            if not name:
                continue

            if self.is_excluded(name):
                continue

            if self.is_structural(name):
                continue

            if self.has_some_valid_identifier(name, object_types):
                results.append(self.get_object_by_name(name))

        return sorted(results, key=lambda x: x.name)

    def is_object_articulable(self, object_name: str) -> bool:
        """
        If it has at least one hinge or slide joint, return True
        else return False
        """
        body_id = self.model.body(object_name).id

        # go through all joints
        for joint_id in range(self.model.njnt):
            # if root body is same as the body_id, then joint is part of object
            if self.model.body(self.model.joint(joint_id).bodyid[0]).rootid[0] == body_id:
                if self.model.joint(joint_id).type in [
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ]:
                    return True

        return False

    def get_object_by_name(self, object_name: str) -> MlSpacesObject | None:
        """Return the top-level object with the specified name, or None if not found."""
        # static or articulable object
        # if articulable and has joints, return the articulation object
        # if static, return the static object

        cache_in_use = self._model_cache

        is_articulable = False

        if "articulable" not in cache_in_use[object_name]:
            is_articulable = self.is_object_articulable(object_name)
            if self._caching_enabled:
                cache_in_use[object_name]["articulable"] = is_articulable
            else:
                cache_in_use.pop(object_name, None)

        if is_articulable or (self._caching_enabled and cache_in_use[object_name]["articulable"]):
            return MlSpacesArticulationObject(data=self.data, object_name=object_name)

        return MlSpacesObject(data=self.data, object_name=object_name)

    def list_top_level_objects(self) -> list[MlSpacesObject]:
        """List all non-structural top-level objects as Object instances."""
        objs: list[MlSpacesObject] = []
        for b in self.top_level_bodies():
            name = self.get_object_name(b)
            if not name or self.is_structural(name) or self.is_excluded(name):
                continue
            objs.append(self.get_object_by_name(name))

        return sorted(objs, key=lambda x: x.name)

    def get_receptacles(self) -> list[MlSpacesObject]:
        """Return top-level receptacle objects (tables, counters, etc.)"""
        return [o for o in self.list_top_level_objects() if self.has_receptacle_site(o)]

    def get_pickup_candidates(self) -> list[MlSpacesObject]:
        """Return top-level candidate small objects for pickup."""
        return [o for o in self.list_top_level_objects() if self.has_free_joint(o)]

    def find_door_names(self) -> list[str]:
        """Find all valid door body names in the scene.

        Valid doors are identified by the naming convention and whether they have joints.

        Returns:
            List of door body names found in the scene.
        """
        door_body_names = []
        for key, value in self.scene_metadata["objects"].items():
            if "doorway" in key:
                name_map = value.get("name_map", {})
                bodies = name_map.get("bodies", {})
                for k, v in bodies.items():
                    if "_door_" in v:
                        door_object = Door(k, self.data)
                        if door_object.njoints > 0:
                            door_body_names.append(k)
        return door_body_names

    def summarize_top_level_bodies(
        self, receptacle_types: Collection[str], limit: int = 50
    ) -> list[str]:
        # No widespread usage, so no cache
        lines: list[str] = []
        count = 0
        for b in self.top_level_bodies():
            if self.is_structural(b) or self.is_excluded(b):
                continue
            try:
                obj = self.get_object(b)
                lines.append(self.object_summary_str(obj, receptacle_types))
                count += 1
                if count >= limit:
                    break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                try:
                    name = self.get_object(b).name
                except KeyboardInterrupt:
                    raise
                print(f"Error summarizing top-level bodies (obj with name '{name}'): {e}")
                continue
        return lines

    def get_objects_that_are_on_top_of_object(
        self,
        object_or_name_or_id: ObjectOrNameOrIdType,
        pickup_types: Collection[str],
        z_above_min: float = 0.02,
        z_above_max: float = 0.60,
        constrain_to_pickupable: bool = True,
    ) -> list[MlSpacesObject]:
        cache_in_use = self._model_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "objects_on_top" not in cache_in_use[oname]:
            # Compute host placement region
            obj = self.get_object(object_or_name_or_id)
            oid = obj.body_id
            region = self.compute_placement_region(object_or_name_or_id)
            xy_min, xy_max, top_z = region["xy_min"], region["xy_max"], float(region["top_z"])
            # Candidate roots: direct children of world, non-structural
            candidates = [
                b
                for b in self.top_level_bodies()
                if b != oid and not (self.is_structural(b) or self.is_excluded(b))
            ]

            roots_seen: set[int] = set()
            results: list[MlSpacesObject] = []
            for b in candidates:
                root_b = MlSpacesObject.find_top_object_body_id(self.model, b)
                if root_b in roots_seen:
                    continue
                roots_seen.add(root_b)

                name = self.get_object_name(root_b)

                if constrain_to_pickupable:
                    if not self.is_pickup_candidate(name, pickup_types):
                        continue

                pos = self.data.xpos[root_b]
                in_xy = (xy_min[0] <= pos[0] <= xy_max[0]) and (xy_min[1] <= pos[1] <= xy_max[1])
                above = (pos[2] >= top_z + z_above_min) and (pos[2] <= top_z + z_above_max)
                if in_xy and above:
                    results.append(self.get_object_by_name(name))

            if self._caching_enabled:
                cache_in_use[oname]["objects_on_top"] = results
            else:
                cache_in_use.pop(oname, None)
                return results

        return cache_in_use[oname]["objects_on_top"]

    def get_support_below(
        self,
        object_or_name_or_id: ObjectOrNameOrIdType,
        receptacle_types: Collection[str],
        z_clearance_eps: float = 1e-1,
    ) -> str | None:
        """Return the name of the support surface below (receptacle or room floor)."""
        if self.is_structural(object_or_name_or_id) or self.is_excluded(object_or_name_or_id):
            return None

        cache_in_use = self._data_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "support_below" not in cache_in_use[oname]:
            obj = self.get_object(object_or_name_or_id)
            obj_bottom = self.object_bottom_z(object_or_name_or_id)
            oid = obj.body_id
            xy = obj.position[:2]

            best_name = None
            best_top = -np.inf

            for b in self.top_level_bodies():
                if b == oid:
                    continue

                name = self.get_object_name(b)
                if not (
                    self.is_receptacle(name, receptacle_types)
                    or any(
                        identifier in {"room", "floor"}
                        for identifier in self.get_possible_object_types(b)
                    )
                ):
                    continue

                region = self.compute_placement_region(name)
                xy_min, xy_max, top_z = region["xy_min"], region["xy_max"], float(region["top_z"])
                in_xy = (xy_min[0] <= xy[0] <= xy_max[0]) and (xy_min[1] <= xy[1] <= xy_max[1])
                clearance = obj_bottom - top_z
                z_check = top_z <= obj_bottom + z_clearance_eps

                is_better = top_z > best_top

                # Debug log for each candidate
                log.debug(
                    f"[get_support_below] {oname} <- {name}: "
                    f"clearance={clearance:.4f}, in_xy={in_xy}, "
                    f"z_check={z_check} (top_z={top_z:.4f} <= obj_bottom={obj_bottom:.4f} + eps={z_clearance_eps:.4f}), "
                    f"is_better={is_better} (top_z={top_z:.4f} vs best={best_top:.4f})"
                )

                # TODO The problem here must be that for objaverse assets, only one body has all the possible shelves
                #  but It could also be due to a suboptimal placement region computation
                if in_xy and (top_z <= obj_bottom + z_clearance_eps) and (top_z > best_top):
                    best_top = top_z
                    best_name = name
                    log.debug(f"[get_support_below] New best support: {best_name}")

            if self._caching_enabled:
                cache_in_use[oname]["support_below"] = best_name
            else:
                cache_in_use.pop(oname, None)
                return best_name

        return cache_in_use[oname]["support_below"]

    def infer_room_name(
        self, object_or_name_or_id: ObjectOrNameOrIdType, receptacle_types: Collection[str]
    ) -> str | None:
        # Note: If passing [] for receptacle types, only finds room or floor
        cache_in_use = self._data_cache

        oname = self.get_object_name(object_or_name_or_id)

        receptacle_key = hashlib.md5(f"{sorted(receptacle_types) or []}".encode()).hexdigest()
        cache_key = f"room_{receptacle_key}"

        if cache_key not in cache_in_use[oname]:

            def dfs(cur_object_or_name_or_id):
                support_name = self.get_support_below(cur_object_or_name_or_id, receptacle_types)
                if support_name is None:
                    return None

                # If the support is a room or floor, return room or map floor->room
                for t in self.get_possible_object_types(support_name):
                    # This should be terminal
                    if t == "room":
                        return support_name

                    # This should be terminal
                    if t == "floor":
                        # Find enclosing room by nearest/first room in parent chain
                        ancestors = self.ancestors(cur_object_or_name_or_id)
                        for a in ancestors:
                            aname = self.get_object_name(a)
                            for tt in self.get_possible_object_types(aname):
                                if tt == "room":
                                    return aname
                        raise ValueError("BUG? Floor has no room ancestor")

                # else: keep searching
                return dfs(support_name)

            res = dfs(object_or_name_or_id)
            if self._caching_enabled:
                cache_in_use[oname][cache_key] = res
            else:
                cache_in_use.pop(oname, None)
                return res

        return cache_in_use[oname][cache_key]

    def object_summary_str(
        self, object_or_name_or_id: ObjectOrNameOrIdType, receptacle_types: Collection[str]
    ) -> str:
        # Rare use, so no cache
        obj = self.get_object(object_or_name_or_id)
        name = obj.name
        category = self.get_annotation_category(obj)
        synset = self.get_annotation_synset(obj)
        pos = obj.position
        support = self.get_support_below(obj, receptacle_types)
        room = self.infer_room_name(obj, receptacle_types)
        on_str = f"on {support}" if support else "on <unknown>"
        room_str = f"in {room}" if room else "in <unknown>"
        return f"{name} (category={category} synset={synset}) center=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) {on_str}, {room_str}"

    @staticmethod
    def prefilter_with_clip(
        target_type: str | list[str], uids: list[str], min_sim: float = 0.25
    ) -> list[str]:
        all_uids = set(ObjectMeta.all_uids())
        kept_uids = [uid for uid in uids if uid in all_uids]
        if not kept_uids:
            return []

        if isinstance(target_type, str):
            expressions = [normalize_expression(target_type)]
        else:
            expressions = [normalize_expression(cur_type) for cur_type in target_type]
        try:
            text_features = compute_text_clip(expressions)
        except ValueError:
            log.warning("No image features, using all uids")
            return uids

        img_features = ObjectMeta.img_features(kept_uids)
        sims = clip_sim(img_features, text_features).max(axis=1).flatten()
        return np.array(kept_uids)[sims >= min_sim].tolist()

    def referral_expression_priority(
        self,
        object_or_name_or_id: ObjectOrNameOrIdType,
        context_object_or_name_or_ids: Collection[ObjectOrNameOrIdType],
    ) -> list[tuple[float, float, str]]:
        """
        Return prioritized referral expressions, each presented as a tuple with:
          - similarity score margin versus highest-scoring distractor in context
          - similarity score for target object
          - referral expression
        Priority is based on margin, similarity to target, expression length,
        """
        # Rare use, so no cache

        target_name = self.get_object_name(object_or_name_or_id)
        context_names = {
            self.get_object_name(maybe_obj) for maybe_obj in context_object_or_name_or_ids
        }
        assert target_name in context_names, f"Target {target_name} must be in context"

        name_to_uid = {
            self.get_object_name(maybe_obj): self.object_metadata(maybe_obj)["asset_id"]
            for maybe_obj in context_object_or_name_or_ids
            if "asset_id" in self.object_metadata(maybe_obj)
        }

        assert target_name in name_to_uid, f"Target {target_name} has no metadata"

        asset_ids = sorted(set(name_to_uid.values()))

        img = []
        kept_asset_ids = []
        for asset_id in asset_ids:
            try:
                img.append(ObjectMeta.img_features(asset_id))
                kept_asset_ids.append(asset_id)
            except ValueError:
                log.warning(f"No image features for {asset_id}, ignoring.")

        if name_to_uid[target_name] not in kept_asset_ids:
            raise ValueError(f"Missing asset id {name_to_uid[target_name]} from {target_name}")

        img = np.concatenate(img, axis=0)
        asset_ids = kept_asset_ids

        descriptions = self.get_natural_object_names(
            object_or_name_or_id,
            self.get_context_synsets(context_object_or_name_or_ids),
        )
        try:
            sim = clip_sim(img, compute_text_clip(descriptions))
        except NameError:
            log.warning("No CLIP module, using dummy description scores.")
            # e.g. when you don't want to install it / no gpu (?)
            names = self.get_natural_object_names(object_or_name_or_id, [])
            return [(1.0, 1.0, name) for name in names]

        target_idx = asset_ids.index(name_to_uid[target_name])
        target_sims = sim[target_idx]

        if sim.shape[0] > 1:
            sorting = np.argsort(sim, axis=0)

            # last row has the indices of largest similarity
            nearest_sim = sim[sorting[sim.shape[0] - 1], np.arange(sim.shape[1])]
            runner_up_sim = sim[sorting[sim.shape[0] - 2], np.arange(sim.shape[1])]

            deltas = target_sims - np.where(target_sims == nearest_sim, runner_up_sim, nearest_sim)
        else:
            deltas = target_sims

        return sorted(
            [
                (delta, target_sim, description)
                for delta, target_sim, description in zip(deltas, target_sims, descriptions)
            ],
            key=lambda x: (x[0], x[1], len(x[2]), x[2]),
            reverse=True,
        )

    @staticmethod
    def thresholded_expression_priority(
        priority: list[tuple[float, float, str]],
        sim_margin_threshold: float = 0.03,
        target_sim_threshold: float = 0.1,
    ) -> list[tuple[float, float, str]]:
        """
        Thresholding to filter out ambiguous expressions.

        Parameters
        ----------
        priority : list[tuple[float, float, str]]
            A list of tuples `(sim_margin, target_similarity, value)`:
            - sim_margin (float):    A priority score (typically in [-1, 1]) used as the
                                     primary softmax logit.
            - target_similarity (float): A secondary score indicating similarity to a
                                         target representation.
            - value (str):           The actual expression or token to be sampled.
        sim_margin_threshold : float, optional
            Minimum allowed sim_margin for an item to be considered when thresholding.
        target_sim_threshold : float, optional
            Minimum required target_similarity for thresholding.

        Returns
        -------
        list[tuple[float, float, str]] (filtered prioritization)
            The sampled expression/value from the filtered list.
        """

        return [
            p for p in priority if p[0] >= sim_margin_threshold and p[1] >= target_sim_threshold
        ]

    @staticmethod
    def expression_probs(
        priority: list[tuple[float, float, str]],
        temperature: float = 2e-2,
    ) -> np.ndarray:
        """
        Compute probabilities for each expression in the priority list.

        Parameters
        ----------
        priority : list[tuple[float, float, str]]
            A list of tuples `(sim_margin, target_similarity, value)`:
            - sim_margin (float):    A priority score (typically in [-1, 1]) used as the
                                     primary softmax logit.
            - target_similarity (float): A secondary score indicating similarity to a
                                         target representation.
            - value (str):           The actual expression or token to be sampled.
        temperature : float, optional
            Softmax temperature. Lower values (< 0.1) make sampling more deterministic by
            amplifying score differences; higher values increase exploration.

        Returns
        -------
        np.ndarray
            A probability distribution over the expressions in the priority list.
        """

        scores = np.asarray([p[0] for p in priority], dtype=float)
        logits = np.exp(scores / max(1e-3, temperature))
        probs = logits / logits.sum()
        return probs

    @staticmethod
    def sample_expression(
        priority: list[tuple[float, float, str]],
        temperature: float = 2e-2,
    ) -> str:
        """
        Sample a candidate expression using a softmax distribution over priority scores.

        Parameters
        ----------
        priority : list[tuple[float, float, str]]
            A list of tuples `(sim_margin, target_similarity, value)`:
            - sim_margin (float):    A priority score (typically in [-1, 1]) used as the
                                     primary softmax logit.
            - target_similarity (float): A secondary score indicating similarity to a
                                         target representation.
            - value (str):           The actual expression or token to be sampled.
        temperature : float, optional
            Softmax temperature. Lower values (< 0.1) make sampling more deterministic by
            amplifying score differences; higher values increase exploration.
            Default is `2e-2`, producing a sharp distribution.

        Returns
        -------
        str
            The sampled expression/value from the given list.

        Notes
        -----
        The final sampling is categorical:
            p(i) = softmax(sim_margin_i / temperature).
        """

        probs = ObjectManager.expression_probs(priority, temperature)
        return np.random.choice([p[-1] for p in priority], p=probs)

    def get_free_objects(self) -> list[MlSpacesObject]:
        """Return list of all bodies with free joints"""
        model = self.model
        freejoints = np.where(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        body_ids = model.jnt_bodyid[freejoints]
        return [self.get_object_by_name(model.body(id).name) for id in body_ids]

    def get_mobile_objects(self) -> list[MlSpacesObject]:
        """Return of list of all task relevant bodies i.e. not robots/policy objects"""

        task_objects = []
        for object_name, object_dict in self.scene_metadata["objects"].items():
            if not object_dict["is_static"]:
                try:
                    task_object = create_mlspaces_body(self.data, object_name)
                except KeyError:
                    log.warning("Could not find object %s in scene", object_name)
                    continue
                task_objects.append(task_object)

        # TODO(Abhay): do we want this?
        for object_name in self._env.config.task_config.added_objects:
            task_object = create_mlspaces_body(self.data, object_name)
            task_objects.append(task_object)

        return task_objects

    @staticmethod
    def uid_to_annotation_for_type(object_type: str) -> dict[str, dict]:
        """
        Return list of uids in entire object library with object_type among the possible types

        Note: for now, we are reusing the functionality based on use of scene
        metadata in this class. We also don't cache results, so best keep them
        cached in your caller
        """
        object_type = object_type.lower()

        valid_uids = {}

        class DummyEnv:
            mj_datas = [None]

        om = ObjectManager(DummyEnv(), -1)  # type:ignore
        om.scene_metadata = {"objects": {}}

        for uid, anno in ObjectMeta.annotation().items():
            category = anno["category"]
            name = f"{category.lower()}_{hashlib.md5(uid.encode()).hexdigest()}_0_0_0"

            om.scene_metadata["objects"][name] = {
                "asset_id": uid,
                "category": category,
                "object_enum": "temp_object",
            }

            possible_types = om.get_possible_object_types(name)
            if object_type in possible_types:
                valid_uids[uid] = anno

            # Avoid wasting memory
            om.scene_metadata["objects"].pop(name)
            om._object_name_to_possible_type_names = {}
            om._object_name_and_context_to_source_to_natural_names = {}

        return valid_uids

    def get_parent_chain_names(self, object_or_name_or_id: ObjectOrNameOrIdType) -> list[str]:
        # Rare use, so no cache

        try:
            parent_chain_names = [
                self.model.body(b).name for b in self.ancestors(object_or_name_or_id)
            ]
        except Exception as e:
            print(f"Error getting parent chain names: {e}")
            parent_chain_names = []

        return parent_chain_names

    def children_lists(self):
        cache_in_use = self._model_cache

        cache_key = "__scene__children_lists__"

        if cache_key not in cache_in_use:
            children_lists = MlSpacesObject.build_children_lists(self.model)
            if self._caching_enabled:
                cache_in_use[cache_key] = children_lists
            else:
                return children_lists

        return cache_in_use[cache_key]

    def descendants(self, object_or_name_or_id: ObjectOrNameOrIdType):
        cache_in_use = self._model_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "descendants" not in cache_in_use[oname]:
            descendants = MlSpacesObject.get_descendants(
                self.children_lists(), self.get_object_body_id(object_or_name_or_id)
            )
            if self._caching_enabled:
                cache_in_use[oname]["descendants"] = descendants
            else:
                cache_in_use.pop(oname, None)
                return descendants

        return cache_in_use[oname]["descendants"]

    def ancestors(self, object_or_name_or_id: ObjectOrNameOrIdType):
        cache_in_use = self._model_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "ancestors" not in cache_in_use[oname]:
            ancestors = MlSpacesObject.get_ancestors(
                self.model, self.get_object_body_id(object_or_name_or_id)
            )
            if self._caching_enabled:
                cache_in_use[oname]["ancestors"] = ancestors
            else:
                cache_in_use.pop(oname, None)
                return ancestors

        return cache_in_use[oname]["ancestors"]

    def get_direct_children_names(self, object_or_name_or_id: ObjectOrNameOrIdType) -> list[str]:
        # Rare use, so no cache
        direct_children_names = [
            self.model.body(c).name
            for c in MlSpacesObject.get_direct_children(
                self.children_lists(), self.get_object_body_id(object_or_name_or_id)
            )
        ]
        return direct_children_names

    def get_geom_infos(
        self,
        object_or_name_or_id: ObjectOrNameOrIdType,
        include_descendants: bool = True,
        max_geoms: int | None = 2048,
    ) -> list[dict[str, object]]:
        cache_in_use = self._data_cache

        oname = self.get_object_name(object_or_name_or_id)

        cache_key = f"geom_infos_{include_descendants}_{max_geoms}"

        if cache_key not in cache_in_use[oname]:
            # Delegate to MlSpacesObject.get_geom_infos
            obj = self.get_object(object_or_name_or_id)
            geoms = obj.get_geom_infos(include_descendants=include_descendants, max_geoms=max_geoms)
            if self._caching_enabled:
                cache_in_use[oname][cache_key] = geoms
            else:
                cache_in_use.pop(oname, None)
                return geoms

        return cache_in_use[oname][cache_key]

    def compute_placement_region(
        self, object_or_name_or_id: ObjectOrNameOrIdType, margin_xy: float = 0.05
    ) -> dict[str, np.ndarray]:
        cache_in_use = self._data_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "placement_region" not in cache_in_use[oname]:
            obj_body_id = self.get_object_body_id(object_or_name_or_id)
            body_ids = {obj_body_id, *self.descendants(obj_body_id)}
            xy_min = np.array([np.inf, np.inf])
            xy_max = np.array([-np.inf, -np.inf])
            top_z = -np.inf
            found_any = False
            for geom_id in range(self.model.ngeom):
                try:
                    if int(self.model.geom_bodyid[geom_id]) in body_ids:
                        center, dims = geom_aabb(self.model, self.data, [int(geom_id)])
                        aabb_min = center - dims / 2.0
                        aabb_max = center + dims / 2.0
                        xy_min = np.minimum(xy_min, aabb_min[:2])
                        xy_max = np.maximum(xy_max, aabb_max[:2])
                        top_z = max(top_z, float(aabb_max[2]))
                        found_any = True
                except Exception as e:
                    print(f"Error computing AABB for geom {geom_id}: {e}")
                    continue
            if (not found_any) or (not np.isfinite(top_z)):
                # Fallback small patch around object center if no valid AABBs
                pos = self.get_object(object_or_name_or_id).position
                placement_region = {
                    "xy_min": pos[:2] - 0.3,
                    "xy_max": pos[:2] + 0.3,
                    "top_z": float(pos[2]),
                }
            else:
                placement_region = {
                    "xy_min": xy_min - margin_xy,
                    "xy_max": xy_max + margin_xy,
                    "top_z": top_z,
                }

            if self._caching_enabled:
                cache_in_use[oname]["placement_region"] = placement_region
            else:
                cache_in_use.pop(oname, None)
                return placement_region

        return cache_in_use[oname]["placement_region"]

    def object_bottom_z(self, object_or_name_or_id: ObjectOrNameOrIdType) -> float:
        cache_in_use = self._data_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "bottom_z" not in cache_in_use[oname]:
            # Bottom Z from aggregated AABB minima
            oid = self.get_object_name(object_or_name_or_id)
            body_ids = {oid, *self.descendants(oid)}
            bottom_z = np.inf
            for geom_id in range(self.model.ngeom):
                try:
                    if int(self.model.geom_bodyid[geom_id]) in body_ids:
                        aabb_min, _ = geom_aabb(self.model, self.data, [geom_id])
                        bottom_z = min(bottom_z, float(aabb_min[2]))
                except Exception as e:
                    print(f"Error getting object bottom z: {e}")
                    continue

            if not np.isfinite(bottom_z):
                bottom_z = float(self.get_object(object_or_name_or_id).position[2])

            if self._caching_enabled:
                cache_in_use[oname]["bottom_z"] = bottom_z
            else:
                cache_in_use.pop(oname, None)
                return bottom_z

        return cache_in_use[oname]["bottom_z"]

    def get_door_bboxes_array(self, object_or_name_or_id: ObjectOrNameOrIdType) -> np.ndarray:
        """Get door collision geometry bounding boxes as an array.
        Returns:
            np.ndarray: Array of AABBs (center, size) for door collision geoms
        """
        cache_in_use = self._model_cache

        oname = self.get_object_name(object_or_name_or_id)

        if "door_bboxes_array" not in cache_in_use[oname]:
            # Get all geoms for the door object
            geom_infos = self.get_geom_infos(object_or_name_or_id, include_descendants=True)
            door_bboxes = []

            for geom_info in geom_infos:
                geom_id = geom_info["id"]
                # Check if it's a collision geom (contype != 0 or conaffinity != 0)
                if (
                    self.model.geom(geom_id).contype != 0
                    or self.model.geom(geom_id).conaffinity != 0
                ):
                    # Get AABB from model (center, size)
                    aabb = self.model.geom_aabb[geom_id]
                    door_bboxes.append(aabb)

            if not door_bboxes:
                door_bboxes = np.array([]).reshape(0, 6)
            else:
                door_bboxes = np.array(door_bboxes)

            if self._caching_enabled:
                cache_in_use[oname]["door_bboxes_array"] = door_bboxes
            else:
                cache_in_use.pop(oname, None)
                return door_bboxes

        return cache_in_use[oname]["door_bboxes_array"]

    def objects_on_receptacle(
        self,
        objs_to_check: list[MlSpacesObject],
        bench_geom_ids: list[int],
        angle_threshold: float = np.radians(30),
        fallback_thres=0.01,  # 1 cm
        attempt_contact: bool = True,
    ) -> list[MlSpacesObject]:
        from shapely.geometry import Point, Polygon

        from molmo_spaces.utils.mujoco_scene_utils import body_aabb

        valid_names = {obj.name for obj in objs_to_check}

        data = self.data
        model = data.model

        cos_threshold = np.cos(angle_threshold)

        object_names = set()

        if attempt_contact:
            # Based on contact with the bench geom id:
            for c in data.contact:
                root_body1, root_body2 = model.body_rootid[model.geom_bodyid[c.geom]]
                if (c.geom[0] in bench_geom_ids) ^ (c.geom[1] in bench_geom_ids):
                    other_body_id = root_body1 if c.geom[0] not in bench_geom_ids else root_body2
                    normal = c.frame[:3] / np.linalg.norm(c.frame[:3])
                    if c.geom[1] in bench_geom_ids:
                        normal = -normal
                    body_aabb_center, _ = body_aabb(model, data, other_body_id)
                    if c.pos[2] < body_aabb_center[2] and normal[2] >= cos_threshold:
                        cname = model.body(other_body_id).name
                        if cname in valid_names:
                            object_names.add(cname)

        contactless_object_names = set()
        seen_poly_z = set()

        # Fallback: list with objects with aabbs overlapping >= 50% in xy with the bench and "just above" in z
        for geom_id in bench_geom_ids:
            # Take full body
            bc, be = body_aabb(model, data, model.body_rootid[model.geom_bodyid[geom_id]])

            # Avoid recomputing if all geom ids are part of the same body
            cur_str = f"{np.round(bc, 3).tolist() + np.round(be, 3).tolist()}"
            if cur_str in seen_poly_z:
                continue
            seen_poly_z.add(cur_str)

            bench_poly = Polygon(
                [
                    Point(bc[0] - be[0] / 2, bc[1] - be[1] / 2),
                    Point(bc[0] + be[0] / 2, bc[1] - be[1] / 2),
                    Point(bc[0] + be[0] / 2, bc[1] + be[1] / 2),
                    Point(bc[0] - be[0] / 2, bc[1] + be[1] / 2),
                ]
            )
            bench_z = bc[2] + be[2] / 2

            # # Debug
            # for object_name in object_names:
            #     body_id = model.body(object_name).id
            #     obj_center, obj_ext = body_aabb(model, data, body_id)
            #     assert bench_poly.contains(Point(*obj_center[:2]))
            #     obj_base_z = obj_center[2] - obj_ext[2] / 2
            #     print(body_id, object_name, abs(bench_z - obj_base_z) <= fallback_thres)

            for obj in objs_to_check:
                object_name = obj.name

                if object_name in object_names:
                    continue
                if object_name not in self.scene_metadata.get("objects", {}):
                    continue
                if object_name in contactless_object_names:
                    continue

                body_id = model.body(object_name).id
                obj_center, obj_ext = body_aabb(model, data, body_id)
                if bench_poly.contains(Point(*obj_center[:2])):
                    obj_base_z = obj_center[2] - obj_ext[2] / 2
                    # Check the base of the object is somewhere between the bbox center below and the fallback thres above
                    if -be[2] / 2 <= obj_base_z - bench_z <= fallback_thres:
                        contactless_object_names.add(object_name)

        # Combine all objects in a single list
        object_list = [
            self.get_object_by_name(object_name)
            for object_name in sorted(object_names | contactless_object_names)
        ]

        return object_list

    def objects_on_bench(
        self,
        bench_geom_ids: list[int],
        angle_threshold: float = np.radians(30),
        fallback_thres=0.01,  # 1 cm
        attempt_contact: bool = True,
    ) -> list[MlSpacesObject]:
        return self.objects_on_receptacle(
            objs_to_check=self.list_top_level_objects(),
            bench_geom_ids=bench_geom_ids,
            angle_threshold=angle_threshold,
            fallback_thres=fallback_thres,
            attempt_contact=attempt_contact,
        )

    def clear(self):
        self.invalidate_all_caches()

    def get_body_to_geoms(self):
        body_to_geom_ids = defaultdict(set)
        for geom_id in range(0, self.model.ngeom):
            body_id = int(self.model.geom(geom_id).bodyid.item())
            root_id = int(self.model.body(body_id).rootid.item())
            body_to_geom_ids[root_id].add(geom_id)
        return {
            key: sorted(values)
            for key, values in body_to_geom_ids.items()
            if not self.is_excluded(key)
        }

    def approximate_supporting_geoms(
        self, object_or_name_or_id: ObjectOrNameOrIdType, body_to_geoms: dict[int, list[int]]
    ):
        from shapely.geometry import Point, Polygon

        from molmo_spaces.utils.mujoco_scene_utils import body_aabb, geom_aabb

        def make_poly_and_zs(c, e):
            poly = Polygon(
                [
                    Point(c[0] - e[0] / 2, c[1] - e[1] / 2),
                    Point(c[0] + e[0] / 2, c[1] - e[1] / 2),
                    Point(c[0] + e[0] / 2, c[1] + e[1] / 2),
                    Point(c[0] - e[0] / 2, c[1] + e[1] / 2),
                ]
            )
            zb = c[2] - e[2] / 2  # bottom
            zt = c[2] + e[2] / 2  # top

            return poly, zb, zt

        def get_body_box(bid):
            c, e = body_aabb(self.model, self.data, bid)
            return make_poly_and_zs(c, e)

        def get_geom_box(geom_id):
            c, e = geom_aabb(self.model, self.data, [geom_id], tight_mesh=True)
            return make_poly_and_zs(c, e)

        oid = self.get_object(object_or_name_or_id).body_id
        opoly, oz, ot = get_body_box(oid)

        half_area = opoly.area / 2

        candidates = []
        for body, geoms in body_to_geoms.items():
            if body == oid:
                continue
            bpoly, bz, bt = get_body_box(body)
            if bpoly.intersects(opoly):
                for geom in geoms:
                    gpoly, gz, gt = get_geom_box(geom)
                    if abs(oz - gt) < 0.02:
                        # Keep geom if it "supports" at least half of the box
                        if gpoly.intersection(opoly).area >= half_area:
                            candidates.append((oz - gt, geom, body))

        candidates = sorted(candidates, key=lambda x: x[0])

        return candidates
