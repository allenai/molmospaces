from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from molmo_spaces.env.data_views import MlSpacesObject


class DynamicsRandomizer:
    """
    Randomizer for dynamics properties of MlSpacesObject instances.

    Randomizes object-level properties: friction (of geoms), mass, and inertia.
    Note: Density cannot be modified at runtime in MuJoCo, only mass can be changed.

    Args:
        random_state (np.random.RandomState | None): Random state for reproducibility.
            If None, uses global numpy random state.
        randomize_friction (bool): If True, randomizes geom friction
        randomize_mass (bool): If True, randomizes object mass
        randomize_inertia (bool): If True, randomizes object inertia
        friction_perturbation_ratio (float): Relative magnitude of friction randomization
        mass_perturbation_ratio (float): Relative magnitude of mass randomization
        inertia_perturbation_ratio (float): Relative magnitude of inertia randomization
    """

    def __init__(
        self,
        random_state: np.random.RandomState | None = None,
        randomize_friction: bool = True,
        randomize_mass: bool = True,
        randomize_inertia: bool = True,
        friction_perturbation_ratio: float = 0.1,
        mass_perturbation_ratio: float = 0.1,
        inertia_perturbation_ratio: float = 0.1,
    ):
        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        self.randomize_friction = randomize_friction
        self.randomize_mass = randomize_mass
        self.randomize_inertia = randomize_inertia

        # Validate that perturbation ratios are < 1 to ensure positive values after perturbation
        if friction_perturbation_ratio >= 1.0:
            raise ValueError(
                f"friction_perturbation_ratio must be < 1.0, got {friction_perturbation_ratio}"
            )
        if mass_perturbation_ratio >= 1.0:
            raise ValueError(
                f"mass_perturbation_ratio must be < 1.0, got {mass_perturbation_ratio}"
            )
        if inertia_perturbation_ratio >= 1.0:
            raise ValueError(
                f"inertia_perturbation_ratio must be < 1.0, got {inertia_perturbation_ratio}"
            )

        self.friction_perturbation_ratio = friction_perturbation_ratio
        self.mass_perturbation_ratio = mass_perturbation_ratio
        self.inertia_perturbation_ratio = inertia_perturbation_ratio

        # Will be populated when randomize_objects is called
        # Keyed by object_id (int) instead of name since objects may not have names
        self._defaults: dict[int, dict] = {}

    def _save_object_defaults(self, obj: "MlSpacesObject") -> None:
        """Save default values for an object."""
        from molmo_spaces.env.data_views import MlSpacesObject

        if not isinstance(obj, MlSpacesObject):
            raise TypeError(f"Expected MlSpacesObject, got {type(obj)}")

        model = obj.mj_model
        object_id = obj.object_id
        object_root_id = model.body(object_id).rootid[0]

        # Get all bodies belonging to this object (root and all descendants)
        from molmo_spaces.utils import mj_model_and_data_utils

        body_ids = mj_model_and_data_utils.descendant_bodies(model, object_id)

        # Get total mass of the object (including all descendant bodies)
        # Use body_subtreemass if available (MuJoCo 3.0+), otherwise sum manually
        total_mass = float(model.body_subtreemass[object_id])

        # Get all geoms belonging to this object
        geom_frictions = {}
        for geom_id in range(model.ngeom):
            geom_body_id = model.geom(geom_id).bodyid.item()
            geom_root_id = model.body(geom_body_id).rootid[0]
            if geom_root_id == object_root_id:
                geom_frictions[geom_id] = np.array(model.geom_friction[geom_id])

        self._defaults[object_id] = {
            "mass": total_mass,
            "inertia": np.array(
                model.body_inertia[object_id]
            ),  # Root inertia (for backward compatibility)
            "body_ids": body_ids,  # Store all body IDs for this object
            # Get mass for all body IDs (root and all children)
            "body_masses": {bid: float(model.body_mass[bid]) for bid in body_ids},
            # Get inertia for all body IDs (root and all children)
            "body_inertias": {bid: np.array(model.body_inertia[bid]) for bid in body_ids},
            "geom_frictions": geom_frictions,
        }

    def randomize_object(self, obj: "MlSpacesObject") -> None:
        """
        Randomize dynamics properties of a single MlSpacesObject.

        Args:
            obj: MlSpacesObject instance to randomize
        """
        from molmo_spaces.env.data_views import MlSpacesObject

        if not isinstance(obj, MlSpacesObject):
            raise TypeError(f"Expected MlSpacesObject, got {type(obj)}")

        # Save defaults if not already saved
        object_id = obj.object_id
        if object_id not in self._defaults:
            self._save_object_defaults(obj)

        model = obj.mj_model
        defaults = self._defaults[object_id]

        # Randomize mass for all descendant bodies (maintaining proportional distribution)
        if self.randomize_mass:
            # Use object's body_ids property which uses descendant_bodies()
            body_ids = obj.body_ids
            body_masses = defaults.get("body_masses", {object_id: defaults["mass"]})
            total_mass = defaults["mass"]

            if total_mass > 0:  # Only randomize if object has non-zero mass
                # Apply perturbation to total mass
                # Since mass_perturbation_ratio < 1, (1.0 + perturbation) > 0, so new_total_mass > 0
                perturbation = self.random_state.uniform(
                    -self.mass_perturbation_ratio, self.mass_perturbation_ratio
                )
                new_total_mass = total_mass * (1.0 + perturbation)

                # Distribute the new total mass proportionally across all descendant bodies
                if len(body_ids) == 1:
                    # Single body: use object's _set_mass method
                    obj._set_mass(new_total_mass)
                else:
                    # Multiple bodies: maintain proportional distribution across all descendant bodies
                    mass_ratio = new_total_mass / total_mass
                    for bid in body_ids:
                        if bid in body_masses and body_masses[bid] > 0:
                            new_body_mass = body_masses[bid] * mass_ratio
                            model.body_mass[bid] = new_body_mass

        # Randomize inertia for all bodies (root and all children)
        if self.randomize_inertia:
            # Use object's body_ids property which uses descendant_bodies()
            body_ids = obj.body_ids
            body_inertias = defaults.get("body_inertias", {object_id: defaults["inertia"]})

            for bid in body_ids:
                if bid in body_inertias:
                    current_inertia = body_inertias[bid]
                    if np.any(current_inertia > 0):  # Only randomize if inertia exists
                        # Since inertia_perturbation_ratio < 1, (1.0 + perturbation) > 0, so new_inertia > 0
                        perturbation = self.random_state.uniform(
                            -self.inertia_perturbation_ratio,
                            self.inertia_perturbation_ratio,
                            size=3,
                        )
                        new_inertia = current_inertia * (1.0 + perturbation)
                        model.body_inertia[bid] = new_inertia

        # Randomize friction using object's _set_friction method
        # Compute average friction value across all geoms and apply perturbation
        if self.randomize_friction:
            if defaults["geom_frictions"]:
                # Get average default friction (using first component - sliding friction)
                avg_default_friction = np.mean(
                    [
                        friction[0]
                        for friction in defaults["geom_frictions"].values()
                        if np.any(friction > 0)
                    ]
                )
                if avg_default_friction > 0:
                    # Apply perturbation to average friction
                    # Since friction_perturbation_ratio < 1, (1.0 + perturbation) > 0, so new_friction > 0
                    perturbation = self.random_state.uniform(
                        -self.friction_perturbation_ratio,
                        self.friction_perturbation_ratio,
                    )
                    new_friction = avg_default_friction * (1.0 + perturbation)

                    # Ensure _geom_ids exists for _set_friction method (geom_ids is a property)
                    if not hasattr(obj, "_geom_ids") or obj._geom_ids is None:
                        object_root_id = model.body(object_id).rootid[0]
                        obj._geom_ids = []
                        for geom_id in range(model.ngeom):
                            geom_body_id = model.geom(geom_id).bodyid.item()
                            geom_root_id = model.body(geom_body_id).rootid[0]
                            if geom_root_id == object_root_id:
                                obj._geom_ids.append(geom_id)

                    # Use object's _set_friction method which sets friction for all geoms
                    obj._set_friction(new_friction)

    def randomize_objects(self, objects: list["MlSpacesObject"]) -> None:
        """
        Randomize dynamics properties of multiple MlSpacesObject instances.

        Args:
            objects: List of MlSpacesObject instances to randomize
        """
        # Save defaults for all objects first
        for obj in objects:
            if obj.object_id not in self._defaults:
                self._save_object_defaults(obj)

        # Then randomize each object
        for obj in objects:
            self.randomize_object(obj)

        # Forward pass to propagate changes
        if objects:
            # All objects should share the same model/data
            model = objects[0].mj_model
            data = objects[0].mj_data
            mujoco.mj_forward(model, data)

    def restore_object(self, obj: "MlSpacesObject") -> None:
        """
        Restore default values for a single object.

        Args:
            obj: MlSpacesObject instance to restore
        """
        object_id = obj.object_id
        if object_id not in self._defaults:
            return  # No defaults saved for this object

        model = obj.mj_model
        defaults = self._defaults[object_id]

        # Restore mass using object's _set_mass method
        # body_ids property uses descendant_bodies() internally
        if "body_masses" in defaults:
            # Sum up all body masses to get total mass
            total_mass = sum(defaults["body_masses"].values())
            obj._set_mass(total_mass)
        else:
            # Fallback to old behavior if body_masses not saved
            obj._set_mass(defaults["mass"])

        # Restore inertia for all bodies (root and all children)
        # Note: No object method for inertia, so use direct access
        # Use object's body_ids property which uses descendant_bodies()
        body_ids = obj.body_ids
        if "body_inertias" in defaults:
            for body_id in body_ids:
                if body_id in defaults["body_inertias"]:
                    model.body_inertia[body_id] = defaults["body_inertias"][body_id]
        else:
            # Fallback to old behavior if body_inertias not saved
            model.body_inertia[object_id] = defaults["inertia"]

        # Restore friction using object's _set_friction method
        # Compute average default friction to restore
        if defaults["geom_frictions"]:
            avg_default_friction = np.mean(
                [
                    friction[0]
                    for friction in defaults["geom_frictions"].values()
                    if np.any(friction > 0)
                ]
            )
            if avg_default_friction > 0:
                # Ensure _geom_ids exists for _set_friction method (geom_ids is a property)
                if not hasattr(obj, "_geom_ids") or obj._geom_ids is None:
                    object_root_id = model.body(object_id).rootid[0]
                    obj._geom_ids = []
                    for geom_id in range(model.ngeom):
                        geom_body_id = model.geom(geom_id).bodyid.item()
                        geom_root_id = model.body(geom_body_id).rootid[0]
                        if geom_root_id == object_root_id:
                            obj._geom_ids.append(geom_id)

                obj._set_friction(avg_default_friction)

        # Forward pass
        mujoco.mj_forward(model, obj.mj_data)

    def restore_objects(self, objects: list["MlSpacesObject"]) -> None:
        """
        Restore default values for multiple objects.

        Args:
            objects: List of MlSpacesObject instances to restore
        """
        for obj in objects:
            self.restore_object(obj)
