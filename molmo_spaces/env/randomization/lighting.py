import mujoco
import numpy as np
from mujoco import MjData, MjModel
from scipy.spatial.transform import Rotation


class LightingRandomizer:
    """
    Randomizer for lighting properties in MuJoCo simulations.

    Based on the mujoco-py LightingModder implementation, adapted to work
    with MjModel and MjData directly (instead of MjSim).

    Args:
        model (MjModel): MuJoCo model
        random_state (np.random.RandomState | None): Random state for reproducibility.
            If None, uses global numpy random state.
        light_names (list[str] | None): List of light names to randomize.
            If None, randomizes all lights in the model.
        randomize_position (bool): If True, randomizes light position
        randomize_direction (bool): If True, randomizes light direction
        randomize_specular (bool): If True, randomizes specular color
        randomize_ambient (bool): If True, randomizes ambient color
        randomize_diffuse (bool): If True, randomizes diffuse color
        randomize_active (bool): If True, randomizes whether light is active
        position_perturbation_size (float): Magnitude of position randomization
        direction_perturbation_size (float): Magnitude of direction randomization in radians
        specular_perturbation_size (float): Magnitude of specular color randomization
        ambient_perturbation_size (float): Magnitude of ambient color randomization
        diffuse_perturbation_size (float): Magnitude of diffuse color randomization

    Note:
        MjData should be passed to the randomize() method, not to __init__.
    """

    def __init__(
        self,
        model: MjModel,
        random_state: np.random.RandomState | None = None,
        light_names: list[str] | None = None,
        randomize_position: bool = True,
        randomize_direction: bool = True,
        randomize_specular: bool = True,
        randomize_ambient: bool = True,
        randomize_diffuse: bool = True,
        randomize_active: bool = True,
        position_perturbation_size: float = 0.1,
        direction_perturbation_size: float = 0.35,  # ~20 degrees
        specular_perturbation_size: float = 0.1,
        ambient_perturbation_size: float = 0.1,
        diffuse_perturbation_size: float = 0.1,
    ):
        self.model = model

        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        # Get light IDs from model (use IDs directly since lights may not have names)
        if light_names is None:
            # Use all light IDs (0 to nlight-1)
            self.light_ids = list(range(model.nlight))
        else:
            # Convert light names to IDs
            self.light_ids = []
            for name in light_names:
                light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
                if light_id >= 0:
                    self.light_ids.append(light_id)

        # Debug: print detected lights
        if model.nlight == 0:
            print(f"   Warning: No lights in model (model.nlight={model.nlight})")
        else:
            print(
                f"   Found {len(self.light_ids)} lights (model.nlight={model.nlight}): IDs {self.light_ids}"
            )

        self.randomize_position = randomize_position
        self.randomize_direction = randomize_direction
        self.randomize_specular = randomize_specular
        self.randomize_ambient = randomize_ambient
        self.randomize_diffuse = randomize_diffuse
        self.randomize_active = randomize_active

        self.position_perturbation_size = position_perturbation_size
        self.direction_perturbation_size = direction_perturbation_size
        self.specular_perturbation_size = specular_perturbation_size
        self.ambient_perturbation_size = ambient_perturbation_size
        self.diffuse_perturbation_size = diffuse_perturbation_size

        # Enable shadow casting for all lights by default (required for shadows to appear)
        # This ensures shadows are visible even when lighting randomization is disabled
        for light_id in self.light_ids:
            if hasattr(self.model, "light_castshadow"):
                self.model.light_castshadow[light_id] = 1

        self.save_defaults()

    def save_defaults(self):
        """
        Save default light parameter values from the current model state.
        """
        self._defaults = {light_id: {} for light_id in self.light_ids}
        for light_id in self.light_ids:
            self._defaults[light_id]["pos"] = np.array(self.model.light_pos[light_id])
            self._defaults[light_id]["dir"] = np.array(self.model.light_dir[light_id])
            self._defaults[light_id]["specular"] = np.array(self.model.light_specular[light_id])
            self._defaults[light_id]["ambient"] = np.array(self.model.light_ambient[light_id])
            self._defaults[light_id]["diffuse"] = np.array(self.model.light_diffuse[light_id])
            self._defaults[light_id]["active"] = int(self.model.light_active[light_id])
            # Save castshadow if available (enables shadow casting)
            if hasattr(self.model, "light_castshadow"):
                self._defaults[light_id]["castshadow"] = int(self.model.light_castshadow[light_id])

    def restore_defaults(self):
        """
        Restore saved default light parameter values.
        """
        for light_id in self.light_ids:
            self.set_pos(light_id, self._defaults[light_id]["pos"])
            self.set_dir(light_id, self._defaults[light_id]["dir"])
            self.set_specular(light_id, self._defaults[light_id]["specular"])
            self.set_ambient(light_id, self._defaults[light_id]["ambient"])
            self.set_diffuse(light_id, self._defaults[light_id]["diffuse"])
            self.set_active(light_id, self._defaults[light_id]["active"])
            # Restore castshadow if it was saved
            if hasattr(self.model, "light_castshadow") and "castshadow" in self._defaults[light_id]:
                self.model.light_castshadow[light_id] = self._defaults[light_id]["castshadow"]

    def randomize(self, data: MjData | None = None):
        """
        Randomize all enabled light properties.

        Args:
            data (MjData | None): MuJoCo data for forward pass. If None, forward pass is skipped.
        """
        # Track which lights are active before randomization
        active_lights_before = []
        for light_id in self.light_ids:
            if self.model.light_active[light_id] > 0:
                active_lights_before.append(light_id)

        for light_id in self.light_ids:
            if self.randomize_position:
                self._randomize_position(light_id)

            if self.randomize_direction:
                self._randomize_direction(light_id)

            if self.randomize_specular:
                self._randomize_specular(light_id)

            if self.randomize_ambient:
                self._randomize_ambient(light_id)

            if self.randomize_diffuse:
                self._randomize_diffuse(light_id)

            if self.randomize_active:
                self._randomize_active(light_id)

            # Enable shadow casting for all lights (required for shadows to appear)
            if hasattr(self.model, "light_castshadow"):
                self.model.light_castshadow[light_id] = 1

        # Ensure at least one light is active for shadows to be visible
        # If randomize_active turned off all lights, re-enable at least one
        active_lights_after = []
        for light_id in self.light_ids:
            if self.model.light_active[light_id] > 0:
                active_lights_after.append(light_id)

        if len(active_lights_after) == 0 and len(self.light_ids) > 0:
            # All lights were turned off - re-enable at least one (prefer the first one that was active before)
            if active_lights_before:
                # Re-enable the first light that was active before
                self.model.light_active[active_lights_before[0]] = 1
            else:
                # If no lights were active before, enable the first light
                self.model.light_active[self.light_ids[0]] = 1

        # Forward pass to propagate changes
        if data is not None:
            mujoco.mj_forward(self.model, data)

    def _randomize_position(self, light_id: int):
        """
        Randomize position of a specific light.

        Args:
            light_id (int): ID of the light
        """
        delta_pos = self.random_state.uniform(
            low=-self.position_perturbation_size,
            high=self.position_perturbation_size,
            size=3,
        )
        new_pos = self._defaults[light_id]["pos"] + delta_pos
        self.set_pos(light_id, new_pos)

    def _randomize_direction(self, light_id: int):
        """
        Randomize direction (orientation) of a specific light.

        Args:
            light_id (int): ID of the light
        """
        # Sample a random axis and angle for rotation
        random_axis = self.random_state.uniform(-1, 1, size=3)
        axis_norm = np.linalg.norm(random_axis)
        if axis_norm > 1e-6:
            random_axis = random_axis / axis_norm
        else:
            random_axis = np.array([0, 0, 1])  # Fallback to z-axis

        random_angle = self.random_state.uniform(
            -self.direction_perturbation_size, self.direction_perturbation_size
        )

        # Create rotation from axis-angle
        rotation = Rotation.from_rotvec(random_axis * random_angle)

        # Apply rotation to default direction
        default_dir = self._defaults[light_id]["dir"]
        default_dir_norm = np.linalg.norm(default_dir)
        if default_dir_norm > 1e-6:
            default_dir_normalized = default_dir / default_dir_norm
        else:
            default_dir_normalized = np.array([0, 0, -1])  # Default downward direction

        new_dir = rotation.apply(default_dir_normalized)
        # Normalize the new direction to ensure it's a unit vector
        new_dir_norm = np.linalg.norm(new_dir)
        if new_dir_norm > 1e-6:
            new_dir = new_dir / new_dir_norm
        else:
            new_dir = default_dir_normalized

        # Scale back to original magnitude if default had non-unit length
        new_dir = new_dir * default_dir_norm if default_dir_norm > 1e-6 else new_dir

        self.set_dir(light_id, new_dir)

    def _randomize_specular(self, light_id: int):
        """
        Randomize specular color of a specific light.

        Args:
            light_id (int): ID of the light
        """
        delta = self.random_state.uniform(
            low=-self.specular_perturbation_size,
            high=self.specular_perturbation_size,
            size=3,
        )
        new_specular = np.clip(self._defaults[light_id]["specular"] + delta, 0.0, 1.0)
        self.set_specular(light_id, new_specular)

    def _randomize_ambient(self, light_id: int):
        """
        Randomize ambient color of a specific light.

        Args:
            light_id (int): ID of the light
        """
        delta = self.random_state.uniform(
            low=-self.ambient_perturbation_size,
            high=self.ambient_perturbation_size,
            size=3,
        )
        new_ambient = np.clip(self._defaults[light_id]["ambient"] + delta, 0.0, 1.0)
        self.set_ambient(light_id, new_ambient)

    def _randomize_diffuse(self, light_id: int):
        """
        Randomize diffuse color of a specific light.

        Args:
            light_id (int): ID of the light
        """
        delta = self.random_state.uniform(
            low=-self.diffuse_perturbation_size,
            high=self.diffuse_perturbation_size,
            size=3,
        )
        new_diffuse = np.clip(self._defaults[light_id]["diffuse"] + delta, 0.0, 1.0)
        self.set_diffuse(light_id, new_diffuse)

    def _randomize_active(self, light_id: int):
        """
        Randomize active state of a specific light.

        Args:
            light_id (int): ID of the light
        """
        active = int(self.random_state.uniform() > 0.5)
        self.set_active(light_id, active)

    def get_pos(self, light_id: int) -> np.ndarray:
        """
        Get position of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (x, y, z) position
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        return np.array(self.model.light_pos[light_id])

    def set_pos(self, light_id: int, value: np.ndarray):
        """
        Set position of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (x, y, z) position
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError(f"Expected 3-dim value, got shape {value.shape}")
        self.model.light_pos[light_id] = value

    def get_dir(self, light_id: int) -> np.ndarray:
        """
        Get direction of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (x, y, z) direction vector
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        return np.array(self.model.light_dir[light_id])

    def set_dir(self, light_id: int, value: np.ndarray):
        """
        Set direction of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (x, y, z) direction vector
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError(f"Expected 3-dim value, got shape {value.shape}")
        # Normalize direction vector
        norm = np.linalg.norm(value)
        if norm > 0:
            value = value / norm
        self.model.light_dir[light_id] = value

    def get_active(self, light_id: int) -> int:
        """
        Get active state of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            int: 1 if active, 0 if inactive
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        return int(self.model.light_active[light_id])

    def set_active(self, light_id: int, value: int):
        """
        Set active state of a specific light.

        Args:
            light_id (int): ID of the light
            value (int): 1 for active, 0 for inactive
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        self.model.light_active[light_id] = value

    def get_specular(self, light_id: int) -> np.ndarray:
        """
        Get specular color of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (r, g, b) specular color
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        return np.array(self.model.light_specular[light_id])

    def set_specular(self, light_id: int, value: np.ndarray):
        """
        Set specular color of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (r, g, b) specular color
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError(f"Expected 3-dim value, got shape {value.shape}")
        self.model.light_specular[light_id] = np.clip(value, 0.0, 1.0)

    def get_ambient(self, light_id: int) -> np.ndarray:
        """
        Get ambient color of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (r, g, b) ambient color
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        return np.array(self.model.light_ambient[light_id])

    def set_ambient(self, light_id: int, value: np.ndarray):
        """
        Set ambient color of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (r, g, b) ambient color
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError(f"Expected 3-dim value, got shape {value.shape}")
        self.model.light_ambient[light_id] = np.clip(value, 0.0, 1.0)

    def get_diffuse(self, light_id: int) -> np.ndarray:
        """
        Get diffuse color of a specific light.

        Args:
            light_id (int): ID of the light

        Returns:
            np.ndarray: (r, g, b) diffuse color
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        return np.array(self.model.light_diffuse[light_id])

    def set_diffuse(self, light_id: int, value: np.ndarray):
        """
        Set diffuse color of a specific light.

        Args:
            light_id (int): ID of the light
            value (np.ndarray): (r, g, b) diffuse color
        """
        if light_id < 0 or light_id >= self.model.nlight:
            raise ValueError(f"Invalid light ID: {light_id}")
        value = np.asarray(value)
        if value.shape != (3,):
            raise ValueError(f"Expected 3-dim value, got shape {value.shape}")
        self.model.light_diffuse[light_id] = np.clip(value, 0.0, 1.0)

    def update_model(self, model: MjModel):
        """
        Update the model reference.

        Args:
            model (MjModel): New MuJoCo model
        """
        self.model = model
        self.save_defaults()
