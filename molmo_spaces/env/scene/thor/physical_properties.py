import numpy as np


class PhysicalProperty:
    def __init__(self, enabled: bool = False, range: tuple[float, float] = (1.0, 3.0)) -> None:
        self.enabled = enabled
        self.range = range


class RandomizePhysicalProperties:
    def __init__(self, property_name: str = None, range: tuple[float, float] = None) -> None:
        self.properties = {
            "friction": PhysicalProperty(),
            "density": PhysicalProperty(),
            "mass": PhysicalProperty(),
            "joint": {
                "stiffness": PhysicalProperty(),
                "damping": PhysicalProperty(),
                "limited": PhysicalProperty(),
                "armature": PhysicalProperty(),
                "frictionloss": PhysicalProperty(),
            },
        }

        # Initialize with specific property if provided
        if property_name is not None:
            if property_name.startswith("joint_"):
                joint_prop = property_name.replace("joint_", "")
                if joint_prop in self.properties["joint"]:
                    self.properties["joint"][joint_prop].enabled = True
                    if range is not None:
                        self.properties["joint"][joint_prop].range = range
            elif property_name in self.properties:
                self.properties[property_name].enabled = True
                if range is not None:
                    self.properties[property_name].range = range

    # Property getters and setters
    @property
    def randomize_friction(self) -> bool:
        return self.properties["friction"].enabled

    @randomize_friction.setter
    def randomize_friction(self, value: bool) -> None:
        self.properties["friction"].enabled = value

    @property
    def randomize_density(self) -> bool:
        return self.properties["density"].enabled

    @randomize_density.setter
    def randomize_density(self, value: bool) -> None:
        self.properties["density"].enabled = value

    @property
    def randomize_mass(self) -> bool:
        return self.properties["mass"].enabled

    @randomize_mass.setter
    def randomize_mass(self, value: bool) -> None:
        self.properties["mass"].enabled = value

    @property
    def randomize_joint_stiffness(self) -> bool:
        return self.properties["joint"]["stiffness"].enabled

    @randomize_joint_stiffness.setter
    def randomize_joint_stiffness(self, value: bool) -> None:
        self.properties["joint"]["stiffness"].enabled = value

    @property
    def randomize_joint_damping(self) -> bool:
        return self.properties["joint"]["damping"].enabled

    @randomize_joint_damping.setter
    def randomize_joint_damping(self, value: bool) -> None:
        self.properties["joint"]["damping"].enabled = value

    @property
    def randomize_joint_limited(self) -> bool:
        return self.properties["joint"]["limited"].enabled

    @randomize_joint_limited.setter
    def randomize_joint_limited(self, value: bool) -> None:
        self.properties["joint"]["limited"].enabled = value

    @property
    def randomize_joint_armature(self) -> bool:
        return self.properties["joint"]["armature"].enabled

    @randomize_joint_armature.setter
    def randomize_joint_armature(self, value: bool) -> None:
        self.properties["joint"]["armature"].enabled = value

    @property
    def randomize_joint_frictionloss(self) -> bool:
        return self.properties["joint"]["frictionloss"].enabled

    @randomize_joint_frictionloss.setter
    def randomize_joint_frictionloss(self, value: bool) -> None:
        self.properties["joint"]["frictionloss"].enabled = value

    # Range properties
    @property
    def range_friction(self) -> tuple[float, float]:
        return self.properties["friction"].range

    @range_friction.setter
    def range_friction(self, value: tuple[float, float]) -> None:
        self.properties["friction"].range = value

    @property
    def range_density(self) -> tuple[float, float]:
        return self.properties["density"].range

    @range_density.setter
    def range_density(self, value: tuple[float, float]) -> None:
        self.properties["density"].range = value

    @property
    def range_mass(self) -> tuple[float, float]:
        return self.properties["mass"].range

    @range_mass.setter
    def range_mass(self, value: tuple[float, float]) -> None:
        self.properties["mass"].range = value

    @property
    def range_joint_stiffness(self) -> tuple[float, float]:
        return self.properties["joint"]["stiffness"].range

    @range_joint_stiffness.setter
    def range_joint_stiffness(self, value: tuple[float, float]) -> None:
        self.properties["joint"]["stiffness"].range = value

    @property
    def range_joint_damping(self) -> tuple[float, float]:
        return self.properties["joint"]["damping"].range

    @range_joint_damping.setter
    def range_joint_damping(self, value: tuple[float, float]) -> None:
        self.properties["joint"]["damping"].range = value

    @property
    def range_joint_limited(self) -> tuple[float, float]:
        return self.properties["joint"]["limited"].range

    @range_joint_limited.setter
    def range_joint_limited(self, value: tuple[float, float]) -> None:
        self.properties["joint"]["limited"].range = value

    @property
    def range_joint_armature(self) -> tuple[float, float]:
        return self.properties["joint"]["armature"].range

    @range_joint_armature.setter
    def range_joint_armature(self, value: tuple[float, float]) -> None:
        self.properties["joint"]["armature"].range = value

    @property
    def range_joint_frictionloss(self) -> tuple[float, float]:
        return self.properties["joint"]["frictionloss"].range

    @range_joint_frictionloss.setter
    def range_joint_frictionloss(self, value: tuple[float, float]) -> None:
        self.properties["joint"]["frictionloss"].range = value

    # Keep the utility methods
    def enable_property(self, property_name: str, joint_property: str = None) -> None:
        """Enable randomization for a specific property.

        Args:
            property_name: Name of the property to enable ('friction', 'density', 'mass', 'joint')
            joint_property: If property_name is 'joint', specify which joint property
        """
        if property_name == "joint" and joint_property:
            self.properties["joint"][joint_property].enabled = True
        elif property_name in self.properties:
            self.properties[property_name].enabled = True

    def set_range(
        self, property_name: str, range: tuple[float, float], joint_property: str = None
    ) -> None:
        """Set the randomization range for a specific property.

        Args:
            property_name: Name of the property to set range for
            range: Tuple of (min, max) values
            joint_property: If property_name is 'joint', specify which joint property
        """
        if property_name == "joint" and joint_property:
            self.properties["joint"][joint_property].range = range
        elif property_name in self.properties:
            self.properties[property_name].range = range

    def is_enabled(self, property_name: str, joint_property: str = None) -> bool:
        """Check if randomization is enabled for a specific property.

        Args:
            property_name: Name of the property to check
            joint_property: If property_name is 'joint', specify which joint property

        Returns:
            bool: True if randomization is enabled for the property
        """
        if property_name == "joint" and joint_property:
            return self.properties["joint"][joint_property].enabled
        return self.properties[property_name].enabled if property_name in self.properties else False

    def get_range(self, property_name: str, joint_property: str = None) -> tuple[float, float]:
        """Get the randomization range for a specific property.

        Args:
            property_name: Name of the property to get range for
            joint_property: If property_name is 'joint', specify which joint property

        Returns:
            Tuple[float, float]: The (min, max) range for the property
        """
        if property_name == "joint" and joint_property:
            return self.properties["joint"][joint_property].range
        return (
            self.properties[property_name].range if property_name in self.properties else (1.0, 3.0)
        )

    def get_random_value(self, property_name: str) -> float:
        """Get a random value for a specific property.

        Args:
            property_name: Name of the property to get random value for
            joint_property: If property_name is 'joint', specify which joint property

        Returns:
            float: A random value for the property
        """
        if property_name.startswith("joint_"):
            joint_property = property_name.replace("joint_", "")
            return np.random.uniform(
                self.properties["joint"][joint_property].range[0],
                self.properties["joint"][joint_property].range[1],
            )
        return np.random.uniform(
            self.properties[property_name].range[0], self.properties[property_name].range[1]
        )
