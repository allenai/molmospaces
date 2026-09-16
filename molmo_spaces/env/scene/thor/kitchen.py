import mujoco

from molmo_spaces.env.data_views import MlSpacesArticulationObject
from molmo_spaces.env.scene.thor.physical_properties import RandomizePhysicalProperties

OVEN = RandomizePhysicalProperties()
OVEN.randomize_friction = True
OVEN.randomize_density = True
OVEN.randomize_joint_stiffness = True
OVEN.randomize_joint_damping = True
OVEN.randomize_joint_armature = True
OVEN.randomize_joint_frictionloss = True

OVEN.range_friction = (0.5, 1.5)
OVEN.range_density = (1000, 1000)
OVEN.range_mass = (3, 6)
OVEN.range_joint_stiffness = (1, 5)  # (195, 205)
OVEN.range_joint_damping = (10, 15)
OVEN.range_joint_armature = (0, 0.1)
OVEN.range_joint_frictionloss = (10, 10)


DISHWASHER = RandomizePhysicalProperties()
DISHWASHER.randomize_friction = True
DISHWASHER.randomize_density = True
DISHWASHER.randomize_joint_stiffness = True
DISHWASHER.randomize_joint_damping = True
DISHWASHER.randomize_joint_armature = True
DISHWASHER.randomize_joint_frictionloss = True

DISHWASHER.range_friction = (1.5, 3.0)
DISHWASHER.range_density = (1000, 1000)
DISHWASHER.range_mass = (1, 2)
DISHWASHER.range_joint_stiffness = (1, 5)  # (10, 15) #$(200, 205)
DISHWASHER.range_joint_damping = (10, 15)
DISHWASHER.range_joint_armature = (0, 0.1)
DISHWASHER.range_joint_frictionloss = (10, 10)


STOVEKNOB = RandomizePhysicalProperties()
STOVEKNOB.randomize_friction = True
STOVEKNOB.randomize_density = True
STOVEKNOB.randomize_joint_stiffness = True
STOVEKNOB.randomize_joint_damping = True
STOVEKNOB.randomize_joint_armature = True
STOVEKNOB.randomize_joint_frictionloss = True

STOVEKNOB.range_friction = (0.5, 1.5)
STOVEKNOB.range_density = (1000, 1000)
STOVEKNOB.range_joint_stiffness = (0, 0)
STOVEKNOB.range_joint_damping = (0, 0.1)
STOVEKNOB.range_joint_armature = (0, 0.1)
STOVEKNOB.range_joint_frictionloss = (0, 1)


class Oven(MlSpacesArticulationObject):
    def __init__(self, object_name: str, data: mujoco.MjData, body_name2id: dict[str, int]) -> None:
        super().__init__(object_name, data)
        self._set_friction(OVEN.get_random_value("friction"))
        # Note: Density cannot be modified at runtime in MuJoCo
        # self._set_density(OVEN.get_random_value("density"))
        self._set_mass(OVEN.get_random_value("mass"))
        self._set_joint_stiffness(OVEN.get_random_value("joint_stiffness"))
        self._set_joint_damping(OVEN.get_random_value("joint_damping"))
        self._set_joint_armature(OVEN.get_random_value("joint_armature"))
        self._set_joint_frictionloss(OVEN.get_random_value("joint_frictionloss"))


class Dishwasher(MlSpacesArticulationObject):
    def __init__(self, object_name: str, data: mujoco.MjData, body_name2id: dict[str, int]) -> None:
        super().__init__(object_name, data)
        self._set_friction(DISHWASHER.get_random_value("friction"))
        # self._set_density(DISHWASHER.get_random_value("density"))
        self._set_mass(DISHWASHER.get_random_value("mass"))
        self._set_joint_stiffness(DISHWASHER.get_random_value("joint_stiffness"))
        self._set_joint_damping(DISHWASHER.get_random_value("joint_damping"))
        self._set_joint_armature(DISHWASHER.get_random_value("joint_armature"))
        self._set_joint_frictionloss(DISHWASHER.get_random_value("joint_frictionloss"))


class Stoveknob(MlSpacesArticulationObject):
    def __init__(self, object_name: str, data: mujoco.MjData, body_name2id: dict[str, int]) -> None:
        super().__init__(object_name, data)
        self._set_friction(STOVEKNOB.get_random_value("friction"))
        # self._set_density(STOVEKNOB.get_random_value("density"))
        self._set_joint_stiffness(STOVEKNOB.get_random_value("joint_stiffness"))
        self._set_joint_damping(STOVEKNOB.get_random_value("joint_damping"))
        self._set_joint_armature(STOVEKNOB.get_random_value("joint_armature"))
        self._set_joint_frictionloss(STOVEKNOB.get_random_value("joint_frictionloss"))
