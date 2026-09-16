import mujoco

from molmo_spaces.env.data_views import MlSpacesArticulationObject
from molmo_spaces.env.scene.thor.physical_properties import RandomizePhysicalProperties

SHOWER_DOOR = RandomizePhysicalProperties()
SHOWER_DOOR.randomize_friction = True
SHOWER_DOOR.randomize_density = True
SHOWER_DOOR.randomize_joint_stiffness = True
SHOWER_DOOR.randomize_joint_damping = True
SHOWER_DOOR.randomize_joint_armature = True
SHOWER_DOOR.randomize_joint_frictionloss = True

SHOWER_DOOR.range_friction = (0.5, 1.5)
SHOWER_DOOR.range_density = (1000, 1000)
SHOWER_DOOR.range_joint_stiffness = (0, 0)
SHOWER_DOOR.range_joint_damping = (0, 1)
SHOWER_DOOR.range_joint_armature = (0, 0.1)
SHOWER_DOOR.range_joint_frictionloss = (0.5, 0.5)


class ShowerDoor(MlSpacesArticulationObject):
    def __init__(self, object_name: str, data: mujoco.MjData, body_name2id: dict[str, int]) -> None:
        super().__init__(object_name, data)
        self._set_friction(SHOWER_DOOR.get_random_value("friction"))
        # Note: Density cannot be modified at runtime in MuJoCo
        # self._set_density(SHOWER_DOOR.get_random_value("density"))
        self._set_joint_stiffness(SHOWER_DOOR.get_random_value("joint_stiffness"))
        self._set_joint_damping(SHOWER_DOOR.get_random_value("joint_damping"))
        self._set_joint_armature(SHOWER_DOOR.get_random_value("joint_armature"))
        self._set_joint_frictionloss(SHOWER_DOOR.get_random_value("joint_frictionloss"))
