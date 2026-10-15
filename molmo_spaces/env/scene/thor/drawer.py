import mujoco

from molmo_spaces.env.data_views import MlSpacesArticulationObject
from molmo_spaces.env.scene.thor.physical_properties import RandomizePhysicalProperties

DRAWER = RandomizePhysicalProperties()
DRAWER.randomize_friction = True
DRAWER.randomize_density = True
DRAWER.randomize_joint_stiffness = True
DRAWER.randomize_joint_damping = True
DRAWER.randomize_joint_armature = True
DRAWER.randomize_joint_frictionloss = True

DRAWER.range_friction = (0.1, 0.3)
DRAWER.range_density = (1000, 1000)
DRAWER.range_joint_stiffness = (0, 0)
DRAWER.range_joint_damping = (0.01, 0.1)
DRAWER.range_joint_armature = (0, 0.1)
DRAWER.range_joint_frictionloss = (1, 2)


class Drawer(MlSpacesArticulationObject):
    def __init__(self, object_name: str, data: mujoco.MjData, body_name2id: dict[str, int]) -> None:
        super().__init__(object_name, data)
        self._set_friction(DRAWER.get_random_value("friction"))
        # Note: Density cannot be modified at runtime in MuJoCo
        # self._set_density(DRAWER.get_random_value("density"))
        self._set_max_mass(1)
        self._set_joint_stiffness(DRAWER.get_random_value("joint_stiffness"))
        self._set_joint_damping(DRAWER.get_random_value("joint_damping"))
        self._set_joint_armature(DRAWER.get_random_value("joint_armature"))
        self._set_joint_frictionloss(DRAWER.get_random_value("joint_frictionloss"))
