import mujoco

from molmo_spaces.env.data_views import MlSpacesArticulationObject
from molmo_spaces.env.scene.thor.physical_properties import RandomizePhysicalProperties

CABINET = RandomizePhysicalProperties()
CABINET.randomize_friction = True
CABINET.randomize_density = True
CABINET.randomize_joint_stiffness = True
CABINET.randomize_joint_damping = True
CABINET.randomize_joint_armature = True
CABINET.randomize_joint_frictionloss = True

CABINET.range_friction = (0.1, 1)
CABINET.range_density = (1000, 1000)
CABINET.range_joint_stiffness = (0, 0)
CABINET.range_joint_damping = (0.01, 0.1)
CABINET.range_joint_armature = (0, 0.01)
CABINET.range_joint_frictionloss = (0.1, 0.3)  # (150, 300)


class Cabinet(MlSpacesArticulationObject):
    def __init__(self, object_name: str, data: mujoco.MjData, body_name2id: dict[str, int]) -> None:
        super().__init__(object_name, data)
        self._set_friction(CABINET.get_random_value("friction"))
        # Note: Density cannot be modified at runtime in MuJoCo
        # self._set_density(CABINET.get_random_value("density"))
        self._set_max_mass(1)
        self._set_joint_stiffness(CABINET.get_random_value("joint_stiffness"))
        self._set_joint_damping(CABINET.get_random_value("joint_damping"))
        self._set_joint_armature(CABINET.get_random_value("joint_armature"))
        self._set_joint_frictionloss(CABINET.get_random_value("joint_frictionloss"))


"""
can only modify these....
<_MjModelGeomViews
  bodyid: array([1], dtype=int32)
  conaffinity: array([15], dtype=int32)
  condim: array([3], dtype=int32)
  contype: array([8], dtype=int32)
  dataid: array([-1], dtype=int32)
  friction: array([0.9  , 0.9  , 0.001])
  gap: array([0.])
  group: array([4], dtype=int32)
  id: 0
  margin: array([0.])
  matid: array([-1], dtype=int32)
  name: 'floor_floor'
  pos: array([0., 0., 0.])
  priority: array([0], dtype=int32)
  quat: array([1., 0., 0., 0.])
  rbound: array([0.])
  rgba: array([0.5, 0.5, 0.5, 1. ], dtype=float32)
  sameframe: array([1], dtype=uint8)
  size: array([0.  , 0.  , 0.01])
  solimp: array([9.98e-01, 9.98e-01, 1.00e-03, 5.00e-01, 2.00e+00])
  solmix: array([1.])
  solref: array([0.025, 1.   ])
  type: array([0], dtype=int32)
  user: array([], dtype=float64)
>
"""
