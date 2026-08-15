from dataclasses import dataclass, field


@dataclass
class Object:
    body_id: int
    name: str
    category: str
    asset_id: str
    is_static: bool
    has_freejoint: bool
    # Articulation metadata (set by Scene.__init__ for objects with hinge/slide
    # joints that survive _optimize). Empty dicts/lists for non-articulated
    # objects so the pick task is unaffected.
    thor_name: str = ""  # e.g. "Dresser_220_1"
    joint_xml_names: list = field(default_factory=list)  # XML joint names (children)
    joint_ids: list = field(default_factory=list)  # mj joint ids (same order)
    joint_thor_names: list = field(default_factory=list)  # THOR names (same order)
    joint_body_ids: list = field(default_factory=list)  # moving body for each joint

    @property
    def is_articulated(self) -> bool:
        return len(self.joint_ids) > 0

    def position(self, data):
        return data.xpos[self.body_id].copy()

    def quat(self, data):
        return data.xquat[self.body_id].copy()
