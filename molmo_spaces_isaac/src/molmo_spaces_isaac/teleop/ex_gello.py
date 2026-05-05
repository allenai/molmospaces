from dataclasses import dataclass
from pathlib import Path

import tyro
from isaaclab.app import AppLauncher


@dataclass
class Args:
    franka_model: Path

    cube_model: Path


# launch omniverse app
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import msgspec
import numpy as np
import torch
from gello.agents.gello_agent import DynamixelRobot

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext


class DynamixelConfigSpec(msgspec.Struct):
    port: str = "/dev/ttyUSB1"
    joint_ids: list[int] = msgspec.field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7])
    joint_offsets: list[float] = msgspec.field(
        default_factory=lambda: [4.712, 3.142, 3.142, 4.712, 3.142, 4.712, 2.357]
    )
    joint_signs: list[int] = msgspec.field(default_factory=lambda: [1, -1, 1, 1, 1, -1, 1])
    gripper_config: tuple[int, float, float] = msgspec.field(
        default_factory=lambda: (8, 159.62109375, 201.42109375)
    )


def main() -> int:
    args = tyro.cli(Args)

    if not args.franka_model.is_file():
        return 1

    gello_cfg = DynamixelConfigSpec()
    gello_driver = DynamixelRobot(
        joint_ids=gello_cfg.joint_ids,
        joint_offsets=gello_cfg.joint_offsets,
        joint_signs=gello_cfg.joint_signs,
        real=True,
        port=gello_cfg.port,
        baudrate=57600,
        gripper_config=gello_cfg.gripper_config,
        start_joints=np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.0]),
    )

    franka_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.franka_model.absolute().as_posix(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": -0.569,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.810,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3.037,
                "fr3_joint7": 0.741,
                "fr3_gripper_left_driver_joint": 0.0,
                "fr3_gripper_left_spring_link_joint": 0.0,
                "fr3_gripper_left_follower_joint": 0.0,
                "fr3_gripper_right_driver_joint": 0.0,
                "fr3_gripper_right_spring_link_joint": 0.0,
                "fr3_gripper_right_follower_joint": 0.0,
            }
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=4000.0,
                damping=450.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=2000.0,
                damping=200.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["fr3_gripper_left_driver_joint"],
                effort_limit_sim=200.0,
                velocity_limit_sim=5.0,
                stiffness=1.0,
                damping=0.1,
            ),
        },
    )

    cube_cfg = RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.cube_model.absolute().as_posix(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.025),
        ),
        prim_path="/World/cube",
    )

    prim_cfg = RigidObjectCfg(
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.025),
        ),
        prim_path="/World/prim",
    )

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim_ctx = SimulationContext(sim_cfg)

    plane_cfg = sim_utils.GroundPlaneCfg()
    plane_cfg.func("/World/floor", plane_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    franka_cfg.prim_path = "/World/robot"
    franka = Articulation(cfg=franka_cfg)

    _ = RigidObject(cfg=cube_cfg)
    _ = RigidObject(cfg=prim_cfg)

    sim_dt = sim_ctx.get_physics_dt()

    sim_ctx.reset()

    while simulation_app.is_running():
        gello_joints = gello_driver.get_joint_state()
        joint_pos = franka.data.default_joint_pos.clone().cpu().squeeze().numpy()
        joint_pos[:7] = gello_joints[:7]
        joint_pos[7] = np.clip(1.0 - gello_joints[7], 0.0, 1.0) * np.deg2rad(51.5662)

        joint_targets = torch.tensor(joint_pos, device=sim_cfg.device)

        franka.set_joint_position_target(joint_targets)

        franka.write_data_to_sim()
        sim_ctx.step()
        franka.update(sim_dt)

    simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
