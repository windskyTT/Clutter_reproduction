# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Clutter grasping DirectRLEnv.

这个文件承接 DemoGrasp 的 hand/task YAML 中最关键的运行参数：
机器人资产、动作/观测维度、关节名、reset 范围、参考轨迹文件和奖励尺度。
IsaacLab 不再在环境代码里临时 load_asset，而是先把资产写成 cfg，再由
`ClutterEnv._setup_scene()` 实例化为 Articulation/RigidObject。
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


# 工程根目录和资产根目录。这样脚本无论从哪里启动，资产都指向 Clutter 仓库内部。
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_ASSET_ROOT = _PROJECT_ROOT / "assets"

# DemoGrasp 默认手型：FR3 机械臂 + Inspire tactile hand。
_ROBOT_ASSET = _ASSET_ROOT / "inspire_tac/fr3_inspire_tac_L_right_safety.urdf"
_TRACKING_REFERENCE = _ASSET_ROOT / "references/grasp_ref_inspire.pkl"
_OBJECT_LIST = _ASSET_ROOT / "union_ycb_unidex/union_ycb_debugset.yaml"
_OBJECT_NAME_LIST = _ASSET_ROOT / "union_ycb_unidex/union_ycb_debugset_names.yaml"

# 旧 hand/fr3_inspire_tac.yaml 里的默认关节位置。前 7 维是 FR3，后 12 维是 Inspire 手。
_DEFAULT_DOF_POS = {
    "fr3_joint1": 0.0,
    "fr3_joint2": 0.0,
    "fr3_joint3": 0.0,
    "fr3_joint4": -1.6,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.6,
    "fr3_joint7": 0.0,
    "right_thumb_1_joint": 0.0,
    "right_thumb_2_joint": 0.0,
    "right_thumb_3_joint": 0.0,
    "right_thumb_4_joint": 0.0,
    "right_index_1_joint": 0.0,
    "right_index_2_joint": 0.0,
    "right_middle_1_joint": 0.0,
    "right_middle_2_joint": 0.0,
    "right_ring_1_joint": 0.0,
    "right_ring_2_joint": 0.0,
    "right_little_1_joint": 0.0,
    "right_little_2_joint": 0.0,
}


@configclass
class ClutterEnvCfg(DirectRLEnvCfg):
    """IsaacLab DirectRLEnv configuration for the migrated grasp task."""

    # -----------------------------
    # DirectRLEnv 基础参数
    # -----------------------------
    num_envs = 8192
    env_spacing = 1.2
    episode_length_steps = 50
    decimation = 1
    episode_length_s = episode_length_steps / 60.0
    action_space = 13
    observation_space = 27
    state_space = 0

    # 仿真步长对应旧配置里的 60 Hz 控制节奏。
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

    # 旧 tasks/config.yaml 顶层 num_envs=8192；命令行仍可通过 --num_envs 覆盖。
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)

    # -----------------------------
    # 机器人、物体和桌面资产
    # -----------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(_ROBOT_ASSET),
            fix_base=True,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=True,
            self_collision=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=_DEFAULT_DOF_POS,
        ),
        actuators={
            # 旧 IsaacGym 里通过 dof_props 设置 PD；IsaacLab 里用 actuator cfg 表达。
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=400.0,
                damping=40.0,
                effort_limit_sim=200.0,
                velocity_limit_sim=8.0,
            ),
        },
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.35, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.10, 0.08), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.95, 0.75, 0.036),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.43, 0.36)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.10, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # -----------------------------
    # DemoGrasp 任务语义参数
    # -----------------------------
    asset_root = str(_ASSET_ROOT)
    tracking_reference_file = str(_TRACKING_REFERENCE)
    tracking_reference_lift_timestep = 13

    hand_name = "fr3_inspire_tac"
    arm_dof_names = [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]
    active_hand_dof_names = [
        "right_little_1_joint",
        "right_ring_1_joint",
        "right_middle_1_joint",
        "right_index_1_joint",
        "right_thumb_2_joint",
        "right_thumb_1_joint",
    ]
    passive_joint_mimic = {
        "right_thumb_3_joint": ("right_thumb_2_joint", 0.6),
        "right_thumb_4_joint": ("right_thumb_2_joint", 0.8),
        "right_index_2_joint": ("right_index_1_joint", 1.05),
        "right_middle_2_joint": ("right_middle_1_joint", 1.05),
        "right_ring_2_joint": ("right_ring_1_joint", 1.05),
        "right_little_2_joint": ("right_little_1_joint", 1.18),
    }
    eef_link_name = "fr3_link8"
    palm_link_name = "base_link"
    fingertip_link_names = [
        "right_thumb_4",
        "right_index_2",
        "right_middle_2",
        "right_ring_2",
        "right_little_2",
    ]
    palm_offset = (0.0, 0.02, -0.05)

    num_arm_dofs = 7
    num_active_hand_dofs = 6
    num_actions = 13
    num_observations = 27
    hand_dof_start_idx = 7
    obs_type = "armdof+handdof+eefpose+objpose"

    # 旧 grasp.yaml 里的资产列表设置。当前 clutter_env.py 先使用 cuboid 占位物体；
    # 这些路径保留下来，供后续把 Object 扩展成 YCB 多物体/RigidObjectCollection。
    multi_object = True
    object_list_file = str(_OBJECT_LIST)
    object_name_list_file = str(_OBJECT_NAME_LIST)
    use_distractor_objects = False
    num_distractor_objects = 5
    random_remove_distractor_objects = 0.5
    object_friction = 1.0

    clip_actions = 1.0
    clip_observations = 5.0
    action_smoothing = 1.0
    use_relative_control = False
    arm_controller = "qpos"
    act_max_ang_vel_arm = 1.57
    act_max_ang_vel_hand = 6.28
    delta_action_scale = (1.0,) * 13
    control_frequency_inv = 1
    limit_control_error = False
    max_pd_error_ee_pos = 0.03
    max_pd_error_hand = 0.35
    pd_param_scale = 1.0

    reset_position_range = ((0.30, 0.80), (-0.35, 0.15), (0.10, 0.12))
    reset_random_rot = "random"
    table_height_range = (0.018, 0.018)
    reset_dof_pos_random_interval = 0.2
    reset_hand_dof_pos_full_range = True
    ee_safe_workspace = ((0.15, -0.45, 0.05), (0.95, 0.25, 0.95))

    randomize_tracking_reference = False
    randomize_tracking_reference_range = (0.05, 0.05, 0.05, 1.57, 1.57, 1.57)
    randomize_grasp_pose = False
    randomize_grasp_pose_range = 1.0
    interpolation_step_scale = 1.0

    # 点云/视觉观测配置，来自旧 grasp.yaml。默认不开启点云，但维度和裁剪范围保留。
    enable_point_cloud = False
    points_per_object = 512
    pc_feature_dim = 64
    pcl_clip_workspace = ((0.25, -0.45, 0.0), (0.85, 0.30, 0.60))
    n_pcl_downsample = 2048

    # 相机和渲染配置保持为普通 dict，方便后续 Camera/TiledCamera 迁移时直接读取。
    render_cfg = {
        "enable": False,
        "appearance_realistic": True,
        "camera_ids": [1, 2],
        "data_type": "rgb",
        "instruction_template": "Grasp the object.",
        "use_advanced_instruction": False,
        "advanced_instruction_template": "Grasp the {COLOR} {OBJ}.",
        "object_name_list": str(_OBJECT_NAME_LIST),
        "save_depth_range": [0.15, 1.0],
        "pcl_clip_workspace": [[0.25, -0.45, 0.0], [0.85, 0.30, 0.60]],
        "n_pcl_downsample": 2048,
        "resize": [256, 256],
        "randomize": False,
        "randomization_params": {
            "camera_pos": [0, 0],
            "camera_quat": [0, 0],
            "depth_range": 0.05,
            "object_random_texture": True,
            "texture_folder": "textures",
            "object_color_choices": [
                "red",
                "green",
                "blue",
                "yellow",
                "cyan",
                "magenta",
                "white",
                "black",
                "gray",
                "orange",
                "purple",
                "pink",
                "brown",
                "olive",
                "teal",
                "navy",
                "maroon",
                "lime",
                "gold",
                "silver",
                "bronze",
            ],
            "color": 0.2,
            "num_lights": 3,
            "light_intensity": [0.1, 0.8],
            "light_ambient": [0.1, 0.8],
            "table_xyz": [0, 0, 0],
        },
    }
    camera_cfg = {
        "camera_1": {
            "type": "D435",
            "mount": "fixed",
            "width": 640,
            "height": 480,
            "depth_range": [0.15, 3.0],
            "intrinsics": [[608.10601807, 0.0, 323.16036987], [0.0, 607.04846191, 245.16740417], [0.0, 0.0, 1.0]],
            "extrinsics": [
                [-0.95458387, -0.22931432, 0.19022242, 0.25351984],
                [-0.29675754, 0.78866419, -0.53846426, 0.47084633],
                [-0.02654405, -0.57045923, -0.82089687, 0.62032344],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "camera_2": {
            "type": "D435",
            "mount": "fixed",
            "width": 640,
            "height": 480,
            "depth_range": [0.15, 3.0],
            "intrinsics": [[610.1685791, 0.0, 327.0413208], [0.0, 609.07666016, 245.29025269], [0.0, 0.0, 1.0]],
            "extrinsics": [
                [0.6464361, 0.58084152, -0.49471557, 0.89060788],
                [0.76291282, -0.4842839, 0.42829095, -0.44560813],
                [0.00918638, -0.65428758, -0.75619004, 0.49017396],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }

    # 旧 IsaacGym randomization_params 暂存为配置字典。IsaacLab 的 EventManager
    # 需要单独写 event terms，后续迁移域随机化时从这里取参数。
    randomize = False
    domain_randomization_cfg = {
        "frequency": 720,
        "observations": {
            "range": [0.0, 0.002],
            "range_correlated": [0.0, 0.001],
            "operation": "additive",
            "distribution": "gaussian",
        },
        "actions": {
            "range": [0.0, 0.001],
            "range_correlated": [0.0, 0.015],
            "operation": "additive",
            "distribution": "gaussian",
        },
        "sim_params": {
            "gravity": {
                "range": [0.0, 0.4],
                "operation": "additive",
                "distribution": "gaussian",
            }
        },
        "actor_params": {
            "hand": {
                "color": True,
                "dof_properties": {
                    "damping": {"range": [0.3, 3.0], "operation": "scaling", "distribution": "loguniform"},
                    "stiffness": {"range": [0.75, 1.5], "operation": "scaling", "distribution": "loguniform"},
                    "lower": {"range": [0.0, 0.01], "operation": "additive", "distribution": "gaussian"},
                    "upper": {"range": [0.0, 0.01], "operation": "additive", "distribution": "gaussian"},
                },
                "rigid_body_properties": {
                    "mass": {"range": [0.5, 1.5], "operation": "scaling", "distribution": "uniform", "setup_only": True}
                },
                "rigid_shape_properties": {
                    "friction": {
                        "num_buckets": 250,
                        "range": [0.7, 1.3],
                        "operation": "scaling",
                        "distribution": "uniform",
                    }
                },
            },
            "object": {
                "scale": {"range": [0.95, 1.05], "operation": "scaling", "distribution": "uniform", "setup_only": True},
                "rigid_body_properties": {
                    "mass": {"range": [0.5, 1.5], "operation": "scaling", "distribution": "uniform", "setup_only": True}
                },
                "rigid_shape_properties": {
                    "friction": {
                        "num_buckets": 250,
                        "range": [0.7, 1.3],
                        "operation": "scaling",
                        "distribution": "uniform",
                    }
                },
            },
        },
    }

    reward_type = "binary"
    lift_success_height = 0.08
    fall_height = -0.05
    reach_reward_scale = 2.0
    lift_reward_scale = 8.0
    action_penalty_scale = 0.01
