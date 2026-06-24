# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab DirectRLEnv implementation of the migrated DemoGrasp task.

旧 DemoGrasp 的 `tasks/grasp.py` 继承 IsaacGym `VecTask`，自己负责创建 sim、
创建 actor、刷新 Gym tensor、执行 simulate/fetch_results。IsaacLab 的直接式环境
把这些底层步骤收进 `DirectRLEnv.step()`，所以本文件只保留任务语义：
场景创建、动作转关节目标、观测拼接、奖励、终止条件、reset 和专家参考动作。
"""

from __future__ import annotations

import math
import pickle
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.utils import bind_physics_material, get_all_matching_child_prims
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_from_euler_xyz, quat_mul, sample_uniform
from pxr import UsdPhysics

from .clutter_env_cfg import ClutterEnvCfg
from .reward import REWARD_DICT
from .utils import batch_linear_interpolate_poses, load_object_point_clouds


class ClutterEnv(DirectRLEnv):
    """FR3 + Inspire grasping task migrated to IsaacLab.

    外部 PPO 只需要调用标准 Gymnasium/IsaacLab 接口 `reset()` 和 `step()`。
    兼容旧 DemoGrasp runner 的少量方法，比如 `reset_idx()`、`compute_reference_actions()`
    和 `compute_real_observation_dict()` 也保留在这里。
    """

    cfg: ClutterEnvCfg

    def __init__(self, cfg: ClutterEnvCfg, render_mode: str | None = None, **kwargs):
        # 父类会创建 SimulationContext、InteractiveScene，并调用本类的 `_setup_scene()`。
        super().__init__(cfg, render_mode, **kwargs)

        # DirectRLEnv 初始化完成后，资产张量句柄已经可用，此时解析关节/body 索引最稳。
        self.init_configs()
        self._init_robot_metadata()
        self._load_tracking_references()
        self._init_runtime_buffers()

    @property
    def max_episode_length(self) -> int:
        """Use the DemoGrasp control-step count directly.

        IsaacLab's base property derives this value from seconds with
        ``ceil(episode_length_s / step_dt)``.  For DemoGrasp's 50 control-step
        clips that can become 51 because of floating-point roundoff, which
        shifts the replay/reset timing and makes the play script end on a reset
        frame instead of the visible grasp frame.
        """

        return int(self.cfg.episode_length_steps)

    def init_configs(self):
        """整理任务运行时常用配置。

        DemoGrasp 旧版在这里解析 Hydra/YAML 字典；IsaacLab 迁移后配置已经在
        `ClutterEnvCfg` 中展开，所以这里主要建立旧字段别名，便于旧 PPO/脚本复用。
        """

        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_obs = self.num_observations
        self.num_acts = self.num_actions
        self.hand_dof_start_idx = self.cfg.hand_dof_start_idx
        self.clip_actions = self.cfg.clip_actions
        self.clip_obs = self.cfg.clip_observations
        self.randomize_tracking_reference = self.cfg.randomize_tracking_reference
        self.randomize_grasp_pose = self.cfg.randomize_grasp_pose
        self.render_cfg = self.cfg.render_cfg
        self.camera_cfg = self.cfg.camera_cfg
        self.use_camera = bool(self.render_cfg.get("enable", False))
        self.enable_pcl = self.cfg.enable_point_cloud
        self.points_per_object = self.cfg.points_per_object
        try:
            # 根据配置选择奖励函数；当前迁移旧 DemoGrasp 的 binary grasp reward。
            self.reward_function = REWARD_DICT[self.cfg.reward_type]
        except KeyError as exc:
            available_rewards = ", ".join(sorted(REWARD_DICT))
            raise ValueError(
                f"Unsupported reward_type={self.cfg.reward_type!r}. Available rewards: {available_rewards}"
            ) from exc
        self.progress_buf = self.episode_length_buf
        self.instructions = ["Grasp the object."] * self.num_envs

    def _setup_scene(self):
        """Create IsaacLab scene assets.

        对应旧代码里的 `create_sim()`、`_create_ground_plane()` 和 `_create_envs()`。
        这里不再调用 `gym.load_asset/create_actor`，而是实例化 cfg 中声明好的资产。
        """

        self._spawn_visual_grid_ground()
        self._scene_object_files = self._load_object_file_list()
        self._scene_env_object_files = self._select_object_files_for_envs(self._scene_object_files)
        self._spawn_multi_object_assets(self._scene_env_object_files)

        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        # 地面只做 GUI 视觉参照，不加入 scene.rigid_objects，也不参与碰撞过滤。

        # 同质场景可复制 env_0；多物体复刻关闭 replicate_physics 后，InteractiveScene
        # 已经提前创建了所有 env prim，Object 也在上面逐个生成，因此这里不再 clone。
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 注册到 scene 后，DirectRLEnv 才会自动写入/更新这些资产的数据。
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _init_robot_metadata(self):
        """Resolve joint/body names from the loaded articulation.

        旧 IsaacGym 代码通过 `gym.find_asset_dof_index` 和
        `gym.find_actor_rigid_body_index` 做这件事；IsaacLab 使用 asset 的
        `find_joints/find_bodies`。
        """

        self.num_robot_dofs = self.robot.num_joints
        self.robot_dof_names = list(self.robot.joint_names)

        self.arm_dof_indices = self._find_joint_ids(self.cfg.arm_dof_names)
        self.active_hand_dof_indices = self._find_joint_ids(self.cfg.active_hand_dof_names)

        passive_ids: list[int] = []
        parent_ids: list[int] = []
        multipliers: list[float] = []
        for passive_name, (parent_name, multiplier) in self.cfg.passive_joint_mimic.items():
            passive_id = self._find_joint_ids([passive_name], required=False)
            parent_id = self._find_joint_ids([parent_name], required=False)
            if passive_id and parent_id:
                passive_ids.append(passive_id[0])
                parent_ids.append(parent_id[0])
                multipliers.append(float(multiplier))

        self.passive_hand_dof_indices = torch.tensor(passive_ids, dtype=torch.long, device=self.device)
        self.mimic_parent_dof_indices = torch.tensor(parent_ids, dtype=torch.long, device=self.device)
        self.mimic_multipliers = torch.tensor(multipliers, dtype=torch.float, device=self.device)
        self.have_passive_joints = len(passive_ids) > 0

        hand_ids = list(dict.fromkeys(self.active_hand_dof_indices.tolist() + passive_ids))
        self.hand_dof_indices = torch.tensor(hand_ids, dtype=torch.long, device=self.device)
        active_robot_ids = self.arm_dof_indices.tolist() + self.active_hand_dof_indices.tolist()
        self.active_robot_dof_indices = torch.tensor(active_robot_ids, dtype=torch.long, device=self.device)
        self.robot_dof_indices = torch.arange(self.num_robot_dofs, dtype=torch.long, device=self.device)

        self.num_arm_dofs = len(self.arm_dof_indices)
        self.num_active_hand_dofs = len(self.active_hand_dof_indices)
        self.num_hand_dofs = len(self.hand_dof_indices)

        self.eef_body_id = self._find_body_id(self.cfg.eef_link_name)
        # IsaacLab/PhysX Jacobian rows omit the fixed base link.  The robot is
        # spawned with fix_base=True, so body state indices and Jacobian body
        # indices are offset by one.
        self.eef_jacobian_body_id = max(self.eef_body_id - 1, 0)
        self.palm_body_id = self._find_body_id(self.cfg.palm_link_name)
        self.fingertip_body_ids = self._find_body_ids(self.cfg.fingertip_link_names)
        self.num_fingers = len(self.fingertip_body_ids)

        joint_limits = self.robot.data.soft_joint_pos_limits[0].clone()
        self.robot_dof_lower_limits = joint_limits[:, 0]
        self.robot_dof_upper_limits = joint_limits[:, 1]
        self._sanitize_joint_limits()
        self.robot_dof_default_pos = self.robot.data.default_joint_pos[0].clone()

        self.palm_offset = torch.tensor(self.cfg.palm_offset, dtype=torch.float, device=self.device).view(1, 3)
        self.ee_safe_workspace = torch.tensor(self.cfg.ee_safe_workspace, dtype=torch.float, device=self.device)
        self.reset_position_range = torch.tensor(self.cfg.reset_position_range, dtype=torch.float, device=self.device)
        self.table_height_range = torch.tensor(self.cfg.table_height_range, dtype=torch.float, device=self.device)
        self.delta_action_scale = torch.tensor(self.cfg.delta_action_scale, dtype=torch.float, device=self.device)

    def _init_runtime_buffers(self):
        """Allocate tensors that change while the RL task runs."""

        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self.no_op_action = unscale(
            self.robot.data.default_joint_pos[:, self.active_robot_dof_indices],
            self.robot_dof_lower_limits[self.active_robot_dof_indices],
            self.robot_dof_upper_limits[self.active_robot_dof_indices],
        )

        self.prev_targets = self.robot.data.default_joint_pos.clone()
        self.cur_targets = self.robot.data.default_joint_pos.clone()
        self.target_joint_pos = self.robot.data.default_joint_pos.clone()

        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.has_hit_table = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.object_init_states = torch.zeros((self.num_envs, 13), dtype=torch.float, device=self.device)
        self.object_init_states[:, 3] = 1.0
        self.table_heights = torch.full(
            (self.num_envs,), float(self.cfg.table_height_range[0]), dtype=torch.float, device=self.device
        )

        self.reaching_plan_ee = torch.zeros(
            (self.num_envs, self.max_episode_length, 7), dtype=torch.float, device=self.device
        )
        self.reaching_plan_ee[..., 3] = 1.0
        self.reaching_plan_timesteps = torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        self.obs_tensor = torch.zeros((self.num_envs, self.num_observations), dtype=torch.float, device=self.device)
        self.obs_dict = {"policy": self.obs_tensor}

        pcl_points = self.points_per_object if self.enable_pcl else 0
        self.object_files = getattr(self, "_scene_object_files", None)
        if self.object_files is None:
            self.object_files = self._load_object_file_list()
        self.env_object_files = getattr(self, "_scene_env_object_files", None)
        if self.env_object_files is None:
            self.env_object_files = self._select_object_files_for_envs(self.object_files)
        self.object_pcl_buf = torch.zeros((self.num_envs, pcl_points, 3), dtype=torch.float, device=self.device)
        if self.enable_pcl:
            self._load_object_point_cloud_buffer()

    def _load_tracking_references(self):
        """Load DemoGrasp wrist/hand tracking reference.

        pkl 来自旧项目的 `grasp_ref_inspire.pkl`。旧 IsaacGym 数据中的四元数通常是
        xyzw，IsaacLab 数学工具使用 wxyz，因此加载时统一转换。
        """

        try:
            with open(self.cfg.tracking_reference_file, "rb") as f:
                reference = pickle.load(f)
            wrist_pos = torch.as_tensor(reference["wrist_initobj_pos"], dtype=torch.float, device=self.device)
            wrist_quat = torch.as_tensor(reference["wrist_quat"], dtype=torch.float, device=self.device)
            hand_qpos = torch.as_tensor(reference["hand_qpos"], dtype=torch.float, device=self.device)
        except FileNotFoundError:
            wrist_pos = torch.zeros((1, 3), dtype=torch.float, device=self.device)
            wrist_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float, device=self.device)
            hand_qpos = self.robot.data.default_joint_pos[0, self.active_hand_dof_indices].view(1, -1)

        self.tracking_reference = {
            "wrist_initobj_pos": wrist_pos,
            "wrist_quat": xyzw_to_wxyz(wrist_quat),
            "hand_qpos": hand_qpos,
        }
        self.T_ref = int(wrist_pos.shape[0])
        self.T_ref_start_lifting = min(self.cfg.tracking_reference_lift_timestep, max(self.T_ref - 1, 0))
        self._refresh_current_tracking_reference()

    def _refresh_current_tracking_reference(self):
        """Expand one reference trajectory to all parallel environments."""

        self.current_tracking_reference = {
            key: value.unsqueeze(0).repeat(self.num_envs, 1, 1) for key, value in self.tracking_reference.items()
        }

    def _spawn_visual_grid_ground(self) -> None:
        """Spawn a black grid ground for the GUI without adding physics contacts.

        IsaacLab 的 `GroundPlaneCfg` 使用 Grid/default_environment.usd，并通过
        `color=(0, 0, 0)` 得到黑色网格地面。这个 USD 通常来自 Nucleus/S3；
        为了让本地离线运行稳定，这里用本地 Cuboid 几何生成等价的黑底网格。
        """

        if not self.cfg.enable_grid_ground:
            return

        ground_root = self.cfg.grid_ground_prim_path
        stage = sim_utils.get_current_stage()
        if stage.GetPrimAtPath(ground_root).IsValid():
            return

        num_envs = max(int(self.cfg.scene.num_envs), 1)
        env_spacing = max(float(self.cfg.scene.env_spacing), 1.0e-3)
        env_grid_width = math.ceil(math.sqrt(num_envs))
        # 覆盖所有 env origin，并在四周留一点余量，避免 GUI 中边缘环境落到网格之外。
        visual_size = max(float(self.cfg.grid_ground_size), (env_grid_width + 2) * env_spacing)
        spacing = max(float(self.cfg.grid_ground_spacing), 1.0e-3)
        half_steps = max(1, math.ceil(visual_size / (2.0 * spacing)))
        visual_size = 2.0 * half_steps * spacing

        line_width = max(float(self.cfg.grid_ground_line_width), 1.0e-4)
        line_height = max(float(self.cfg.grid_ground_line_height), 1.0e-4)
        z = float(self.cfg.grid_ground_z)

        sim_utils.create_prim(ground_root, "Xform")
        # 黑色底板提供暗背景；不设置 collision_props/rigid_props，避免改变抓取物理。
        base_cfg = sim_utils.CuboidCfg(
            size=(visual_size, visual_size, line_height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.grid_ground_base_color, roughness=0.8),
        )
        base_cfg.func(f"{ground_root}/Base", base_cfg, translation=(0.0, 0.0, z - line_height))

        normal_line_cfg = sim_utils.CuboidCfg(
            size=(visual_size, line_width, line_height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.grid_ground_line_color, roughness=0.8),
        )
        axis_line_cfg = sim_utils.CuboidCfg(
            size=(visual_size, line_width * 1.5, line_height),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.grid_ground_axis_line_color, roughness=0.8),
        )

        for index in range(-half_steps, half_steps + 1):
            coord = index * spacing
            x_line_cfg = axis_line_cfg if index == 0 else normal_line_cfg
            y_line_cfg = axis_line_cfg if index == 0 else normal_line_cfg

            # 与 X 轴平行的线：长度沿 x 方向，位置沿 y 方向平移。
            x_line_cfg.func(
                f"{ground_root}/Lines/X_{index + half_steps:04d}",
                x_line_cfg,
                translation=(0.0, coord, z),
            )
            # 与 Y 轴平行的线：复用 cuboid，但绕 z 轴旋转 90 度。
            y_line_cfg.func(
                f"{ground_root}/Lines/Y_{index + half_steps:04d}",
                y_line_cfg,
                translation=(coord, 0.0, z),
                orientation=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            )

    def _load_object_file_list(self) -> list[str]:
        """Load object URDF list using the old DemoGrasp asset-list convention.

        DemoGrasp 的列表项形如 `002_master_chef_can.urdf`，真实文件在
        `union_ycb_unidex/urdf/002_master_chef_can.urdf`，对应点云在
        `union_ycb_unidex/pointclouds/002_master_chef_can.npy`。
        """

        object_list_path = Path(self.cfg.object_list_file)
        if not object_list_path.is_file():
            raise FileNotFoundError(f"Object list file not found: {object_list_path}")

        with object_list_path.open("r", encoding="utf-8") as f:
            object_names = [line.strip()[2:].strip() for line in f if line.strip().startswith("- ")]

        if not object_names:
            raise RuntimeError(f"Object list file is empty or has no '- ' entries: {object_list_path}")

        object_files = [f"union_ycb_unidex/urdf/{name}" for name in sorted(object_names)]
        print(f"[DEBUG] Loaded {len(object_files)} object URDF entries from {object_list_path}", flush=True)
        print(f"[DEBUG] First object file: {object_files[0]}", flush=True)
        return object_files

    def _select_object_files_for_envs(self, object_files: list[str]) -> list[str]:
        """Return the DemoGrasp env-to-object assignment.

        旧 DemoGrasp 在 `_create_envs()` 中用 `object_assets[i % len(object_assets)]`，
        所以 `num_envs=175` 时每个 debugset 物体正好出现一次。
        """

        if not object_files:
            return []
        return [object_files[i % len(object_files)] for i in range(self.num_envs)]

    def _spawn_multi_object_assets(self, env_object_files: list[str]) -> None:
        """Spawn one true UnionYCB object mesh under each environment.

        IsaacLab 的 URDF importer 无法直接导入这些 URDF：mesh 文件名如
        `002_master_chef_can.stl` 会产生非法 USD prim path。这里保留 URDF 作为
        DemoGrasp 的资产列表来源，解析其中真实 mesh，再通过 MeshConverter 生成
        带 RigidBodyAPI 的 USD 刚体。
        """

        if not env_object_files:
            return

        for env_id, object_file in enumerate(env_object_files):
            usd_path = self._ensure_object_mesh_usd(object_file)
            prim_path = f"/World/envs/env_{env_id}/Object"
            if env_id < 3:
                print(f"[DEBUG] Spawning object env={env_id}: {object_file} -> {usd_path} at {prim_path}", flush=True)
            spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    disable_gravity=False,
                    angular_damping=float(self.cfg.object_angular_damping),
                    max_depenetration_velocity=float(self.cfg.object_max_depenetration_velocity),
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=float(self.cfg.object_mass)),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=float(self.cfg.object_contact_offset),
                    rest_offset=float(self.cfg.object_rest_offset),
                ),
            )
            spawn_cfg.func(
                prim_path,
                spawn_cfg,
                translation=self.cfg.object_cfg.init_state.pos,
                orientation=self.cfg.object_cfg.init_state.rot,
            )
            self._bind_object_physics_material(prim_path)

    def _bind_object_physics_material(self, prim_path: str) -> None:
        """Apply DemoGrasp's object rigid-shape friction to spawned USD colliders."""

        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.cfg.object_friction),
            dynamic_friction=float(self.cfg.object_friction),
            restitution=0.0,
        )
        material_path = f"{prim_path}/physicsMaterial"
        material_cfg.func(material_path, material_cfg)

        collision_prims = get_all_matching_child_prims(
            prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI),
        )
        if not collision_prims:
            bind_physics_material(prim_path, material_path)
            return

        for prim in collision_prims:
            bind_physics_material(str(prim.GetPath()), material_path)

    def _ensure_object_mesh_usd(self, object_file: str) -> str:
        """Convert one DemoGrasp object URDF's mesh to a cached IsaacLab USD file."""

        urdf_path = Path(self.cfg.asset_root) / object_file
        mesh_path, mesh_scale = self._read_urdf_mesh(urdf_path)
        usd_dir = Path(self.cfg.object_usd_cache_dir)
        usd_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", urdf_path.stem)
        # v2 includes the old DemoGrasp object friction/contact settings in the cached USD.
        usd_name = f"obj_{safe_name}_rigid_v2.usd"
        converter_cfg = MeshConverterCfg(
            asset_path=str(mesh_path),
            usd_dir=str(usd_dir),
            usd_file_name=usd_name,
            force_usd_conversion=False,
            make_instanceable=False,
            scale=mesh_scale,
            mass_props=sim_utils.MassPropertiesCfg(mass=float(self.cfg.object_mass)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                angular_damping=float(self.cfg.object_angular_damping),
                max_depenetration_velocity=float(self.cfg.object_max_depenetration_velocity),
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=float(self.cfg.object_contact_offset),
                rest_offset=float(self.cfg.object_rest_offset),
            ),
            mesh_collision_props=sim_utils.ConvexHullPropertiesCfg(),
        )
        return MeshConverter(converter_cfg).usd_path

    @staticmethod
    def _read_urdf_mesh(urdf_path: Path) -> tuple[Path, tuple[float, float, float]]:
        """Read the first mesh path and scale from a single-link object URDF."""

        tree = ET.parse(urdf_path)
        mesh = tree.getroot().find(".//mesh")
        if mesh is None or "filename" not in mesh.attrib:
            raise RuntimeError(f"Object URDF has no mesh filename: {urdf_path}")

        filename = mesh.attrib["filename"]
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (urdf_path.parent / mesh_path).resolve()

        scale_text = mesh.attrib.get("scale", "1.0 1.0 1.0")
        scale = tuple(float(x) for x in scale_text.split())
        if len(scale) != 3:
            raise RuntimeError(f"Invalid mesh scale in {urdf_path}: {scale_text!r}")
        return mesh_path, scale

    def _load_object_point_cloud_buffer(self):
        """Fill `object_pcl_buf` with the same object/env cycling rule as DemoGrasp."""

        if not self.env_object_files:
            return

        point_clouds = load_object_point_clouds(self.env_object_files, self.cfg.asset_root)
        for env_id in range(self.num_envs):
            pcl_np = np.asarray(point_clouds[env_id], dtype=np.float32)
            if pcl_np.shape[0] != self.points_per_object:
                # 资产理论上已经是 512 点；这里保留一个确定性裁剪/补零兜底，避免坏文件直接打断播放。
                fixed = np.zeros((self.points_per_object, 3), dtype=np.float32)
                count = min(self.points_per_object, pcl_np.shape[0])
                fixed[:count] = pcl_np[:count, :3]
                pcl_np = fixed
            self.object_pcl_buf[env_id] = torch.as_tensor(pcl_np[:, :3], dtype=torch.float, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert policy actions to full robot joint targets.

        输入动作是 13 维：前 7 维控制 FR3 机械臂，后 6 维控制 Inspire 主动手指关节。
        qpos 模式下，动作先从 [-1, 1] 映射到关节上下限，再按最大角速度限制每步变化。
        """

        self.actions = self._normalize_action_shape(actions).clamp(-self.clip_actions, self.clip_actions)
        self.target_joint_pos = self.cur_targets.clone()

        if self.cfg.use_relative_control:
            hand_delta = self.actions[:, self.hand_dof_start_idx :] * self.delta_action_scale[self.hand_dof_start_idx :]
            self.target_joint_pos[:, self.active_hand_dof_indices] = (
                self.prev_targets[:, self.active_hand_dof_indices] + hand_delta
            )
        else:
            self.target_joint_pos[:, self.active_hand_dof_indices] = scale(
                self.actions[:, self.hand_dof_start_idx :],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )

        if self.cfg.arm_controller == "qpos":
            self.target_joint_pos[:, self.arm_dof_indices] = scale(
                self.actions[:, : self.num_arm_dofs],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
        elif "pose" in self.cfg.arm_controller:
            if self.cfg.arm_controller == "pose":
                arm_delta = self.compute_arm_ik(self.actions[:, :7], is_delta_pose=False)
            else:
                arm_action = self.actions[:, :6] * self.delta_action_scale[:6]
                arm_delta = self.compute_arm_ik(
                    arm_action,
                    is_delta_pose=True,
                    is_delta_pose_in_world=("world" in self.cfg.arm_controller),
                )
            self.target_joint_pos[:, self.arm_dof_indices] = self.robot.data.joint_pos[:, self.arm_dof_indices] + arm_delta
        else:
            raise ValueError(f"Unsupported arm_controller: {self.cfg.arm_controller}")

        self._apply_mimic_hand_joints()
        self._limit_target_velocity_and_position()

        alpha = float(self.cfg.action_smoothing)
        self.cur_targets = alpha * self.target_joint_pos + (1.0 - alpha) * self.prev_targets

    def _apply_action(self) -> None:
        """Write joint position targets to the IsaacLab articulation."""

        self.robot.set_joint_position_target(self.cur_targets)
        self.prev_targets.copy_(self.cur_targets)

    def _get_observations(self) -> dict:
        """Return policy observations in IsaacLab's expected dictionary format."""

        obs = self.compute_observations()
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        self.obs_tensor = obs
        self.obs_dict = {"policy": obs}
        return self.obs_dict

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward for the current post-physics state."""

        return self.compute_reward()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and timeout flags."""

        self._compute_intermediate_values()
        # DirectRLEnv increments episode_length_buf before calling _get_dones().
        # Timeout at max_episode_length keeps the DemoGrasp t=max-2 success frame
        # visible instead of immediately auto-resetting to the initial pose.
        time_out = self.episode_length_buf >= self.max_episode_length

        object_dropped = self.object_pos[:, 2] < self.table_heights + self.cfg.fall_height
        workspace_low = self.ee_safe_workspace[0]
        workspace_high = self.ee_safe_workspace[1]
        eef_outside = torch.any((self.eef_pos < workspace_low) | (self.eef_pos > workspace_high), dim=-1)

        if self.num_fingers > 0:
            finger_min_z = torch.amin(self.fingertip_pos[..., 2], dim=-1)
            self.has_hit_table |= finger_min_z < self.table_heights + 0.003

        terminated = object_dropped | eef_outside
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Reset selected environments.

        替代旧 DemoGrasp 的 `reset_idx()` 中对 root tensor/dof tensor 的直接索引写入。
        IsaacLab 里分别调用 articulation/rigid object 的 write 方法。
        """

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        pos_noise = sample_uniform(
            -self.cfg.reset_dof_pos_random_interval,
            self.cfg.reset_dof_pos_random_interval,
            joint_pos.shape,
            self.device,
        )
        joint_pos += pos_noise

        if self.cfg.reset_hand_dof_pos_full_range and len(self.hand_dof_indices) > 0:
            hand_lower = self.robot_dof_lower_limits[self.hand_dof_indices]
            hand_upper = self.robot_dof_upper_limits[self.hand_dof_indices]
            hand_action = sample_uniform(-1.0, 1.0, (num_resets, len(self.hand_dof_indices)), self.device)
            joint_pos[:, self.hand_dof_indices] = scale(hand_action, hand_lower, hand_upper)

        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel.zero_()

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        robot_root_state[:, :3] += env_origins
        self.robot.write_root_pose_to_sim(robot_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(robot_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        self.prev_targets[env_ids] = joint_pos
        self.cur_targets[env_ids] = joint_pos
        self.target_joint_pos[env_ids] = joint_pos

        table_height = sample_uniform(
            float(self.table_height_range[0]),
            float(self.table_height_range[1]),
            (num_resets,),
            self.device,
        )
        table_state = self.table.data.default_root_state[env_ids].clone()
        table_state[:, 0] = env_origins[:, 0] + float(self.cfg.table_cfg.init_state.pos[0])
        table_state[:, 1] = env_origins[:, 1] + float(self.cfg.table_cfg.init_state.pos[1])
        table_state[:, 2] = env_origins[:, 2] + table_height - 0.018
        table_state[:, 3:7] = identity_quat(num_resets, self.device)
        # 桌面是 kinematic body，只需要写位姿；给 kinematic 刚体写速度会触发
        # PhysX 的 "Body must be non-kinematic" 报错。
        self.table.write_root_pose_to_sim(table_state[:, :7], env_ids)
        self.table_heights[env_ids] = table_height

        object_state = self.object.data.default_root_state[env_ids].clone()
        object_state[:, 0] = env_origins[:, 0] + sample_uniform(
            float(self.reset_position_range[0, 0]), float(self.reset_position_range[0, 1]), (num_resets,), self.device
        )
        object_state[:, 1] = env_origins[:, 1] + sample_uniform(
            float(self.reset_position_range[1, 0]), float(self.reset_position_range[1, 1]), (num_resets,), self.device
        )
        object_state[:, 2] = env_origins[:, 2] + sample_uniform(
            float(self.reset_position_range[2, 0]), float(self.reset_position_range[2, 1]), (num_resets,), self.device
        )
        object_state[:, 3:7] = self._sample_object_quat(num_resets)
        object_state[:, 7:].zero_()
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)

        self.object_init_states[env_ids] = object_state
        self.object_init_states[env_ids, :3] -= env_origins
        self.actions[env_ids].zero_()
        self.rew_buf[env_ids].zero_()
        self.successes[env_ids].zero_()
        self.current_successes[env_ids].zero_()
        self.has_hit_table[env_ids] = False

        self._refresh_current_tracking_reference()
        self.generate_reaching_plan_idx(env_ids)

    def reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Compatibility wrapper for old DemoGrasp scripts."""

        self._reset_idx(env_ids)
        return self._get_observations()

    def pre_physics_step(self, actions: torch.Tensor):
        """Compatibility wrapper for the old method name."""

        self._pre_physics_step(actions)

    def post_physics_step(self):
        """Compatibility wrapper; DirectRLEnv normally handles this internally."""

        self.compute_observations()
        self.compute_reward()

    def generate_reaching_plan_idx(self, env_ids: Sequence[int] | torch.Tensor, actions: torch.Tensor | None = None):
        """Generate a smooth wrist plan from current EE pose to reference start.

        旧 DemoGrasp 的 one-step PPO 输出不是底层关节动作，而是 12 维 planner action：
        前 6 维随机化 wrist 参考轨迹，后 6 维随机化 Inspire 主动手指抓取姿态。
        如果这里忽略 `actions`，加载 `inspire.pt` 后会退化成固定参考轨迹，表现为
        “模型加载了但机械臂抓不到”。
        """

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return

        self._compute_intermediate_values()

        def get_random_value(interval: slice) -> torch.Tensor:
            """Use policy planner actions when provided; otherwise sample random values."""
            width = int(interval.stop - interval.start)
            if actions is None:
                return sample_uniform(-1.0, 1.0, (len(env_ids), width), self.device)
            action_tensor = actions.to(self.device)
            if action_tensor.shape[-1] < int(interval.stop):
                raise RuntimeError(f"Planner action shape {action_tensor.shape} is too small for slice {interval}.")
            return action_tensor[env_ids, interval]

        self._refresh_current_tracking_reference()
        if self.cfg.randomize_tracking_reference:
            ref_range = torch.tensor(self.cfg.randomize_tracking_reference_range, dtype=torch.float, device=self.device)

            rand_rpy = get_random_value(slice(3, 6)) * ref_range[3:6]
            rand_quat = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2]).unsqueeze(1)
            rand_quat = rand_quat.expand(-1, self.T_ref, -1)

            self.current_tracking_reference["wrist_quat"][env_ids] = quat_mul(
                rand_quat,
                self.current_tracking_reference["wrist_quat"][env_ids],
            )
            self.current_tracking_reference["wrist_initobj_pos"][env_ids] = quat_apply(
                rand_quat,
                self.current_tracking_reference["wrist_initobj_pos"][env_ids],
            )

            rand_xyz = get_random_value(slice(0, 3))
            self.current_tracking_reference["wrist_initobj_pos"][env_ids] += (rand_xyz * ref_range[0:3]).unsqueeze(1)

            # 保持 lift 阶段相对轨迹与 demo 一致，只平移到新的 pre-lift 末端位置。
            lift_t = self.T_ref_start_lifting
            if lift_t > 0:
                base_lift = (
                    self.tracking_reference["wrist_initobj_pos"][lift_t:]
                    - self.tracking_reference["wrist_initobj_pos"][lift_t - 1 : lift_t]
                )
                self.current_tracking_reference["wrist_initobj_pos"][env_ids, lift_t:] = (
                    base_lift.unsqueeze(0)
                    + self.current_tracking_reference["wrist_initobj_pos"][env_ids, lift_t - 1 : lift_t]
                )

        if self.cfg.randomize_grasp_pose:
            rand_hand = get_random_value(slice(6, 6 + self.num_active_hand_dofs))
            lift_t = self.T_ref_start_lifting
            ref_hand = self.current_tracking_reference["hand_qpos"][env_ids]
            grasp_pose = ref_hand[:, lift_t - 1] + rand_hand * float(self.cfg.randomize_grasp_pose_range)
            grasp_pose = torch.clamp(
                grasp_pose,
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )

            if lift_t > 1:
                hand_t0 = ref_hand[:, 0].unsqueeze(1).repeat(1, lift_t - 1, 1)
                denom = ref_hand[:, lift_t - 1] - ref_hand[:, 0] + 1.0e-6
                fraction = (grasp_pose - ref_hand[:, 0]) / denom
                ref_hand[:, : lift_t - 1] = hand_t0 + (ref_hand[:, : lift_t - 1] - hand_t0) * fraction.unsqueeze(1)
            ref_hand[:, lift_t - 1 :] = grasp_pose.unsqueeze(1).repeat(1, self.T_ref - lift_t + 1, 1)
            self.current_tracking_reference["hand_qpos"][env_ids] = torch.clamp(
                ref_hand,
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )

        start_pose = self.eef_pose[env_ids]
        target_pose = torch.cat(
            (
                self.current_tracking_reference["wrist_initobj_pos"][env_ids, 0]
                + self.object_init_states[env_ids, 0:3],
                self.current_tracking_reference["wrist_quat"][env_ids, 0],
            ),
            dim=-1,
        )

        full_plan, plan_steps = batch_linear_interpolate_poses(
            start_pose,
            target_pose,
            max_trans_step=0.04 * float(self.cfg.interpolation_step_scale),
            max_rot_step=0.1 * float(self.cfg.interpolation_step_scale),
        )
        # DemoGrasp 去掉起点，让第 0 个控制步就开始向目标移动。
        full_plan = full_plan[:, 1 : min(self.max_episode_length, full_plan.shape[1])]
        self.reaching_plan_ee[env_ids].zero_()
        self.reaching_plan_ee[env_ids, :, 3] = 1.0
        if full_plan.shape[1] > 0:
            self.reaching_plan_ee[env_ids, : full_plan.shape[1]] = full_plan
        self.reaching_plan_timesteps[env_ids] = (plan_steps - 1).clamp(min=1, max=self.max_episode_length - 1)

    def compute_reference_actions(self) -> torch.Tensor:
        """Return a DemoGrasp-style expert/reference action.

        reaching 阶段跟踪插值得到的 wrist pose；到达后跟踪 pkl 中的 wrist/hand 参考。
        qpos 控制器会用 Jacobian IK 把 wrist pose 转成机械臂关节目标，再统一 unscale
        成 [-1, 1] 动作。
        """

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        reach_t = torch.minimum(self.episode_length_buf, self.reaching_plan_timesteps)
        track_t = (self.episode_length_buf - self.reaching_plan_timesteps).clamp(min=0, max=self.T_ref - 1)

        wrist_reach = self.reaching_plan_ee[env_ids, reach_t]
        wrist_track = torch.cat(
            (
                self.current_tracking_reference["wrist_initobj_pos"][env_ids, track_t]
                + self.object_init_states[:, 0:3],
                self.current_tracking_reference["wrist_quat"][env_ids, track_t],
            ),
            dim=-1,
        )
        in_reaching = (self.episode_length_buf < self.reaching_plan_timesteps).unsqueeze(-1)
        wrist_target = torch.where(in_reaching, wrist_reach, wrist_track)
        hand_qpos_target = self.current_tracking_reference["hand_qpos"][env_ids, track_t]

        if self.cfg.arm_controller == "qpos" and not self.cfg.use_relative_control:
            arm_delta = self.compute_arm_ik(wrist_target, is_delta_pose=False)
            arm_qpos_target = self.robot.data.joint_pos[:, self.arm_dof_indices] + arm_delta
            qpos_target = torch.cat((arm_qpos_target, hand_qpos_target), dim=-1)
            action = unscale(
                qpos_target,
                self.robot_dof_lower_limits[self.active_robot_dof_indices],
                self.robot_dof_upper_limits[self.active_robot_dof_indices],
            )
        else:
            action = self.actions.clone()
            action[:, :7] = wrist_target
            action[:, self.hand_dof_start_idx :] = unscale(
                hand_qpos_target,
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
        return action.clamp(-self.clip_actions, self.clip_actions)

    def compute_observations(self) -> torch.Tensor:
        """Compute and cache policy observation tensor."""

        self._compute_intermediate_values()
        obs_buf = torch.zeros((self.num_envs, self.num_observations), dtype=torch.float, device=self.device)
        self.compute_required_observations(obs_buf, self.cfg.obs_type, self.num_observations)
        self.obs_tensor = obs_buf
        return obs_buf

    def compute_required_observations(self, obs_buf: torch.Tensor, obs_type: str, num_obs: int):
        """Concatenate observation fields requested by `obs_type`.

        默认 `armdof+handdof+eefpose+objpose` 的维度是 7 + 6 + 7 + 7 = 27。
        """

        obs_end = 0
        if "armdof" in obs_type:
            obs_buf[:, obs_end : obs_end + self.num_arm_dofs] = unscale(
                self.robot_dof_pos[:, self.arm_dof_indices],
                self.robot_dof_lower_limits[self.arm_dof_indices],
                self.robot_dof_upper_limits[self.arm_dof_indices],
            )
            obs_end += self.num_arm_dofs

        if "handdof" in obs_type:
            obs_buf[:, obs_end : obs_end + self.num_active_hand_dofs] = unscale(
                self.robot_dof_pos[:, self.active_hand_dof_indices],
                self.robot_dof_lower_limits[self.active_hand_dof_indices],
                self.robot_dof_upper_limits[self.active_hand_dof_indices],
            )
            obs_end += self.num_active_hand_dofs

        if "fulldof" in obs_type:
            obs_buf[:, obs_end : obs_end + self.num_robot_dofs] = unscale(
                self.robot_dof_pos[:, self.robot_dof_indices],
                self.robot_dof_lower_limits[self.robot_dof_indices],
                self.robot_dof_upper_limits[self.robot_dof_indices],
            )
            obs_end += self.num_robot_dofs

        if "eefpose" in obs_type:
            obs_buf[:, obs_end : obs_end + 7] = policy_pose_from_wxyz(self.eef_pose)
            obs_end += 7

        if "ftpos" in obs_type:
            num_ft_states = self.num_fingers * 3
            obs_buf[:, obs_end : obs_end + num_ft_states] = self.fingertip_pos.reshape(self.num_envs, num_ft_states)
            obs_end += num_ft_states

        if "palmpose" in obs_type:
            obs_buf[:, obs_end : obs_end + 7] = policy_pose_from_wxyz(self.palm_pose)
            obs_end += 7

        if "lastact" in obs_type:
            obs_buf[:, obs_end : obs_end + self.num_actions] = self.actions
            obs_end += self.num_actions

        if "objxyz" in obs_type:
            obs_buf[:, obs_end : obs_end + 3] = self.object_pos
            obs_end += 3

        if "objpose" in obs_type:
            obs_buf[:, obs_end : obs_end + 7] = policy_pose_from_wxyz(self.object_pose)
            obs_end += 7

        if "objinitpose" in obs_type:
            obs_buf[:, obs_end : obs_end + 7] = policy_pose_from_wxyz(self.object_init_states[:, 0:7])
            obs_end += 7

        if "objpcl" in obs_type:
            pcl = self.transform_obj_pcl_2_world()
            pcl_flat = pcl.reshape(self.num_envs, -1)
            obs_buf[:, obs_end : obs_end + pcl_flat.shape[-1]] = pcl_flat
            obs_end += pcl_flat.shape[-1]

        if obs_end != num_obs:
            raise RuntimeError(f"Observation shape mismatch: built {obs_end} values, cfg expects {num_obs}.")

    def transform_obj_pcl_2_world(self) -> torch.Tensor:
        """Transform object-local point cloud to each environment's local world frame."""

        if self.object_pcl_buf.numel() == 0:
            return self.object_pcl_buf

        flat_points = self.object_pcl_buf.reshape(-1, 3)
        quat = self.object_rot[:, None, :].expand(-1, self.object_pcl_buf.shape[1], -1).reshape(-1, 4)
        rotated = quat_apply(quat, flat_points).reshape_as(self.object_pcl_buf)
        return rotated + self.object_pos[:, None, :]

    def compute_arm_ik(
        self,
        action: torch.Tensor,
        is_delta_pose: bool = True,
        is_delta_pose_in_world: bool = True,
        reference_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert desired EE pose/delta-pose to arm joint deltas with damped least squares IK."""

        if reference_state is None:
            self._compute_intermediate_values()
            reference_state = self.eef_pose

        if is_delta_pose:
            if is_delta_pose_in_world:
                pos_err = action[:, 0:3]
                rot_axis_angle = action[:, 3:6]
            else:
                pos_err = quat_apply(reference_state[:, 3:7], action[:, 0:3])
                rot_axis_angle = action[:, 3:6]
            angle = torch.linalg.norm(rot_axis_angle, dim=-1)
            axis = rot_axis_angle / (angle.unsqueeze(-1) + 1.0e-6)
            desired_quat = quat_mul(quat_from_angle_axis(angle, axis), reference_state[:, 3:7])
            orn_err = orientation_error(desired_quat, reference_state[:, 3:7])
        else:
            pos_err = action[:, 0:3] - reference_state[:, 0:3]
            orn_err = orientation_error(normalize_quat(action[:, 3:7]), reference_state[:, 3:7])

        dpose = torch.cat((pos_err, orn_err), dim=-1).unsqueeze(-1)
        return self._control_ik(dpose)

    def _control_ik(self, dpose: torch.Tensor) -> torch.Tensor:
        """Solve `J^T (J J^T + lambda I)^-1 dpose` for arm joint deltas."""

        try:
            jacobians = self.robot.root_physx_view.get_jacobians()
            j_eef = jacobians[:, self.eef_jacobian_body_id, :, self.arm_dof_indices]
        except Exception:
            return torch.zeros((self.num_envs, self.num_arm_dofs), dtype=torch.float, device=self.device)

        damping = 0.1
        j_eef_t = torch.transpose(j_eef, 1, 2)
        identity = torch.eye(6, dtype=torch.float, device=self.device).unsqueeze(0)
        lhs = j_eef @ j_eef_t + identity * (damping**2)
        return (j_eef_t @ torch.linalg.solve(lhs, dpose)).squeeze(-1)

    def compute_reward(self) -> torch.Tensor:
        """Call the migrated DemoGrasp reward and publish IsaacLab logs."""

        self._compute_intermediate_values()

        (
            reward,
            reward_resets,
            reward_progress_buf,
            self.successes,
            self.current_successes,
            self.has_hit_table,
            reward_info,
        ) = self.reward_function(
            reset_buf=self.reset_buf,
            progress_buf=self.episode_length_buf,
            successes=self.successes,
            current_successes=self.current_successes,
            has_hit_table=self.has_hit_table,
            max_episode_length=self.max_episode_length,
            table_heights=self.table_heights,
            object_pos=self.object_pos,
            palm_pos=self.palm_center_pos,
            fingertip_pos=self.fingertip_pos,
            num_fingers=self.num_fingers,
            object_init_states=self.object_init_states,
            eef_pose=self.eef_pose,
            actions=self.actions,
        )

        self.rew_buf = reward

        # 旧奖励会返回 reset/progress；IsaacLab 仍负责真实 episode 计数，这里只合并 reset 标志。
        reward_resets = reward_resets.to(dtype=torch.bool)
        self.reset_buf |= reward_resets
        self.progress_buf = self.episode_length_buf
        self.reward_progress_buf = reward_progress_buf

        self.extras.setdefault("log", {})
        for key, value in reward_info.items():
            self.extras[key] = value
            if torch.is_tensor(value):
                # 每个环境一份的指标取均值后写入 log，避免 logger 收到大矩阵。
                self.extras["log"][key] = value.float().mean() if value.numel() > 1 else value.float()
            else:
                self.extras["log"][key] = value
        self.extras["log"]["success_rate"] = self.current_successes.float().mean()
        self.extras["successes"] = self.successes
        self.extras["current_successes"] = self.current_successes
        self.extras["has_hit_table"] = self.has_hit_table
        return self.rew_buf

    def compute_real_observation_dict(self) -> dict:
        """Return observations in a real-robot-like dictionary layout."""

        self._compute_intermediate_values()
        return {
            "instruction": self.instructions,
            "right_arm_qpos": self.robot_dof_pos[:, self.arm_dof_indices].detach().cpu().numpy(),
            "right_arm_eef_pose": self.eef_pose.detach().cpu().numpy(),
            "right_hand_qpos": self.robot_dof_pos[:, self.active_hand_dof_indices].detach().cpu().numpy(),
            "object_pose": self.object_pose.detach().cpu().numpy(),
        }

    def _compute_intermediate_values(self):
        """Refresh convenient state tensors from IsaacLab asset data."""

        self.robot_dof_pos = self.robot.data.joint_pos
        self.robot_dof_vel = self.robot.data.joint_vel

        env_origins = self.scene.env_origins
        self.eef_pos = self.robot.data.body_pos_w[:, self.eef_body_id] - env_origins
        self.eef_rot = self.robot.data.body_quat_w[:, self.eef_body_id]
        self.eef_pose = torch.cat((self.eef_pos, self.eef_rot), dim=-1)

        self.palm_pos = self.robot.data.body_pos_w[:, self.palm_body_id] - env_origins
        self.palm_rot = self.robot.data.body_quat_w[:, self.palm_body_id]
        self.palm_pose = torch.cat((self.palm_pos, self.palm_rot), dim=-1)
        self.palm_center_pos = self.palm_pos + quat_apply(self.palm_rot, self.palm_offset.repeat(self.num_envs, 1))

        if self.num_fingers > 0:
            self.fingertip_pos = self.robot.data.body_pos_w[:, self.fingertip_body_ids] - env_origins[:, None, :]
            self.fingertip_rot = self.robot.data.body_quat_w[:, self.fingertip_body_ids]
        else:
            self.fingertip_pos = torch.empty((self.num_envs, 0, 3), dtype=torch.float, device=self.device)
            self.fingertip_rot = torch.empty((self.num_envs, 0, 4), dtype=torch.float, device=self.device)

        self.object_pos = self.object.data.root_pos_w - env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_pose = torch.cat((self.object_pos, self.object_rot), dim=-1)
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def _find_joint_ids(self, names: Sequence[str], required: bool = True) -> torch.Tensor | list[int]:
        """Resolve joint ids while preserving the order requested by config."""

        ids, resolved = self.robot.find_joints(list(names), preserve_order=True)
        if required and len(ids) != len(names):
            missing = sorted(set(names) - set(resolved))
            raise RuntimeError(f"Missing joints in robot asset: {missing}")
        if required:
            return torch.tensor(ids, dtype=torch.long, device=self.device)
        return ids

    def _find_body_id(self, name: str) -> int:
        """Resolve one body id and fall back to the last body for development-time robustness."""

        ids, _ = self.robot.find_bodies(name, preserve_order=True)
        return ids[0] if ids else len(self.robot.body_names) - 1

    def _find_body_ids(self, names: Sequence[str]) -> list[int]:
        """Resolve multiple body ids, skipping names that do not exist in the current URDF."""

        body_ids: list[int] = []
        for name in names:
            ids, _ = self.robot.find_bodies(name, preserve_order=True)
            if ids:
                body_ids.append(ids[0])
        return body_ids

    def _sanitize_joint_limits(self):
        """Replace invalid/continuous joint limits with a finite fallback window."""

        lower = self.robot_dof_lower_limits
        upper = self.robot_dof_upper_limits
        default = self.robot.data.default_joint_pos[0]
        bad = (~torch.isfinite(lower)) | (~torch.isfinite(upper)) | (upper <= lower)
        lower[bad] = default[bad] - 1.0
        upper[bad] = default[bad] + 1.0

    def _normalize_action_shape(self, actions: torch.Tensor) -> torch.Tensor:
        """Pad or truncate actions so old/new runners both get a 13-D tensor."""

        if actions.shape[-1] == self.num_actions:
            return actions.clone()
        fixed = torch.zeros((actions.shape[0], self.num_actions), dtype=actions.dtype, device=self.device)
        width = min(actions.shape[-1], self.num_actions)
        fixed[:, :width] = actions[:, :width]
        return fixed

    def _apply_mimic_hand_joints(self):
        """Copy active hand targets to passive mimic joints."""

        if self.have_passive_joints:
            self.target_joint_pos[:, self.passive_hand_dof_indices] = (
                self.target_joint_pos[:, self.mimic_parent_dof_indices] * self.mimic_multipliers
            )

    def _limit_target_velocity_and_position(self):
        """Clamp target motion per step and enforce joint limits."""

        arm_step = self.cfg.act_max_ang_vel_arm * self.step_dt
        hand_step = self.cfg.act_max_ang_vel_hand * self.step_dt

        self.target_joint_pos[:, self.arm_dof_indices] = tensor_clamp(
            self.target_joint_pos[:, self.arm_dof_indices],
            self.prev_targets[:, self.arm_dof_indices] - arm_step,
            self.prev_targets[:, self.arm_dof_indices] + arm_step,
        )
        self.target_joint_pos[:, self.hand_dof_indices] = tensor_clamp(
            self.target_joint_pos[:, self.hand_dof_indices],
            self.prev_targets[:, self.hand_dof_indices] - hand_step,
            self.prev_targets[:, self.hand_dof_indices] + hand_step,
        )
        self.target_joint_pos = tensor_clamp(
            self.target_joint_pos,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

    def _sample_object_quat(self, num_samples: int) -> torch.Tensor:
        """Sample reset orientation in IsaacLab wxyz quaternion order."""

        if self.cfg.reset_random_rot == "z":
            angle = sample_uniform(-math.pi, math.pi, (num_samples,), self.device)
            axis = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float, device=self.device).repeat(num_samples, 1)
            return quat_from_angle_axis(angle, axis)
        if self.cfg.reset_random_rot == "random":
            quat = torch.randn((num_samples, 4), dtype=torch.float, device=self.device)
            return normalize_quat(quat)
        return identity_quat(num_samples, self.device)


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map normalized actions from [-1, 1] to joint limits."""

    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map joint positions from joint limits back to [-1, 1]."""

    return 2.0 * (x - lower) / (upper - lower + 1.0e-6) - 1.0


def tensor_clamp(x: torch.Tensor, lower: torch.Tensor | float, upper: torch.Tensor | float) -> torch.Tensor:
    """Clamp tensors with tensor-valued lower/upper bounds."""

    return torch.maximum(torch.minimum(x, torch.as_tensor(upper, device=x.device)), torch.as_tensor(lower, device=x.device))


def identity_quat(num: int, device: str) -> torch.Tensor:
    """Create identity quaternions in IsaacLab wxyz order."""

    quat = torch.zeros((num, 4), dtype=torch.float, device=device)
    quat[:, 0] = 1.0
    return quat


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion tensors and avoid division by zero."""

    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-6)


def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert old IsaacGym xyzw quaternions to IsaacLab wxyz quaternions."""

    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion tensor must end with 4 values, got {quat.shape}.")
    return normalize_quat(torch.cat((quat[..., 3:4], quat[..., 0:3]), dim=-1))


def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Convert IsaacLab wxyz quaternions to old DemoGrasp/IsaacGym xyzw order."""

    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion tensor must end with 4 values, got {quat.shape}.")
    return normalize_quat(torch.cat((quat[..., 1:4], quat[..., 0:1]), dim=-1))


def policy_pose_from_wxyz(pose: torch.Tensor) -> torch.Tensor:
    """Return `xyz + xyzw` policy observations expected by old DemoGrasp checkpoints."""

    return torch.cat((pose[..., :3], wxyz_to_xyzw(pose[..., 3:7])), dim=-1)


def orientation_error(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Return 3-D orientation error between desired and current wxyz quaternions."""

    q_r = quat_mul(desired, quat_conjugate(current))
    return q_r[:, 1:4] * torch.sign(q_r[:, 0]).unsqueeze(-1)


def linear_interpolate_poses(pose1: torch.Tensor, pose2: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Interpolate batched poses with linear position interpolation and normalized quat lerp."""

    t = torch.linspace(0.0, 1.0, n_steps, dtype=torch.float, device=pose1.device).view(1, n_steps, 1)
    interp_pos = pose1[:, None, :3] + t * (pose2[:, None, :3] - pose1[:, None, :3])
    interp_quat = pose1[:, None, 3:7] + t * (pose2[:, None, 3:7] - pose1[:, None, 3:7])
    interp_quat = normalize_quat(interp_quat)
    return torch.cat((interp_pos, interp_quat), dim=-1)
