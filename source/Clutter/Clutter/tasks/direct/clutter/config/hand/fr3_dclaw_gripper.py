"""FR3 arm with DClaw three-finger gripper."""

from __future__ import annotations

from .hand_cfg import HandCfg, joint_pos_from_order

ARM_DOF_NAMES = (
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
)

ACTIVE_HAND_DOF_NAMES = (
    "joint_f1_0",
    "joint_f1_1",
    "joint_f1_2",
    "joint_f2_0",
    "joint_f2_1",
    "joint_f2_2",
    "joint_f3_0",
    "joint_f3_1",
    "joint_f3_2",
)

HAND_CFG = HandCfg(
    name="fr3_dclaw_gripper",
    robot_asset_file="fr3_gripper/fr3_dclaw_gripper.usd",
    robot_asset_file_visual_realistic="fr3_gripper/fr3_dclaw_gripper.usd",
    arm_dof_names=ARM_DOF_NAMES,
    active_hand_dof_names=ACTIVE_HAND_DOF_NAMES,
    passive_joint_mimic={},
    eef_link_name="fr3_link8",
    palm_link_name="base_link",
    fingertip_link_names=("link_f1_3", "link_f2_3", "link_f3_3"),
    palm_offset=(0.0, 0.0, 0.0),
    default_joint_pos=joint_pos_from_order(
        ARM_DOF_NAMES + ACTIVE_HAND_DOF_NAMES,
        (0.0, 0.0, 0.0, -1.6, 0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    num_actions=16,
)

__all__ = ["HAND_CFG"]
