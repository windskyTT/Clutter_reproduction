"""FR3 arm with Panda parallel gripper."""

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

ACTIVE_HAND_DOF_NAMES = ("panda_finger_joint1",)
PASSIVE_JOINT_MIMIC = {"panda_finger_joint2": ("panda_finger_joint1", 1.0)}

HAND_CFG = HandCfg(
    name="fr3_panda_gripper",
    robot_asset_file="fr3_gripper/fr3_panda_gripper.usd",
    robot_asset_file_visual_realistic="fr3_gripper/fr3_panda_gripper.usd",
    arm_dof_names=ARM_DOF_NAMES,
    active_hand_dof_names=ACTIVE_HAND_DOF_NAMES,
    passive_joint_mimic=PASSIVE_JOINT_MIMIC,
    eef_link_name="fr3_link8",
    palm_link_name="fr3_link8",
    fingertip_link_names=("right_gripper",),
    palm_offset=(0.0, 0.0, 0.0),
    default_joint_pos=joint_pos_from_order(
        ARM_DOF_NAMES + ACTIVE_HAND_DOF_NAMES + tuple(PASSIVE_JOINT_MIMIC),
        (0.0, 0.0, 0.0, -1.6, 0.0, 1.6, 0.0, 0.04, 0.04),
    ),
    num_actions=8,
)

__all__ = ["HAND_CFG"]
