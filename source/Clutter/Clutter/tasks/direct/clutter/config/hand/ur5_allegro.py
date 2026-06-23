"""UR5 arm with Allegro hand."""

from __future__ import annotations

from .hand_cfg import HandCfg, joint_pos_from_order

ARM_DOF_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

ACTIVE_HAND_DOF_NAMES = (
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
)

HAND_CFG = HandCfg(
    name="ur5_allegro",
    robot_asset_file="ur5_allegro/ur5_allegro.usd",
    robot_asset_file_visual_realistic="ur5_allegro/ur5_allegro.usd",
    arm_dof_names=ARM_DOF_NAMES,
    active_hand_dof_names=ACTIVE_HAND_DOF_NAMES,
    passive_joint_mimic={},
    eef_link_name="ee_link",
    palm_link_name="Allegro_base_link",
    fingertip_link_names=("link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"),
    palm_offset=(0.02, 0.0, 0.03),
    default_joint_pos=joint_pos_from_order(
        ARM_DOF_NAMES + ACTIVE_HAND_DOF_NAMES,
        (
            0.0,
            -2.0,
            1.8,
            0.0,
            1.57,
            -1.57,
            0.0,
            0.0,
            0.0,
            0.0,
            1.365,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ),
    num_actions=23,
    hand_dof_start_idx=7,
)

__all__ = ["HAND_CFG"]
