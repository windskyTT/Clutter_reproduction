"""UR5 arm with SVH hand."""

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
    "right_hand_Thumb_Opposition",
    "right_hand_Thumb_Flexion",
    "right_hand_Index_Finger_Proximal",
    "right_hand_Index_Finger_Distal",
    "right_hand_Finger_Spread",
    "right_hand_Pinky",
    "right_hand_Ring_Finger",
    "right_hand_Middle_Finger_Proximal",
    "right_hand_Middle_Finger_Distal",
)

PASSIVE_JOINT_MIMIC = {
    "right_hand_j5": ("right_hand_Thumb_Opposition", 1.0),
    "right_hand_j3": ("right_hand_Thumb_Flexion", 1.01511),
    "right_hand_j4": ("right_hand_Thumb_Flexion", 1.44889),
    "right_hand_j14": ("right_hand_Index_Finger_Distal", 1.0450),
    "right_hand_j15": ("right_hand_Middle_Finger_Distal", 1.0454),
    "right_hand_j12": ("right_hand_Ring_Finger", 1.3588),
    "right_hand_j16": ("right_hand_Ring_Finger", 1.42093),
    "right_hand_j13": ("right_hand_Pinky", 1.3588),
    "right_hand_j17": ("right_hand_Pinky", 1.42307),
    "right_hand_index_spread": ("right_hand_Finger_Spread", 0.5),
    "right_hand_ring_spread": ("right_hand_Finger_Spread", 0.5),
}

DEFAULT_JOINT_ORDER = ARM_DOF_NAMES + ACTIVE_HAND_DOF_NAMES + tuple(PASSIVE_JOINT_MIMIC)

HAND_CFG = HandCfg(
    name="ur5_svh",
    robot_asset_file="ur5_svh/ur5_svh.usd",
    robot_asset_file_visual_realistic="ur5_svh/ur5_svh.usd",
    arm_dof_names=ARM_DOF_NAMES,
    active_hand_dof_names=ACTIVE_HAND_DOF_NAMES,
    passive_joint_mimic=PASSIVE_JOINT_MIMIC,
    eef_link_name="ee_link",
    palm_link_name="svh_base_link",
    fingertip_link_names=("right_hand_c", "right_hand_t", "right_hand_s", "right_hand_r", "right_hand_q"),
    palm_offset=(0.045, 0.0, 0.01),
    default_joint_pos=joint_pos_from_order(
        DEFAULT_JOINT_ORDER,
        (0.0, -2.0, 1.8, 0.0, 1.57, -1.57) + (0.0,) * 20,
    ),
    num_actions=16,
    hand_dof_start_idx=7,
)

__all__ = ["HAND_CFG"]
