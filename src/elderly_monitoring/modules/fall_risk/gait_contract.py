"""Shared tensor contract for the retained gait TCN candidate."""

from __future__ import annotations


CANONICAL_GAIT_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "pelvis_center",
    "shoulder_center",
)

GAIT_TCN_CHANNELS = ("x", "y", "dx", "dy", "quality")
