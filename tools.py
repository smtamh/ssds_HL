import os, json, time
import numpy as np

# ROS
import rclpy
from rclpy.node import Node

import asyncio
import threading


async def joint_move(
    arm_names: str = "both",
    left_target_positions: list[float] = None,
    right_target_positions: list[float] = None,
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.move_to_joint import run_move_to_joint

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    run_apple_vision_pro(disable=True)
    try:
        run_move_to_joint(
            arm=arm_names,
            left_target_positions=left_target_positions,
            right_target_positions=right_target_positions,
        )

    finally:
        run_apple_vision_pro(disable=False)
    
    if arm_names == "left":
        return f"Joint move completed: left arm moved to {left_target_positions}."
    if arm_names == "right":
        return f"Joint move completed: right arm moved to {right_target_positions}."
    return f"Joint move completed: left arm moved to {left_target_positions}, right arm moved to {right_target_positions}."


async def init_pose(
    arm_names: str = "both",
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.move_to_joint import run_move_to_joint

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    run_apple_vision_pro(disable=True)
    try:
        run_move_to_joint(
            arm=arm_names,
            left_target_positions=[0.25, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            right_target_positions=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        )

    finally:
        run_apple_vision_pro(disable=False)

    if arm_names == "left":
        return f"Init pose completed: left arm moved to the initial pose."
    if arm_names == "right":
        return f"Init pose completed: right arm moved to the initial pose."
    return f"Init pose completed: both arms moved to the initial pose."


async def gripper_command(
    arm_names: str = "both",
    command: str = "open",
    width: float | None=None,
    speed: float = 0.1,
    force: float = 30.0,
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.gripper_move import run_gripper_move

    run_apple_vision_pro(disable=True)
    try:
        result = run_gripper_move(
            arm_names=arm_names,
            command=command,
            width=width,
            speed=speed,
            force=force,
        )
    finally:
        run_apple_vision_pro(disable=False)

    return result
