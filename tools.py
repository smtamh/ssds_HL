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
    except Exception as e:
        return f"Joint move failed: {str(e)}"

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
    except Exception as e:
        return f"Init pose failed: {str(e)}"

    finally:
        run_apple_vision_pro(disable=False)

    if arm_names == "left":
        return f"Init pose completed: left arm moved to the initial pose."
    if arm_names == "right":
        return f"Init pose completed: right arm moved to the initial pose."
    return f"Init pose completed: both arms moved to the initial pose."




async def apple_vision_pro():
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro

    result = run_apple_vision_pro(disable=False)

    if result:
        return "Apple Vision Pro tracking enabled."
    return "Failed to enable Apple Vision Pro tracking."


async def husky_pedal():
    from fr3_husky_task_manager.husky_pedal import run_husky_pedal

    result = run_husky_pedal()

    if result:
        return "Husky pedal teleoperation enabled."
    return "Failed to enable Husky pedal teleoperation."


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
    except Exception as e:
        return f"Gripper command failed: {str(e)}"
    finally:
        run_apple_vision_pro(disable=False)

    return result


async def onoff_vision_pro(
    arm_names: str = "both",
    command: str = "on",
    image=None
):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    if command not in ["on", "off"]:
        return f"Invalid command value: {command}"

    if arm_names == "both":
        if command == "on":
            run_apple_vision_pro(disable=False, left_off=False, right_off=False)
        else:
            run_apple_vision_pro(disable=True)
        return f"{arm_names} vision pro tracking turned {command}."

    if arm_names == "left":
        left_off = command == "off"
        right_off = command != "off"
    else:
        left_off = command != "off"
        right_off = command == "off"

    run_apple_vision_pro(
        disable=False,
        left_off=left_off,
        right_off=right_off,
    )

    return f"{arm_names} vision pro tracking turned {command}."
