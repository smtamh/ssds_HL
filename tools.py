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
        run_apple_vision_pro(disable=False, left_off=True)
    
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
        run_apple_vision_pro(disable=False, left_off=True)

    if arm_names == "left":
        return f"Init pose completed: left arm moved to the initial pose."
    if arm_names == "right":
        return f"Init pose completed: right arm moved to the initial pose."
    return f"Init pose completed: both arms moved to the initial pose."




async def apple_vision_pro():
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro

    result = run_apple_vision_pro(disable=False, left_off=True)

    if result:
        return "Apple Vision Pro tracking enabled."
    return "Failed to enable Apple Vision Pro tracking."


async def husky_pedal():
    from fr3_husky_task_manager.husky_pedal import run_husky_pedal

    result = run_husky_pedal()

    if result:
        return "Husky pedal teleoperation enabled."
    return "Failed to enable Husky pedal teleoperation."


async def nut_tightening(
    arm_name: str = "right",
    rotation_angle: float = -60.0,
    image=None,
):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.nut_tightening import run_nut_tightening

    if arm_name not in ["left", "right"]:
        return f"Invalid arm_name value: {arm_name}"
    if abs(rotation_angle) < 60.0:
        return "rotation_angle must be at least 60 degrees in magnitude."

    if not run_apple_vision_pro(disable=True):
        return "Nut tightening failed: unable to disable Apple Vision Pro tracking."

    try:
        return run_nut_tightening(
            arm=arm_name,
            rotation_angle=rotation_angle,
        )
    except Exception as e:
        return f"Nut tightening failed: {str(e)}"
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        run_apple_vision_pro(disable=False, left_off=True)


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
        run_apple_vision_pro(disable=False, left_off=True)

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


async def task_space_delta_move(
    arm_names: str = "both",
    left_position: list[float] | None = None,
    left_rpy: list[float] | None = None,
    right_position: list[float] | None = None,
    right_rpy: list[float] | None = None,
    image=None,
):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.task_space_delta_move import run_task_space_delta_move

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    left_position = left_position if left_position is not None else [0.0, 0.0, 0.0]
    left_rpy = left_rpy if left_rpy is not None else [0.0, 0.0, 0.0]
    right_position = right_position if right_position is not None else [0.0, 0.0, 0.0]
    right_rpy = right_rpy if right_rpy is not None else [0.0, 0.0, 0.0]

    values = {
        "left_position": left_position,
        "left_rpy": left_rpy,
        "right_position": right_position,
        "right_rpy": right_rpy,
    }
    for name, value in values.items():
        if len(value) != 3:
            return f"{name} must contain exactly three values."

    run_apple_vision_pro(disable=True)
    try:
        result = run_task_space_delta_move(
            arm=arm_names,
            left_position=left_position,
            left_rpy=left_rpy,
            right_position=right_position,
            right_rpy=right_rpy,
        )
    except Exception as e:
        return f"Task-space delta move failed: {str(e)}"
    finally:
        run_apple_vision_pro(disable=False, left_off=True)

    return result


async def rotate_yaw(
    arm_names: str = "both",
    yaw: float = 0.0,
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.task_space_delta_move import run_task_space_delta_move

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"
    
    if yaw > 1.57 or yaw < -1.57:
        return f"Invalid yaw value: {yaw}. Yaw should be in the range of [-1.57, 1.57] radians."

    run_apple_vision_pro(disable=True)
    try:
        run_task_space_delta_move(
            arm=arm_names,
            left_position=[0.0, 0.0, 0.0],
            left_rpy=[0.0, 0.0,yaw],
            right_position=[0.0, 0.0, 0.0],
            right_rpy=[0.0, 0.0,yaw],
        )

    finally:
        run_apple_vision_pro(disable=False, left_off=True)

    return f"rotate_yaw completed."


async def rotate_pitch(
    arm_names: str = "both",
    pitch: float = 0.0,
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.task_space_delta_move import run_task_space_delta_move

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    if pitch > 0.32 or pitch < -0.32:
        return f"Invalid pitch value: {pitch}. Pitch should be in the range of [-0.32, 0.32] radians."

    run_apple_vision_pro(disable=True)
    try:
        run_task_space_delta_move(
            arm=arm_names,
            left_position=[0.0, 0.0, 0.0],
            left_rpy=[0.0, pitch, 0.0],
            right_position=[0.0, 0.0, 0.0],
            right_rpy=[0.0, pitch, 0.0],
        )

    finally:
        run_apple_vision_pro(disable=False, left_off=True)

    return f"rotate_pitch completed."

async def rotate_roll(
    arm_names: str = "both",
    roll: float = 0.0,
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.task_space_delta_move import run_task_space_delta_move

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    if roll > 0.32 or roll < -0.32:
        return f"Invalid roll value: {roll}. Roll should be in the range of [-0.32, 0.32] radians."

    run_apple_vision_pro(disable=True)
    try:
        run_task_space_delta_move(
            arm=arm_names,
            left_position=[0.0, 0.0, 0.0],
            left_rpy=[roll, 0.0,  0.0],
            right_position=[0.0, 0.0, 0.0],
            right_rpy=[roll, 0.0,  0.0],
        )

    finally:
        run_apple_vision_pro(disable=False, left_off=True)

    return f"rotate_roll completed."


async def move_arm_forward_backward(
    arm_names: str = "both",
    value: float = 0.0,
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.task_space_delta_move import run_task_space_delta_move
    """
    Move robot arm(s) along the local x-axis direction.

    Args:
    - arm_names:
        Target arm selection.
        Options: "left", "right", "both"

    - value:
        Relative translation distance [m] along local x-axis.
        Positive value  -> move forward
        Negative value  -> move backward

    - image:
        Optional image input.
    """

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"
    
    if value > 0.1 or value < -0.1:
        return f"Invalid value: {value}. Value should be in the range of [-0.1, 0.1] meters."

    run_apple_vision_pro(disable=True)
    try:
        run_task_space_delta_move(
            arm=arm_names,
            left_position=[value, 0.0, 0.0],
            left_rpy=[0.0, 0.0, 0.0],
            right_position=[value, 0.0, 0.0],
            right_rpy=[0.0, 0.0, 0.0],
        )

    finally:
        run_apple_vision_pro(disable=False, left_off=True)

    return f"move_arm_forward_backward completed. [arm:{arm_names}, value:{value}]"



async def move_arm_down_up(
    arm_names: str = "both",
    value: float = 0.0,
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.task_space_delta_move import run_task_space_delta_move
    """
    Move robot arm(s) along the local z-axis direction.

    Args:
    - arm_names:
        Target arm selection.
        Options: "left", "right", "both"

    - value:
        Relative translation distance [m] along local z-axis.
        Positive value  -> move down
        Negative value  -> move up

    - image:
        Optional image input.
    """


    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"
    

    if value > 0.1 or value < -0.1:
        return f"Invalid value: {value}. Value should be in the range of [-0.1, 0.1] meters."

    run_apple_vision_pro(disable=True)
    try:
        run_task_space_delta_move(
            arm=arm_names,
            left_position=[0.0, 0.0, value],
            left_rpy=[0.0, 0.0, 0.0],
            right_position=[0.0, 0.0, value],
            right_rpy=[0.0, 0.0, 0.0],
        )

    finally:
        run_apple_vision_pro(disable=False, left_off=True)

    return f"move_arm_down_up completed. [arm:{arm_names}, value:{value}]"




async def move_arm_right_left(
    arm_names: str = "both",
    value: float = 0.0,
    image=None):
    from fr3_husky_task_manager.apple_vision_pro import run_apple_vision_pro
    from fr3_husky_task_manager.task_space_delta_move import run_task_space_delta_move
    """
    Move robot arm(s) along the local y-axis direction.

    Args:
    - arm_names:
        Target arm selection.
        Options: "left", "right", "both"

    - value:
        Relative translation distance [m] along local y-axis.
        Positive value  -> move right
        Negative value  -> move left

    - image:
        Optional image input.
    """
    
    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"
    

    if value > 0.1 or value < -0.1:
        return f"Invalid value: {value}. Value should be in the range of [-0.1, 0.1] meters."

    run_apple_vision_pro(disable=True)
    try:
        run_task_space_delta_move(
            arm=arm_names,
            left_position=[0.0, value, 0.0],
            left_rpy=[0.0, 0.0, 0.0],
            right_position=[0.0, value, 0.0],
            right_rpy=[0.0, 0.0, 0.0],
        )

    finally:
        run_apple_vision_pro(disable=False, left_off=True)

    return f"move_arm_right_left completed. [arm:{arm_names}, value:{value}]"
