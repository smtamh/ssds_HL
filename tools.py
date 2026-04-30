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
    from fr3_husky_task_manager.move_to_joint import MoveToJointClient

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    rclpy.init()
    node = MoveToJointClient(
        arm=arm_names,
        left_target_positions=left_target_positions,
        right_target_positions=right_target_positions,
    )

    try:
        node.send_goal_and_wait()
    except KeyboardInterrupt:
        cancel_future = node.cancel_goal()
        if cancel_future is not None:
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if arm_names == "left":
        return f"Joint move completed: left arm moved to {left_target_positions}."
    if arm_names == "right":
        return f"Joint move completed: right arm moved to {right_target_positions}."
    return f"Joint move completed: left arm moved to {left_target_positions}, right arm moved to {right_target_positions}."


async def init_pose(
    arm_names: str = "both",
    image=None):
    from fr3_husky_task_manager.move_to_joint import MoveToJointClient

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    rclpy.init()
    node = MoveToJointClient(
        arm=arm_names,
        left_target_positions=[0.25, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        right_target_positions=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    )

    try:
        node.send_goal_and_wait()
    except KeyboardInterrupt:
        cancel_future = node.cancel_goal()
        if cancel_future is not None:
            rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

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
    from fr3_husky_msgs.action import GripperCommand
    from rclpy.action import ActionClient

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    if command not in ["open", "grasp", "move"]:
        return f"Invalid command value: {command}"

    if command == "open":
        target_width = 0.08
    elif command == "grasp":
        target_width = 0.0
    else:
        if width is None:
            return "Invalid width value: width is required when command is 'move'."
        target_width = width

    if target_width < 0.0 or target_width > 0.08:
        return f"Invalid width value: {target_width}. Use a value from 0.0 to 0.08 meters."

    if speed <= 0.0:
        return f"Invalid speed value: {speed}. Use a positive speed."

    if force <= 0.0 or force > 140.0:
        return f"Invalid force value: {force}. Use a value from 0.0 to 140.0 newtons."

    rclpy.init()
    node = Node("gripper_command_client")

    try:
        action_client = ActionClient(node, GripperCommand, "/fr3_husky_gripper_command")
        action_client.wait_for_server()

        goal = GripperCommand.Goal()
        goal.arm_names = arm_names
        goal.command = command
        goal.width = float(target_width)
        goal.speed = float(speed)
        goal.force = float(force)
        goal.epsilon_inner = 0.08
        goal.epsilon_outer = 0.08
        goal.use_weld = True
        goal.weld_name = "weld_blue_right_tcp"

        send_goal_future = action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_goal_future)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            return "Gripper command failed: goal response is None."
        if not goal_handle.accepted:
            return "Gripper command failed: goal was rejected."

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)

        wrapped_result = result_future.result()
        if wrapped_result is None:
            return "Gripper command failed: result is None."

        result = wrapped_result.result
        if result.success:
            return f"Gripper command completed: {result.message}"
        return f"Gripper command failed: {result.message}"
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
