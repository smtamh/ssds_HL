import rclpy


def _run_apple_vision_pro(
    disable: bool = False,
    left_off: bool = False,
    right_off: bool = False,
):
    from fr3_husky_task_manager.apple_vision_pro import AppleVisionProClient

    should_shutdown = False
    if not rclpy.ok():
        rclpy.init()
        should_shutdown = True

    node = AppleVisionProClient(
        disable=disable,
        left_off=left_off,
        right_off=right_off,
    )

    try:
        return node.run()
    finally:
        node.destroy_node()
        if should_shutdown and rclpy.ok():
            rclpy.shutdown()


async def joint_move(
    arm_names: str = "both",
    left_target_positions: list[float] = None,
    right_target_positions: list[float] = None,
    image=None):
    from fr3_husky_task_manager.move_to_joint import run_move_to_joint

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    _run_apple_vision_pro(disable=True)
    try:
        run_move_to_joint(
            arm=arm_names,
            left_target_positions=left_target_positions,
            right_target_positions=right_target_positions,
        )
    except Exception as e:
        return f"Joint move failed: {str(e)}"

    finally:
        _run_apple_vision_pro(disable=False)
    
    if arm_names == "left":
        return f"Joint move completed: left arm moved to {left_target_positions}."
    if arm_names == "right":
        return f"Joint move completed: right arm moved to {right_target_positions}."
    return f"Joint move completed: left arm moved to {left_target_positions}, right arm moved to {right_target_positions}."


async def init_pose(
    arm_names: str = "both",
    image=None):
    from fr3_husky_task_manager.move_to_joint import run_move_to_joint

    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    _run_apple_vision_pro(disable=True)
    try:
        run_move_to_joint(
            arm=arm_names,
            left_target_positions=[0.25, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            right_target_positions=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        )

    finally:
        _run_apple_vision_pro(disable=False)

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
    from fr3_husky_task_manager.gripper_move import run_gripper_move

    _run_apple_vision_pro(disable=True)
    try:
        result = run_gripper_move(
            arm_names=arm_names,
            command=command,
            width=width,
            speed=speed,
            force=force,
        )
    finally:
        _run_apple_vision_pro(disable=False)

    return result


async def onoff_vision_pro(
    arm_names: str = "both",
    command: str = "on",
    image=None
):
    if arm_names not in ["left", "right", "both"]:
        return f"Invalid arm_names value: {arm_names}"

    if command not in ["on", "off"]:
        return f"Invalid command value: {command}"

    if arm_names == "both":
        if command == "on":
            _run_apple_vision_pro(disable=False, left_off=False, right_off=False)
        else:
            _run_apple_vision_pro(disable=True)
        return f"{arm_names} vision pro tracking turned {command}."

    if arm_names == "left":
        left_off = command == "off"
        right_off = command != "off"
    else:
        left_off = command != "off"
        right_off = command == "off"

    _run_apple_vision_pro(
        disable=False,
        left_off=left_off,
        right_off=right_off,
    )

    return f"{arm_names} vision pro tracking turned {command}."
