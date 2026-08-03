from mcp.server.fastmcp import FastMCP
import tools
import argparse

mcp = FastMCP(name="ssds_hl")


@mcp.tool()
async def joint_move(
    arm_names: str = "both",
    left_target_positions: list[float] = None,
    right_target_positions: list[float] = None
    ) -> str:
    
    """Move the robot to the specified joint target positions
    
    Args:
    - arm_names: Name of the robot arm (default: "both"). Options: "left", "right", "both"
    - left_target_positions: List of target joint positions.
        If "arm_names" is "right", provide this value as None.
    - right_target_positions: List of target joint positions.
        If "arm_names" is "left", provide this value as None.
    """
    return await tools.joint_move(
        arm_names.strip().lower(),
        left_target_positions,
        right_target_positions,
    )


@mcp.tool()
async def init_pose(
    arm_names: str = "both"
    ) -> str:

    """Move the robot to the initial position

    Args:
    - arm_names: Name of the robot arm (default: "both"). Options: "left", "right", "both"
    """
    return await tools.init_pose(
        arm_names.strip().lower(),
    )

"""
@mcp.tool()
async def enable_apple_vision_pro() -> str:
    return await tools.apple_vision_pro()
"""


@mcp.tool()
async def enable_husky_pedal() -> str:
    """Enable the Husky pedal for teleoperation."""
    return await tools.husky_pedal()


@mcp.tool()
async def nut_tightening(
    arm_name: str = "right",
    rotation_angle: float = -60.0,
) -> str:
    """Tighten the configured default nut with the specified arm.

    Args:
    - arm_name: Arm holding the spanner. Options: "left" or "right".
    - rotation_angle: Tightening angle in degrees; negative values tighten about +Z.
    """
    return await tools.nut_tightening(
        arm_name.strip().lower(),
        rotation_angle,
    )


@mcp.tool()
async def gripper_command(
    arm_names: str = "both",
    command: str = "open",
    width: float | None=None,
    speed: float = 0.1,
    force: float = 30.0,
    ) -> str:
    
    """Send the gripper command action
    
    Args:
    - arm_names: Name of the robot arm (default: "both"). Options: "left", "right", "both"
    - command: Gripper command. Options: "open", "grasp", "move"
    - width: Target gripper width in meters. Required only for command="move".
    - speed: Gripper movement speed in meters per second.
    - force: Grasp force in newtons. Used for "grasp".
    """
    return await tools.gripper_command(
        arm_names.strip().lower(),
        command.strip().lower(),
        width,
        speed,
        force,
    )


@mcp.tool()
async def onoff_vision_pro(
    arm_names: str = "both",
    command: str = "on",
    ) -> str:

    """Enable or disable Apple Vision Pro tracking per arm.

    Args:
    - arm_names: Name of the robot arm (default: "both"). Options: "left", "right", "both"
    - command: Tracking command. Options: "on", "off"
    """
    return await tools.onoff_vision_pro(
        arm_names.strip().lower(),
        command.strip().lower(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP transport type (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the MCP HTTP server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the MCP HTTP server"
    )

    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.settings.transport_security = None
        mcp.run(transport="streamable-http")
    else:
        mcp.run()

if __name__ == "__main__":
    main()
