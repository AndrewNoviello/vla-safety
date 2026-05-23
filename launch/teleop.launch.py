"""teleop.launch.py — hardware bridge + leader teleop.

Brings up the SO101 follower (robot_node + webcam) and the leader bridge that
republishes the leader arm's joints onto so101real/joint_command. Move the
leader arm; the follower mirrors it.

Launch arguments:
  follower_port   (default /dev/ttyACM0)   USB port for the SO101 follower.
  leader_port     (default /dev/ttyACM1)   USB port for the SO101 leader.
  camera_device   (default /dev/video0)
  camera_topic    (default so101real/camera/image)
  camera_rate_hz  (default 30)
  robot_name      (default so101real)

Example:
  ros2 launch launch/teleop.launch.py leader_port:=/dev/ttyACM3
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler, EmitEvent)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_DIR = REPO_ROOT / "launch"


def generate_launch_description():
    follower_port  = LaunchConfiguration("follower_port")
    leader_port    = LaunchConfiguration("leader_port")
    camera_device  = LaunchConfiguration("camera_device")
    camera_topic   = LaunchConfiguration("camera_topic")
    camera_rate_hz = LaunchConfiguration("camera_rate_hz")
    robot_name     = LaunchConfiguration("robot_name")

    args = [
        DeclareLaunchArgument("follower_port",  default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("leader_port",    default_value="/dev/ttyACM1",
                              description="USB serial port for the SO101 leader arm."),
        DeclareLaunchArgument("camera_device",  default_value="/dev/video0"),
        DeclareLaunchArgument("camera_topic",   default_value="so101real/camera/image"),
        DeclareLaunchArgument("camera_rate_hz", default_value="30"),
        DeclareLaunchArgument("robot_name",     default_value="so101real"),
    ]

    hardware = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(LAUNCH_DIR / "hardware.launch.py")),
        launch_arguments={
            "follower_port":  follower_port,
            "camera_device":  camera_device,
            "camera_topic":   camera_topic,
            "camera_rate_hz": camera_rate_hz,
            "robot_name":     robot_name,
        }.items(),
    )

    teleop = ExecuteProcess(
        name="so101_leader_teleop",
        cmd=[
            "python", str(REPO_ROOT / "scripts" / "teleop.py"),
            "--leader-port", leader_port,
        ],
        cwd=str(REPO_ROOT),
        output="screen",
        emulate_tty=True,
    )

    shutdown_on_teleop_exit = RegisterEventHandler(
        OnProcessExit(target_action=teleop, on_exit=[EmitEvent(event=Shutdown())])
    )

    return LaunchDescription([*args, hardware, teleop, shutdown_on_teleop_exit])
