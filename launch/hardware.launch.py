"""hardware.launch.py — bring up the SO101 follower node and the wrist webcam.

This is the foundation launch file. Every other pipeline (teleop, recorder,
deploy, safety_monitor) includes this one to spin up the hardware bridge.

Topics published / subscribed (in the so101real namespace):
  so101real/joint_state    (RobotNode)
  so101real/joint_command  (RobotNode subscribes)
  so101real/camera/image   (CameraNode)
  so101real/robot_status   (RobotNode)
  so101real/estop          (RobotNode subscribes)

Launch arguments (`ros2 launch launch/hardware.launch.py --show-args`):
  follower_port    (default /dev/ttyACM0)   USB serial port for the SO101 follower arm.
  camera_device    (default /dev/video0)    V4L2 device for the wrist webcam.
  camera_topic     (default so101real/camera/image)
  camera_rate_hz   (default 30)             Webcam publish rate.
  robot_name       (default so101real)      ROS namespace for the robot node.

Example:
  ros2 launch launch/hardware.launch.py follower_port:=/dev/ttyACM2 camera_device:=/dev/video2
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration


REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_launch_description():
    follower_port  = LaunchConfiguration("follower_port")
    camera_device  = LaunchConfiguration("camera_device")
    camera_topic   = LaunchConfiguration("camera_topic")
    camera_rate_hz = LaunchConfiguration("camera_rate_hz")
    robot_name     = LaunchConfiguration("robot_name")

    args = [
        DeclareLaunchArgument(
            "follower_port", default_value="/dev/ttyACM0",
            description="USB serial port for the SO101 follower arm.",
        ),
        DeclareLaunchArgument(
            "camera_device", default_value="/dev/video0",
            description="V4L2 device path for the USB webcam.",
        ),
        DeclareLaunchArgument(
            "camera_topic", default_value="so101real/camera/image",
            description="ROS topic the webcam publishes to.",
        ),
        DeclareLaunchArgument(
            "camera_rate_hz", default_value="30",
            description="Webcam publish rate in Hz.",
        ),
        DeclareLaunchArgument(
            "robot_name", default_value="so101real",
            description="ROS namespace and config name for the SO101 follower.",
        ),
    ]

    robot_node = ExecuteProcess(
        name="so101_robot_node",
        cmd=[
            "python", "-m", "hardware.ros.scripts.robot_node_launcher",
            "robot=so101real",
            ["robot.config.port=", follower_port],
            ["robot.config.name=", robot_name],
        ],
        cwd=str(REPO_ROOT),
        output="screen",
        emulate_tty=True,
    )

    camera_node = ExecuteProcess(
        name="so101_camera_node",
        cmd=[
            "python", str(REPO_ROOT / "hardware" / "ros" / "examples" / "webcam.py"),
            "--device", camera_device,
            "--topic", camera_topic,
            "--rate-hz", camera_rate_hz,
        ],
        cwd=str(REPO_ROOT),
        output="screen",
        emulate_tty=True,
    )

    # If either hardware process dies, tear down the whole launch instead of
    # leaving a half-running stack.
    shutdown_on_robot_exit = RegisterEventHandler(
        OnProcessExit(target_action=robot_node, on_exit=[EmitEvent(event=Shutdown())])
    )
    shutdown_on_camera_exit = RegisterEventHandler(
        OnProcessExit(target_action=camera_node, on_exit=[EmitEvent(event=Shutdown())])
    )

    return LaunchDescription([*args, robot_node, camera_node,
                              shutdown_on_robot_exit, shutdown_on_camera_exit])
