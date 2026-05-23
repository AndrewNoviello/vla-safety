"""safety_monitor.launch.py — hardware bridge + teleop + live SAFE/UNSAFE
classifier readout.

Use this to drive the arm with the leader and watch the classifier print
SAFE / UNSAFE per tick. The monitor only observes — it does not gate or
override teleop commands. (See `safety_filter` in deploy.launch.py for the
gating variant on top of PI0.)

Launch arguments:
  follower_port           (default /dev/ttyACM0)
  leader_port             (default /dev/ttyACM1)
  camera_device           (default /dev/video0)
  classifier_checkpoint   (default runs/classifier_exp_merged/classifier_best.pt)
  dataset_root            (default data/exp_merged)
  policy_dir              (default '')   — optional safety_ddpg dir for V(s) display.

Example:
  ros2 launch launch/safety_monitor.launch.py \\
      classifier_checkpoint:=runs/classifier_exp_merged/classifier_best.pt \\
      policy_dir:=outputs/safety_ddpg/checkpoints/epoch_0015
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler, EmitEvent,
                            OpaqueFunction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_DIR = REPO_ROOT / "launch"


def _build_monitor(context, *args, **kwargs):
    """Resolve LaunchConfigurations to strings so we can omit --policy_dir
    when the user didn't supply one."""
    classifier_checkpoint = LaunchConfiguration("classifier_checkpoint").perform(context)
    dataset_root          = LaunchConfiguration("dataset_root").perform(context)
    policy_dir            = LaunchConfiguration("policy_dir").perform(context)

    cmd = [
        "python", str(REPO_ROOT / "scripts" / "safety_monitor.py"),
        "--checkpoint",   classifier_checkpoint,
        "--dataset_root", dataset_root,
    ]
    if policy_dir:
        cmd.extend(["--policy_dir", policy_dir])

    monitor = ExecuteProcess(
        name="so101_safety_monitor",
        cmd=cmd,
        cwd=str(REPO_ROOT),
        output="screen",
        emulate_tty=True,
    )
    return [
        monitor,
        RegisterEventHandler(
            OnProcessExit(target_action=monitor, on_exit=[EmitEvent(event=Shutdown())])
        ),
    ]


def generate_launch_description():
    follower_port  = LaunchConfiguration("follower_port")
    leader_port    = LaunchConfiguration("leader_port")
    camera_device  = LaunchConfiguration("camera_device")
    camera_topic   = LaunchConfiguration("camera_topic")
    camera_rate_hz = LaunchConfiguration("camera_rate_hz")
    robot_name     = LaunchConfiguration("robot_name")

    args = [
        DeclareLaunchArgument("follower_port",  default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("leader_port",    default_value="/dev/ttyACM1"),
        DeclareLaunchArgument("camera_device",  default_value="/dev/video0"),
        DeclareLaunchArgument("camera_topic",   default_value="so101real/camera/image"),
        DeclareLaunchArgument("camera_rate_hz", default_value="30"),
        DeclareLaunchArgument("robot_name",     default_value="so101real"),
        DeclareLaunchArgument(
            "classifier_checkpoint",
            default_value="runs/classifier_exp_merged/classifier_best.pt",
            description="Path to failure_head classifier checkpoint.",
        ),
        DeclareLaunchArgument(
            "dataset_root",
            default_value="data/exp_merged",
            description="Dataset root used for normalization stats (meta/stats.json).",
        ),
        DeclareLaunchArgument(
            "policy_dir",
            default_value="",
            description="Optional dir with actor.pt/critic.pt for V(s) display (default: skip).",
        ),
    ]

    teleop = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(LAUNCH_DIR / "teleop.launch.py")),
        launch_arguments={
            "follower_port":  follower_port,
            "leader_port":    leader_port,
            "camera_device":  camera_device,
            "camera_topic":   camera_topic,
            "camera_rate_hz": camera_rate_hz,
            "robot_name":     robot_name,
        }.items(),
    )

    return LaunchDescription([*args, teleop, OpaqueFunction(function=_build_monitor)])
