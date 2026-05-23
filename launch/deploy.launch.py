"""deploy.launch.py — hardware bridge + PI0 deployment, with optional safety filter.

The safety filter is **opt-in**: by default the launch file runs PI0 unfiltered
(deploy.py --disable-safety). To engage the latent reach-avoid filter, pass
`enable_safety:=true`.

Launch arguments:
  follower_port        (default /dev/ttyACM0)
  camera_device        (default /dev/video0)
  camera_topic         (default so101real/camera/image)
  camera_rate_hz       (default 30)
  robot_name           (default so101real)
  pi0_model            (default AndrewNoviello/vla-safety-task-4)
  prompt               (default "pick up the middle domino...")
  enable_safety        (default false)  — set true to engage the safety filter.
  wm_checkpoint        (default outputs/dino_wm_v2/checkpoints/latest/model.pt)
  actor_checkpoint     (default outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt)
  critic_checkpoint    (default outputs/safety_ddpg/checkpoints/epoch_0015/critic.pt)
  stats_path           (default data/exp_success_v3_clean/meta/stats.json)
  epsilon              (default 0.3)
  control_hz           (default 15.0)
  device               (default cuda)
  enable_rtc           (default true)   — RTC prefix-guided denoising.
  execution_horizon    (default 10)     — refill watermark in controller ticks.
  num_inference_steps  (default 10)     — flow-matching denoising steps.
  use_amp              (default true)   — CUDA autocast for PI0.

Examples:
  # PI0 alone with RTC (default)
  ros2 launch launch/deploy.launch.py prompt:='pick up a domino'

  # PI0 with the latent safety filter
  ros2 launch launch/deploy.launch.py enable_safety:=true epsilon:=0.3

  # PI0 with async inference but no RTC guidance
  ros2 launch launch/deploy.launch.py enable_rtc:=false
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

DEFAULT_PROMPT = (
    "pick up the middle domino from the three domino row and place it flat "
    "on top of the other two dominos to form an arch"
)


def _build_deploy(context, *args, **kwargs):
    pi0_model           = LaunchConfiguration("pi0_model").perform(context)
    prompt              = LaunchConfiguration("prompt").perform(context)
    enable_safety       = LaunchConfiguration("enable_safety").perform(context).lower()
    wm_checkpoint       = LaunchConfiguration("wm_checkpoint").perform(context)
    actor_checkpoint    = LaunchConfiguration("actor_checkpoint").perform(context)
    critic_checkpoint   = LaunchConfiguration("critic_checkpoint").perform(context)
    stats_path          = LaunchConfiguration("stats_path").perform(context)
    epsilon             = LaunchConfiguration("epsilon").perform(context)
    control_hz          = LaunchConfiguration("control_hz").perform(context)
    device              = LaunchConfiguration("device").perform(context)
    enable_rtc          = LaunchConfiguration("enable_rtc").perform(context).lower()
    execution_horizon   = LaunchConfiguration("execution_horizon").perform(context)
    num_inference_steps = LaunchConfiguration("num_inference_steps").perform(context)
    use_amp             = LaunchConfiguration("use_amp").perform(context).lower()

    safety_on = enable_safety in ("true", "1", "yes", "on")
    rtc_on    = enable_rtc    in ("true", "1", "yes", "on")
    amp_on    = use_amp       in ("true", "1", "yes", "on")

    cmd = [
        "python", str(REPO_ROOT / "scripts" / "deploy.py"),
        "--pi0-model",           pi0_model,
        "--prompt",              prompt,
        "--device",              device,
        "--control-hz",          control_hz,
        "--execution-horizon",   execution_horizon,
        "--num-inference-steps", num_inference_steps,
    ]

    cmd.append("--rtc" if rtc_on else "--no-rtc")
    cmd.append("--use-amp" if amp_on else "--no-amp")

    if safety_on:
        cmd.extend([
            "--wm-checkpoint",     wm_checkpoint,
            "--actor-checkpoint",  actor_checkpoint,
            "--critic-checkpoint", critic_checkpoint,
            "--stats-path",        stats_path,
            "--epsilon",           epsilon,
        ])
    else:
        cmd.append("--disable-safety")

    deploy = ExecuteProcess(
        name="so101_pi0_deploy",
        cmd=cmd,
        cwd=str(REPO_ROOT),
        output="screen",
        emulate_tty=True,
    )
    return [
        deploy,
        RegisterEventHandler(
            OnProcessExit(target_action=deploy, on_exit=[EmitEvent(event=Shutdown())])
        ),
    ]


def generate_launch_description():
    follower_port  = LaunchConfiguration("follower_port")
    camera_device  = LaunchConfiguration("camera_device")
    camera_topic   = LaunchConfiguration("camera_topic")
    camera_rate_hz = LaunchConfiguration("camera_rate_hz")
    robot_name     = LaunchConfiguration("robot_name")

    args = [
        DeclareLaunchArgument("follower_port",  default_value="/dev/ttyACM0"),
        DeclareLaunchArgument("camera_device",  default_value="/dev/video0"),
        DeclareLaunchArgument("camera_topic",   default_value="so101real/camera/image"),
        DeclareLaunchArgument("camera_rate_hz", default_value="30"),
        DeclareLaunchArgument("robot_name",     default_value="so101real"),
        DeclareLaunchArgument(
            "pi0_model", default_value="AndrewNoviello/vla-safety-task-4",
            description="HuggingFace model ID for PI0.",
        ),
        DeclareLaunchArgument(
            "prompt", default_value=DEFAULT_PROMPT,
            description="Natural-language task prompt for PI0.",
        ),
        DeclareLaunchArgument(
            "enable_safety", default_value="false",
            description="If true, gate PI0's actions through the latent safety filter.",
        ),
        DeclareLaunchArgument(
            "wm_checkpoint",
            default_value="outputs/dino_wm_v2/checkpoints/latest/model.pt",
            description="DINO-WM checkpoint (used only when enable_safety is true).",
        ),
        DeclareLaunchArgument(
            "actor_checkpoint",
            default_value="outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt",
            description="Safety actor checkpoint (used only when enable_safety is true).",
        ),
        DeclareLaunchArgument(
            "critic_checkpoint",
            default_value="outputs/safety_ddpg/checkpoints/epoch_0015/critic.pt",
            description="Safety critic checkpoint (used only when enable_safety is true).",
        ),
        DeclareLaunchArgument(
            "stats_path",
            default_value="data/exp_success_v3_clean/meta/stats.json",
            description="Dataset stats JSON for normalization.",
        ),
        DeclareLaunchArgument(
            "epsilon", default_value="0.3",
            description="Safety value threshold (override only when enable_safety is true).",
        ),
        DeclareLaunchArgument(
            "control_hz", default_value="15.0",
            description="Control loop rate.",
        ),
        DeclareLaunchArgument(
            "device", default_value="cuda",
            description="Torch device for PI0 + safety filter.",
        ),
        DeclareLaunchArgument(
            "enable_rtc", default_value="true",
            description="Enable real-time chunking (RTC) prefix-guided denoising.",
        ),
        DeclareLaunchArgument(
            "execution_horizon", default_value="10",
            description="Refill watermark in controller ticks (RTC execution horizon).",
        ),
        DeclareLaunchArgument(
            "num_inference_steps", default_value="10",
            description="Number of flow-matching denoising steps per PI0 chunk.",
        ),
        DeclareLaunchArgument(
            "use_amp", default_value="true",
            description="Enable CUDA autocast (mixed precision) for PI0 inference.",
        ),
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

    return LaunchDescription([*args, hardware, OpaqueFunction(function=_build_deploy)])
