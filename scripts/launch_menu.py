"""Interactive launcher for the vla-safety ROS pipelines.

Wraps the launch files under <repo>/launch/ with an interactive prompt that
shows defaults, validates obvious things (port present in /dev/, checkpoint
files exist), prints the equivalent `ros2 launch` command, and runs it.

Usage:
  python scripts/launch_menu.py              # full interactive menu
  python scripts/launch_menu.py --menu 3     # jump straight to pipeline 3
  python scripts/launch_menu.py --dry-run    # print commands without executing
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
LAUNCH_DIR = REPO_ROOT / "launch"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ─── Validators ───────────────────────────────────────────────────────────

def validate_port(value: str) -> str | None:
    if not value.startswith("/dev/"):
        return f"port should start with /dev/ (got {value!r})"
    if not Path(value).exists():
        return f"{value} not present right now (will fail unless plugged in before launch)"
    return None


def validate_path_exists(value: str) -> str | None:
    if not value:
        return None
    p = (REPO_ROOT / value) if not Path(value).is_absolute() else Path(value)
    if not p.exists():
        return f"path not found: {p}"
    return None


def validate_optional_path(value: str) -> str | None:
    if not value:
        return None  # optional, empty is fine
    return validate_path_exists(value)


def validate_bool(value: str) -> str | None:
    if value.lower() not in ("true", "false", "yes", "no", "1", "0", "on", "off"):
        return f"expected true/false (got {value!r})"
    return None


# ─── Custom runners ───────────────────────────────────────────────────────
# Most pipelines just run `ros2 launch <file> arg:=value ...` in the foreground.
# A pipeline can opt into a custom runner by setting `runner=<callable>` in its
# dict; that callable replaces the default build/display/confirm/run path.

def _run_with_background_hardware(
    hw_cmd: list[str],
    fg_cmd: list[str],
    fg_env: dict[str, str] | None = None,
) -> int:
    """Run hw_cmd as a backgrounded `ros2 launch` (stdout/stderr → log file),
    then run fg_cmd in the foreground attached to the menu's TTY. On exit
    (foreground returns or Ctrl-C), SIGINT the hardware launch's process group
    and wait for it to tear down. Use when the foreground command needs a real
    terminal (recorder.py: termios keypresses, `\\r` overwrite, input() prompts)."""
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"hardware-{int(time.time())}.log"
    print(f"{DIM}Hardware logs → {log_path}{RESET}", flush=True)

    with log_path.open("w") as log:
        hw = subprocess.Popen(
            hw_cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            start_new_session=True,  # own process group so we can SIGINT all children
        )
        print(f"{DIM}Waiting ~3 s for hardware to settle…{RESET}", flush=True)
        time.sleep(3)

        if hw.poll() is not None:
            print(f"{RED}Hardware launch exited early (rc={hw.returncode}). "
                  f"Tail of {log_path}:{RESET}")
            try:
                tail = log_path.read_text().splitlines()[-15:]
                for line in tail:
                    print(f"  {DIM}{line}{RESET}")
            except OSError:
                pass
            return hw.returncode or 1

        rc = 0
        try:
            rc = subprocess.run(
                fg_cmd,
                cwd=str(REPO_ROOT),
                env={**os.environ, **(fg_env or {})},
            ).returncode
        except KeyboardInterrupt:
            rc = 130
        finally:
            print(f"\n{DIM}Tearing down hardware launch…{RESET}", flush=True)
            try:
                os.killpg(os.getpgid(hw.pid), signal.SIGINT)
                try:
                    hw.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"{YELLOW}Hardware didn't exit in 10 s, killing.{RESET}")
                    os.killpg(os.getpgid(hw.pid), signal.SIGKILL)
                    hw.wait(timeout=5)
            except ProcessLookupError:
                pass  # already gone
        return rc


def recorder_runner(pipeline: dict, values: dict[str, str], dry_run: bool) -> int:
    """Pipeline 3 runner: hardware/teleop in the background, recorder in the
    foreground attached to the user's TTY. recorder.py needs a real terminal
    for its `\\r` live counter, termios keypresses, and `input()` prompts —
    none of which work as a child of `ros2 launch`."""
    hw_cmd = ["ros2", "launch", str(LAUNCH_DIR / "teleop.launch.py")]
    for k in ("follower_port", "leader_port", "camera_device"):
        v = values.get(k, "")
        if v:
            hw_cmd.append(f"{k}:={v}")

    rec_cmd = ["python", str(REPO_ROOT / "scripts" / "recorder.py")]
    if values.get("experiment_id"):
        rec_cmd.extend(["--experiment-id", values["experiment_id"]])
    if values.get("task_name"):
        rec_cmd.extend(["--task-name", values["task_name"]])

    rec_env: dict[str, str] = {}
    if values.get("data_root"):
        rec_env["VLA_RECORD_DATA_ROOT"] = values["data_root"]
    if values.get("failure_extra_seconds"):
        rec_env["VLA_RECORD_FAILURE_EXTRA"] = values["failure_extra_seconds"]

    print(f"\n{DIM}Background hardware:{RESET}")
    print(f"{DIM}{' '.join(shlex.quote(c) for c in hw_cmd)}{RESET}")
    env_blurb = "  ".join(f"{k}={v}" for k, v in rec_env.items()) or "(none)"
    print(f"\n{DIM}Foreground recorder (env: {env_blurb}):{RESET}")
    print(f"{DIM}{' '.join(shlex.quote(c) for c in rec_cmd)}{RESET}\n")

    if dry_run:
        print(f"{YELLOW}--dry-run: not executing.{RESET}")
        return 0

    answer = input(f"{BOLD}Run now? [Y/n] {RESET}").strip().lower()
    if answer not in ("", "y", "yes"):
        print(f"{DIM}Not running.{RESET}")
        return 0

    print(f"{GREEN}{BOLD}▶ launching…{RESET}\n")
    return _run_with_background_hardware(hw_cmd, rec_cmd, rec_env)


def lerobot_record_runner(pipeline: dict, values: dict[str, str], dry_run: bool) -> int:
    """Pipeline runner: invoke `scripts/lerobot_record_with_stub.py`. Drives the
    SO101 follower directly via its own serial connection — does **not** go
    through the ROS bridge, so do not start the hardware launch in parallel.
    The wrapper registers no-op stubs for processor steps that exist in the
    saved checkpoint pipeline but not in upstream lerobot==0.4.4."""
    cameras = (
        f"{{front: {{type: opencv, "
        f"index_or_path: {values['camera_device']}, "
        f"width: 640, height: 480, fps: 30}}}}"
    )
    cmd = [
        "python", str(REPO_ROOT / "scripts" / "lerobot_record_with_stub.py"),
        "--robot.type=so101_follower",
        f"--robot.port={values['follower_port']}",
        "--robot.id=follower_v1",
        f"--robot.calibration_dir={REPO_ROOT}/calibration/robots/so_follower",
        f"--robot.cameras={cameras}",
        "--policy.type=pi0",
        f"--policy.pretrained_path={values['pi0_model']}",
        f"--policy.device={values['device']}",
        f"--dataset.repo_id={values['dataset_repo_id']}",
        f"--dataset.single_task={values['prompt']}",
        "--dataset.fps=20",
        f"--dataset.episode_time_s={values['episode_time_s']}",
        f"--dataset.num_episodes={values['num_episodes']}",
        "--dataset.push_to_hub=false",
        "--display_data=false",
    ]

    print(f"\n{DIM}Foreground command:{RESET}")
    print(f"{DIM}{' '.join(shlex.quote(c) for c in cmd)}{RESET}\n")

    if dry_run:
        print(f"{YELLOW}--dry-run: not executing.{RESET}")
        return 0

    answer = input(f"{BOLD}Run now? [Y/n] {RESET}").strip().lower()
    if answer not in ("", "y", "yes"):
        print(f"{DIM}Not running.{RESET}")
        return 0

    print(f"{GREEN}{BOLD}▶ launching…{RESET}\n")
    try:
        return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    except KeyboardInterrupt:
        return 130


# ─── Argument descriptors ─────────────────────────────────────────────────

class Arg:
    __slots__ = ("name", "default", "description", "validator")

    def __init__(self, name: str, default: str, description: str = "",
                 validator=None):
        self.name = name
        self.default = default
        self.description = description
        self.validator = validator


# ─── Pipelines ────────────────────────────────────────────────────────────

PIPELINE_HARDWARE = dict(
    title="Hardware only",
    blurb="Robot node + USB webcam. The base for everything else.",
    launch="hardware.launch.py",
    args=[
        Arg("follower_port",  "/dev/ttyACM0", "USB port for SO101 follower", validate_port),
        Arg("camera_device",  "/dev/video0",  "V4L2 device for the wrist webcam", validate_port),
        Arg("camera_rate_hz", "30",           "Webcam publish rate (Hz)"),
        Arg("robot_name",     "so101real",    "Robot ROS namespace"),
    ],
)

PIPELINE_TELEOP = dict(
    title="Teleop session",
    blurb="Hardware + leader teleop. Move the leader, the follower mirrors.",
    launch="teleop.launch.py",
    args=[
        Arg("follower_port",  "/dev/ttyACM0", "USB port for SO101 follower", validate_port),
        Arg("leader_port",    "/dev/ttyACM1", "USB port for SO101 leader",   validate_port),
        Arg("camera_device",  "/dev/video0",  "V4L2 device for the webcam",  validate_port),
    ],
)

PIPELINE_RECORDER = dict(
    title="Record dataset",
    blurb="Hardware + teleop + recorder. Use 's'/'b' to start, 'g'/'f'/'d' to save, 'q' to quit.",
    # Custom runner: hardware in background, recorder in the foreground attached
    # to the user's TTY (needed for the recorder's keyboard input + live display).
    runner=recorder_runner,
    args=[
        Arg("follower_port",         "/dev/ttyACM0", "USB port for SO101 follower",  validate_port),
        Arg("leader_port",           "/dev/ttyACM1", "USB port for SO101 leader",    validate_port),
        Arg("camera_device",         "/dev/video0",  "V4L2 device for the webcam",   validate_port),
        Arg("data_root",             str(REPO_ROOT / "data" / "recordings"),
                                     "Output directory for recordings"),
        Arg("experiment_id",         "",             "Experiment ID (empty: prompt in recorder)"),
        Arg("task_name",             "",             "Task description (empty: prompt in recorder)"),
        Arg("failure_extra_seconds", "5.0",          "Seconds to record after pressing 'f'"),
    ],
)

PIPELINE_SAFETY_MONITOR = dict(
    title="Live safety monitor with teleop",
    blurb="Hardware + teleop + per-tick SAFE/UNSAFE classifier readout (observe-only).",
    launch="safety_monitor.launch.py",
    args=[
        Arg("follower_port",         "/dev/ttyACM0", "USB port for SO101 follower", validate_port),
        Arg("leader_port",           "/dev/ttyACM1", "USB port for SO101 leader",   validate_port),
        Arg("camera_device",         "/dev/video0",  "V4L2 device for the webcam",  validate_port),
        Arg("classifier_checkpoint", "runs/classifier_exp_merged/classifier_best.pt",
                                     "Failure_head classifier checkpoint",         validate_path_exists),
        Arg("dataset_root",          "data/exp_merged",
                                     "Dataset dir for normalization stats"),
        Arg("policy_dir",            "",
                                     "Optional: dir with actor.pt/critic.pt to also show V(s) (empty: skip)",
                                     validate_optional_path),
    ],
)

PIPELINE_DEPLOY_NO_SAFETY = dict(
    title="Deploy PI0 (no safety filter)  ← default deploy path",
    blurb="Hardware + PI0 driving the arm directly. The safety filter is OFF.",
    launch="deploy.launch.py",
    args=[
        Arg("follower_port", "/dev/ttyACM0", "USB port for SO101 follower", validate_port),
        Arg("camera_device", "/dev/video0",  "V4L2 device for the webcam",  validate_port),
        Arg("pi0_model",     "AndrewNoviello/vla-safety-task-4",
                             "PI0 HuggingFace model ID"),
        Arg("prompt",        "pick up the middle domino from the three domino row "
                             "and place it flat on top of the other two dominos to form an arch",
                             "Natural-language task prompt"),
        Arg("control_hz",    "15.0", "Control loop rate (Hz)"),
        Arg("device",        "cuda", "Torch device"),
    ],
    fixed={"enable_safety": "false"},
)

PIPELINE_DEPLOY_SAFETY = dict(
    title="Deploy PI0 with safety filter",
    blurb="Hardware + PI0 + latent reach-avoid safety filter gating each action.",
    launch="deploy.launch.py",
    args=[
        Arg("follower_port",     "/dev/ttyACM0", "USB port for SO101 follower", validate_port),
        Arg("camera_device",     "/dev/video0",  "V4L2 device for the webcam",  validate_port),
        Arg("pi0_model",         "AndrewNoviello/vla-safety-task-4",
                                 "PI0 HuggingFace model ID"),
        Arg("prompt",            "pick up the middle domino from the three domino row "
                                 "and place it flat on top of the other two dominos to form an arch",
                                 "Natural-language task prompt"),
        Arg("wm_checkpoint",     "outputs/dino_wm_v2/checkpoints/latest/model.pt",
                                 "DINO-WM checkpoint", validate_path_exists),
        Arg("actor_checkpoint",  "outputs/safety_ddpg/checkpoints/epoch_0015/actor.pt",
                                 "Safety actor checkpoint",  validate_path_exists),
        Arg("critic_checkpoint", "outputs/safety_ddpg/checkpoints/epoch_0015/critic.pt",
                                 "Safety critic checkpoint", validate_path_exists),
        Arg("stats_path",        "data/exp_success_v3_clean/meta/stats.json",
                                 "Dataset stats JSON",       validate_path_exists),
        Arg("epsilon",           "0.3",  "Safety value threshold ε"),
        Arg("control_hz",        "15.0", "Control loop rate (Hz)"),
        Arg("device",            "cuda", "Torch device"),
    ],
    fixed={"enable_safety": "true"},
)

PIPELINE_LEROBOT_RECORD = dict(
    title="PI0 inference via lerobot-record",
    blurb="Drive the SO101 follower with PI0 using upstream lerobot-record (no ROS bridge). "
          "Records evaluation episodes to a local HF dataset.",
    runner=lerobot_record_runner,
    args=[
        Arg("follower_port",    "/dev/ttyACM0", "USB port for SO101 follower",   validate_port),
        Arg("camera_device",    "/dev/video0",  "V4L2 device for the wrist webcam", validate_port),
        Arg("pi0_model",        "AndrewNoviello/vla-safety-task-5",
                                "PI0 HuggingFace model ID"),
        Arg("prompt",           "pick up the middle domino",
                                "Natural-language task prompt"),
        Arg("dataset_repo_id",  "local/eval_pi0_dominos",
                                "HF dataset repo id (use 'local/...' for local-only, no push)"),
        Arg("episode_time_s",   "20",   "Seconds per episode"),
        Arg("num_episodes",     "1",    "Number of episodes to record"),
        Arg("device",           "cuda", "Torch device"),
    ],
)

PIPELINES = [
    PIPELINE_HARDWARE,
    PIPELINE_TELEOP,
    PIPELINE_RECORDER,
    PIPELINE_SAFETY_MONITOR,
    PIPELINE_DEPLOY_NO_SAFETY,
    PIPELINE_DEPLOY_SAFETY,
    PIPELINE_LEROBOT_RECORD,
]


# ─── UI ───────────────────────────────────────────────────────────────────

def header():
    print(f"\n{BOLD}{BLUE}vla-safety launcher{RESET}  "
          f"{DIM}repo: {REPO_ROOT}{RESET}\n")


def show_menu():
    header()
    print(f"{BOLD}Pick a pipeline:{RESET}\n")
    for i, p in enumerate(PIPELINES, start=1):
        print(f"  {BOLD}{i}.{RESET} {GREEN}{p['title']}{RESET}")
        print(f"     {DIM}{p['blurb']}{RESET}")
    print()
    print(f"  {BOLD}q.{RESET} {DIM}quit{RESET}\n")


def pick_pipeline() -> dict | None:
    while True:
        show_menu()
        choice = input(f"{BOLD}> {RESET}").strip().lower()
        if choice in ("q", "quit", "exit"):
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(PIPELINES):
            return PIPELINES[int(choice) - 1]
        print(f"{YELLOW}Please enter 1–{len(PIPELINES)} or q.{RESET}")


def prompt_for_args(pipeline: dict) -> dict[str, str]:
    print(f"\n{BOLD}{pipeline['title']}{RESET}")
    print(f"{DIM}{pipeline['blurb']}{RESET}")
    print(f"{DIM}(press Enter to accept the default in [brackets]){RESET}\n")

    values: dict[str, str] = {}
    for arg in pipeline["args"]:
        prompt = f"  {CYAN}{arg.name}{RESET}"
        if arg.description:
            prompt += f" {DIM}— {arg.description}{RESET}"
        prompt += f"\n    [{arg.default}] "
        raw = input(prompt).strip()
        value = raw if raw else arg.default
        if arg.validator:
            warning = arg.validator(value)
            if warning:
                print(f"    {YELLOW}⚠  {warning}{RESET}")
        values[arg.name] = value
    return values


def build_command(pipeline: dict, values: dict[str, str]) -> list[str]:
    cmd = ["ros2", "launch", str(LAUNCH_DIR / pipeline["launch"])]
    for arg in pipeline["args"]:
        value = values[arg.name]
        # ros2 launch rejects `name:=` (empty value) as malformed. Empty input
        # from the user means "fall back to whatever the launch file declares
        # as default", so just omit the override.
        if value == "":
            continue
        cmd.append(f"{arg.name}:={value}")
    for fixed_name, fixed_val in pipeline.get("fixed", {}).items():
        cmd.append(f"{fixed_name}:={fixed_val}")
    return cmd


def display_command(cmd: list[str]):
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n{DIM}Command:{RESET}")
    print(f"{DIM}{pretty}{RESET}\n")


def confirm_and_run(cmd: list[str], dry_run: bool):
    if dry_run:
        print(f"{YELLOW}--dry-run: not executing.{RESET}")
        return 0
    answer = input(f"{BOLD}Run now? [Y/n] {RESET}").strip().lower()
    if answer in ("", "y", "yes"):
        print(f"{GREEN}{BOLD}▶ launching…{RESET}\n")
        return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
    print(f"{DIM}Not running. You can paste the command above to run it later.{RESET}")
    return 0


# ─── Entry point ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--menu", type=int, default=None,
                   help=f"Jump straight to a pipeline number (1–{len(PIPELINES)}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the assembled command without running it.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.menu is not None:
        if not (1 <= args.menu <= len(PIPELINES)):
            print(f"{RED}--menu must be 1–{len(PIPELINES)}{RESET}")
            sys.exit(2)
        pipeline = PIPELINES[args.menu - 1]
    else:
        pipeline = pick_pipeline()
        if pipeline is None:
            print(f"{DIM}bye{RESET}")
            return

    values = prompt_for_args(pipeline)
    if "runner" in pipeline:
        rc = pipeline["runner"](pipeline, values, args.dry_run)
    else:
        cmd = build_command(pipeline, values)
        display_command(cmd)
        rc = confirm_and_run(cmd, dry_run=args.dry_run)
    sys.exit(rc)


if __name__ == "__main__":
    main()
