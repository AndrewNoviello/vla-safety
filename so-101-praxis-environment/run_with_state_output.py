import numpy as np
import pyarrow.parquet as pq
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from pathlib import Path
import time

DATA_ROOT = Path("/workspace/praxis-core/data")

# ANSI color codes
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def print_frame(i, n, cmd, label, duration, fps):
    pct     = (i + 1) / n
    bar_len = 40
    filled  = int(bar_len * pct)
    bar     = "█" * filled + "░" * (bar_len - filled)

    if label == 1.0:
        state_str = f"{GREEN}{BOLD}  ✓ GOOD  {RESET}"
        bar_color = GREEN
    else:
        state_str = f"{RED}{BOLD}  ✗ BAD   {RESET}"
        bar_color = RED

    elapsed = duration * pct
    print(
        f"\r{state_str} "
        f"{bar_color}[{bar}]{RESET} "
        f"{CYAN}{i+1:>5}/{n}{RESET} "
        f"({elapsed:.1f}s / {duration:.1f}s) "
        f"{YELLOW}{fps:.1f}hz{RESET}  "
        f"joints: [{', '.join(f'{v:6.2f}' for v in cmd)}]",
        end="",
        flush=True,
    )


def load_episode(parquet_path: Path):
    """Read a parquet episode file and return arrays for playback."""
    table = pq.read_table(parquet_path)
    df    = table.to_pydict()

    commands = np.array(
        [[df[f"action_j{i+1}"][t] for i in range(6)] for t in range(len(df["timestamp"]))],
        dtype=np.float32,
    )
    timestamps = np.array(df["timestamp"], dtype=np.float64)
    labels     = np.array(df["label"],     dtype=np.float32)

    return commands, timestamps, labels


def main():
    # ── List available experiments ─────────────────────────────────────────
    experiments = sorted(DATA_ROOT.glob("*/data")) if DATA_ROOT.exists() else []
    experiments = [p.parent for p in experiments]
    if not experiments:
        print(f"{RED}No experiments found in {DATA_ROOT}{RESET}")
        return

    print(f"\n{BOLD}Available experiments:{RESET}")
    for exp in experiments:
        episodes = sorted((exp / "data").glob("episode_*.parquet"))
        print(f"  {CYAN}{exp.name}{RESET}  ({len(episodes)} episodes)")

    experiment_id = input(f"\n{BOLD}Enter experiment ID (e.g. exp_01): {RESET}").strip()
    exp_dir       = DATA_ROOT / experiment_id
    data_dir      = exp_dir / "data"

    if not data_dir.exists():
        print(f"{RED}No data directory found for '{experiment_id}'{RESET}")
        return

    episodes = sorted(data_dir.glob("episode_*.parquet"))
    if not episodes:
        print(f"{RED}No episodes found in {data_dir}{RESET}")
        return

    # ── List available episodes ────────────────────────────────────────────
    print(f"\n{BOLD}Episodes in {experiment_id}:{RESET}")
    for ep in episodes:
        try:
            cmds, tss, lbls = load_episode(ep)
            n        = len(cmds)
            duration = float(tss[-1] - tss[0]) if n > 1 else 0.0
            fps      = n / duration if duration > 0 else 0.0
            n_good   = int(np.sum(lbls == 1.0))
            n_bad    = int(np.sum(lbls == 0.0))
            outcome  = "success" if n_bad == 0 else "failure"
            color    = GREEN if outcome == "success" else RED
            print(
                f"  {CYAN}{ep.stem}{RESET}  "
                f"{color}{BOLD}{outcome.upper()}{RESET}  "
                f"{n} frames  {duration:.1f}s @ {fps:.1f}fps  "
                f"{GREEN}{n_good} good{RESET} / {RED}{n_bad} bad{RESET}"
            )
        except Exception as e:
            print(f"  {CYAN}{ep.stem}{RESET}  {YELLOW}(could not read: {e}){RESET}")

    # ── Select episode ─────────────────────────────────────────────────────
    ep_num   = input(f"\n{BOLD}Enter episode number (e.g. 0): {RESET}").strip()
    ep_path  = data_dir / f"episode_{int(ep_num):03d}.parquet"
    if not ep_path.exists():
        print(f"{RED}Episode not found: {ep_path}{RESET}")
        return

    commands, timestamps, labels = load_episode(ep_path)

    n        = len(commands)
    duration = float(timestamps[-1] - timestamps[0]) if n > 1 else 0.0
    fps      = n / duration if duration > 0 else 0.0
    n_good   = int(np.sum(labels == 1.0))
    n_bad    = int(np.sum(labels == 0.0))

    print(f"\n{BOLD}Loaded:{RESET} {n} frames  {duration:.1f}s  "
          f"{GREEN}{n_good} good{RESET} / {RED}{n_bad} bad{RESET}")

    transitions = [i for i in range(1, n) if labels[i] != labels[i - 1]]
    if transitions:
        print(f"{YELLOW}State transitions at frames: {transitions}{RESET}")

    # ── Playback ───────────────────────────────────────────────────────────
    print(f"\n{BOLD}Make sure the arm is in its starting position.{RESET}")
    input(f"Press {BOLD}ENTER{RESET} to start playback (Ctrl+C to abort)...")

    rclpy.init()
    node = Node("playback")
    pub  = node.create_publisher(Float64MultiArray, "so101real/joint_command", 10)
    time.sleep(0.5)

    print()
    prev_label = None
    try:
        for i, (cmd, ts, label) in enumerate(zip(commands, timestamps, labels)):
            if label != prev_label and prev_label is not None:
                if label == 0.0:
                    print(f"\n{RED}{BOLD}  ── FAILURE STATE ──{RESET}")
                else:
                    print(f"\n{GREEN}{BOLD}  ── GOOD STATE ──{RESET}")
            prev_label = label

            msg      = Float64MultiArray()
            msg.data = cmd.tolist()
            pub.publish(msg)

            print_frame(i, n, cmd, label, duration, fps)

            if i < n - 1:
                dt = float(timestamps[i + 1]) - float(ts)
                time.sleep(max(0.0, dt))

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Playback interrupted.{RESET}")

    print(f"\n{GREEN}{BOLD}Done.{RESET}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
