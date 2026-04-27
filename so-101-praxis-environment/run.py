import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from pathlib import Path
import time
import json

DATA_ROOT = Path("/workspace/praxis-core/data")

def list_available(experiment_id: str):
    exp_dir = DATA_ROOT / f"exp_{experiment_id}"
    if not exp_dir.exists():
        return []
    trajs = sorted(exp_dir.glob("traj_*"), key=lambda p: int(p.name.split("_")[1]))
    return trajs

def main():
    # List available experiments
    experiments = sorted(DATA_ROOT.glob("exp_*")) if DATA_ROOT.exists() else []
    if experiments:
        print("Available experiments:")
        for exp in experiments:
            trajs = list(exp.glob("traj_*"))
            print(f"  {exp.name}  ({len(trajs)} trajectories)")
    else:
        print("No experiments found in", DATA_ROOT)
        return

    experiment_id = input("\nEnter experiment ID (e.g. 01): ").strip()
    trajs = list_available(experiment_id)
    if not trajs:
        print(f"No trajectories found for exp_{experiment_id}")
        return

    print(f"\nAvailable trajectories in exp_{experiment_id}:")
    for traj in trajs:
        meta_path = traj / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"  {traj.name}  ({meta['num_frames']} frames, {meta['duration_s']}s @ {meta['fps']}fps)")
        else:
            print(f"  {traj.name}")

    traj_num = input("\nEnter trajectory number (e.g. 0): ").strip()
    traj_dir = DATA_ROOT / f"exp_{experiment_id}" / f"traj_{traj_num}"
    if not traj_dir.exists():
        print(f"Trajectory not found: {traj_dir}")
        return

    commands   = np.load(traj_dir / "joint_commands.npy")
    timestamps = np.load(traj_dir / "timestamps.npy")

    print(f"\nLoaded {len(commands)} frames from {traj_dir}")
    print("Make sure the arm is in its starting position before continuing.")
    input("Press ENTER to start playback (Ctrl+C to abort)...")

    rclpy.init()
    node = Node("playback")
    pub  = node.create_publisher(Float64MultiArray, "so101real/joint_command", 10)
    time.sleep(0.5)  # let publisher connect

    print(f"Playing back {len(commands)} frames...")
    try:
        for i, (cmd, ts) in enumerate(zip(commands, timestamps)):
            msg = Float64MultiArray()
            msg.data = cmd.tolist()
            pub.publish(msg)
            print(f"  frame {i+1}/{len(commands)}: {np.round(cmd, 3)}", end="\r")
            if i < len(timestamps) - 1:
                dt = timestamps[i+1] - timestamps[i]
                time.sleep(max(0.0, dt))
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")

    print("\nDone.")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
