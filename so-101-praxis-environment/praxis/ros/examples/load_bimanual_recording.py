"""
Helper script to load and verify synchronized bimanual recordings.

This script demonstrates how to load the synchronized data recorded by
bimanual_recorder.py and verify that all data sources have the same
number of frames.

Usage:
  # List all experiments
  python -m praxis.ros.examples.load_bimanual_recording --list
  
  # List trajectories in an experiment
  python -m praxis.ros.examples.load_bimanual_recording --list --exp 1
  
  # Load specific trajectory
  python -m praxis.ros.examples.load_bimanual_recording --exp 1 --traj 0
  
  # Or use full path
  python -m praxis.ros.examples.load_bimanual_recording /home/Shared/data/exp_01/traj_0
  
  # Play back recording
  python -m praxis.ros.examples.load_bimanual_recording --exp 1 --traj 0 --play
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import json


def list_experiments(base_dir: str = "/home/Shared/data"):
    """
    List all experiments in the data directory.
    
    Args:
        base_dir: Base directory containing experiments
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Data directory does not exist: {base_path}")
        return
    
    experiments = sorted(base_path.glob("exp_*"))
    
    if not experiments:
        print("No experiments found.")
        return
    
    print("=" * 60)
    print("Available Experiments:")
    print("=" * 60)
    
    for exp_dir in experiments:
        try:
            exp_num = int(exp_dir.name.split('_')[1])
            trajectories = sorted(exp_dir.glob("traj_*"))
            num_trajs = len(trajectories)
            
            print(f"\nExperiment {exp_num:02d}: {exp_dir.name}")
            print(f"  Path: {exp_dir}")
            print(f"  Trajectories: {num_trajs}")
            
            # Show first few trajectory names
            if trajectories:
                traj_names = [t.name for t in trajectories[:5]]
                if num_trajs > 5:
                    traj_names.append("...")
                print(f"  └─ {', '.join(traj_names)}")
                
        except (IndexError, ValueError):
            continue
    
    print("=" * 60)


def list_trajectories(experiment_id: int, base_dir: str = "/home/Shared/data"):
    """
    List all trajectories in an experiment.
    
    Args:
        experiment_id: Experiment ID number
        base_dir: Base directory containing experiments
    """
    exp_dir = Path(base_dir) / f"exp_{experiment_id:02d}"
    
    if not exp_dir.exists():
        print(f"Experiment directory does not exist: {exp_dir}")
        return
    
    trajectories = sorted(exp_dir.glob("traj_*"))
    
    if not trajectories:
        print(f"No trajectories found in experiment {experiment_id:02d}.")
        return
    
    print("=" * 60)
    print(f"Trajectories in Experiment {experiment_id:02d}:")
    print("=" * 60)
    
    for traj_dir in trajectories:
        try:
            traj_num = int(traj_dir.name.split('_')[1])
            
            # Load metadata if available
            metadata_path = traj_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                num_frames = metadata.get('num_frames', '?')
                duration = metadata.get('duration_seconds', '?')
                fps = metadata.get('actual_fps', '?')
                
                print(f"\nTrajectory {traj_num}: {traj_dir.name}")
                print(f"  Frames: {num_frames}")
                print(f"  Duration: {duration:.2f}s" if isinstance(duration, (int, float)) else f"  Duration: {duration}")
                print(f"  FPS: {fps:.2f}" if isinstance(fps, (int, float)) else f"  FPS: {fps}")
            else:
                print(f"\nTrajectory {traj_num}: {traj_dir.name}")
                print(f"  (no metadata)")
                
        except (IndexError, ValueError):
            continue
    
    print("=" * 60)


def load_recording(recording_path: str):
    """
    Load a synchronized bimanual recording.
    
    Args:
        recording_path: Path to the recording directory
        
    Returns:
        dict: Dictionary containing all loaded data
    """
    recording_dir = Path(recording_path)
    
    if not recording_dir.exists():
        raise ValueError(f"Recording directory does not exist: {recording_dir}")
    
    print(f"Loading recording from: {recording_dir}")
    print("=" * 60)
    
    # Load metadata
    metadata_path = recording_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        print("=" * 60)
    else:
        metadata = None
        print("No metadata.json found")
    
    # Load timestamps
    timestamps = np.load(recording_dir / "timestamps.npy")
    num_frames = len(timestamps)
    
    print(f"\nLoaded {num_frames} synchronized frames")
    print(f"Timestamps shape: {timestamps.shape}")
    
    # Load joint states (positions only - velocities and efforts removed)
    rowlet_positions = np.load(recording_dir / "rowlet_positions.npy")
    piplup_positions = np.load(recording_dir / "piplup_positions.npy")
    
    # Load joint commands (if available)
    rowlet_commands = None
    piplup_commands = None
    if (recording_dir / "rowlet_commands.npy").exists():
        rowlet_commands = np.load(recording_dir / "rowlet_commands.npy")
    if (recording_dir / "piplup_commands.npy").exists():
        piplup_commands = np.load(recording_dir / "piplup_commands.npy")
    
    # Try to load velocities/efforts if they exist (for backward compatibility with old recordings)
    rowlet_velocities = None
    rowlet_efforts = None
    piplup_velocities = None
    piplup_efforts = None
    
    if (recording_dir / "rowlet_velocities.npy").exists():
        rowlet_velocities = np.load(recording_dir / "rowlet_velocities.npy")
    if (recording_dir / "rowlet_efforts.npy").exists():
        rowlet_efforts = np.load(recording_dir / "rowlet_efforts.npy")
    if (recording_dir / "piplup_velocities.npy").exists():
        piplup_velocities = np.load(recording_dir / "piplup_velocities.npy")
    if (recording_dir / "piplup_efforts.npy").exists():
        piplup_efforts = np.load(recording_dir / "piplup_efforts.npy")
    
    print(f"\nJoint States:")
    print(f"  Rowlet positions: {rowlet_positions.shape}")
    if rowlet_commands is not None:
        print(f"  Rowlet commands: {rowlet_commands.shape}")
    if rowlet_velocities is not None:
        print(f"  Rowlet velocities: {rowlet_velocities.shape} (legacy)")
    if rowlet_efforts is not None:
        print(f"  Rowlet efforts: {rowlet_efforts.shape} (legacy)")
    print(f"  Piplup positions: {piplup_positions.shape}")
    if piplup_commands is not None:
        print(f"  Piplup commands: {piplup_commands.shape}")
    if piplup_velocities is not None:
        print(f"  Piplup velocities: {piplup_velocities.shape} (legacy)")
    if piplup_efforts is not None:
        print(f"  Piplup efforts: {piplup_efforts.shape} (legacy)")
    
    # Count sensor images (check both lowercase and capitalized versions for compatibility)
    sensorR_dir = recording_dir / "sensorR"
    sensorL_dir = recording_dir / "sensorL"
    if not sensorR_dir.exists():
        sensorR_dir = recording_dir / "SensorR"
    if not sensorL_dir.exists():
        sensorL_dir = recording_dir / "SensorL"
    
    sensorR_images = sorted(sensorR_dir.glob("frame_*.png")) if sensorR_dir.exists() else []
    sensorL_images = sorted(sensorL_dir.glob("frame_*.png")) if sensorL_dir.exists() else []
    
    print(f"\nSensor Images:")
    print(f"  SensorR images: {len(sensorR_images)}")
    print(f"  SensorL images: {len(sensorL_images)}")
    
    # Verify synchronization
    print("\n" + "=" * 60)
    print("Verification:")
    
    all_counts = {
        'timestamps': len(timestamps),
        'rowlet_positions': len(rowlet_positions),
        'piplup_positions': len(piplup_positions),
        'sensorR_images': len(sensorR_images),
        'sensorL_images': len(sensorL_images),
    }
    
    # Add optional commands if they exist
    if rowlet_commands is not None:
        all_counts['rowlet_commands'] = len(rowlet_commands)
    if piplup_commands is not None:
        all_counts['piplup_commands'] = len(piplup_commands)
    
    # Add optional velocities/efforts if they exist
    if rowlet_velocities is not None:
        all_counts['rowlet_velocities'] = len(rowlet_velocities)
    if rowlet_efforts is not None:
        all_counts['rowlet_efforts'] = len(rowlet_efforts)
    if piplup_velocities is not None:
        all_counts['piplup_velocities'] = len(piplup_velocities)
    if piplup_efforts is not None:
        all_counts['piplup_efforts'] = len(piplup_efforts)
    
    # Check if all have the same count
    counts = set(all_counts.values())
    if len(counts) == 1:
        print(f"✓ All data sources have {num_frames} frames (SYNCHRONIZED)")
    else:
        print("✗ Data sources have different frame counts:")
        for name, count in all_counts.items():
            status = "✓" if count == num_frames else "✗"
            print(f"  {status} {name}: {count}")
    
    print("=" * 60)
    
    # Return loaded data
    rowlet_data = {'positions': rowlet_positions}
    piplup_data = {'positions': piplup_positions}
    
    # Add optional commands if they exist
    if rowlet_commands is not None:
        rowlet_data['commands'] = rowlet_commands
    if piplup_commands is not None:
        piplup_data['commands'] = piplup_commands
    
    # Add optional velocities/efforts if they exist
    if rowlet_velocities is not None:
        rowlet_data['velocities'] = rowlet_velocities
    if rowlet_efforts is not None:
        rowlet_data['efforts'] = rowlet_efforts
    if piplup_velocities is not None:
        piplup_data['velocities'] = piplup_velocities
    if piplup_efforts is not None:
        piplup_data['efforts'] = piplup_efforts
    
    return {
        'metadata': metadata,
        'timestamps': timestamps,
        'rowlet': rowlet_data,
        'piplup': piplup_data,
        'sensorR_image_paths': sensorR_images,
        'sensorL_image_paths': sensorL_images,
        'recording_dir': recording_dir,
    }


def visualize_frame(data: dict, frame_idx: int):
    """
    Visualize a single synchronized frame.
    
    Args:
        data: Dictionary returned by load_recording()
        frame_idx: Frame index to visualize
    """
    num_frames = len(data['timestamps'])
    if frame_idx < 0 or frame_idx >= num_frames:
        raise ValueError(f"Frame index {frame_idx} out of range [0, {num_frames})")
    
    # Load images
    sensorL_img = cv2.imread(str(data['sensorL_image_paths'][frame_idx]))
    sensorR_img = cv2.imread(str(data['sensorR_image_paths'][frame_idx]))
    
    # Create combined display
    display = np.hstack([sensorL_img, sensorR_img])
    
    # Add frame information
    timestamp = data['timestamps'][frame_idx]
    cv2.putText(
        display,
        f"Frame {frame_idx}/{num_frames-1} | Time: {timestamp:.3f}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    # Add joint positions
    rowlet_pos = data['rowlet']['positions'][frame_idx]
    piplup_pos = data['piplup']['positions'][frame_idx]
    
    cv2.putText(
        display,
        f"Rowlet: [{', '.join([f'{p:.2f}' for p in rowlet_pos[:3]])}...]",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        display,
        f"Piplup: [{', '.join([f'{p:.2f}' for p in piplup_pos[:3]])}...]",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return display


def play_recording(data: dict, fps: float = 15.0):
    """
    Play back a synchronized recording.
    
    Args:
        data: Dictionary returned by load_recording()
        fps: Playback frame rate (default: 15.0)
    """
    num_frames = len(data['timestamps'])
    delay_ms = int(1000 / fps)
    
    print(f"\nPlaying back recording at {fps} fps")
    print("Press 'q' to quit, space to pause, 'n' for next frame")
    
    frame_idx = 0
    paused = False
    
    while frame_idx < num_frames:
        display = visualize_frame(data, frame_idx)
        
        if paused:
            cv2.putText(
                display,
                "PAUSED",
                (display.shape[1] // 2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )
        
        cv2.imshow("Bimanual Recording Playback", display)
        
        key = cv2.waitKey(delay_ms if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n'):
            frame_idx += 1
        elif not paused:
            frame_idx += 1
    
    cv2.destroyAllWindows()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Load and verify synchronized bimanual recordings"
    )
    parser.add_argument(
        'recording_path',
        type=str,
        nargs='?',
        help='Path to the recording directory (optional if using --exp and --traj)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available experiments (or trajectories if --exp is specified)'
    )
    parser.add_argument(
        '--exp',
        type=int,
        help='Experiment ID number'
    )
    parser.add_argument(
        '--traj',
        type=int,
        help='Trajectory number within experiment'
    )
    parser.add_argument(
        '--play',
        action='store_true',
        help='Play back the recording'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=15.0,
        help='Playback frame rate (default: 15.0)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/home/Shared/data',
        help='Base directory for experiments (default: /home/Shared/data)'
    )
    
    args = parser.parse_args()
    
    try:
        # Handle list command
        if args.list:
            if args.exp is not None:
                list_trajectories(args.exp, args.base_dir)
            else:
                list_experiments(args.base_dir)
            return
        
        # Determine recording path
        if args.recording_path:
            recording_path = args.recording_path
        elif args.exp is not None and args.traj is not None:
            recording_path = f"{args.base_dir}/exp_{args.exp:02d}/traj_{args.traj}"
        else:
            parser.error("Either provide recording_path or both --exp and --traj")
            return
        
        # Load the recording
        data = load_recording(recording_path)
        
        # Play if requested
        if args.play:
            play_recording(data, fps=args.fps)
        else:
            print("\nUse --play to visualize the recording")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

