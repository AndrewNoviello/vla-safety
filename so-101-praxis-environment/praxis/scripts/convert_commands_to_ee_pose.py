"""
Convert joint commands to end-effector poses.

This script converts all rowlet_commands.npy files in trajectory directories
to end-effector poses using MuJoCo forward kinematics.

Usage:
    python -m praxis.scripts.convert_commands_to_ee_pose --exp 9
    python -m praxis.scripts.convert_commands_to_ee_pose --base-dir /home/Shared/data/exp_09
    python -m praxis.scripts.convert_commands_to_ee_pose --traj-dir /home/Shared/data/exp_09/traj_01
"""

import argparse
import numpy as np
from pathlib import Path
import mujoco
from mujoco import MjData, MjSpec
from praxis import MODELS_DIR
import os

# SO101 MuJoCo model configuration
SO101_XML_PATH = os.path.join(MODELS_DIR, "so101/mjcf/so101_new_calib.xml")
SO101_EE_SITE_NAME = "gripperframe"
SO101_DOF = 6


class JointToEEPoseConverter:
    """Converter from joint positions (percentage) to end-effector poses."""
    
    def __init__(self):
        """Initialize MuJoCo model for forward kinematics."""
        self._spec = MjSpec.from_file(SO101_XML_PATH)
        self._model = self._spec.compile()
        self._data = MjData(self._model)
        
        # Get end-effector site ID
        self._ee_site_id = mujoco.mj_name2id(
            self._model, 
            mujoco.mjtObj.mjOBJ_SITE, 
            SO101_EE_SITE_NAME
        )
        assert self._ee_site_id >= 0, f"End-effector site '{SO101_EE_SITE_NAME}' not found"
        
        # Get joint limits for percentage conversion
        self._lower_bound = self._model.jnt_range[:SO101_DOF, 0]
        self._upper_bound = self._model.jnt_range[:SO101_DOF, 1]
        
        print(f"Initialized converter with joint limits:")
        print(f"  Lower: {self._lower_bound}")
        print(f"  Upper: {self._upper_bound}")
    
    def _percentage_to_radians(self, q_percentage: np.ndarray) -> np.ndarray:
        """Convert joint positions from percentage format to radians."""
        q_rad = np.zeros(SO101_DOF)
        gripper_id = SO101_DOF - 1
        
        for i in range(SO101_DOF - 1):
            q_rad[i] = self._lower_bound[i] + \
                      ((q_percentage[i] + 100) / 200) * \
                      (self._upper_bound[i] - self._lower_bound[i])
        
        q_rad[gripper_id] = self._lower_bound[gripper_id] + \
                           (q_percentage[gripper_id] / 100) * \
                           (self._upper_bound[gripper_id] - self._lower_bound[gripper_id])
        
        return q_rad
    
    def convert_joints_to_ee_pose(self, q_percentage: np.ndarray) -> np.ndarray:
        """
        Convert joint positions (percentage) to end-effector pose with gripper.
        
        Args:
            q_percentage: Joint positions in percentage format [num_joints] or [N, num_joints]
            
        Returns:
            EE pose as [x, y, z, qw, qx, qy, qz, gripper] or [N, 8] if batch input
        """
        # Handle both single and batch inputs
        is_batch = q_percentage.ndim == 2
        if not is_batch:
            q_percentage = q_percentage[np.newaxis, :]
        
        num_frames = q_percentage.shape[0]
        ee_poses = np.zeros((num_frames, 8), dtype=np.float64)
        
        for i in range(num_frames):
            # Convert percentage to radians
            q_rad = self._percentage_to_radians(q_percentage[i])
            
            # Update MuJoCo model
            self._data.qpos[:SO101_DOF] = q_rad
            mujoco.mj_forward(self._model, self._data)
            
            # Get EE pose (position + orientation)
            ee_poses[i, :3] = self._data.site(self._ee_site_id).xpos
            mat = self._data.site(self._ee_site_id).xmat
            mujoco.mju_mat2Quat(ee_poses[i, 3:7], mat)
            
            # Add gripper joint value (last joint, in percentage format)
            ee_poses[i, 7] = q_percentage[i, SO101_DOF - 1]
        
        if not is_batch:
            return ee_poses[0]
        return ee_poses


def convert_trajectory(traj_dir: Path, converter: JointToEEPoseConverter, overwrite: bool = False):
    """
    Convert joint commands to EE poses for a single trajectory.
    
    Args:
        traj_dir: Path to trajectory directory
        converter: JointToEEPoseConverter instance
        overwrite: Whether to overwrite existing files
        
    Returns:
        bool: True if conversion was successful
    """
    commands_file = traj_dir / "rowlet_commands.npy"
    output_file = traj_dir / "rowlet_ee_commands.npy"
    
    if not commands_file.exists():
        print(f"  ⚠️  Skipping {traj_dir.name}: {commands_file} not found")
        return False
    
    if output_file.exists() and not overwrite:
        print(f"  ⏭️  Skipping {traj_dir.name}: {output_file} already exists (use --overwrite to replace)")
        return False
    
    try:
        # Load joint commands
        joint_commands = np.load(commands_file)
        print(f"  📂 {traj_dir.name}: Loaded {joint_commands.shape}")
        
        # Validate shape
        if joint_commands.ndim != 2:
            print(f"  ❌ {traj_dir.name}: Expected 2D array [num_frames, num_joints], got shape {joint_commands.shape}")
            return False
        
        if joint_commands.shape[1] != SO101_DOF:
            print(f"  ❌ {traj_dir.name}: Expected {SO101_DOF} joints, got {joint_commands.shape[1]}")
            return False
        
        # Convert to EE poses
        print(f"  🔄 Converting {joint_commands.shape[0]} frames...", end=" ", flush=True)
        ee_poses = converter.convert_joints_to_ee_pose(joint_commands)
        print("✓")
        
        # Save EE poses (shape: [num_frames, 8] where 8 = [x, y, z, qw, qx, qy, qz, gripper])
        np.save(output_file, ee_poses)
        print(f"  💾 Saved to {output_file.name} (shape: {ee_poses.shape})")
        print(f"     Format: [x, y, z, qw, qx, qy, qz, gripper_percentage]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {traj_dir.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert joint commands to end-effector poses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--exp",
        type=int,
        help="Experiment ID (e.g., 9 for exp_09)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/Shared/data",
        help="Base directory containing experiments (default: /home/Shared/data)"
    )
    
    parser.add_argument(
        "--traj-dir",
        type=str,
        help="Specific trajectory directory to convert (overrides --exp)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rowlet_ee_poses.npy files"
    )
    
    args = parser.parse_args()
    
    # Initialize converter
    print("=" * 60)
    print("Joint Commands to EE Pose Converter")
    print("=" * 60)
    converter = JointToEEPoseConverter()
    print()
    
    # Determine which trajectories to process
    if args.traj_dir:
        # Single trajectory
        traj_dir = Path(args.traj_dir)
        if not traj_dir.exists():
            print(f"Error: Trajectory directory does not exist: {traj_dir}")
            return
        
        print(f"Processing single trajectory: {traj_dir}")
        print("=" * 60)
        success = convert_trajectory(traj_dir, converter, args.overwrite)
        print("=" * 60)
        if success:
            print("✓ Conversion completed successfully")
        else:
            print("✗ Conversion failed")
            
    elif args.exp is not None:
        # All trajectories in experiment
        exp_dir = Path(args.base_dir) / f"exp_{args.exp:02d}"
        if not exp_dir.exists():
            print(f"Error: Experiment directory does not exist: {exp_dir}")
            return
        
        trajectories = sorted(exp_dir.glob("traj_*"))
        if not trajectories:
            print(f"No trajectories found in {exp_dir}")
            return
        
        print(f"Processing experiment: exp_{args.exp:02d}")
        print(f"Found {len(trajectories)} trajectories")
        print("=" * 60)
        
        successful = 0
        failed = 0
        skipped = 0
        
        for traj_dir in trajectories:
            output_file = traj_dir / "rowlet_ee_poses.npy"
            result = convert_trajectory(traj_dir, converter, args.overwrite)
            if result:
                successful += 1
            elif output_file.exists() and not args.overwrite:
                skipped += 1
            else:
                failed += 1
            print()
        
        print("=" * 60)
        print(f"Summary:")
        print(f"  ✓ Successful: {successful}")
        print(f"  ⏭️  Skipped: {skipped}")
        print(f"  ❌ Failed: {failed}")
        print(f"  Total: {len(trajectories)}")
        
    else:
        parser.error("Either --exp or --traj-dir must be provided")


if __name__ == "__main__":
    main()

