"""
Bimanual Recording Script with Synchronized Data Collection

This script subscribes to joint states and sensor readings from two robots
(rowlet and piplup) and their associated sensors (SensorR and SensorL).

The script uses SensorL as the master clock (15fps) to ensure all data sources
are synchronized. When a SensorL frame arrives, it captures the latest data from
all other sources, ensuring that each timestamp has exactly one sample from
each source.

Features:
  - Synchronized data collection at 15fps
  - Live display of sensor readings
  - Experiment ID set once at startup (applies to entire session)
  - Press 's' to start/stop recording (auto-increments trajectory)
  - Automatic trajectory numbering within experiments
  - Saves joint states as .npy files (shape: [num_frames, num_joints])
  - Saves sensor images as .png files
  - Single timestamps array shared across all data sources

Workflow:
  1. Start script and enter experiment ID at prompt
  2. All recordings in this session go to exp_XX/
  3. Press 's' to start recording → saves to exp_XX/traj_0/
  4. Press 's' to stop and save
  5. Press 's' again to start new recording → saves to exp_XX/traj_1/
  6. Repeat as needed...

Save Path Structure:
  /home/Shared/data/exp_XX/traj_Y/
  - User inputs experiment ID (XX) once at startup
  - Script automatically increments trajectory number (Y) after each recording
  - Example session: exp_01/traj_0, exp_01/traj_1, exp_01/traj_2, ...

Data Structure (in each trajectory folder):
  - timestamps.npy: [num_frames] - synchronized timestamps
  - rowlet_positions.npy: [num_frames, num_joints]
  - piplup_positions.npy: [num_frames, num_joints]
  - rowlet_commands.npy: [num_frames, num_joints]
  - piplup_commands.npy: [num_frames, num_joints]
  - sensorR/frame_000000.png, frame_000001.png, ...
  - sensorL/frame_000000.png, frame_000001.png, ...
  - metadata.json: recording metadata (exp_id, traj_num, duration, fps, etc.)

Usage:
  python -m praxis.ros.examples.bimanual_recorder

Enter experiment ID at startup.
Press 's' to start/stop recording.
Press 'q' to quit.
"""

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from datetime import datetime
from pathlib import Path


class BimanualRecorder(Node):
    """ROS2 node to record bimanual robot data and sensor readings."""
    
    def __init__(self, experiment_id: int):
        """
        Initialize the recorder node.
        
        Args:
            experiment_id: Experiment ID for all recordings in this session
        """
        super().__init__('bimanual_recorder')
        
        # Set experiment ID for this session
        self.session_experiment_id = experiment_id
        
        # Image dimensions for sensors
        self.imgh = 240
        self.imgw = 320
        
        # Storage for latest data (used for synchronization)
        self.latest_rowlet_joint_state = None
        self.latest_piplup_joint_state = None
        self.latest_rowlet_joint_command = None
        self.latest_piplup_joint_command = None
        self.latest_sensorR_image = None
        self.latest_sensorL_image = None
        
        # Recording state
        self.is_recording = False
        self.recording_data = {
            'timestamps': [],           # Synchronized timestamps (one per frame)
            'rowlet_positions': [],     # Synchronized rowlet positions
            'piplup_positions': [],     # Synchronized piplup positions
            'rowlet_commands': [],     # Synchronized rowlet joint commands
            'piplup_commands': [],      # Synchronized piplup joint commands
            'sensorR_images': [],        # Synchronized sensorR images
            'sensorL_images': [],        # Synchronized sensorL images
        }
        self.recording_count = 0
        self.recording_start_time = None
        self.sync_frame_count = 0  # Count of synchronized frames
        self.current_trajectory_num = None  # Trajectory number for current recording
        self.next_trajectory_num = self._get_next_trajectory_number()  # Next available trajectory
        
        # QoS profile for subscriptions
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribe to joint states
        self.rowlet_joint_sub = self.create_subscription(
            JointState,
            '/rowlet/joint_state',
            self._rowlet_joint_callback,
            qos
        )
        
        self.piplup_joint_sub = self.create_subscription(
            JointState,
            '/piplup/joint_state',
            self._piplup_joint_callback,
            qos
        )
        
        # Subscribe to joint commands
        self.rowlet_joint_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/rowlet/joint_command',
            self._rowlet_joint_command_callback,
            qos
        )
        
        self.piplup_joint_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/piplup/joint_command',
            self._piplup_joint_command_callback,
            qos
        )
        
        # Subscribe to sensor readings
        self.sensorR_sub = self.create_subscription(
            Float64MultiArray,
            '/sensorR/sensor_reading',
            self._sensorR_callback,
            qos
        )
        
        self.sensorL_sub = self.create_subscription(
            Float64MultiArray,
            '/sensorL/sensor_reading',
            self._sensorL_callback,
            qos
        )
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Bimanual Recorder initialized")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Experiment ID: {self.session_experiment_id}")
        self.get_logger().info(f"Next trajectory: exp_{self.session_experiment_id:02d}/traj_{self.next_trajectory_num}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Subscribed to:")
        self.get_logger().info("  - /rowlet/joint_state")
        self.get_logger().info("  - /piplup/joint_state")
        self.get_logger().info("  - /rowlet/joint_command")
        self.get_logger().info("  - /piplup/joint_command")
        self.get_logger().info("  - /sensorR/sensor_reading")
        self.get_logger().info("  - /sensorL/sensor_reading")
        self.get_logger().info("")
        self.get_logger().info("Press 's' to start/stop recording")
        self.get_logger().info("Press 'q' to quit")
        self.get_logger().info("=" * 60)
    
    def _rowlet_joint_callback(self, msg: JointState):
        """Callback for rowlet joint state - stores latest for synchronization."""
        self.latest_rowlet_joint_state = msg
    
    def _piplup_joint_callback(self, msg: JointState):
        """Callback for piplup joint state - stores latest for synchronization."""
        self.latest_piplup_joint_state = msg
    
    def _rowlet_joint_command_callback(self, msg: Float64MultiArray):
        """Callback for rowlet joint command - stores latest for synchronization."""
        self.latest_rowlet_joint_command = np.array(msg.data)
    
    def _piplup_joint_command_callback(self, msg: Float64MultiArray):
        """Callback for piplup joint command - stores latest for synchronization."""
        self.latest_piplup_joint_command = np.array(msg.data)
    
    def _sensorR_callback(self, msg: Float64MultiArray):
        """Callback for sensorR sensor reading - stores latest for synchronization."""
        try:
            # Convert Float64MultiArray to image
            data = np.array(msg.data, dtype=np.float64)
            expected_size = self.imgh * self.imgw * 3
            
            if len(data) == expected_size:
                image = data.reshape((self.imgh, self.imgw, 3)).astype(np.uint8)
                self.latest_sensorR_image = image
            else:
                self.get_logger().warn(
                    f"sensorR: Received {len(data)} values, expected {expected_size}"
                )
        except Exception as e:
            self.get_logger().error(f"Error processing sensorR data: {e}")
    
    def _sensorL_callback(self, msg: Float64MultiArray):
        """
        Callback for sensorL sensor reading - MASTER CLOCK for synchronization.
        
        This callback runs at 15fps and triggers synchronized recording of all data sources.
        When a sensorL frame arrives, we capture the latest data from all sources.
        """
        try:
            # Convert Float64MultiArray to image
            data = np.array(msg.data, dtype=np.float64)
            expected_size = self.imgh * self.imgw * 3
            
            if len(data) == expected_size:
                image = data.reshape((self.imgh, self.imgw, 3)).astype(np.uint8)
                self.latest_sensorL_image = image
                
                # If recording, synchronize and save all data at this timestamp
                if self.is_recording:
                    self._record_synchronized_frame()
            else:
                self.get_logger().warn(
                    f"sensorL: Received {len(data)} values, expected {expected_size}"
                )
        except Exception as e:
            self.get_logger().error(f"Error processing sensorL data: {e}")
    
    def _record_synchronized_frame(self):
        """
        Record a synchronized frame from all data sources.
        
        This is called at 15fps when sensorL frames arrive, ensuring all data
        is synchronized to the same timestamp.
        """
        # Only record if we have data from all sources
        if (self.latest_sensorL_image is None or 
            self.latest_sensorR_image is None or
            self.latest_rowlet_joint_state is None or
            self.latest_piplup_joint_state is None):
            
            # Log warning on first few frames only
            if self.sync_frame_count < 5:
                missing = []
                if self.latest_sensorL_image is None:
                    missing.append("sensorL")
                if self.latest_sensorR_image is None:
                    missing.append("sensorR")
                if self.latest_rowlet_joint_state is None:
                    missing.append("rowlet")
                if self.latest_piplup_joint_state is None:
                    missing.append("piplup")
                self.get_logger().warn(
                    f"Waiting for all data sources. Missing: {', '.join(missing)}"
                )
            return
        
        # Get current timestamp
        timestamp = self.get_clock().now().nanoseconds * 1e-9
        
        # Record synchronized data
        self.recording_data['timestamps'].append(timestamp)
        
        # Record joint states (positions only)
        self.recording_data['rowlet_positions'].append(
            np.array(self.latest_rowlet_joint_state.position)
        )
        
        self.recording_data['piplup_positions'].append(
            np.array(self.latest_piplup_joint_state.position)
        )
        
        # Record joint commands (if available)
        if self.latest_rowlet_joint_command is not None:
            self.recording_data['rowlet_commands'].append(
                self.latest_rowlet_joint_command.copy()
            )
        else:
            # Use zeros if no command received yet
            num_joints = len(self.latest_rowlet_joint_state.position)
            self.recording_data['rowlet_commands'].append(
                np.zeros(num_joints)
            )
        
        if self.latest_piplup_joint_command is not None:
            self.recording_data['piplup_commands'].append(
                self.latest_piplup_joint_command.copy()
            )
        else:
            # Use zeros if no command received yet
            num_joints = len(self.latest_piplup_joint_state.position)
            self.recording_data['piplup_commands'].append(
                np.zeros(num_joints)
            )
        
        # Record sensor images
        self.recording_data['sensorL_images'].append(self.latest_sensorL_image.copy())
        self.recording_data['sensorR_images'].append(self.latest_sensorR_image.copy())
        
        self.sync_frame_count += 1
    
    def _get_next_trajectory_number(self) -> int:
        """
        Find the next available trajectory number for the current experiment.
        
        Returns:
            int: Next available trajectory number
        """
        base_dir = Path("/home/Shared/data")
        exp_dir = base_dir / f"exp_{self.session_experiment_id:02d}"
        
        # Create experiment directory if it doesn't exist
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the next available trajectory number
        existing_trajs = list(exp_dir.glob("traj_*"))
        if not existing_trajs:
            return 0
        
        # Extract trajectory numbers and find the max
        traj_nums = []
        for traj_dir in existing_trajs:
            try:
                num = int(traj_dir.name.split('_')[1])
                traj_nums.append(num)
            except (IndexError, ValueError):
                continue
        
        return max(traj_nums) + 1 if traj_nums else 0
    
    def _get_trajectory_path(self, traj_num: int) -> Path:
        """
        Get the path for a specific trajectory in the current experiment.
        
        Args:
            traj_num: Trajectory number
            
        Returns:
            Path: Path to trajectory directory
        """
        base_dir = Path("/home/Shared/data")
        exp_dir = base_dir / f"exp_{self.session_experiment_id:02d}"
        traj_dir = exp_dir / f"traj_{traj_num}"
        return traj_dir
    
    def _toggle_recording(self):
        """Toggle recording state."""
        if not self.is_recording:
            # Start recording with the next trajectory number
            self.current_trajectory_num = self.next_trajectory_num
            save_path = self._get_trajectory_path(self.current_trajectory_num)
            
            # Start recording
            self.is_recording = True
            self.recording_start_time = datetime.now()
            self.recording_count += 1
            self.sync_frame_count = 0
            
            # Clear previous data
            for key in self.recording_data.keys():
                self.recording_data[key] = []
            
            self.get_logger().info("=" * 60)
            self.get_logger().info(
                f"RECORDING STARTED (exp_{self.session_experiment_id:02d}/traj_{self.current_trajectory_num})"
            )
            self.get_logger().info(f"Save path: {save_path}")
            self.get_logger().info("15fps synchronized recording")
            self.get_logger().info("=" * 60)
        else:
            # Stop recording and save
            self.is_recording = False
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"RECORDING STOPPED ({self.sync_frame_count} frames)")
            self.get_logger().info("=" * 60)
            self._save_recording()
            
            # Update next trajectory number for next recording
            self.next_trajectory_num += 1
            self.get_logger().info("")
            self.get_logger().info(
                f"Next recording will be: exp_{self.session_experiment_id:02d}/traj_{self.next_trajectory_num}"
            )
            self.get_logger().info("")
    
    def _save_recording(self):
        """Save synchronized recorded data to disk."""
        try:
            # Check if we have any data
            if not self.recording_data['timestamps']:
                self.get_logger().warn("No data to save!")
                return
            
            # Check if we have trajectory info
            if self.current_trajectory_num is None:
                self.get_logger().error("No trajectory number set!")
                return
            
            # Create output directory
            output_dir = self._get_trajectory_path(self.current_trajectory_num)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            num_frames = len(self.recording_data['timestamps'])
            
            self.get_logger().info(f"Saving {num_frames} synchronized frames to: {output_dir}")
            
            # Save synchronized timestamps (one per frame, shared by all data)
            np.save(
                output_dir / "timestamps.npy",
                np.array(self.recording_data['timestamps'])
            )
            
            # Save joint states as .npy files (shape: [num_frames, num_joints])
            np.save(
                output_dir / "rowlet_positions.npy",
                np.array(self.recording_data['rowlet_positions'])
            )
            
            np.save(
                output_dir / "piplup_positions.npy",
                np.array(self.recording_data['piplup_positions'])
            )
            
            # Save joint commands as .npy files (shape: [num_frames, num_joints])
            np.save(
                output_dir / "rowlet_commands.npy",
                np.array(self.recording_data['rowlet_commands'])
            )
            
            np.save(
                output_dir / "piplup_commands.npy",
                np.array(self.recording_data['piplup_commands'])
            )
            
            self.get_logger().info(
                f"  ✓ Saved {num_frames} rowlet joint positions"
            )
            self.get_logger().info(
                f"  ✓ Saved {num_frames} piplup joint positions"
            )
            self.get_logger().info(
                f"  ✓ Saved {num_frames} rowlet joint commands"
            )
            self.get_logger().info(
                f"  ✓ Saved {num_frames} piplup joint commands"
            )
            
            # Save sensor images as .png files
            sensorR_dir = output_dir / "sensorR"
            sensorR_dir.mkdir(exist_ok=True)
            
            for idx, image in enumerate(self.recording_data['sensorR_images']):
                cv2.imwrite(
                    str(sensorR_dir / f"frame_{idx:06d}.png"),
                    image
                )
            
            self.get_logger().info(
                f"  ✓ Saved {num_frames} sensorR images"
            )
            
            sensorL_dir = output_dir / "sensorL"
            sensorL_dir.mkdir(exist_ok=True)
            
            for idx, image in enumerate(self.recording_data['sensorL_images']):
                cv2.imwrite(
                    str(sensorL_dir / f"frame_{idx:06d}.png"),
                    image
                )
            
            self.get_logger().info(
                f"  ✓ Saved {num_frames} sensorL images"
            )
            
            # Calculate and save metadata
            duration = self.recording_data['timestamps'][-1] - self.recording_data['timestamps'][0]
            actual_fps = num_frames / duration if duration > 0 else 0
            
            metadata = {
                'experiment_id': self.session_experiment_id,
                'trajectory_num': self.current_trajectory_num,
                'num_frames': num_frames,
                'duration_seconds': duration,
                'actual_fps': actual_fps,
                'target_fps': 15.0,
                'recording_start': self.recording_start_time.isoformat(),
            }
            
            import json
            with open(output_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"Recording saved successfully!")
            self.get_logger().info(f"Duration: {duration:.2f}s | Frames: {num_frames} | FPS: {actual_fps:.2f}")
            self.get_logger().info("=" * 60)
            
        except Exception as e:
            self.get_logger().error(f"Error saving recording: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def display_loop(self):
        """Display live sensor readings and handle keyboard input."""
        window_name = "Bimanual Recorder - Live View"
        
        while rclpy.ok():
            # Create display image
            display_image = self._create_display_image()
            
            # Show the image
            cv2.imshow(window_name, display_image)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                self.get_logger().info("Quit requested. Exiting...")
                break
            elif key == ord('s'):
                self._toggle_recording()
            
            # Spin ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.001)
        
        cv2.destroyAllWindows()
    
    def _create_display_image(self):
        """Create a combined display image showing both sensors and status."""
        # Create base image (larger to accommodate status text)
        display_height = self.imgh + 150  # Extra space for text
        display_width = self.imgw * 2 + 30  # Two images side by side with spacing
        display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Draw sensorL image (left side)
        if self.latest_sensorL_image is not None:
            display[60:60+self.imgh, 10:10+self.imgw] = self.latest_sensorL_image
        else:
            # Placeholder for sensorL
            cv2.putText(
                display,
                "Waiting for sensorL...",
                (20, 60 + self.imgh // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),
                1
            )
        
        # Draw sensorR image (right side)
        if self.latest_sensorR_image is not None:
            display[60:60+self.imgh, 20+self.imgw:20+self.imgw*2] = self.latest_sensorR_image
        else:
            # Placeholder for sensorR
            cv2.putText(
                display,
                "Waiting for sensorR...",
                (30 + self.imgw, 60 + self.imgh // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),
                1
            )
        
        # Add labels
        cv2.putText(
            display,
            "sensorL",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        cv2.putText(
            display,
            "sensorR",
            (20 + self.imgw, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        # Add recording status
        y_offset = 60 + self.imgh + 30
        if self.is_recording:
            status_text = f"RECORDING exp_{self.session_experiment_id:02d}/traj_{self.current_trajectory_num}"
            status_color = (0, 0, 255)  # Red
            # Calculate recording duration
            duration = (datetime.now() - self.recording_start_time).total_seconds()
            duration_text = f"Frames: {self.sync_frame_count} | Duration: {duration:.1f}s | FPS: {self.sync_frame_count/duration:.1f}" if duration > 0 else "Duration: 0.0s"
        else:
            status_text = f"IDLE - exp_{self.session_experiment_id:02d}/traj_{self.next_trajectory_num} (15fps sync)"
            status_color = (0, 255, 0)  # Green
            duration_text = "Press 's' to start recording"
        
        cv2.putText(
            display,
            f"Status: {status_text}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )
        
        if duration_text:
            cv2.putText(
                display,
                duration_text,
                (10, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        # Add instructions
        cv2.putText(
            display,
            "Press 's' to start/stop recording | Press 'q' to quit",
            (10, y_offset + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
        
        # Add data status indicators
        y_offset += 90
        indicators = [
            (f"Rowlet: {'OK' if self.latest_rowlet_joint_state else 'NO DATA'}", 
             (0, 255, 0) if self.latest_rowlet_joint_state else (128, 128, 128)),
            (f"Piplup: {'OK' if self.latest_piplup_joint_state else 'NO DATA'}", 
             (0, 255, 0) if self.latest_piplup_joint_state else (128, 128, 128)),
        ]
        
        x_offset = 10
        for text, color in indicators:
            cv2.putText(
                display,
                text,
                (x_offset, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            x_offset += 150
        
        return display


def main():
    """Main function to run the bimanual recorder."""
    print("=" * 60)
    print("Bimanual Recording Script - Synchronized at 15fps")
    print("=" * 60)
    print("\nThis script records synchronized data from:")
    print("  - rowlet robot joint states")
    print("  - piplup robot joint states")
    print("  - sensorR tactile sensor")
    print("  - sensorL tactile sensor (master clock)")
    print("\nSynchronization:")
    print("  - sensorL runs at 15fps and serves as the master clock")
    print("  - At each sensorL frame, latest data from all sources is captured")
    print("  - Ensures exactly 1 sample from each source per timestamp")
    print("\nSave Path Structure:")
    print("  - Data saved to: /home/Shared/data/exp_XX/traj_Y/")
    print("  - Experiment ID is set at startup (applies to all recordings)")
    print("  - Trajectory number auto-increments with each recording")
    print("  - Example: exp_01/traj_0, exp_01/traj_1, exp_01/traj_2, ...")
    print("=" * 60)
    print("\nMake sure the following nodes are running:")
    print("  - rowlet robot node")
    print("  - piplup robot node")
    print("  - sensorR sensor node (configured at 15fps)")
    print("  - sensorL sensor node (configured at 15fps)")
    print("=" * 60)
    
    # Prompt for experiment ID at startup
    print("\n")
    while True:
        try:
            experiment_id_str = input("Enter experiment ID for this session (integer): ")
            experiment_id = int(experiment_id_str)
            
            # Confirm with user
            print(f"\nAll recordings in this session will be saved to:")
            print(f"  /home/Shared/data/exp_{experiment_id:02d}/traj_X")
            confirm = input("Proceed? [Y/n]: ").strip().lower()
            
            if confirm in ['', 'y', 'yes']:
                break
            else:
                print("Let's try again...\n")
                
        except ValueError:
            print("Invalid input. Please enter an integer.\n")
        except KeyboardInterrupt:
            print("\n\nStartup cancelled. Exiting...")
            return
    
    print("\n" + "=" * 60)
    print("\nControls:")
    print("  Press 's' - Start recording")
    print("  Press 's' - Stop recording and save (auto-increments trajectory)")
    print("  Press 'q' - Quit application")
    print("=" * 60)
    print("\nInitializing recorder...")
    print()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create recorder node with experiment ID
        recorder = BimanualRecorder(experiment_id)
        
        # Run display loop
        recorder.display_loop()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        rclpy.shutdown()
        print("Recorder stopped.")


if __name__ == '__main__':
    main()

