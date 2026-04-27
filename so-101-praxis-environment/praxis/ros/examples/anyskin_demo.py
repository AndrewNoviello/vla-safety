"""
AnySkin ROS subscriber visualization demo.

This script subscribes to the AnySkin sensor ROS topic and displays
the tactile magnetometer data using pygame visualization.

Controls:
  - Press 'b' key to recalibrate baseline
  - Close window to quit
"""

import argparse
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame


class AnySkinVisualizer(Node):
    """ROS2 node to visualize AnySkin sensor data."""
    
    def __init__(self, namespace: str, topic: str, viz_mode: str = "3axis", scaling: float = 7.0):
        """
        Initialize the visualizer node.
        
        Args:
            namespace: ROS namespace for the sensor
            topic: Topic name to subscribe to
            viz_mode: Visualization mode ("magnitude" or "3axis")
            scaling: Scaling factor for visualization
        """
        super().__init__('anyskin_visualizer')
        
        self.viz_mode = viz_mode
        self.scaling = scaling
        self.latest_data = None
        self.baseline = np.zeros(15)  # 5 magnetometers * 3 axes
        self.frame_count = 0
        self.baseline_samples = []
        self.recalibrating = False
        
        # QoS profile for subscription
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribe to sensor reading topic
        self.subscription = self.create_subscription(
            Float64MultiArray,
            f'/{namespace}/{topic}',
            self._sensor_callback,
            qos
        )
        
        # Setup pygame visualization
        self._setup_pygame()
        
        self.get_logger().info(f"Subscribed to /{namespace}/{topic}")
        self.get_logger().info(f"Visualization mode: {viz_mode}, scaling: {scaling}")
        self.get_logger().info("Press 'b' key to recalibrate baseline")
    
    def _setup_pygame(self):
        """Setup pygame window and visualization elements."""
        pygame.init()
        
        # Load background image
        dir_path = os.path.dirname(os.path.realpath(__file__))
        bg_image_path = os.path.join(dir_path, "data/viz_bg.png")
        
        # Check if background image exists, otherwise create a blank one
        if os.path.exists(bg_image_path):
            bg_image = pygame.image.load(bg_image_path)
        else:
            # Create a simple blank background
            bg_image = pygame.Surface((408, 348))
            bg_image.fill((234, 237, 232))
        
        image_width, image_height = bg_image.get_size()
        aspect_ratio = image_height / image_width
        desired_width = 400
        desired_height = int(desired_width * aspect_ratio)
        
        # Chip locations on the sensor (pixel coordinates)
        self.chip_locations = np.array([
            [204, 222],  # center
            [130, 222],  # left
            [279, 222],  # right
            [204, 157],  # up
            [204, 290],  # down
        ])
        
        # Rotation angles for each chip's x-y plane
        self.chip_xy_rotations = np.array([
            -np.pi / 2,  # center
            -np.pi / 2,  # left
            np.pi,       # right
            np.pi / 2,   # up
            0.0          # down
        ])
        
        # Resize background and create window
        bg_image = pygame.transform.scale(bg_image, (desired_width, desired_height))
        self.window = pygame.display.set_mode((desired_width, desired_height), pygame.SRCALPHA)
        
        # Create background surface
        self.background_surface = pygame.Surface(self.window.get_size(), pygame.SRCALPHA)
        background_color = (234, 237, 232, 255)
        self.background_surface.fill(background_color)
        self.background_surface.blit(bg_image, (0, 0))
        
        pygame.display.set_caption("AnySkin Sensor Data Visualization")
    
    def _sensor_callback(self, msg: Float64MultiArray):
        """
        Callback for sensor reading messages.
        
        Args:
            msg: Float64MultiArray message containing flattened magnetometer data
        """
        try:
            # Convert Float64MultiArray back to numpy array
            data = np.array(msg.data, dtype=np.float64)
            
            # Expected size: 5 magnetometers * 3 axes = 15 values
            # Plus potential timestamp as first element
            if len(data) == 16:
                # Remove timestamp if present
                data = data[1:]
            elif len(data) != 15:
                self.get_logger().warn(
                    f"Received data size {len(data)} does not match expected "
                    f"size 15. Expected 5 magnetometers * 3 axes."
                )
                return
            
            self.latest_data = data
            self.frame_count += 1
            
            # Collect samples for baseline calibration
            if self.recalibrating:
                self.baseline_samples.append(data)
                if len(self.baseline_samples) >= 5:
                    self.baseline = np.mean(self.baseline_samples, axis=0)
                    self.get_logger().info("Baseline recalibrated")
                    self.recalibrating = False
                    self.baseline_samples = []
            
        except Exception as e:
            self.get_logger().error(f"Error processing sensor data: {e}")
    
    def _visualize_data(self, data: np.ndarray):
        """
        Visualize the magnetometer data using pygame.
        
        Args:
            data: Numpy array of shape (15,) containing 5 magnetometers * 3 axes
        """
        # Reshape to (5, 3) for 5 magnetometers with x, y, z components
        data = data.reshape(-1, 3)
        data_mag = np.linalg.norm(data, axis=1)
        
        # Draw the chip locations
        for magid, chip_location in enumerate(self.chip_locations):
            if self.viz_mode == "magnitude":
                # Draw circle with radius proportional to magnitude
                pygame.draw.circle(
                    self.window, 
                    (255, 83, 72), 
                    chip_location.astype(int), 
                    int(data_mag[magid] / self.scaling)
                )
            elif self.viz_mode == "3axis":
                # Draw z-axis as circle (filled if positive, outline if negative)
                if data[magid, -1] < 0:
                    width = 2  # outline
                else:
                    width = 0  # filled
                
                pygame.draw.circle(
                    self.window,
                    (255, 0, 0),
                    chip_location.astype(int),
                    int(np.abs(data[magid, -1]) / self.scaling),
                    width
                )
                
                # Draw x-y components as arrow
                arrow_start = chip_location
                rotation_mat = np.array([
                    [np.cos(self.chip_xy_rotations[magid]), -np.sin(self.chip_xy_rotations[magid])],
                    [np.sin(self.chip_xy_rotations[magid]), np.cos(self.chip_xy_rotations[magid])]
                ])
                data_xy = np.dot(rotation_mat, data[magid, :2])
                arrow_end = (
                    int(chip_location[0] + data_xy[0] / self.scaling),
                    int(chip_location[1] + data_xy[1] / self.scaling)
                )
                pygame.draw.line(
                    self.window, 
                    (0, 255, 0), 
                    arrow_start.astype(int), 
                    arrow_end,
                    2
                )
    
    def display_loop(self):
        """Display data in a loop until user quits."""
        clock = pygame.time.Clock()
        FPS = 60
        running = True
        
        while running and rclpy.ok():
            # Redraw background
            self.window.blit(self.background_surface, (0, 0))
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.get_logger().info("User closed window. Exiting...")
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_b:
                        # Start baseline recalibration
                        self.get_logger().info("Recalibrating baseline...")
                        self.recalibrating = True
                        self.baseline_samples = []
            
            # Visualize latest data
            if self.latest_data is not None:
                # Subtract baseline
                adjusted_data = self.latest_data - self.baseline
                self._visualize_data(adjusted_data)
                
                # Display frame count
                font = pygame.font.Font(None, 24)
                text = font.render(f"Frame: {self.frame_count}", True, (0, 0, 0))
                self.window.blit(text, (10, 10))
                
                # Print data to console
                print(adjusted_data)
            else:
                # Display waiting message
                font = pygame.font.Font(None, 36)
                text = font.render("Waiting for sensor data...", True, (100, 100, 100))
                text_rect = text.get_rect(center=(self.window.get_width() // 2, 
                                                   self.window.get_height() // 2))
                self.window.blit(text, text_rect)
            
            # Update display
            pygame.display.update()
            clock.tick(FPS)
            
            # Spin ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.001)
        
        pygame.quit()


def main():
    """Main function to run the AnySkin visualizer."""
    parser = argparse.ArgumentParser(
        description="Visualize AnySkin sensor data from ROS topics"
    )
    parser.add_argument(
        '--sensor', '-s', 
        type=str, 
        default='anyskin',
        help='ROS namespace for the sensor (default: anyskin)'
    )
    parser.add_argument(
        '--topic', '-t', 
        type=str, 
        default='sensor_reading',
        help='Topic name to subscribe to (default: sensor_reading)'
    )
    parser.add_argument(
        '--viz_mode', '-v',
        type=str,
        default='3axis',
        choices=['magnitude', '3axis'],
        help='Visualization mode (default: 3axis)'
    )
    parser.add_argument(
        '--scaling', '-c',
        type=float,
        default=7.0,
        help='Scaling factor for visualization (default: 7.0)'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("AnySkin ROS Visualizer")
    print("=" * 60)
    print(f"Namespace: {args.sensor}")
    print(f"Topic: /{args.sensor}/{args.topic}")
    print(f"Visualization mode: {args.viz_mode}")
    print(f"Scaling: {args.scaling}")
    print("=" * 60)
    print("\nMake sure the sensor node is running:")
    print("  python -m praxis.ros.scripts.sensor_node_launcher")
    print("=" * 60)
    print("\nControls:")
    print("  - Press 'b' to recalibrate baseline")
    print("  - Close window to quit")
    print("=" * 60)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create visualizer node
        visualizer = AnySkinVisualizer(
            namespace=args.sensor,
            topic=args.topic,
            viz_mode=args.viz_mode,
            scaling=args.scaling
        )
        
        # Run display loop
        visualizer.display_loop()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        rclpy.shutdown()
        print("Visualizer stopped.")


if __name__ == '__main__':
    main()