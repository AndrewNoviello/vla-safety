"""
GelSight Mini ROS subscriber test script.

This script subscribes to the GelSight Mini sensor ROS topic and displays
the tactile images using OpenCV.

Requirements:
  - The sensor node must be running (praxis.ros.scripts.sensor_node_launcher)

Usage:
  python -m praxis.ros.examples.gsmini_test
  python -m praxis.ros.examples.gsmini_test --namespace gsmini
  python -m praxis.ros.examples.gsmini_test --topic sensor_reading

Press any key in the image window to quit.
"""

import argparse
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray


class GSMiniVisualizer(Node):
    """ROS2 node to visualize GelSight Mini sensor data."""
    
    def __init__(self, namespace: str, topic: str, imgh: int, imgw: int):
        """
        Initialize the visualizer node.
        
        Args:
            namespace: ROS namespace for the sensor
            topic: Topic name to subscribe to
            imgh: Image height
            imgw: Image width
        """
        super().__init__('gsmini_visualizer')
        
        self.imgh = imgh
        self.imgw = imgw
        self.frame_count = 0
        self.latest_image = None
        
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
        
        self.get_logger().info(f"Subscribed to /{namespace}/{topic}")
        self.get_logger().info(f"Expected image size: {imgw}x{imgh}")
        self.get_logger().info("Press any key in the image window to quit.")
    
    def _sensor_callback(self, msg: Float64MultiArray):
        """
        Callback for sensor reading messages.
        
        Args:
            msg: Float64MultiArray message containing flattened image data
        """
        try:
            # Convert Float64MultiArray back to numpy array
            data = np.array(msg.data, dtype=np.float64)
            
            # Reshape to image dimensions (height, width, channels)
            expected_size = self.imgh * self.imgw * 3
            
            if len(data) != expected_size:
                self.get_logger().warn(
                    f"Received data size {len(data)} does not match expected "
                    f"size {expected_size}. Skipping frame."
                )
                return
            
            # Reshape and convert to uint8
            image = data.reshape((self.imgh, self.imgw, 3)).astype(np.uint8)
            
            self.latest_image = image
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error processing sensor data: {e}")
    
    def display_loop(self):
        """Display images in a loop until user quits."""
        window_name = "GelSight Mini - ROS Viewer"
        
        while rclpy.ok():
            if self.latest_image is not None:
                # Add frame counter and info to image
                display_image = self.latest_image.copy()
                
                cv2.putText(
                    display_image, 
                    f"Frame: {self.frame_count}", 
                    (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
                
                cv2.putText(
                    display_image, 
                    f"Size: {self.imgw}x{self.imgh}", 
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    1
                )
                
                cv2.imshow(window_name, display_image)
            else:
                # Show waiting message
                waiting_img = np.zeros((self.imgh, self.imgw, 3), dtype=np.uint8)
                cv2.putText(
                    waiting_img, 
                    "Waiting for sensor data...", 
                    (10, self.imgh // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                cv2.imshow(window_name, waiting_img)
            
            # Check for key press (1ms wait)
            key = cv2.waitKey(1)
            if key != -1:
                self.get_logger().info("User pressed key. Exiting...")
                break
            
            # Spin ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.001)
        
        cv2.destroyAllWindows()


def main():
    """Main function to run the GelSight Mini visualizer."""
    parser = argparse.ArgumentParser(
        description="Visualize GelSight Mini sensor data from ROS topics"
    )
    parser.add_argument(
        '--sensor', '-s', 
        type=str, 
        default='gsmini',
        help='ROS namespace for the sensor (default: gsmini)'
    )
    parser.add_argument(
        '--topic', '-t', 
        type=str, 
        default='sensor_reading',
        help='Topic name to subscribe to (default: sensor_reading)'
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=240,
        help='Image height (default: 240)'
    )
    parser.add_argument(
        '--width', '-w', 
        type=int, 
        default=320,
        help='Image width (default: 320)'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("GelSight Mini ROS Visualizer")
    print("=" * 60)
    print(f"Namespace: {args.sensor}")
    print(f"Topic: /{args.sensor}/{args.topic}")
    print(f"Image size: {args.width}x{args.height}")
    print("=" * 60)
    print("\nMake sure the sensor node is running:")
    print("  python -m praxis.ros.scripts.sensor_node_launcher")
    print("=" * 60)
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create visualizer node
        visualizer = GSMiniVisualizer(
            namespace=args.sensor,
            topic=args.topic,
            imgh=args.height,
            imgw=args.width
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

