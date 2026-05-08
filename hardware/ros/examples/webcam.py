"""
USB Camera Node
Publishes frames from a USB camera as sensor_msgs/Image on:
  so101real/camera/image   (raw BGR, 30hz)
Usage:
  python hardware/ros/examples/webcam.py [--device /dev/video0]
"""
import argparse
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

TOPIC      = "so101real/camera/image"
RATE_HZ    = 30
WIDTH      = 640   # downscale from 1920 for policy training
HEIGHT     = 480

class CameraNode(Node):
    def __init__(self, device="/dev/video0"):
        super().__init__("camera_node")
        self.device = device
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open camera {device}")
            raise RuntimeError(f"Cannot open camera {device}")
        # request native resolution; we'll downscale in software
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(Image, TOPIC, qos)
        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.get_logger().info(
            f"Camera node ({device}) publishing {WIDTH}x{HEIGHT} @ {RATE_HZ}hz -> {TOPIC}"
        )
    def _tick(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return
        # downscale to 640x480 for storage / training efficiency
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        msg              = Image()
        now              = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = "camera"
        msg.height       = HEIGHT
        msg.width        = WIDTH
        msg.encoding     = "bgr8"
        msg.is_bigendian = False
        msg.step         = WIDTH * 3
        msg.data         = frame.tobytes()
        self._pub.publish(msg)
    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser(description="USB Camera ROS2 publisher")
    parser.add_argument(
        "--device",
        default="/dev/video0",
        help="Video device path (e.g. /dev/video0) or index (e.g. 0). Default: /dev/video0",
    )
    # parse_known_args so ROS args (--ros-args ...) pass through cleanly
    args, ros_args = parser.parse_known_args()

    # Allow either a device path or a numeric index
    device = int(args.device) if args.device.isdigit() else args.device

    rclpy.init(args=ros_args)
    node = CameraNode(device=device)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


