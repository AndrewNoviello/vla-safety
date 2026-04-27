"""
USB Camera Node

Publishes frames from a USB camera as sensor_msgs/Image on:
  so101real/camera/image   (raw BGR, 30hz)

Usage:
  python praxis/ros/examples/camera_node.py
"""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time

DEVICE     = "/dev/video2"
TOPIC      = "so101real/camera/image"
RATE_HZ    = 30
WIDTH      = 640   # downscale from 1920 — plenty for policy training
HEIGHT     = 480


class CameraNode(Node):

    def __init__(self):
        super().__init__("camera_node")

        self.cap = cv2.VideoCapture(DEVICE)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open camera {DEVICE}")
            raise RuntimeError(f"Cannot open camera {DEVICE}")

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
            f"Camera node publishing {WIDTH}x{HEIGHT} @ {RATE_HZ}hz → {TOPIC}"
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
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
