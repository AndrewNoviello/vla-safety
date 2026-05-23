"""
USB Camera Node
Publishes frames from a USB camera as sensor_msgs/Image on:
  so101real/camera/image   (raw BGR, 30 Hz by default)

Usage:
  python hardware/ros/examples/webcam.py
  python hardware/ros/examples/webcam.py --device /dev/video2
  python hardware/ros/examples/webcam.py --device /dev/video0 --rate-hz 15 --width 320 --height 240
"""

import argparse

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

DEVICE     = "/dev/video0"
TOPIC      = "so101real/camera/image"
RATE_HZ    = 30
WIDTH      = 640   # downscale from 1920 for policy training
HEIGHT     = 480

class CameraNode(Node):

    def __init__(self, device: str, topic: str, rate_hz: float, width: int, height: int):
        super().__init__("camera_node")

        self._device = device
        self._topic = topic
        self._rate_hz = rate_hz
        self._width = width
        self._height = height

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
        self._pub = self.create_publisher(Image, topic, qos)
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"Camera node publishing {width}x{height} @ {rate_hz}hz from {device} -> {topic}"
        )
    def _tick(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to grab frame")
            return

        frame = cv2.resize(frame, (self._width, self._height))

        msg              = Image()
        now              = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = "camera"
        msg.height       = self._height
        msg.width        = self._width
        msg.encoding     = "bgr8"
        msg.is_bigendian = False
        msg.step         = self._width * 3
        msg.data         = frame.tobytes()
        self._pub.publish(msg)
    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def parse_args():
    p = argparse.ArgumentParser(description="USB camera publisher for the SO101 wrist webcam.")
    p.add_argument("--device",   default=DEVICE,  help=f"V4L2 device path (default {DEVICE}).")
    p.add_argument("--topic",    default=TOPIC,   help=f"ROS topic to publish on (default {TOPIC}).")
    p.add_argument("--rate-hz",  type=float, default=RATE_HZ, help=f"Publish rate in Hz (default {RATE_HZ}).")
    p.add_argument("--width",    type=int,   default=WIDTH,   help=f"Output frame width  (default {WIDTH}).")
    p.add_argument("--height",   type=int,   default=HEIGHT,  help=f"Output frame height (default {HEIGHT}).")
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CameraNode(
        device=args.device,
        topic=args.topic,
        rate_hz=args.rate_hz,
        width=args.width,
        height=args.height,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()


