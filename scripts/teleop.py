"""Leader teleoperator — republishes the SO101 leader arm's joint positions
as joint commands for the follower on so101real/joint_command at 40 Hz.

Usage:
  python scripts/teleop.py
  python scripts/teleop.py --leader-port /dev/ttyACM3
"""

import argparse

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import Float64MultiArray
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader


DEFAULT_LEADER_PORT = "/dev/ttyACM1"


class Leader(Node):
    def __init__(self, leader_port: str = DEFAULT_LEADER_PORT):
        super().__init__(node_name="rowlet_leader")
        self.pub_real = self.create_publisher(
            msg_type=Float64MultiArray,
            topic="so101real/joint_command",
            qos_profile=10,
        )
        self.clock = Clock()
        leader_config = SO101LeaderConfig(id="leader_arm", port=leader_port)
        self.leader = SO101Leader(config=leader_config)
        self.leader.connect(calibrate=False)
        self.get_logger().info(f"Leader connected on {leader_port}")
        timer_period = 0.025
        self.timer = self.create_timer(timer_period, self.pub_callback)

    def pub_callback(self):
        msg = Float64MultiArray()
        action = self.leader.get_action().values()
        msg.data = [val for val in action]
        self.pub_real.publish(msg)


def parse_args():
    p = argparse.ArgumentParser(description="SO101 leader-to-follower teleop bridge.")
    p.add_argument(
        "--leader-port",
        default=DEFAULT_LEADER_PORT,
        help=f"Serial port for the SO101 leader arm (default {DEFAULT_LEADER_PORT}).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    leader = Leader(leader_port=args.leader_port)
    try:
        rclpy.spin(leader)
    except KeyboardInterrupt:
        pass
    finally:
        leader.leader.disconnect()
        leader.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
