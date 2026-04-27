from dataclasses import dataclass, field
from typing import Optional, get_args
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from std_msgs.msg import Float64MultiArray
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from pathlib import Path


class Leader(Node):
    def __init__(self):
        super().__init__(node_name="rowlet_leader")
        self.pub_real = self.create_publisher(msg_type=Float64MultiArray, topic="so101real/joint_command", qos_profile=10)

        # self.pub_sim = self.create_publisher(msg_type=Float64MultiArray, topic="so101sim/joint_command", qos_profile=10)        self.clock = Clock()
        leader_config = SO101LeaderConfig(id="leader_arm", port="/dev/ttyACM0")
        self.leader = SO101Leader(config=leader_config)
        self.leader.connect(calibrate=False)
        timer_period = 0.025
        self.timer = self.create_timer(timer_period, self.pub_callback)
        
    def pub_callback(self):
        msg = Float64MultiArray()        
        action = self.leader.get_action().values()
        act = [val for val in action]
        msg.data = act
        self.pub_real.publish(msg)
        # self.pub_sim.publish(msg)


def main():
    rclpy.init()
    leader = Leader()
    rclpy.spin(leader)
    leader.leader.disconnect()    
    leader.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
