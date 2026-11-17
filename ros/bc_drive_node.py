#!/usr/bin/env python3
"""ROS 2 node that drives the car with the behavioural-cloning steering MLP.

Every control period (26 ms, the real lidar's scan period) it:
    1. takes the newest lidar scan and the scan from the previous control step,
    2. takes the forward speed and yaw rate from odometry,
    3. builds the observation with rl_driving.preprocess.make_observation (the same function
       used to build the training data, so the network sees exactly what it was trained on),
    4. runs the MLP to get a steering angle,
    5. publishes the steering angle with a fixed speed on /drive.

Speed is fixed (1.0 m/s by default) because the expert drove at a fixed 1.0 m/s, so the model
only learned steering.

Why a timer instead of reacting to every scan: the sim publishes scans every 4 ms (repeating the
same scan between physics steps), but the model was trained on pairs of scans ~26 ms apart. The
timer makes "previous scan" mean the same thing as in the training data.

Currently written for the simulator (f1tenth_gym_ros). On the real car the odometry is
sign-flipped and the IMU yaw rate is in deg/s (see CLAUDE.md), so the speed and yaw-rate inputs
would need converting first.

Run it with the ROS Python (sim_ws/.venv), not the project venv; see ros/README.md:
    python3 ros/bc_drive_node.py
    python3 ros/bc_drive_node.py --ros-args -p speed:=1.0 -p policy_path:=/abs/path/model.npz
"""
import math
import os
import sys

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

# The rl_driving package is installed in the project venv, not in the ROS Python, so import it
# straight from this repo's src/ folder. It only needs numpy.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
from rl_driving.numpy_policy import NumpyPolicy                                # noqa: E402
from rl_driving.preprocess import N_BEAMS, action_to_steer, make_observation   # noqa: E402

DEFAULT_POLICY = os.path.join(REPO_DIR, 'checkpoints', 'bc_v1.npz')


class BCDriveNode(Node):

    def __init__(self):
        super().__init__('bc_drive')
        self.declare_parameter('policy_path', DEFAULT_POLICY)
        self.declare_parameter('speed', 1.0)                 # m/s, what the expert drove at
        self.declare_parameter('control_period', 0.026)      # s, the real lidar's scan period
        self.declare_parameter('status_period', 2.0)         # s between status log lines
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('collision_topic', '/ego_racecar/collision')

        policy_path = self.get_parameter('policy_path').value
        self.policy = NumpyPolicy.load(policy_path)
        self.speed = float(self.get_parameter('speed').value)
        self.get_logger().info(f'loaded policy {policy_path}, driving at {self.speed} m/s')

        # Latest sensor values, written by the callbacks and read by the control loop.
        self.scan = None              # newest scan, (N_BEAMS,) in m
        self.prev_scan = None         # the scan used at the previous control step
        self.forward_speed = None     # m/s
        self.yaw_rate = None          # rad/s, positive = turning left
        self.position = None          # (x, y) in m, only used for the status line

        # Numbers for the status line.
        self.distance_driven = 0.0
        self.collisions = 0
        self.in_collision = False
        self.steer = 0.0

        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.on_scan, 10)
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.on_odom, 10)
        self.create_subscription(Bool, self.get_parameter('collision_topic').value, self.on_collision, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.get_parameter('drive_topic').value, 10)
        self.create_timer(self.get_parameter('control_period').value, self.control_step)
        self.create_timer(self.get_parameter('status_period').value, self.log_status)

    # ----- sensor callbacks: only store the data -----

    def on_scan(self, msg):
        if len(msg.ranges) != N_BEAMS:
            self.get_logger().error(
                f'scan has {len(msg.ranges)} beams but the model needs {N_BEAMS} '
                '(set scan_num_beams: 1081 in the sim config)', throttle_duration_sec=5.0)
            return
        self.scan = np.asarray(msg.ranges, dtype=np.float32)

    def on_odom(self, msg):
        self.forward_speed = msg.twist.twist.linear.x
        self.yaw_rate = msg.twist.twist.angular.z
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if self.position is not None:
            self.distance_driven += math.hypot(x - self.position[0], y - self.position[1])
        self.position = (x, y)

    def on_collision(self, msg):
        if msg.data and not self.in_collision:
            self.collisions += 1
            where = f' at x={self.position[0]:.2f}, y={self.position[1]:.2f}' if self.position else ''
            self.get_logger().warn(f'collision #{self.collisions}{where}')
        self.in_collision = msg.data

    # ----- control loop -----

    def control_step(self):
        if self.scan is None or self.forward_speed is None:
            return                                          # wait until both sensors have published
        if self.prev_scan is None:
            self.prev_scan = self.scan                      # very first step: no previous scan yet

        obs = make_observation(self.scan, self.prev_scan, self.forward_speed, self.yaw_rate)
        self.steer = float(action_to_steer(self.policy(obs)))
        self.prev_scan = self.scan
        self.publish_drive(self.steer, self.speed)

    def publish_drive(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def log_status(self):
        if self.position is None:
            self.get_logger().info('waiting for scan and odometry ...')
            return
        self.get_logger().info(
            f'x={self.position[0]:6.2f} y={self.position[1]:6.2f}  speed {self.forward_speed:4.2f} m/s  '
            f'steer {self.steer:+.3f} rad  driven {self.distance_driven:6.1f} m  collisions {self.collisions}')

    def stop_car(self):
        self.publish_drive(0.0, 0.0)


def main():
    # By default rclpy shuts ROS down as soon as Ctrl-C arrives, and then the final "stop" message
    # cannot be published. Without its handler, Ctrl-C is a normal KeyboardInterrupt and ROS is
    # shut down below, after the car has been told to stop.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = BCDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_car()                                     # leave the car stopped, not driving blind
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
