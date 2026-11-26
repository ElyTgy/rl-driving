#!/usr/bin/env python3
"""ROS 2 node that drives the car with the behavioural-cloning steering MLP.

Every control period (26 ms, the real lidar's scan period) it:
    1. takes the newest lidar scan and the scan from the previous control step,
    2. takes the forward speed and yaw rate (from odometry in the sim, odometry + IMU on the car),
    3. builds the observation with rl_driving.preprocess.make_observation (the same function
       used to build the training data, so the network sees exactly what it was trained on),
    4. runs the MLP to get a steering angle and a speed,
    5. publishes them, with the speed capped at the `max_speed` parameter.

Why a timer instead of reacting to every scan: the model was trained on pairs of scans ~26 ms
apart. The sim publishes scans every 4 ms (repeating the same scan between physics steps), and the
timer keeps "previous scan" meaning the same thing on the sim and the car.

Modes (parameter `mode`):
    sim  odometry on /ego_racecar/odom with standard signs; yaw rate from odometry (no IMU in the sim)
    car  odometry on /odom, sign-flipped; yaw rate from the IMU on /sensors/imu/raw, in deg/s.
         Converted with rl_driving.car_signals, the same functions the training data went through.

Where the commands go (parameter `drive_topic`):
    The car must be driven through the drive_arbiter, which is what applies the safety node's
    slow-downs and stops. ros/bc_drive_launch.py starts the
    safety node and arbiter and points this node at /gap_follow_drive (an arbiter input). In car
    mode the node refuses to publish straight to /drive, because that would bypass the AEB.
    Publishing to /drive is only allowed in the sim, for quick tests without the arbiter.

Run it with the ROS Python, not the project venv (see ros/README.md):
    ros2 launch ros/bc_drive_launch.py mode:=car          # car: safety node + arbiter + this node
    python3 ros/bc_drive_node.py                         # sim only, straight to /drive
"""
import math
import os
import sys

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool

# The rl_driving package is installed in the project venv, not in the ROS Python, so import it
# straight from this repo's src/ folder. It only needs numpy.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
from rl_driving.car_signals import speed_from_car_odom, yaw_rate_from_car_imu  # noqa: E402
from rl_driving.numpy_policy import NumpyPolicy                                # noqa: E402
from rl_driving.preprocess import N_BEAMS, action_to_speed, action_to_steer, make_observation  # noqa: E402

DEFAULT_POLICY = os.path.join(REPO_DIR, 'checkpoints', 'bc_v2.npz')

# Topics that differ between the simulator and the real car.
TOPICS = {
    'sim': {'odom': '/ego_racecar/odom', 'imu': None, 'collision': '/ego_racecar/collision'},
    'car': {'odom': '/odom', 'imu': '/sensors/imu/raw', 'collision': None},
}


class BCDriveNode(Node):

    def __init__(self):
        super().__init__('bc_drive')
        self.declare_parameter('mode', 'sim')                # 'sim' or 'car', see the module docstring
        self.declare_parameter('policy_path', DEFAULT_POLICY)
        self.declare_parameter('max_speed', 1.0)             # m/s, cap on the model's speed (expert: 1.0-1.8)
        self.declare_parameter('control_period', 0.026)      # s, the real lidar's scan period
        self.declare_parameter('status_period', 2.0)         # s between status log lines
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')      # the launch file sets an arbiter input

        self.mode = self.get_parameter('mode').value
        if self.mode not in TOPICS:
            raise ValueError(f"mode must be 'sim' or 'car', got {self.mode!r}")
        topics = TOPICS[self.mode]
        drive_topic = self.get_parameter('drive_topic').value
        if self.mode == 'car' and drive_topic.lstrip('/') == 'drive':
            raise ValueError('on the car, publishing straight to /drive would bypass the drive_arbiter '
                             'and with it the AEB. Start the node with ros/bc_drive_launch.py instead.')

        policy_path = self.get_parameter('policy_path').value
        self.policy = NumpyPolicy.load(policy_path)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.get_logger().info(f'mode {self.mode}: loaded {policy_path}, speed capped at {self.max_speed} m/s, '
                               f'commands on {drive_topic}')

        # Latest sensor values, written by the callbacks and read by the control loop.
        self.scan = None              # newest scan, (N_BEAMS,) in m
        self.prev_scan = None         # the scan used at the previous control step
        self.forward_speed = None     # m/s, positive driving forward
        self.yaw_rate = None          # rad/s, positive turning left
        self.position = None          # (x, y) in m, only used for the status line

        # Numbers for the status line.
        self.distance_driven = 0.0
        self.collisions = 0
        self.in_collision = False
        self.aeb_seen = False
        self.steer = 0.0
        self.speed = 0.0

        self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.on_scan, 10)
        self.create_subscription(Odometry, topics['odom'], self.on_odom, 10)
        if topics['imu']:
            # Same QoS as the data recorder used for the IMU.
            self.create_subscription(Imu, topics['imu'], self.on_imu, qos_profile_sensor_data)
        if topics['collision']:
            self.create_subscription(Bool, topics['collision'], self.on_collision, 10)
        # Only to report it in the log; the arbiter is what actually stops the car.
        self.create_subscription(AckermannDriveStamped, '/safety_drive', self.on_safety_drive, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.create_timer(self.get_parameter('control_period').value, self.control_step)
        self.create_timer(self.get_parameter('status_period').value, self.log_status)

    # ----- sensor callbacks: convert to the model's units and store -----

    def on_scan(self, msg):
        if len(msg.ranges) != N_BEAMS:
            self.get_logger().error(
                f'scan has {len(msg.ranges)} beams but the model needs {N_BEAMS} '
                '(in the sim, set scan_num_beams: 1081 in the sim config)', throttle_duration_sec=5.0)
            return
        self.scan = np.asarray(msg.ranges, dtype=np.float32)

    def on_odom(self, msg):
        vx = msg.twist.twist.linear.x
        if self.mode == 'car':
            self.forward_speed = speed_from_car_odom(vx)
        else:
            self.forward_speed = vx
            self.yaw_rate = msg.twist.twist.angular.z     # the sim has no IMU; its odometry is exact

        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if self.position is not None:
            self.distance_driven += math.hypot(x - self.position[0], y - self.position[1])
        self.position = (x, y)

    def on_imu(self, msg):
        self.yaw_rate = yaw_rate_from_car_imu(msg.angular_velocity.z)

    def on_collision(self, msg):
        if msg.data and not self.in_collision:
            self.collisions += 1
            where = f' at x={self.position[0]:.2f}, y={self.position[1]:.2f}' if self.position else ''
            self.get_logger().warn(f'collision #{self.collisions}{where}')
        self.in_collision = msg.data

    def on_safety_drive(self, msg):
        if not self.aeb_seen:
            self.aeb_seen = True
            self.get_logger().warn('safety node fired (MS5: it slows or stops the car, then lets it drive on)')

    # ----- control loop -----

    def missing_inputs(self):
        names = {'scan': self.scan, 'speed (odometry)': self.forward_speed,
                 'yaw rate (IMU)' if self.mode == 'car' else 'yaw rate (odometry)': self.yaw_rate}
        return [name for name, value in names.items() if value is None]

    def control_step(self):
        if self.missing_inputs():
            return                                          # never drive on partial inputs
        if self.prev_scan is None:
            self.prev_scan = self.scan                      # very first step: no previous scan yet

        obs = make_observation(self.scan, self.prev_scan, self.forward_speed, self.yaw_rate)
        steer_action, speed_action = self.policy(obs)
        self.steer = float(action_to_steer(steer_action))
        self.speed = min(float(action_to_speed(speed_action)), self.max_speed)
        self.prev_scan = self.scan
        self.publish_drive(self.steer, self.speed)

    def publish_drive(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.drive_pub.publish(msg)

    def log_status(self):
        missing = self.missing_inputs()
        if missing:
            self.get_logger().info(f'waiting for: {", ".join(missing)}')
            return
        self.get_logger().info(
            f'speed {self.forward_speed:4.2f} m/s  yaw rate {self.yaw_rate:+.2f} rad/s  '
            f'steer {self.steer:+.3f} rad  cmd speed {self.speed:4.2f} m/s  driven {self.distance_driven:6.1f} m'
            + (f'  collisions {self.collisions}' if self.mode == 'sim' else '')
            + ('  [AEB stopped]' if self.aeb_seen else ''))

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
