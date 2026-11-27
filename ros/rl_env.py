"""Gymnasium environment around the ROS 2 F1TENTH simulator (f1tenth_gym_ros).

This is where gymnasium is used: F1TenthROSEnv subclasses gym.Env, so any Gymnasium-compatible
algorithm (Stable-Baselines3 PPO here) can train on it through reset() / step().

Run it with the ROS Python (sim_ws/.venv): it needs rclpy, gymnasium, numpy.

The sim must already be running, with scan_num_beams: 1081. Do NOT run the safety node,
drive_arbiter, or bc_drive_node while training: this env is the only thing publishing on /drive.

Assumptions (check against your code):
  * action is 1-D in [-1, 1] and rl_driving.preprocess.action_to_steer() maps it to radians
  * speed is fixed (steering-only policy, like the BC expert at 1.0 m/s)
  * /initialpose teleports the car in the sim and the collision flag clears afterwards
"""
import math
import os
import sys
import threading
import time

import gymnasium as gym
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from gymnasium import spaces
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_DIR, 'src'))
from rl_driving.preprocess import N_BEAMS, OBS_DIM, action_to_steer, make_observation  # noqa: E402


class SimBridge(Node):
    """Stores the newest sensor data; publishes drive commands and reset poses."""

    def __init__(self, scan_topic, odom_topic, drive_topic, collision_topic):
        super().__init__('rl_bridge')
        self.scan = None
        self.forward_speed = None
        self.yaw_rate = None
        self.pos = None
        self.collided = False
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(Bool, collision_topic, self._on_collision, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    def _on_scan(self, msg):
        if len(msg.ranges) == N_BEAMS:
            self.scan = np.asarray(msg.ranges, dtype=np.float32)

    def _on_odom(self, msg):
        self.forward_speed = msg.twist.twist.linear.x
        self.yaw_rate = msg.twist.twist.angular.z
        self.pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _on_collision(self, msg):
        self.collided = bool(msg.data)

    def drive(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def teleport(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        self.pose_pub.publish(msg)


class F1TenthROSEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, start_poses, centerline=None, speed=1.0, dt=0.026, max_steps=2000,
                 scan_topic='/scan', odom_topic='/ego_racecar/odom', drive_topic='/drive',
                 collision_topic='/ego_racecar/collision'):
        super().__init__()
        self.start_poses = np.asarray(start_poses, dtype=np.float64).reshape(-1, 3)  # x, y, yaw
        self.speed, self.dt, self.max_steps = speed, dt, max_steps

        # Progress reward needs a centerline (M, 2) in track order; without it, reward survival.
        self.cl = None if centerline is None else np.asarray(centerline, dtype=np.float64)
        if self.cl is not None:
            seg = np.linalg.norm(np.diff(self.cl, axis=0), axis=1)
            self.cl_s = np.r_[0.0, np.cumsum(seg)]
            self.lap_len = self.cl_s[-1]

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

        if not rclpy.ok():
            rclpy.init()
        self.bridge = SimBridge(scan_topic, odom_topic, drive_topic, collision_topic)
        threading.Thread(target=rclpy.spin, args=(self.bridge,), daemon=True).start()
        self._wait_for(lambda: self.bridge.scan is not None and self.bridge.forward_speed is not None,
                       timeout=15.0, what='scan and odometry (is the sim running, with 1081 beams?)')

        self.t = 0
        self.last_scan = None
        self.last_s = 0.0

    # ----- helpers -----

    def _wait_for(self, cond, timeout, what):
        t0 = time.time()
        while not cond():
            if time.time() - t0 > timeout:
                raise TimeoutError(f'timed out waiting for {what}')
            time.sleep(0.01)

    def _progress_s(self):
        d = np.linalg.norm(self.cl - np.asarray(self.bridge.pos), axis=1)
        return self.cl_s[int(np.argmin(d))]

    def _obs(self):
        b = self.bridge
        scan = b.scan.copy()
        obs = make_observation(scan, self.last_scan, b.forward_speed, b.yaw_rate)
        self.last_scan = scan
        return obs

    # ----- gymnasium API -----

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        b = self.bridge
        b.drive(0.0, 0.0)
        x, y, yaw = self.start_poses[self.np_random.integers(len(self.start_poses))]
        b.teleport(x, y, yaw)
        time.sleep(0.3)                                          # let the sim apply the pose
        self._wait_for(lambda: not b.collided, timeout=3.0,
                       what='collision flag to clear after reset (does /initialpose move the car?)')
        self.t = 0
        self.last_scan = b.scan.copy()                           # first step: previous scan = current
        if self.cl is not None:
            self.last_s = self._progress_s()
        return self._obs(), {}

    def step(self, action):
        b = self.bridge
        steer = float(action_to_steer(np.asarray(action, dtype=np.float32)))
        b.drive(steer, self.speed)
        time.sleep(self.dt)                                      # real-time pacing, one control period
        self.t += 1

        if self.cl is not None:
            s = self._progress_s()
            ds = s - self.last_s
            if ds < -self.lap_len / 2:                           # crossed the start line
                ds += self.lap_len
            elif ds > self.lap_len / 2:                          # went backwards over it
                ds -= self.lap_len
            self.last_s = s
            reward = ds
        else:
            reward = 0.1                                         # survival bonus per step
        reward -= 0.01 * abs(steer)                              # discourage jittery steering

        terminated = bool(b.collided)
        if terminated:
            reward -= 10.0
        truncated = self.t >= self.max_steps                     # time limit, not a failure
        return self._obs(), float(reward), terminated, truncated, {}

    def close(self):
        try:
            self.bridge.drive(0.0, 0.0)
        finally:
            self.bridge.destroy_node()