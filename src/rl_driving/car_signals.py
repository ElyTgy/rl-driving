"""Conversions from the real car's raw sensor values to the units the model uses.

The real car has two verified sensor quirks (see CLAUDE.md, checked on all recorded runs):
    - odometry is sign-flipped: linear.x is negative while driving forward
    - the IMU yaw rate (angular_velocity.z) is in deg/s, not rad/s (its sign is correct)

Both the training data (cleaning.calibrate_signals) and the ROS node on the car
(ros/bc_drive_node.py) convert through these functions, so the model sees the same values in
training and on the car. Works on single numbers and numpy arrays. The simulator needs neither
conversion: its odometry has standard signs and rad/s.
"""
import math


def speed_from_car_odom(odom_vx):
    """Odometry linear.x (sign-flipped) -> forward speed in m/s, positive when driving forward."""
    return -odom_vx


def yaw_rate_from_car_imu(imu_wz_deg):
    """IMU angular_velocity.z in deg/s -> yaw rate in rad/s, positive when turning left."""
    return imu_wz_deg * math.pi / 180     # same operation order as the dataset was built with
