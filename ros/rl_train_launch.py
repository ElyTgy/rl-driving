"""Start RL training in the ROS 2 F1TENTH simulator.

The simulator must already be running (scan_num_beams: 1081). This launch file starts ONLY the
training process (ros/rl_train.py), which drives the car through ros/rl_env.py. It deliberately
does not start the safety node, drive_arbiter, lap_count, stop_node or bc_drive_node: they would
publish on /drive too and corrupt the actions the learner thinks it took.

Usage (from the repo root, with ROS sourced):
    ros2 launch ros/rl_train_launch.py start_poses:=/abs/poses.csv
    ros2 launch ros/rl_train_launch.py start_poses:=/abs/poses.csv centerline:=/abs/cl.csv \
        init_policy:=/abs/checkpoints/bc_v2.npz python:=/abs/sim_ws/.venv/bin/python

Running ros/rl_train.py directly in a terminal does the same and is easier while debugging.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_DIR, 'ros', 'rl_train.py')
DEFAULT_POLICY = os.path.join(REPO_DIR, 'checkpoints', 'bc_v2.npz')


def launch_setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)

    cmd = [arg('python'), TRAIN_SCRIPT,
           '--start-poses', arg('start_poses'),
           '--timesteps', arg('total_timesteps'),
           '--speed', arg('speed'),
           '--lr', arg('lr'),
           '--activation', arg('activation'),
           '--out-dir', os.path.expanduser(arg('out_dir'))]
    if arg('centerline'):
        cmd += ['--centerline', arg('centerline')]
    if arg('init_policy'):
        cmd += ['--init-policy', arg('init_policy')]

    return [ExecuteProcess(cmd=cmd, output='screen')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('python', default_value='python3',
                              description='Python with rclpy, gymnasium, stable-baselines3 (e.g. sim_ws/.venv/bin/python)'),
        DeclareLaunchArgument('start_poses',
                              description='csv/npy file, rows of x,y,yaw in the map frame (required)'),
        DeclareLaunchArgument('centerline', default_value='',
                              description='csv/npy file, rows of x,y in track order; empty = survival reward only'),
        DeclareLaunchArgument('init_policy', default_value='',
                              description=f'BC .npz to warm-start the actor, e.g. {DEFAULT_POLICY}; empty = from scratch'),
        DeclareLaunchArgument('activation', default_value='tanh', choices=['tanh', 'relu'],
                              description='activation of the BC network (must match when init_policy is set)'),
        DeclareLaunchArgument('total_timesteps', default_value='200000',
                              description='control steps (26 ms each, real time): 200000 is about 1.5 hours'),
        DeclareLaunchArgument('speed', default_value='1.0', description='fixed forward speed, m/s'),
        DeclareLaunchArgument('lr', default_value='1e-4', description='PPO learning rate'),
        DeclareLaunchArgument('out_dir', default_value='~/rl_runs/run1',
                              description='checkpoints and the final model are saved here'),
        OpaqueFunction(function=launch_setup),
    ])