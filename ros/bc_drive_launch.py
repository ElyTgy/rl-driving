"""Drive with the BC model, with the milestone 4 safety stack (AEB) in control of /drive.

Same setup as milestone4_wf_py.py (safety_node, drive_arbiter, lap_count, then the driver after
1 s), with the BC node in wall_follow's place:

    bc_drive_node ──/wall_follow_drive──► drive_arbiter ──/drive──► car (or sim)
    safety_node ────/safety_drive──────►      ▲
    lap_count ──────/safety_drive───────────┘ (car only: stops after target_laps)

The BC node publishes on /wall_follow_drive because the milestone 4 arbiter already forwards that
topic (so its code does not need to change). Never run wall_follow at the same time.
Once the safety node fires, the arbiter keeps the car stopped until everything is relaunched.

Needs the milestone4 package built and sourced (colcon build --packages-select milestone4).

Usage (from the repo root, with ROS and the workspace sourced):
    ros2 launch ros/bc_drive_launch.py mode:=car
    ros2 launch ros/bc_drive_launch.py mode:=sim            # sim must already be running
    ros2 launch ros/bc_drive_launch.py mode:=car speed:=1.0 target_laps:=2 record:=false

With record:=true (default) a rosbag of the inputs and commands goes to <log_dir>/bc_run_<time>/,
for checking afterwards what the model saw and did.
"""
import os
from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BC_NODE = os.path.join(REPO_DIR, 'ros', 'bc_drive_node.py')
DEFAULT_POLICY = os.path.join(REPO_DIR, 'checkpoints', 'bc_v1.npz')

# An input of the milestone 4 drive_arbiter (see the module docstring for why this one).
BC_DRIVE_TOPIC = '/wall_follow_drive'

ODOM_TOPIC = {'car': 'odom', 'sim': 'ego_racecar/odom'}
IMU_TOPIC = '/sensors/imu/raw'


def launch_setup(context):
    """Build the nodes once the launch arguments are known."""
    arg = lambda name: LaunchConfiguration(name).perform(context)
    mode = arg('mode')
    odom_topic = ODOM_TOPIC[mode]

    safety_node = Node(
        package='milestone4',
        executable='safety_node',
        output='screen',
        parameters=[{'odom': odom_topic}],
    )
    drive_arbiter = Node(
        package='milestone4',
        executable='drive_arbiter',
        output='screen',
    )
    actions = [safety_node, drive_arbiter]

    if mode == 'car':
        # Stops the car through /safety_drive after target_laps (it counts laps with the camera,
        # which the sim does not have).
        actions.append(Node(
            package='milestone4',
            executable='lap_count',
            output='screen',
            parameters=[{'target_laps': int(arg('target_laps'))}],
        ))

    # The BC node is a plain script (rl-driving is not a ROS package), so it is started as a process.
    bc_drive = ExecuteProcess(
        cmd=['python3', BC_NODE, '--ros-args',
             '-p', f'mode:={mode}',
             '-p', f'speed:={float(arg("speed"))}',
             '-p', f'policy_path:={arg("policy_path")}',
             '-p', f'drive_topic:={BC_DRIVE_TOPIC}'],
        output='screen',
    )
    # Small delay so the safety node and arbiter are up before the car starts moving.
    actions.append(TimerAction(period=1.0, actions=[bc_drive]))

    if arg('record') == 'true':
        bag_dir = Path(arg('log_dir')).expanduser() / datetime.now().strftime('bc_run_%Y%m%d_%H%M%S')
        topics = ['/scan', '/' + odom_topic, BC_DRIVE_TOPIC, '/drive', '/safety_drive', '/tf', '/tf_static']
        if mode == 'car':
            topics.append(IMU_TOPIC)
        actions.append(ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o', str(bag_dir)] + topics,
                                      output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='car', choices=['car', 'sim'],
                              description='car: /odom + IMU, lap_count on. sim: /ego_racecar/odom.'),
        DeclareLaunchArgument('speed', default_value='1.0',
                              description='m/s, fixed. The model was trained at 1.0 m/s.'),
        DeclareLaunchArgument('policy_path', default_value=DEFAULT_POLICY,
                              description='numpy policy made by scripts/export_policy.py'),
        DeclareLaunchArgument('target_laps', default_value='1',
                              description='car only: lap_count stops the car after this many laps'),
        DeclareLaunchArgument('record', default_value='true', choices=['true', 'false'],
                              description='record a rosbag of the inputs and commands'),
        DeclareLaunchArgument('log_dir', default_value='~/bc_logs',
                              description='each recorded run goes to <log_dir>/bc_run_<time>/'),
        OpaqueFunction(function=launch_setup),
    ])
