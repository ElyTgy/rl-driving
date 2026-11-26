"""Drive with the BC model, with the milestone 5 safety stack in control of /drive.

Same setup as milestone5_py.py (safety_node, drive_arbiter, lap_count, then the driver after
1 s), with the BC node in gap_follow's place:

    bc_drive_node ──/gap_follow_drive──► drive_arbiter ──/drive──► car (or sim)
    safety_node ────/safety_drive, /aeb_true──►  ▲   (stage 1 slows to 0.85 m/s, stage 2 stops; both recover)
    lap_count ──────/safety_drive───────────────┤   (car only: stops after target_laps)
    stop_node ──────/stop───────────────────────┘   (Space; run it in its own terminal, see below)

The BC node publishes on /gap_follow_drive because the MS5 arbiter already forwards that topic
(so its code does not need to change). Never run gap_follow at the same time. bc_v2 was trained
on gap_follow runs with this same safety node, so the car behaves near obstacles as in the data.

The stop node is not started here: it reads Space from the terminal's keyboard, which a launch
file cannot give it. Run `ros2 run milestone5 stop_node` in a second terminal. After a stop, the
arbiter keeps the car stopped until it is relaunched.

Needs the milestone5 package built and sourced.

Usage (from the repo root, with ROS and the workspace sourced):
    ros2 launch ros/bc_drive_launch.py mode:=car
    ros2 launch ros/bc_drive_launch.py mode:=sim            # sim must already be running
    ros2 launch ros/bc_drive_launch.py mode:=car max_speed:=1.0 target_laps:=2 record:=false

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
DEFAULT_POLICY = os.path.join(REPO_DIR, 'checkpoints', 'bc_v2.npz')

# An input of the milestone 5 drive_arbiter (see the module docstring for why this one).
BC_DRIVE_TOPIC = '/gap_follow_drive'

ODOM_TOPIC = {'car': 'odom', 'sim': 'ego_racecar/odom'}
IMU_TOPIC = '/sensors/imu/raw'


def launch_setup(context):
    """Build the nodes once the launch arguments are known."""
    arg = lambda name: LaunchConfiguration(name).perform(context)
    mode = arg('mode')
    odom_topic = ODOM_TOPIC[mode]

    safety_node = Node(
        package='milestone5',
        executable='safety_node',
        output='screen',
        parameters=[{'odom': odom_topic}],
    )
    drive_arbiter = Node(
        package='milestone5',
        executable='drive_arbiter',
        output='screen',
    )
    actions = [safety_node, drive_arbiter]

    if mode == 'car':
        # Stops the car through /safety_drive after target_laps (it counts laps with the camera,
        # which the sim does not have).
        actions.append(Node(
            package='milestone5',
            executable='lap_count',
            output='screen',
            parameters=[{'target_laps': int(arg('target_laps'))}],
        ))

    # The BC node is a plain script (rl-driving is not a ROS package), so it is started as a process.
    bc_drive = ExecuteProcess(
        cmd=['python3', BC_NODE, '--ros-args',
             '-p', f'mode:={mode}',
             '-p', f'max_speed:={float(arg("max_speed"))}',
             '-p', f'policy_path:={arg("policy_path")}',
             '-p', f'drive_topic:={BC_DRIVE_TOPIC}'],
        output='screen',
    )
    # Small delay so the safety node and arbiter are up before the car starts moving.
    actions.append(TimerAction(period=1.0, actions=[bc_drive]))

    if arg('record') == 'true':
        bag_dir = Path(arg('log_dir')).expanduser() / datetime.now().strftime('bc_run_%Y%m%d_%H%M%S')
        topics = ['/scan', '/' + odom_topic, BC_DRIVE_TOPIC, '/drive', '/safety_drive', '/aeb_true', '/stop',
                  '/tf', '/tf_static']
        if mode == 'car':
            topics.append(IMU_TOPIC)
        actions.append(ExecuteProcess(cmd=['ros2', 'bag', 'record', '-o', str(bag_dir)] + topics,
                                      output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='car', choices=['car', 'sim'],
                              description='car: /odom + IMU, lap_count on. sim: /ego_racecar/odom.'),
        DeclareLaunchArgument('max_speed', default_value='1.0',
                              description='m/s, cap on the speed the model asks for (the expert drove 1.0-1.8 m/s).'),
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
