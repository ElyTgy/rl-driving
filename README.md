Pull the branch.
Copy the model over; it's gitignored. On the laptop: scp checkpoints/bc_v1.npz <jetson-user>@<jetson-ip>:~/sim_ws/src/rl-driving/checkpoints/
Build milestone4 if it isn't built yet: colcon build --packages-select milestone4
Start the car's drivers as usual, then ros2 launch ros/bc_drive_launch.py mode:=car
