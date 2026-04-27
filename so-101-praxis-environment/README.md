# praxis-core
Base code for the Praxis Lab. Includes shared API defintions for robots/sensors, hardware/sim bindings for each, and robot descriptions/sim assets.

## Installation


Set up conda environment for [ROS2 Kilted](https://docs.ros.org/en/kilted/index.html), following instructions from [RoboStack](https://robostack.github.io/GettingStarted.html).  
(Note: Do not source the system ROS environment. 
When there is an installation available of ROS on the system, in non-conda environments, there will be interference with the environments as the `PYTHONPATH` set in the setup script conflicts with the conda environment.)

  ```bash
  conda create -n praxis -c conda-forge -c robostack-kilted ros-kilted-desktop  # create env & install ros2
  conda activate praxis  # activate env
  conda install -c conda-forge compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep  # install development tools
  ```
Test installation:
  ```bash
  rviz2
  ```
Install `praxis`:
  ```bash
  pip install -e .
  ```

For real UR5e robot support, install with optional dependencies:
  ```bash
  pip install -e ".[real-ur5]"
  ```

For Gelsight Mini support, install with optional dependecies:
 ```bash
  pip install -e ".[gsmini]"
  ```

## Space mouse teleoperation instructions

### Space mouse device preparation  
Install [libhidapi-dev](https://spacemouse.kubaandrysek.cz/) to access HID data.
  ```bash
  sudo apt-get install libhidapi-dev
  ```
Add a udev rule for hidraw access. Create etc/udev/rules.d/99-spacemouse.rules with
  ```
  KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666"
  ```
Then reload rules.
  ```bash
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```

### Running teleoperation in simulation
#### Launch robot node
  UR5
  ```bash
  python praxis/ros/scripts/robot_node_launcher.py robot=ur5sim
  ```
  SO101
  ```bash
  python praxis/ros/scripts/robot_node_launcher.py robot=so101sim
  ```

#### Launch teleoperation node
  UR5
  ```bash
  python praxis/ros/examples/spacemouse_teleop.py robot=ur5sim
  ```

### Running teleoperation in real world
#### Launch robot node
  UR5
  ```bash
  python praxis/ros/scripts/robot_node_launcher.py robot=ur5real
  ```
  SO101
  ```bash
  python praxis/ros/scripts/robot_node_launcher.py robot=so101real
  ```
#### Launch teleoperation node
  UR5
  ```bash
  python praxis/ros/examples/spacemouse_teleop.py robot=ur5real
  ```

  ### Real UR5 Usage Guide
  https://coda.io/d/_dts6r_E0lJs/Robot-Usage-Guide_suVVPRH8

## Sample GelSight Mini Code Usage
Launch sensor node
  ```bash
  python praxis/ros/scripts/sensor_node_launcher.py sensor=[gsmini/digit]
  ```
Launch teleoperation node
  ```bash
  python praxis/ros/examples/gsmini_test.py sensor=[gsmini/digit]
  ```
