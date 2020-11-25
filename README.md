# uncertainty_planning_core

This package is the core of our framework for motion planning and execution with actuation uncertainty. More information on the planning and execution methods can be found in our WAFR 2016 [paper](http://arm.eecs.umich.edu/download.php?p=54) and [presentation](https://www.youtube.com/watch?v=42rwqAUTlbo&list=PL24TB_XE22Jvx6Ozhmdwl5kRClbWjUS0m).

## This package provides several core components:

- The core templated motion planner
- Templated execution policy that updates during execution
- Interfaces for robot models, samplers, outcome clustering, and robot simulators to integrate with the planner
- Concrete instantiations of the planner and execution policy for SE(2), SE(3), and linked robots (multiple configuration representations)

While the planner and execution policy are themselves template-based, this package provides a library containing concrete instantiations of the planner for different types of robot. When possible, you should use these rather than interfacing with the planner directly.

## Setup

`uncertainty_planning_core` is a ROS package.

Thus, it is best to build it within a ROS workspace:

```sh
mkdir -p ~/ws/src
cd ~/ws/src
git clone https://github.com/calderpg/uncertainty_planning_core
```

This package officially supports [ROS 1 Melodic](http://wiki.ros.org/ROS/Installation)
and [ROS 2 Dashing+](https://index.ros.org/doc/ros2/Installation/) distributions, but
ROS 1 Kinetic should also be compatible.
Make sure to symlink the corresponding `CMakeLists.txt` and `package.xml` files
for the ROS distribution of choice:

*For ROS 1 Melodic*
```sh
cd ~/ws/src/uncertainty_planning_core
ln -sT CMakeLists.txt.ros1 CMakeLists.txt
ln -sT package.xml.ros1 package.xml
```

*For ROS 2 Dashing+*
```sh
cd ~/ws/src/uncertainty_planning_core
ln -sT CMakeLists.txt.ros2 CMakeLists.txt
ln -sT package.xml.ros2 package.xml
```

Finally, use [`rosdep`](https://docs.ros.org/independent/api/rosdep/html/)
to ensure all dependencies in the `package.xml` are satisfied:

```sh
cd ~/ws
rosdep install -i -y --from-path src
```

## Building

Use [`catkin_make`](http://wiki.ros.org/catkin/commands/catkin_make) or
[`colcon`](https://colcon.readthedocs.io/en/released/) accordingly.

*For ROS 1 Melodic*
```sh
cd ~/ws
catkin_make  # the entire workspace
catkin_make --pkg uncertainty_planning_core  # the package only
```

*For ROS 2 Dashing +*
```sh
cd ~/ws
colcon build  # the entire workspace
colcon build --packages-select uncertainty_planning_core  # the package only
```

## Dependencies

- [common_robotics_utilities](https://github.com/calderpg/common_robotics_utilities)

Provides a range of utility and math functions, as well as templated implementations of kinodynamic RRT, Dijkstra's algorithm, and hierarchical clustering.

- [ROS](http://ros.org)

ROS is required for the build system and for RViz, which the planner uses as an optional visualization interface.

## Examples

A task planning example is included in this package.

To see several old examples of using the planner and execution policy, see [uncertainty_planning_examples](https://github.com/UM-ARM-LAB/uncertainty_planning_examples) *Note - these examples are not up-to-date*
