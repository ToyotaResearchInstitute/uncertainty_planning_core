# uncertainty_planning_core

This package is the core of our framework for motion planning and execution with actuation uncertainty. More information on the planning and execution methods can be found in our WAFR 2016 [paper](http://arm.eecs.umich.edu/download.php?p=54) and [presentation](https://www.youtube.com/watch?v=42rwqAUTlbo&list=PL24TB_XE22Jvx6Ozhmdwl5kRClbWjUS0m)

## This package provides several core components:

- The core templated motion planner
- Templated execution policy that updates during execution
- Lightweight models of SE(2), SE(3), and linked robots
- Models of uncertain sensors and actuators
- Simple samplers for SE(2), SE(3), and linked problems
- Interface for robot simulators to integrate with the planner
- Concrete instantiations of the planner and execution policy for SE(2), SE(3), and linked robots

While the planner and execution policy are themselves template-based, this package provides a library containing concrete instantiations of the planner for different types of robot. When possible, you should use these rather interfacing with the planner directly.

## Dependencies

- [arc_utilities](github.com/UM-ARM-LAB/arc_utilities)
 
Provides a range of utility and math functions, as well as templated implementations of kinodynamic RRT, Dijkstra's algorithm, and hierarchical clustering.

- [sdf_tools](github.com/UM-ARM-LAB/sdf_tools)

Tools for modeling environments using voxel grids, including several types of collision maps, signed distance fields, and optional integration with MoveIt!

- [ROS Kinetic](ros.org)

ROS is required for the build system, Catkin, and for RViz, which the planner uses as an optional visualization interface.

## Examples

To see several examples of using the planner and execution policy, see [uncertainty_planning_examples](github.com/UM-ARM-LAB/uncertainty_planning_examples)
