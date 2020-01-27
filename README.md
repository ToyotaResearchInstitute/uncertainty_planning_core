# uncertainty_planning_core

This package is the core of our framework for motion planning and execution with actuation uncertainty. More information on the planning and execution methods can be found in our WAFR 2016 [paper](http://arm.eecs.umich.edu/download.php?p=54) and [presentation](https://www.youtube.com/watch?v=42rwqAUTlbo&list=PL24TB_XE22Jvx6Ozhmdwl5kRClbWjUS0m).

## This package provides several core components:

- The core templated motion planner
- Templated execution policy that updates during execution
- Interfaces for robot models, samplers, outcome clustering, and robot simulators to integrate with the planner
- Concrete instantiations of the planner and execution policy for SE(2), SE(3), and linked robots (multiple configuration representations)

While the planner and execution policy are themselves template-based, this package provides a library containing concrete instantiations of the planner for different types of robot. When possible, you should use these rather than interfacing with the planner directly.

## Dependencies

- [common_robotics_utilities](https://github.com/calderpg/common_robotics_utilities)

Provides a range of utility and math functions, as well as templated implementations of kinodynamic RRT, Dijkstra's algorithm, and hierarchical clustering.

- [ROS](http://ros.org)

ROS is required for the build system, Catkin, and for RViz, which the planner uses as an optional visualization interface. ROS Melodic is officially supported, but ROS Kinetic should also be compatible.

## Examples

A task planning example is included in this package.

To see several old examples of using the planner and execution policy, see [uncertainty_planning_examples](https://github.com/UM-ARM-LAB/uncertainty_planning_examples) *Note - these examples are not up-to-date*
