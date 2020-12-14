#pragma once

#if UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 2
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#elif UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 1
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>
#else
#error "Undefined or unknown UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION"
#endif

namespace uncertainty_planning_core
{

#if UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 2
using ColorRGBA = std_msgs::msg::ColorRGBA;
using Marker = visualization_msgs::msg::Marker;
using MarkerArray = visualization_msgs::msg::MarkerArray;
#elif UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 1
using ColorRGBA = std_msgs::ColorRGBA;
using Marker = visualization_msgs::Marker;
using MarkerArray = visualization_msgs::MarkerArray;
#endif

using DisplayFunction = std::function<void(const MarkerArray&)>;

}  // namespace uncertainty_planning_core
