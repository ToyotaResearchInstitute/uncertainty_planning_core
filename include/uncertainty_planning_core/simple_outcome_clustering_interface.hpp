#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <common_robotics_utilities/math.hpp>
#include <common_robotics_utilities/simple_robot_model_interface.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#if UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 2
#include <visualization_msgs/msg/marker_array.hpp>
#elif UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 1
#include <visualization_msgs/MarkerArray.h>
#else
#error "Undefined or unknown UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION"
#endif

namespace uncertainty_planning_core
{
template<typename Configuration,
         typename ConfigAlloc=std::allocator<Configuration>>
class SimpleOutcomeClusteringInterface
{
protected:
  using Robot = common_robotics_utilities::simple_robot_model_interface
      ::SimpleRobotModelInterface<Configuration, ConfigAlloc>;

public:
#if UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 2
  using MarkerArray = visualization_msgs::msg::MarkerArray;
#elif UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 1
  using MarkerArray = visualization_msgs::MarkerArray;
#endif

  virtual ~SimpleOutcomeClusteringInterface() {}

  virtual int32_t GetDebugLevel() const = 0;

  virtual int32_t SetDebugLevel(const int32_t debug_level) = 0;

  virtual std::map<std::string, double> GetStatistics() const = 0;

  virtual void ResetStatistics() = 0;

  virtual std::vector<std::vector<int64_t>> ClusterParticles(
      const std::shared_ptr<Robot>& robot,
      const std::vector<SimulationResult<Configuration>>& particles,
      const std::function<void(const MarkerArray&)>& display_fn) = 0;

  virtual std::vector<uint8_t> IdentifyClusterMembers(
      const std::shared_ptr<Robot>& robot,
      const std::vector<Configuration, ConfigAlloc>& cluster,
      const std::vector<SimulationResult<Configuration>>& particles,
      const std::function<void(const MarkerArray&)>& display_fn) = 0;
};
}  // namespace uncertainty_planning_core
