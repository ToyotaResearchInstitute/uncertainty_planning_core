#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <common_robotics_utilities/math.hpp>
#include <common_robotics_utilities/simple_robot_model_interface.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#include <visualization_msgs/MarkerArray.h>

namespace uncertainty_planning_core
{
template<typename Configuration,
         typename ConfigAlloc=std::allocator<Configuration>>
class SimpleOutcomeClusteringInterface
{
protected:
  typedef common_robotics_utilities::simple_robot_model_interface
      ::SimpleRobotModelInterface<Configuration, ConfigAlloc> Robot;

public:
  virtual ~SimpleOutcomeClusteringInterface() {}

  virtual int32_t GetDebugLevel() const = 0;

  virtual int32_t SetDebugLevel(const int32_t debug_level) = 0;

  virtual std::map<std::string, double> GetStatistics() const = 0;

  virtual void ResetStatistics() = 0;

  virtual std::vector<std::vector<int64_t>> ClusterParticles(
      const std::shared_ptr<Robot>& robot,
      const std::vector<SimulationResult<Configuration>>& particles,
      const std::function<void(
          const visualization_msgs::MarkerArray&)>& display_fn) = 0;

  virtual std::vector<uint8_t> IdentifyClusterMembers(
      const std::shared_ptr<Robot>& robot,
      const std::vector<Configuration, ConfigAlloc>& cluster,
      const std::vector<SimulationResult<Configuration>>& particles,
      const std::function<void(
          const visualization_msgs::MarkerArray&)>& display_fn) = 0;
};
}  // namespace uncertainty_planning_core
