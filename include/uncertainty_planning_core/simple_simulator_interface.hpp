#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <memory>
#include <chrono>
#include <random>
#include <common_robotics_utilities/math.hpp>
#include <common_robotics_utilities/print.hpp>
#include <common_robotics_utilities/conversions.hpp>
#include <common_robotics_utilities/simple_robot_model_interface.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <omp.h>

namespace uncertainty_planning_core
{
template<typename Configuration,
         typename ConfigAlloc=std::allocator<Configuration>>
struct ForwardSimulationContactResolverStepTrace
{
  std::vector<Configuration, ConfigAlloc> contact_resolution_steps;
};

template<typename Configuration,
         typename ConfigAlloc=std::allocator<Configuration>>
struct ForwardSimulationResolverTrace
{
  Eigen::VectorXd control_input;
  Eigen::VectorXd control_input_step;
  std::vector<ForwardSimulationContactResolverStepTrace
      <Configuration, ConfigAlloc>> contact_resolver_steps;
};

template<typename Configuration,
         typename ConfigAlloc=std::allocator<Configuration>>
struct ForwardSimulationStepTrace
{
  std::vector<ForwardSimulationResolverTrace
      <Configuration, ConfigAlloc>> resolver_steps;

  void Reset() { resolver_steps.clear(); }
};

template<typename Configuration,
         typename ConfigAlloc=std::allocator<Configuration>>
inline std::vector<Configuration, ConfigAlloc> ExtractTrajectoryFromTrace(
    const ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace)
{
  std::vector<Configuration, ConfigAlloc> execution_trajectory;
  execution_trajectory.reserve(trace.resolver_steps.size());
  // Each step corresponds to a controller interval timestep in the real world
  for (const auto& step_trace : trace.resolver_steps)
  {
    // Each step trace is the entire resolver history of the motion
    // Get the current trace segment
    if (step_trace.contact_resolver_steps.empty())
    {
      throw std::runtime_error("step_trace.contact_resolver_steps is empty");
    }
    // The last contact resolution step is the final result of resolving the
    // contacts in that timestep
    const ForwardSimulationContactResolverStepTrace<Configuration, ConfigAlloc>&
        contact_resolution_trace = step_trace.contact_resolver_steps.back();
    // Get the last (collision-free resolved) config of the last resolution step
    if (contact_resolution_trace.contact_resolution_steps.empty())
    {
      throw std::runtime_error(
          "contact_resolution_trace.contact_resolution_steps is empty");
    }
    const Configuration& resolved_config
        = contact_resolution_trace.contact_resolution_steps.back();
    execution_trajectory.push_back(resolved_config);
  }
  execution_trajectory.shrink_to_fit();
  return execution_trajectory;
}

template<typename Configuration>
class SimulationResult
{
private:
  Configuration result_config_;
  Configuration actual_target_;
  bool did_contact_;
  bool outcome_is_nominally_independent_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SimulationResult()
      : did_contact_(false), outcome_is_nominally_independent_(false) {}

  SimulationResult(const Configuration& result_config,
                   const Configuration& actual_target,
                   const bool did_contact,
                   const bool outcome_is_nominally_independent)
    : result_config_(result_config), actual_target_(actual_target),
      did_contact_(did_contact),
      outcome_is_nominally_independent_(outcome_is_nominally_independent) {}

  const Configuration& ResultConfig() const { return result_config_; }

  const Configuration& ActualTarget() const { return actual_target_; }

  bool DidContact() const { return did_contact_; }

  bool OutcomeIsNominallyIndependent() const
  {
    return outcome_is_nominally_independent_;
  }

  std::string Print() const
  {
    std::ostringstream strm;
    strm << "Result config: "
         << common_robotics_utilities::print::Print(ResultConfig())
         << " Actual target: "
         << common_robotics_utilities::print::Print(ActualTarget())
         << " Did contact ["
         << common_robotics_utilities::print::Print(DidContact())
         << "] Outcome is nominally independent ["
         << common_robotics_utilities::print::Print(
             OutcomeIsNominallyIndependent()) << "]";
    return strm.str();
  }
};

template<typename Configuration, typename RNG,
         typename ConfigAlloc=std::allocator<Configuration>>
class SimpleSimulatorInterface
{
protected:
  using Robot = common_robotics_utilities::simple_robot_model_interface
      ::SimpleRobotModelInterface<Configuration, ConfigAlloc>;

public:
  virtual ~SimpleSimulatorInterface() {}

  virtual int32_t GetDebugLevel() const = 0;

  virtual int32_t SetDebugLevel(const int32_t debug_level) = 0;

  virtual RNG& GetRandomGenerator() = 0;

  virtual std::string GetFrame() const = 0;

  virtual visualization_msgs::MarkerArray MakeEnvironmentDisplayRep() const = 0;

  virtual visualization_msgs::MarkerArray MakeConfigurationDisplayRep(
      const std::shared_ptr<Robot>& immutable_robot,
      const Configuration& configuration, const std_msgs::ColorRGBA& color,
      const int32_t starting_index,
      const std::string& config_marker_ns) const = 0;

  virtual visualization_msgs::MarkerArray MakeControlInputDisplayRep(
      const std::shared_ptr<Robot>& immutable_robot,
      const Configuration& configuration,
      const Eigen::VectorXd& control_input,
      const std_msgs::ColorRGBA& color, const int32_t starting_index,
      const std::string& control_input_marker_ns) const = 0;

  virtual Eigen::Vector4d Get3dPointForConfig(
      const std::shared_ptr<Robot>& immutable_robot,
      const Configuration& config) const = 0;

  virtual std::map<std::string, double> GetStatistics() const = 0;

  virtual void ResetStatistics() = 0;

  virtual bool CheckConfigCollision(
      const std::shared_ptr<Robot>& immutable_robot,
      const Configuration& config, const double inflation_ratio=0.0) const = 0;

  virtual SimulationResult<Configuration> ForwardSimulateMutableRobot(
      const std::shared_ptr<Robot>& mutable_robot,
      const Configuration& target_position, const bool allow_contacts,
      ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace,
      const bool enable_tracing,
      const std::function<void(
          const visualization_msgs::MarkerArray&)>& display_fn) = 0;

  virtual SimulationResult<Configuration> ForwardSimulateRobot(
      const std::shared_ptr<Robot>& immutable_robot,
      const Configuration& start_position, const Configuration& target_position,
      const bool allow_contacts,
      ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace,
      const bool enable_tracing,
      const std::function<void(
          const visualization_msgs::MarkerArray&)>& display_fn) = 0;

  virtual std::vector<SimulationResult<Configuration>> ForwardSimulateRobots(
      const std::shared_ptr<Robot>& immutable_robot,
      const std::vector<Configuration, ConfigAlloc>& start_positions,
      const std::vector<Configuration, ConfigAlloc>& target_positions,
      const bool allow_contacts,
      const std::function<void(
          const visualization_msgs::MarkerArray&)>& display_fn) = 0;

  virtual SimulationResult<Configuration> ReverseSimulateMutableRobot(
      const std::shared_ptr<Robot>& mutable_robot,
      const Configuration& target_position, const bool allow_contacts,
      ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace,
      const bool enable_tracing,
      const std::function<void(
          const visualization_msgs::MarkerArray&)>& display_fn) = 0;

  virtual SimulationResult<Configuration> ReverseSimulateRobot(
      const std::shared_ptr<Robot>& immutable_robot,
      const Configuration& start_position, const Configuration& target_position,
      const bool allow_contacts,
      ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace,
      const bool enable_tracing,
      const std::function<void(
          const visualization_msgs::MarkerArray&)>& display_fn) = 0;

  virtual std::vector<SimulationResult<Configuration>> ReverseSimulateRobots(
      const std::shared_ptr<Robot>& immutable_robot,
      const std::vector<Configuration, ConfigAlloc>& start_positions,
      const std::vector<Configuration, ConfigAlloc>& target_positions,
      const bool allow_contacts,
      const std::function<void(
          const visualization_msgs::MarkerArray&)>& display_fn) = 0;
};
}  // namespace uncertainty_planning_core

template<typename Configuration>
std::ostream& operator<<(
    std::ostream& strm,
    const uncertainty_planning_core::SimulationResult<Configuration>& result)
{
  strm << result.Print();
  return strm;
}
