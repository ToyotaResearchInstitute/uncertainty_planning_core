#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <random>
#include <mutex>
#include <thread>
#include <atomic>
#include <common_robotics_utilities/math.hpp>
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/simple_robot_model_interface.hpp>
#include <common_robotics_utilities/zlib_helpers.hpp>
#include <uncertainty_planning_core/execution_policy.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#include <uncertainty_planning_core/uncertainty_planner_state.hpp>
#include <uncertainty_planning_core/uncertainty_contact_planning.hpp>
#if UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 2
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#elif UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 1
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#else
#error "Undefined or unknown UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION"
#endif

namespace uncertainty_planning_core
{
struct PLANNING_AND_EXECUTION_OPTIONS
{
  // Time limits
  double planner_time_limit = 0.0;
  // P(goal reached) termination threshold
  double p_goal_reached_termination_threshold = 0.0;
  // Standard planner control params
  double goal_bias = 0.0;
  double step_size = 0.0;
  double goal_probability_threshold = 0.0;
  double goal_distance_threshold = 0.0;
  double connect_after_first_solution = 0.0;
  // Distance function control params/weights
  double feasibility_alpha = 0.0;
  double variance_alpha = 0.0;
  // Reverse/repeat params
  uint32_t edge_attempt_count = 0u;
  // Particle/execution limits
  uint32_t num_particles = 0u;
  // Execution limits
  uint32_t num_policy_simulations = 0u;
  uint32_t num_policy_executions = 0u;
  // How many attempts does a policy action count for?
  uint32_t policy_action_attempt_count = 0u;
  // Execution limits
  uint32_t max_exec_actions = 0u;
  double max_policy_exec_time = 0.0;
  // Control flags
  int32_t debug_level = 0;
  bool use_contact = false;
  bool use_reverse = false;
  bool use_spur_actions = false;
  // Log & data files
  std::string planner_log_file;
  std::string policy_log_file;
  std::string planned_policy_file;
  std::string executed_policy_file;
};

#if UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 2
inline PLANNING_AND_EXECUTION_OPTIONS GetOptions(
    rclcpp::Node::SharedPtr node,
    const PLANNING_AND_EXECUTION_OPTIONS& initial_options)
{
  PLANNING_AND_EXECUTION_OPTIONS options = initial_options;
  // Get options via ROS params
  options.planner_time_limit
      = node->declare_parameter("planner_time_limit",
                                options.planner_time_limit);
  options.p_goal_reached_termination_threshold
      = node->declare_parameter("p_goal_reached_termination_threshold",
                  options.p_goal_reached_termination_threshold);
  options.goal_bias = node->declare_parameter("goal_bias", options.goal_bias);
  options.step_size = node->declare_parameter("step_size", options.step_size);
  options.goal_probability_threshold
      = node->declare_parameter("goal_probability_threshold",
                                options.goal_probability_threshold);
  options.goal_distance_threshold
      = node->declare_parameter("goal_distance_threshold",
                                options.goal_distance_threshold);
  options.connect_after_first_solution
      = node->declare_parameter("connect_after_first_solution",
                                options.connect_after_first_solution);
  options.feasibility_alpha
      = node->declare_parameter("feasibility_alpha",
                                options.feasibility_alpha);
  options.variance_alpha
      = node->declare_parameter("variance_alpha", options.variance_alpha);
  options.num_particles
      = static_cast<uint32_t>(node->declare_parameter("num_particles",
                              static_cast<int>(options.num_particles)));
  options.planner_log_file
      = node->declare_parameter("planner_log_file", options.planner_log_file);
  options.planned_policy_file
      = node->declare_parameter("planned_policy_file",
                                options.planned_policy_file);
  options.policy_action_attempt_count
      = static_cast<uint32_t>(
          node->declare_parameter("policy_action_attempt_count",
              static_cast<int>(options.policy_action_attempt_count)));
  options.debug_level
      = node->declare_parameter("debug_level", options.debug_level);
  options.use_contact
      = node->declare_parameter("use_contact", options.use_contact);
  options.use_reverse
      = node->declare_parameter("use_reverse", options.use_reverse);
  options.num_policy_simulations
      = static_cast<uint32_t>(
          node->declare_parameter(("num_policy_simulations",
              static_cast<int>(options.num_policy_simulations)));
  options.num_policy_executions
      = static_cast<uint32_t>(
          node->declare_parameter("num_policy_executions",
              static_cast<int>(options.num_policy_executions)));
  options.policy_log_file
      = node->declare_parameter("policy_log_file", options.policy_log_file);
  options.executed_policy_file
      = node->declare_parameter("executed_policy_file",
                                options.executed_policy_file);
  options.max_exec_actions
      = static_cast<uint32_t>(
          node->declare_parameter("max_exec_actions",
              static_cast<int>(options.max_exec_actions)));
  options.max_policy_exec_time
      = node->declare_parameter("max_policy_exec_time",
                                options.max_policy_exec_time);
  options.policy_action_attempt_count
      = static_cast<uint32_t>(
          node->declare_parameter("policy_action_attempt_count",
              static_cast<int>(options.policy_action_attempt_count)));
  return options;
}
#elif UNCERTAINTY_PLANNING_CORE__SUPPORTED_ROS_VERSION == 1
inline PLANNING_AND_EXECUTION_OPTIONS GetOptions(
    const PLANNING_AND_EXECUTION_OPTIONS& initial_options)
{
  PLANNING_AND_EXECUTION_OPTIONS options = initial_options;
  // Get options via ROS params
  ros::NodeHandle nhp("~");
  options.planner_time_limit
      = nhp.param(std::string("planner_time_limit"),
                  options.planner_time_limit);
  options.p_goal_reached_termination_threshold
      = nhp.param(std::string("p_goal_reached_termination_threshold"),
                  options.p_goal_reached_termination_threshold);
  options.goal_bias = nhp.param(std::string("goal_bias"), options.goal_bias);
  options.step_size = nhp.param(std::string("step_size"), options.step_size);
  options.goal_probability_threshold
      = nhp.param(std::string("goal_probability_threshold"),
                  options.goal_probability_threshold);
  options.goal_distance_threshold
      = nhp.param(std::string("goal_distance_threshold"),
                  options.goal_distance_threshold);
  options.connect_after_first_solution
      = nhp.param(std::string("connect_after_first_solution"),
                  options.connect_after_first_solution);
  options.feasibility_alpha
      = nhp.param(std::string("feasibility_alpha"),
                  options.feasibility_alpha);
  options.variance_alpha
      = nhp.param(std::string("variance_alpha"), options.variance_alpha);
  options.num_particles
      = static_cast<uint32_t>(nhp.param(std::string("num_particles"),
                              static_cast<int>(options.num_particles)));
  options.planner_log_file
      = nhp.param(std::string("planner_log_file"), options.planner_log_file);
  options.planned_policy_file
      = nhp.param(std::string("planned_policy_file"),
                  options.planned_policy_file);
  options.policy_action_attempt_count
      = static_cast<uint32_t>(
          nhp.param(std::string("policy_action_attempt_count"),
                    static_cast<int>(options.policy_action_attempt_count)));
  options.debug_level
      = nhp.param(std::string("debug_level"), options.debug_level);
  options.use_contact
      = nhp.param(std::string("use_contact"), options.use_contact);
  options.use_reverse
      = nhp.param(std::string("use_reverse"), options.use_reverse);
  options.num_policy_simulations
      = static_cast<uint32_t>(
          nhp.param(std::string("num_policy_simulations"),
                    static_cast<int>(options.num_policy_simulations)));
  options.num_policy_executions
      = static_cast<uint32_t>(
          nhp.param(std::string("num_policy_executions"),
                    static_cast<int>(options.num_policy_executions)));
  options.policy_log_file
      = nhp.param(std::string("policy_log_file"), options.policy_log_file);
  options.executed_policy_file
      = nhp.param(std::string("executed_policy_file"),
                  options.executed_policy_file);
  options.max_exec_actions
      = static_cast<uint32_t>(
          nhp.param(std::string("max_exec_actions"),
                    static_cast<int>(options.max_exec_actions)));
  options.max_policy_exec_time
      = nhp.param(std::string("max_policy_exec_time"),
                  options.max_policy_exec_time);
  options.policy_action_attempt_count
      = static_cast<uint32_t>(
          nhp.param(std::string("policy_action_attempt_count"),
                    static_cast<int>(options.policy_action_attempt_count)));
  return options;
}
#endif

// Policy and tree type definitions

using PRNG = std::mt19937_64;

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
using UncertaintyPlanningState
    = UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>;

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
using UncertaintyPlanningPolicy
    = ExecutionPolicy<Configuration, ConfigSerializer, ConfigAlloc>;

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
using UncertaintyPlanningTreeState
    = common_robotics_utilities::simple_rrt_planner::SimpleRRTPlannerState<
        UncertaintyPlanningState<Configuration, ConfigSerializer, ConfigAlloc>>;

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
using UncertaintyPlanningTreeStateSerializer
    = common_robotics_utilities::serialization
        ::Serializer<UncertaintyPlanningTreeState<
            Configuration, ConfigSerializer, ConfigAlloc>>;

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
using UncertaintyPlanningTreeStateDeserializer
    = common_robotics_utilities::serialization
        ::Deserializer<UncertaintyPlanningTreeState<
            Configuration, ConfigSerializer, ConfigAlloc>>;

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
using UncertaintyPlanningTree
    = common_robotics_utilities::simple_rrt_planner::PlanningTree<
        UncertaintyPlanningState<Configuration, ConfigSerializer, ConfigAlloc>>;

// Typedefs and helpers for a Eigen::VectorXd configuration type.

class VectorXdConfigSerializer
{
public:
  static inline std::string TypeName()
  {
    return std::string("EigenVectorXdSerializer");
  }

  static inline uint64_t Serialize(
      const Eigen::VectorXd& value, std::vector<uint8_t>& buffer)
  {
    return common_robotics_utilities::serialization::SerializeVectorXd(
        value, buffer);
  }

  static inline
  common_robotics_utilities::serialization::Deserialized<Eigen::VectorXd>
  Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
  {
    return common_robotics_utilities::serialization::DeserializeVectorXd(
        buffer, current);
  }
};

using VectorXdConfig = Eigen::VectorXd;
using VectorXdConfigAlloc = std::allocator<Eigen::VectorXd>;
using VectorXdConfigVector
    = std::vector<VectorXdConfig, VectorXdConfigAlloc>;
using VectorXdPolicy = ExecutionPolicy<
    VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>;
using VectorXdPolicyPlanningResult
    = UncertaintyPolicyPlanningResult<
        VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>;
using VectorXdPolicyExecutionResult
    = UncertaintyPolicyExecutionResult<
        VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>;
using VectorXdSampler = SimpleSamplerInterface<VectorXdConfig, PRNG>;
using VectorXdSamplerPtr = std::shared_ptr<VectorXdSampler>;
using VectorXdRobot
    = common_robotics_utilities::simple_robot_model_interface
        ::SimpleRobotModelInterface<VectorXdConfig, VectorXdConfigAlloc>;
using VectorXdRobotPtr = std::shared_ptr<VectorXdRobot>;
using VectorXdSimulator
    = SimpleSimulatorInterface<VectorXdConfig, PRNG, VectorXdConfigAlloc>;
using VectorXdSimulatorPtr = std::shared_ptr<VectorXdSimulator>;
using VectorXdClustering
    = SimpleOutcomeClusteringInterface<VectorXdConfig, VectorXdConfigAlloc>;
using VectorXdClusteringPtr = std::shared_ptr<VectorXdClustering>;
using VectorXdPlanningState = UncertaintyPlanningState<
    VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>;
using VectorXdPlanningSpace = UncertaintyPlanningSpace<
    VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc, PRNG>;
using VectorXdPolicyActionExecutionFunction
    = PolicyActionExecutionFunction<VectorXdConfig, VectorXdConfigAlloc>;

// Typedefs for user-provided goal check functions

using VectorXdUserGoalStateCheckFn
    = std::function<double(const VectorXdPlanningState&)>;

using VectorXdUserGoalConfigCheckFn
    = std::function<bool(const VectorXdConfig&)>;

// Implementations of basic user goal config check -> user goal state check
// functions

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
inline double UserGoalCheckWrapperFn(
    const UncertaintyPlanningState<
        Configuration, ConfigSerializer, ConfigAlloc>& state,
    const std::function<bool(const Configuration&)>& user_goal_config_check_fn)
{
  if (state.HasParticles())
  {
    const std::vector<Configuration, ConfigAlloc>& particle_positions
        = state.GetParticlePositionsImmutable().Value();
    const size_t num_particles = state.GetNumParticles();
    if (num_particles > 0)
    {
      size_t reached_goal = 0;
      for (size_t idx = 0; idx < num_particles; idx++)
      {
        const bool particle_reached_goal
            = user_goal_config_check_fn(particle_positions[idx]);
        if (particle_reached_goal)
        {
          reached_goal++;
        }
      }
      const double p_goal_reached = static_cast<double>(reached_goal)
                                    / static_cast<double>(num_particles);
      return p_goal_reached;
    }
    else
    {
      return 0.0;
    }
  }
  else
  {
    if (user_goal_config_check_fn(state.GetExpectation()))
    {
      return 1.0;
    }
    else
    {
      return 0.0;
    }
  }
}

inline double VectorXdUserGoalCheckWrapperFn(
  const VectorXdPlanningState& state,
  const VectorXdUserGoalConfigCheckFn& user_goal_config_check_fn);

// Policy saving and loading

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
inline uint64_t SerializePlannerTree(
    const UncertaintyPlanningTree<Configuration, ConfigSerializer, ConfigAlloc>&
        planner_tree,
    std::vector<uint8_t>& buffer)
{
  std::cout << "Serializing planner tree..." << std::endl;
  UncertaintyPlanningTreeStateSerializer<
      Configuration, ConfigSerializer, ConfigAlloc>
          planning_tree_state_serializer_fn
      = [] (const UncertaintyPlanningTreeState<
                Configuration, ConfigSerializer, ConfigAlloc>& state,
            std::vector<uint8_t>& ser_buffer)
  {
    return UncertaintyPlanningTreeState<
        Configuration, ConfigSerializer, ConfigAlloc>::Serialize(
            state, ser_buffer,
            UncertaintyPlanningState<
                Configuration, ConfigSerializer, ConfigAlloc>::Serialize);
  };
  const uint64_t size
      = common_robotics_utilities::serialization::SerializeVectorLike(
          planner_tree, buffer, planning_tree_state_serializer_fn);
  std::cout << "...planner tree of " << planner_tree.size()
            << " states serialized into " << buffer.size() << " bytes"
            << std::endl;
  return size;
}

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
inline common_robotics_utilities::serialization::Deserialized<
    UncertaintyPlanningTree<Configuration, ConfigSerializer, ConfigAlloc>>
DeserializePlannerTree(
    const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
{
  std::cout << "Deserializing planner tree..." << std::endl;
  UncertaintyPlanningTreeStateSerializer<
      Configuration, ConfigSerializer, ConfigAlloc>
          planning_tree_state_deserializer_fn
      = [] (const std::vector<uint8_t>& deserialize_buffer,
            const uint64_t deserialize_starting_offset)
  {
    return UncertaintyPlanningTreeState<
        Configuration, ConfigSerializer, ConfigAlloc>::Deserialize(
            deserialize_buffer, deserialize_starting_offset,
            UncertaintyPlanningState<
                Configuration, ConfigSerializer, ConfigAlloc>::Deserialize);
  };
  const auto deserialized_tree
      = common_robotics_utilities::serialization::DeserializeVectorLike<
          UncertaintyPlanningTreeState<
              Configuration, ConfigSerializer, ConfigAlloc>,
          UncertaintyPlanningTree<
              Configuration, ConfigSerializer, ConfigAlloc>>(
                  buffer, starting_offset, planning_tree_state_deserializer_fn);
  std::cout << "...planner tree of " << deserialized_tree.Value().size()
            << " states deserialized from " << deserialized_tree.BytesRead()
            << " bytes" << std::endl;
  return deserialized_tree;
}

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
inline bool SavePlannerTree(
      const UncertaintyPlanningTree<
          Configuration, ConfigSerializer, ConfigAlloc>& planner_tree,
      const std::string& filepath)
{
  try
  {
    std::cout << "Attempting to serialize tree..." << std::endl;
    std::vector<uint8_t> buffer;
    SerializePlannerTree<Configuration, ConfigSerializer, ConfigAlloc>(
        planner_tree, buffer);
    std::cout << "Compressing for storage..." << std::endl;
    const std::vector<uint8_t> compressed_serialized_tree
        = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
    std::cout << "Attempting to save to file..." << std::endl;
    std::ofstream output_file(filepath, std::ios::out|std::ios::binary);
    const size_t serialized_size = compressed_serialized_tree.size();
    output_file.write(
          reinterpret_cast<const char*>(compressed_serialized_tree.data()),
          static_cast<std::streamsize>(serialized_size));
    output_file.close();
    return true;
  }
  catch (...)
  {
    std::cerr << "Saving planner tree failed" << std::endl;
    return false;
  }
}

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
inline UncertaintyPlanningTree<Configuration, ConfigSerializer, ConfigAlloc>
LoadPlannerTree(const std::string& filepath)
{
  std::cout << "Attempting to load from file..." << std::endl;
  std::ifstream input_file(filepath, std::ios::in|std::ios::binary);
  if (input_file.good() == false)
  {
    throw std::invalid_argument("Planner tree file does not exist");
  }
  input_file.seekg(0, std::ios::end);
  std::streampos end = input_file.tellg();
  input_file.seekg(0, std::ios::beg);
  std::streampos begin = input_file.tellg();
  const std::streamsize serialized_size = end - begin;
  std::vector<uint8_t> file_buffer(static_cast<size_t>(serialized_size), 0x00);
  input_file.read(reinterpret_cast<char*>(file_buffer.data()), serialized_size);
  std::cout << "Decompressing from storage..." << std::endl;
  const std::vector<uint8_t> decompressed_serialized_tree
      = common_robotics_utilities::zlib_helpers::DecompressBytes(file_buffer);
  std::cout << "Attempting to deserialize tree..." << std::endl;
  return DeserializePlannerTree<Configuration, ConfigSerializer, ConfigAlloc>(
      decompressed_serialized_tree, 0u).Value();
}

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
inline bool SavePolicy(
    const UncertaintyPlanningPolicy<
        Configuration, ConfigSerializer, ConfigAlloc>& policy,
    const std::string& filepath)
{
  try
  {
    std::cout << "Attempting to serialize policy..." << std::endl;
    std::vector<uint8_t> buffer;
    UncertaintyPlanningPolicy<Configuration, ConfigSerializer, ConfigAlloc>
        ::Serialize(policy, buffer);
    std::cout << "Compressing for storage..." << std::endl;
    const std::vector<uint8_t> compressed_serialized_policy
        = common_robotics_utilities::zlib_helpers::CompressBytes(buffer);
    std::cout << "Attempting to save to file..." << std::endl;
    std::ofstream output_file(filepath, std::ios::out|std::ios::binary);
    const size_t serialized_size = compressed_serialized_policy.size();
    output_file.write(reinterpret_cast<const char*>(
        compressed_serialized_policy.data()),
        static_cast<std::streamsize>(serialized_size));
    output_file.close();
    return true;
  }
  catch (...)
  {
    std::cerr << "Saving policy failed" << std::endl;
    return false;
  }
}

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc>
inline UncertaintyPlanningPolicy<Configuration, ConfigSerializer, ConfigAlloc>
LoadPolicy(const std::string& filepath)
{
  std::cout << "Attempting to load from file..." << std::endl;
  std::ifstream input_file(filepath, std::ios::in|std::ios::binary);
  if (input_file.good() == false)
  {
    throw std::invalid_argument("Policy file does not exist");
  }
  input_file.seekg(0, std::ios::end);
  std::streampos end = input_file.tellg();
  input_file.seekg(0, std::ios::beg);
  std::streampos begin = input_file.tellg();
  const std::streamsize serialized_size = end - begin;
  std::vector<uint8_t> file_buffer(static_cast<size_t>(serialized_size), 0x00);
  input_file.read(reinterpret_cast<char*>(file_buffer.data()), serialized_size);
  std::cout << "Decompressing from storage..." << std::endl;
  const std::vector<uint8_t> decompressed_serialized_policy
      = common_robotics_utilities::zlib_helpers::DecompressBytes(file_buffer);
  std::cout << "Attempting to deserialize policy..." << std::endl;
  return UncertaintyPlanningPolicy<Configuration, ConfigSerializer, ConfigAlloc>
      ::Deserialize(decompressed_serialized_policy, 0u).Value();
}

// Policy saving and loading concrete implementations

bool SaveVectorXdPolicy(
    const VectorXdPolicy& policy, const std::string& filename);

VectorXdPolicy LoadVectorXdPolicy(const std::string& filename);

// VectorXd Interface

VectorXdConfigVector DemonstrateVectorXdSimulator(
    const PLANNING_AND_EXECUTION_OPTIONS& options,
    const VectorXdRobotPtr& robot,
    const VectorXdSimulatorPtr& simulator,
    const VectorXdSamplerPtr& sampler,
    const VectorXdClusteringPtr& clustering,
    const VectorXdConfig& start,
    const VectorXdConfig& goal,
    const LoggingFunction& logging_fn,
    const DisplayFunction& display_fn);

VectorXdPolicyPlanningResult PlanVectorXdUncertainty(
    const PLANNING_AND_EXECUTION_OPTIONS& options,
    const VectorXdRobotPtr& robot,
    const VectorXdSimulatorPtr& simulator,
    const VectorXdSamplerPtr& sampler,
    const VectorXdClusteringPtr& clustering,
    const VectorXdConfig& start,
    const VectorXdConfig& goal,
    const double policy_marker_size,
    const LoggingFunction& logging_fn,
    const DisplayFunction& display_fn);

VectorXdPolicyPlanningResult PlanVectorXdUncertainty(
    const PLANNING_AND_EXECUTION_OPTIONS& options,
    const VectorXdRobotPtr& robot,
    const VectorXdSimulatorPtr& simulator,
    const VectorXdSamplerPtr& sampler,
    const VectorXdClusteringPtr& clustering,
    const VectorXdConfig& start,
    const VectorXdUserGoalStateCheckFn& user_goal_check_fn,
    const double policy_marker_size,
    const LoggingFunction& logging_fn,
    const DisplayFunction& display_fn);

VectorXdPolicyExecutionResult SimulateVectorXdUncertaintyPolicy(
    const PLANNING_AND_EXECUTION_OPTIONS& options,
    const VectorXdRobotPtr& robot,
    const VectorXdSimulatorPtr& simulator,
    const VectorXdSamplerPtr& sampler,
    const VectorXdClusteringPtr& clustering,
    const VectorXdPolicy& policy,
    const bool allow_branch_jumping,
    const bool link_runtime_states_to_planned_parent,
    const VectorXdConfig& start,
    const VectorXdConfig& goal,
    const double policy_marker_size,
    const LoggingFunction& logging_fn,
    const DisplayFunction& display_fn);

VectorXdPolicyExecutionResult ExecuteVectorXdUncertaintyPolicy(
    const PLANNING_AND_EXECUTION_OPTIONS& options,
    const VectorXdRobotPtr& robot,
    const VectorXdSimulatorPtr& simulator,
    const VectorXdSamplerPtr& sampler,
    const VectorXdClusteringPtr& clustering,
    const VectorXdPolicy& policy,
    const bool allow_branch_jumping,
    const bool link_runtime_states_to_planned_parent,
    const VectorXdConfig& start,
    const VectorXdConfig& goal,
    const double policy_marker_size,
    const VectorXdPolicyActionExecutionFunction& robot_execution_fn,
    const LoggingFunction& logging_fn,
    const DisplayFunction& display_fn);

VectorXdPolicyExecutionResult SimulateVectorXdUncertaintyPolicy(
    const PLANNING_AND_EXECUTION_OPTIONS& options,
    const VectorXdRobotPtr& robot,
    const VectorXdSimulatorPtr& simulator,
    const VectorXdSamplerPtr& sampler,
    const VectorXdClusteringPtr& clustering,
    const VectorXdPolicy& policy,
    const bool allow_branch_jumping,
    const bool link_runtime_states_to_planned_parent,
    const VectorXdConfig& start,
    const VectorXdUserGoalConfigCheckFn& user_goal_check_fn,
    const double policy_marker_size,
    const LoggingFunction& logging_fn,
    const DisplayFunction& display_fn);

VectorXdPolicyExecutionResult ExecuteVectorXdUncertaintyPolicy(
    const PLANNING_AND_EXECUTION_OPTIONS& options,
    const VectorXdRobotPtr& robot,
    const VectorXdSimulatorPtr& simulator,
    const VectorXdSamplerPtr& sampler,
    const VectorXdClusteringPtr& clustering,
    const VectorXdPolicy& policy,
    const bool allow_branch_jumping,
    const bool link_runtime_states_to_planned_parent,
    const VectorXdConfig& start,
    const VectorXdUserGoalConfigCheckFn& user_goal_check_fn,
    const double policy_marker_size,
    const VectorXdPolicyActionExecutionFunction& robot_execution_fn,
    const LoggingFunction& logging_fn,
    const DisplayFunction& display_fn);

inline std::ostream& operator<<(
    std::ostream& strm, const PLANNING_AND_EXECUTION_OPTIONS& options)
{
  strm << "OPTIONS:";
  strm << "\nplanner_time_limit: " << options.planner_time_limit;
  strm << "\np_goal_reached_termination_threshold: ";
  strm << options.p_goal_reached_termination_threshold;
  strm << "\ngoal_bias: " << options.goal_bias;
  strm << "\nstep_size: " << options.step_size;
  strm << "\ngoal_probability_threshold: ";
  strm << options.goal_probability_threshold;
  strm << "\ngoal_distance_threshold: " << options.goal_distance_threshold;
  strm << "\nconnect_after_first_solution: ";
  strm << options.connect_after_first_solution;
  strm << "\nfeasibility_alpha: " << options.feasibility_alpha;
  strm << "\nvariance_alpha: " << options.variance_alpha;
  strm << "\nedge_attempt_count: " << options.edge_attempt_count;
  strm << "\npolicy_action_attempt_count: ";
  strm << options.policy_action_attempt_count;
  strm << "\nnum_particles: " << options.num_particles;
  strm << "\nnum_policy_simulations: " << options.num_policy_simulations;
  strm << "\nnum_policy_executions: " << options.num_policy_executions;
  strm << "\nmax_exec_actions: " << options.max_exec_actions;
  strm << "\nmax_policy_exec_time: " << options.max_policy_exec_time;
  strm << "\ndebug_level: " << options.debug_level;
  strm << "\nuse_contact: " << options.use_contact;
  strm << "\nuse_reverse: " << options.use_reverse;
  strm << "\nuse_spur_actions: " << options.use_spur_actions;
  strm << "\nplanner_log_file: " << options.planner_log_file;
  strm << "\npolicy_log_file: " << options.policy_log_file;
  strm << "\nplanned_policy_file: " << options.planned_policy_file;
  strm << "\nexecuted_policy_file: " << options.executed_policy_file;
  return strm;
}
}  // namespace uncertainty_planning_core
