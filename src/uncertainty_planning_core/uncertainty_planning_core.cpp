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
#include <common_robotics_utilities/utility.hpp>
#include <common_robotics_utilities/math.hpp>
#include <uncertainty_planning_core/execution_policy.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#include <uncertainty_planning_core/uncertainty_contact_planning.hpp>
#include <uncertainty_planning_core/uncertainty_planner_state.hpp>
#include <uncertainty_planning_core/uncertainty_planning_core.hpp>

namespace uncertainty_planning_core
{
bool SaveVectorXdPolicy(
    const VectorXdPolicy& policy, const std::string& filename)
{
  return SavePolicy<
      VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>(
          policy, filename);
}

VectorXdPolicy LoadVectorXdPolicy(const std::string& filename)
{
  return LoadPolicy<
      VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>(filename);
}

inline double VectorXdUserGoalCheckWrapperFn(
    const VectorXdPlanningState& state,
    const VectorXdUserGoalConfigCheckFn& user_goal_config_check_fn)
{
  return UserGoalCheckWrapperFn<
      VectorXdConfig, VectorXdConfigSerializer, VectorXdConfigAlloc>(
          state, user_goal_config_check_fn);
}

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
    const DisplayFunction& display_fn)
{
  VectorXdPlanningSpace planning_space(
      options.debug_level, options.num_particles, options.step_size,
      options.goal_distance_threshold, options.goal_probability_threshold,
      options.feasibility_alpha, options.variance_alpha,
      options.connect_after_first_solution, robot, sampler, simulator,
      clustering, logging_fn);
  const auto trace
      = planning_space.DemonstrateSimulator(start, goal, display_fn);
  return ExtractTrajectoryFromTrace(trace);
}

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
    const DisplayFunction& display_fn)
{
    VectorXdPlanningSpace planning_space(
        options.debug_level, options.num_particles, options.step_size,
        options.goal_distance_threshold, options.goal_probability_threshold,
        options.feasibility_alpha, options.variance_alpha,
        options.connect_after_first_solution, robot, sampler, simulator,
        clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(
        options.planner_time_limit);
    return planning_space.PlanGoalState(
        start, goal, options.goal_bias, planner_time_limit,
        options.edge_attempt_count, options.policy_action_attempt_count,
        options.use_contact, options.use_reverse, options.use_spur_actions,
        policy_marker_size, options.p_goal_reached_termination_threshold,
        display_fn);
}

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
    const DisplayFunction& display_fn)
{
    VectorXdPlanningSpace planning_space(
        options.debug_level, options.num_particles, options.step_size,
        options.goal_distance_threshold, options.goal_probability_threshold,
        options.feasibility_alpha, options.variance_alpha,
        options.connect_after_first_solution, robot, sampler, simulator,
        clustering, logging_fn);
    const std::chrono::duration<double> planner_time_limit(
        options.planner_time_limit);
    return planning_space.PlanGoalSampling(
        start, options.goal_bias, user_goal_check_fn, planner_time_limit,
        options.edge_attempt_count, options.policy_action_attempt_count,
        options.use_contact, options.use_reverse, options.use_spur_actions,
        policy_marker_size, options.p_goal_reached_termination_threshold,
        display_fn);
}

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
    const DisplayFunction& display_fn)
{
    VectorXdPolicy working_policy = policy;
    VectorXdPlanningSpace planning_space(
        options.debug_level, options.num_particles, options.step_size,
        options.goal_distance_threshold, options.goal_probability_threshold,
        options.feasibility_alpha, options.variance_alpha,
        options.connect_after_first_solution, robot, sampler, simulator,
        clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(
        options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(
        working_policy, allow_branch_jumping,
        link_runtime_states_to_planned_parent, start, goal,
        options.num_policy_simulations, options.max_exec_actions, display_fn,
        policy_marker_size, true, 0.001);
}

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
    const DisplayFunction& display_fn)
{
    VectorXdPolicy working_policy = policy;
    VectorXdPlanningSpace planning_space(
        options.debug_level, options.num_particles, options.step_size,
        options.goal_distance_threshold, options.goal_probability_threshold,
        options.feasibility_alpha, options.variance_alpha,
        options.connect_after_first_solution, robot, sampler, simulator,
        clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(
        options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(
        working_policy, allow_branch_jumping,
        link_runtime_states_to_planned_parent, start, goal, robot_execution_fn,
        options.num_policy_executions, options.max_policy_exec_time, display_fn,
        policy_marker_size, false, 0.001);
}

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
    const DisplayFunction& display_fn)
{
    VectorXdPolicy working_policy = policy;
    VectorXdPlanningSpace planning_space(
        options.debug_level, options.num_particles, options.step_size,
        options.goal_distance_threshold, options.goal_probability_threshold,
        options.feasibility_alpha, options.variance_alpha,
        options.connect_after_first_solution, robot, sampler, simulator,
        clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(
        options.policy_action_attempt_count);
    return planning_space.SimulateExectionPolicy(
        working_policy, allow_branch_jumping,
        link_runtime_states_to_planned_parent, start, user_goal_check_fn,
        options.num_policy_simulations, options.max_exec_actions, display_fn,
        policy_marker_size, true, 0.001);
}

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
    const DisplayFunction& display_fn)
{
    VectorXdPolicy working_policy = policy;
    VectorXdPlanningSpace planning_space(
        options.debug_level, options.num_particles, options.step_size,
        options.goal_distance_threshold, options.goal_probability_threshold,
        options.feasibility_alpha, options.variance_alpha,
        options.connect_after_first_solution, robot, sampler, simulator,
        clustering, logging_fn);
    working_policy.SetPolicyActionAttemptCount(
        options.policy_action_attempt_count);
    return planning_space.ExecuteExectionPolicy(
        working_policy, allow_branch_jumping,
        link_runtime_states_to_planned_parent, start, user_goal_check_fn,
        robot_execution_fn, options.num_policy_executions,
        options.max_policy_exec_time, display_fn, policy_marker_size, false,
        0.001);
}
}  // namespace uncertainty_planning_core
