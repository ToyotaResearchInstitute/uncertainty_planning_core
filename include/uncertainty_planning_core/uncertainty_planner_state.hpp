#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <random>
#include <memory>
#include <common_robotics_utilities/maybe.hpp>
#include <common_robotics_utilities/print.hpp>
#include <common_robotics_utilities/math.hpp>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/simple_robot_model_interface.hpp>

namespace uncertainty_planning_core
{
template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc=std::allocator<Configuration>>
class UncertaintyPlannerState
{
protected:
  using Robot = common_robotics_utilities::simple_robot_model_interface
      ::SimpleRobotModelInterface<Configuration, ConfigAlloc>;

  Configuration expectation_;
  Configuration command_;
  Eigen::VectorXd variances_;
  Eigen::VectorXd space_independent_variances_;
  std::vector<Configuration, ConfigAlloc> particles_;
  double step_size_;
  double parent_motion_Pfeasibility_;
  double raw_edge_Pfeasibility_;
  double effective_edge_Pfeasibility_;
  double reverse_edge_Pfeasibility_;
  double motion_Pfeasibility_;
  double variance_;
  double space_independent_variance_;
  double goal_Pfeasibility_;
  uint64_t state_id_;
  uint64_t transition_id_;
  uint64_t reverse_transition_id_;
  uint64_t split_id_;
  uint32_t attempt_count_;
  uint32_t reached_count_;
  uint32_t reverse_attempt_count_;
  uint32_t reverse_reached_count_;
  bool initialized_;
  bool has_particles_;
  bool use_for_nearest_neighbors_;
  bool action_outcome_is_nominally_independent_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static uint64_t Serialize(
      const UncertaintyPlannerState<
          Configuration, ConfigSerializer, ConfigAlloc>& state,
      std::vector<uint8_t>& buffer)
  {
      return state.SerializeSelf(buffer);
  }

  static std::string GetConfigurationType()
  {
      return ConfigSerializer::TypeName();
  }

  uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const
  {
    using common_robotics_utilities::serialization::SerializeMemcpyable;
    using common_robotics_utilities::serialization::SerializeString;
    using common_robotics_utilities::serialization::SerializeVectorXd;
    using common_robotics_utilities::serialization::SerializeVectorLike;
    // Takes a state to serialize and a buffer to serialize into
    // Return number of bytes written to buffer
    if (initialized_ == false)
    {
        throw std::runtime_error("Cannot serialize an unitialized state");
    }
    const uint64_t start_buffer_size = buffer.size();
    // First thing we save is the qualified type id
    SerializeMemcpyable<uint64_t>(std::numeric_limits<uint64_t>::max(), buffer);
    SerializeString<char>(GetConfigurationType(), buffer);
    SerializeMemcpyable<uint8_t>(
        static_cast<uint8_t>(has_particles_), buffer);
    SerializeMemcpyable<uint8_t>(
        static_cast<uint8_t>(use_for_nearest_neighbors_), buffer);
    SerializeMemcpyable<uint8_t>(
        static_cast<uint8_t>(action_outcome_is_nominally_independent_), buffer);
    SerializeMemcpyable<uint32_t>(attempt_count_, buffer);
    SerializeMemcpyable<uint32_t>(reached_count_, buffer);
    SerializeMemcpyable<uint32_t>(reverse_attempt_count_, buffer);
    SerializeMemcpyable<uint32_t>(reverse_reached_count_, buffer);
    SerializeMemcpyable<double>(step_size_, buffer);
    SerializeMemcpyable<double>(parent_motion_Pfeasibility_, buffer);
    SerializeMemcpyable<double>(effective_edge_Pfeasibility_, buffer);
    SerializeMemcpyable<double>(motion_Pfeasibility_, buffer);
    SerializeMemcpyable<double>(variance_, buffer);
    SerializeMemcpyable<double>(space_independent_variance_, buffer);
    SerializeMemcpyable<uint64_t>(state_id_, buffer);
    SerializeMemcpyable<uint64_t>(transition_id_, buffer);
    SerializeMemcpyable<uint64_t>(reverse_transition_id_, buffer);
    SerializeMemcpyable<uint64_t>(split_id_, buffer);
    SerializeMemcpyable<double>(goal_Pfeasibility_, buffer);
    ConfigSerializer::Serialize(expectation_, buffer);
    ConfigSerializer::Serialize(command_, buffer);
    SerializeVectorXd(variances_, buffer);
    SerializeVectorXd(space_independent_variances_, buffer);
    // Serialize the particles
    SerializeVectorLike<Configuration, std::vector<Configuration, ConfigAlloc>>(
        particles_, buffer, &ConfigSerializer::Serialize);
    // Figure out how many bytes we wrote
    const uint64_t end_buffer_size = buffer.size();
    const uint64_t bytes_written = end_buffer_size - start_buffer_size;
    return bytes_written;
  }

  static common_robotics_utilities::serialization::Deserialized<
      UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>>
  Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
  {
    UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>
        temp_state;
    const uint64_t bytes_read = temp_state.DeserializeSelf(buffer, current);
    return common_robotics_utilities::serialization::MakeDeserialized(
        temp_state, bytes_read);
  }

  uint64_t DeserializeSelf(
      const std::vector<uint8_t>& buffer, const uint64_t current)
  {
    using common_robotics_utilities::serialization::DeserializeMemcpyable;
    using common_robotics_utilities::serialization::DeserializeString;
    using common_robotics_utilities::serialization::DeserializeVectorXd;
    using common_robotics_utilities::serialization::DeserializeVectorLike;
    uint64_t current_position = current;
    // First thing we load and check is the qualified type ID so we know that
    // we're loading our state properly
    // First thing we save is the qualified type id
    const uint64_t reference_qualified_type_id_hash
        = std::numeric_limits<uint64_t>::max();
    const std::string reference_configuration_type = GetConfigurationType();
    const auto deserialized_qualified_type_id_hash
        = DeserializeMemcpyable<uint64_t>(buffer, current_position);
    const uint64_t qualified_type_id_hash
        = deserialized_qualified_type_id_hash.Value();
    current_position += deserialized_qualified_type_id_hash.BytesRead();
    // Check types
    // If the file used the legacy type ID, we can't safely check it
    // (std::hash is not required to be consistent across program executions!)
    // so we warn the user and continue
    if (qualified_type_id_hash == reference_qualified_type_id_hash)
    {
      const auto deserialized_configuration_type
          = DeserializeString<char>(buffer, current_position);
      const std::string& configuration_type
          = deserialized_configuration_type.Value();
      current_position += deserialized_configuration_type.BytesRead();
      if (configuration_type != reference_configuration_type)
      {
        std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                     "!!!!!!!!!!!!!!!!!!!!!!!\nLoaded configuration type: ["
                  << configuration_type << "] does not match expected ["
                  << reference_configuration_type << "]\nPROCEED WITH CAUTION -"
                  << " THIS MAY CAUSE UNDEFINED BEHAVIOR IN LOADING\n!!!!!!!!!!"
                  << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                  << "!!!!!!!!!!!!!" << std::endl;
      }
    }
    else
    {
      std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                << "!!!!!!!!!!!!!!!!!!!!!\nLoaded file uses old TypeId hash and"
                << " cannot be safely checked\nPROCEED WITH CAUTION - THIS MAY"
                << " CAUSE UNDEFINED BEHAVIOR IN LOADING\n!!!!!!!!!!!!!!!!!!!!!"
                << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                << std::endl;
    }
    // Load fixed size members
    const auto deserialized_has_particles
        = DeserializeMemcpyable<uint8_t>(buffer, current_position);
    has_particles_ = static_cast<bool>(deserialized_has_particles.Value());
    current_position += deserialized_has_particles.BytesRead();
    const auto deserialized_use_for_nearest_neighbors
        = DeserializeMemcpyable<uint8_t>(buffer, current_position);
    use_for_nearest_neighbors_
        = static_cast<bool>(deserialized_use_for_nearest_neighbors.Value());
    current_position += deserialized_use_for_nearest_neighbors.BytesRead();
    const auto deserialized_action_outcome_is_nominally_independent
        = DeserializeMemcpyable<uint8_t>(buffer, current_position);
    action_outcome_is_nominally_independent_
        = static_cast<bool>(
            deserialized_action_outcome_is_nominally_independent.Value());
    current_position
        += deserialized_action_outcome_is_nominally_independent.BytesRead();
    const auto deserialized_attempt_count
        = DeserializeMemcpyable<uint32_t>(buffer, current_position);
    attempt_count_ = deserialized_attempt_count.Value();
    current_position += deserialized_attempt_count.BytesRead();
    const auto deserialized_reached_count
        = DeserializeMemcpyable<uint32_t>(buffer, current_position);
    reached_count_ = deserialized_reached_count.Value();
    current_position += deserialized_reached_count.BytesRead();
    raw_edge_Pfeasibility_
        = static_cast<double>(reached_count_)
            / static_cast<double>(attempt_count_);
    const auto deserialized_reverse_attempt_count
        = DeserializeMemcpyable<uint32_t>(buffer, current_position);
    reverse_attempt_count_ = deserialized_reverse_attempt_count.Value();
    current_position += deserialized_reverse_attempt_count.BytesRead();
    const auto deserialized_reverse_reached_count
        = DeserializeMemcpyable<uint32_t>(buffer, current_position);
    reverse_reached_count_ = deserialized_reverse_reached_count.Value();
    current_position += deserialized_reverse_reached_count.BytesRead();
    reverse_edge_Pfeasibility_
        = static_cast<double>(reverse_reached_count_)
            / static_cast<double>(reverse_attempt_count_);
    const auto deserialized_step_size
        = DeserializeMemcpyable<double>(buffer, current_position);
    step_size_ = deserialized_step_size.Value();
    current_position += deserialized_step_size.BytesRead();
    const auto deserialized_parent_motion_Pfeasibility
        = DeserializeMemcpyable<double>(buffer, current_position);
    parent_motion_Pfeasibility_ = deserialized_parent_motion_Pfeasibility.Value();
    current_position += deserialized_parent_motion_Pfeasibility.BytesRead();
    const auto deserialized_effective_edge_Pfeasibility
        = DeserializeMemcpyable<double>(buffer, current_position);
    effective_edge_Pfeasibility_
        = deserialized_effective_edge_Pfeasibility.Value();
    current_position += deserialized_effective_edge_Pfeasibility.BytesRead();
    const auto deserialized_motion_Pfeasibility
        = DeserializeMemcpyable<double>(buffer, current_position);
    motion_Pfeasibility_ = deserialized_motion_Pfeasibility.Value();
    current_position += deserialized_motion_Pfeasibility.BytesRead();
    const auto deserialized_variance
        = DeserializeMemcpyable<double>(buffer, current_position);
    variance_ = deserialized_variance.Value();
    current_position += deserialized_variance.BytesRead();
    const auto deserialized_space_independent_variance
        = DeserializeMemcpyable<double>(buffer, current_position);
    space_independent_variance_ = deserialized_space_independent_variance.Value();
    current_position += deserialized_space_independent_variance.BytesRead();
    const auto deserialized_state_id
        = DeserializeMemcpyable<uint64_t>(buffer, current_position);
    state_id_ = deserialized_state_id.Value();
    current_position += deserialized_state_id.BytesRead();
    const auto deserialized_transition_id
        = DeserializeMemcpyable<uint64_t>(buffer, current_position);
    transition_id_ = deserialized_transition_id.Value();
    current_position += deserialized_transition_id.BytesRead();
    const auto deserialized_reverse_transition_id
        = DeserializeMemcpyable<uint64_t>(buffer, current_position);
    reverse_transition_id_ = deserialized_reverse_transition_id.Value();
    current_position += deserialized_reverse_transition_id.BytesRead();
    const auto deserialized_split_id
        = DeserializeMemcpyable<uint64_t>(buffer, current_position);
    split_id_ = deserialized_split_id.Value();
    current_position += deserialized_split_id.BytesRead();
    const auto deserialized_goal_Pfeasibility
        = DeserializeMemcpyable<double>(buffer, current_position);
    goal_Pfeasibility_ = deserialized_goal_Pfeasibility.Value();
    current_position += deserialized_goal_Pfeasibility.BytesRead();
    // Load the variable-sized components
    const auto deserialized_expectation
        = ConfigSerializer::Deserialize(buffer, current_position);
    expectation_ = deserialized_expectation.Value();
    current_position += deserialized_expectation.BytesRead();
    const auto deserialized_command
        = ConfigSerializer::Deserialize(buffer, current_position);
    command_ = deserialized_command.Value();
    current_position += deserialized_command.BytesRead();
    const auto deserialized_variances
        = DeserializeVectorXd(buffer, current_position);
    variances_ = deserialized_variances.Value();
    current_position += deserialized_variances.BytesRead();
    const auto deserialized_space_independent_variances
        = DeserializeVectorXd(buffer, current_position);
    space_independent_variances_
        = deserialized_space_independent_variances.Value();
    current_position += deserialized_space_independent_variances.BytesRead();
    // Load the particles
    const auto deserialized_particles
        = DeserializeVectorLike<Configuration,
                                std::vector<Configuration, ConfigAlloc>>(
            buffer, current_position, &ConfigSerializer::Deserialize);
    particles_ = deserialized_particles.Value();
    current_position += deserialized_particles.BytesRead();
    // Initialize the state
    initialized_ = true;
    // Return how many bytes we read from the buffer
    const uint64_t bytes_read = current_position - current;
    return bytes_read;
  }

  inline UncertaintyPlannerState(const Configuration& expectation)
  {
    state_id_ = 0u;
    step_size_ = 0.0;
    expectation_ = expectation;
    particles_.push_back(expectation_);
    variance_ = 0.0;
    variances_ = Eigen::VectorXd();
    space_independent_variance_ = 0.0;
    space_independent_variances_ = Eigen::VectorXd();
    attempt_count_ = 1u;
    reached_count_ = 1u;
    reverse_attempt_count_ = 1u;
    reverse_reached_count_ = 1u;
    parent_motion_Pfeasibility_ = 1.0;
    raw_edge_Pfeasibility_ = 1.0;
    effective_edge_Pfeasibility_ = 1.0;
    reverse_edge_Pfeasibility_ = 1.0;
    motion_Pfeasibility_ = 1.0;
    initialized_ = true;
    has_particles_ = true;
    use_for_nearest_neighbors_ = true;
    action_outcome_is_nominally_independent_ = true;
    command_ = expectation_;
    split_id_ = 0u;
    transition_id_ = 0;
    reverse_transition_id_ = 0;
    goal_Pfeasibility_ = 0.0;
  }

  inline UncertaintyPlannerState(
      const std::vector<Configuration, ConfigAlloc>& particles,
      const double step_size)
  {
    state_id_ = 0u;
    step_size_ = step_size;
    particles_ = particles;
    variance_ = 0.0;
    variances_ = Eigen::VectorXd();
    space_independent_variance_ = 0.0;
    space_independent_variances_ = Eigen::VectorXd();
    attempt_count_ = 1u;
    reached_count_ = 1u;
    reverse_attempt_count_ = 1u;
    reverse_reached_count_ = 1u;
    parent_motion_Pfeasibility_ = 1.0;
    raw_edge_Pfeasibility_ = 1.0;
    effective_edge_Pfeasibility_ = 1.0;
    reverse_edge_Pfeasibility_ = 1.0;
    motion_Pfeasibility_ = 1.0;
    initialized_ = true;
    has_particles_ = true;
    use_for_nearest_neighbors_ = true;
    action_outcome_is_nominally_independent_ = true;
    command_ = expectation_;
    split_id_ = 0u;
    transition_id_ = 0;
    reverse_transition_id_ = 0;
    goal_Pfeasibility_ = 0.0;
  }

  UncertaintyPlannerState(
      const uint64_t state_id, const Configuration& particle,
      const uint32_t attempt_count, const uint32_t reached_count,
      const double effective_edge_Pfeasibility,
      const uint32_t reverse_attempt_count,
      const uint32_t reverse_reached_count,
      const double parent_motion_Pfeasibility, const double step_size,
      const Configuration& command, const uint64_t transition_id,
      const uint64_t reverse_transition_id, const uint64_t split_id,
      const bool action_outcome_is_nominally_independent)
  {
    state_id_ = state_id;
    step_size_ = step_size;
    expectation_ = particle;
    particles_.push_back(expectation_);
    variance_ = 0.0;
    variances_ = Eigen::VectorXd();
    space_independent_variance_ = 0.0;
    space_independent_variances_ = Eigen::VectorXd();
    attempt_count_ = attempt_count;
    reached_count_ = reached_count;
    reverse_attempt_count_ = reverse_attempt_count;
    reverse_reached_count_ = reverse_reached_count;
    parent_motion_Pfeasibility_ = parent_motion_Pfeasibility;
    raw_edge_Pfeasibility_
        = static_cast<double>(reached_count_)
            / static_cast<double>(attempt_count_);
    effective_edge_Pfeasibility_ = effective_edge_Pfeasibility;
    reverse_edge_Pfeasibility_
        = static_cast<double>(reverse_reached_count_)
            / static_cast<double>(reverse_attempt_count_);
    motion_Pfeasibility_
        = effective_edge_Pfeasibility_ * parent_motion_Pfeasibility_;
    initialized_ = true;
    has_particles_ = true;
    use_for_nearest_neighbors_ = true;
    action_outcome_is_nominally_independent_
        = action_outcome_is_nominally_independent;
    command_ = command;
    transition_id_ = transition_id;
    reverse_transition_id_ = reverse_transition_id;
    split_id_ = split_id;
    goal_Pfeasibility_ = 0.0;
  }

  UncertaintyPlannerState(
      const uint64_t state_id,
      const std::vector<Configuration, ConfigAlloc>& particles,
      const uint32_t attempt_count, const uint32_t reached_count,
      const double effective_edge_Pfeasibility,
      const uint32_t reverse_attempt_count,
      const uint32_t reverse_reached_count,
      const double parent_motion_Pfeasibility, const double step_size,
      const Configuration& command, const uint64_t transition_id,
      const uint64_t reverse_transition_id, const uint64_t split_id,
      const bool action_outcome_is_nominally_independent)
  {
      state_id_ = state_id;
      step_size_ = step_size;
      particles_ = particles;
      attempt_count_ = attempt_count;
      reached_count_ = reached_count;
      reverse_attempt_count_ = reverse_attempt_count;
      reverse_reached_count_ = reverse_reached_count;
      parent_motion_Pfeasibility_ = parent_motion_Pfeasibility;
      raw_edge_Pfeasibility_
          = static_cast<double>(reached_count_)
              / static_cast<double>(attempt_count_);
      effective_edge_Pfeasibility_ = effective_edge_Pfeasibility;
      reverse_edge_Pfeasibility_
          = static_cast<double>(reverse_reached_count_)
              / static_cast<double>(reverse_attempt_count_);
      motion_Pfeasibility_
          = effective_edge_Pfeasibility_ * parent_motion_Pfeasibility_;
      initialized_ = true;
      has_particles_ = true;
      use_for_nearest_neighbors_ = true;
      action_outcome_is_nominally_independent_
          = action_outcome_is_nominally_independent;
      command_ = command;
      transition_id_ = transition_id;
      reverse_transition_id_ = reverse_transition_id;
      split_id_ = split_id;
      goal_Pfeasibility_ = 0.0;
  }

  void UpdateStatistics(const std::shared_ptr<Robot>& robot_ptr)
  {
      std::function<Configuration(
          const std::vector<Configuration, ConfigAlloc>&)> average_fn
          = [&] (const std::vector<Configuration, ConfigAlloc>& particles)
      {
        return robot_ptr->AverageConfigurations(particles);
      };
      std::function<double(
          const Configuration&, const Configuration&)> distance_fn
          = [&] (const Configuration& config1, const Configuration& config2)
      {
        return robot_ptr->ComputeConfigurationDistance(config1, config2);
      };
      std::function<Eigen::VectorXd(
          const Configuration&, const Configuration&)> dim_distance_fn
          = [&] (const Configuration& config1, const Configuration& config2)
      {
        return robot_ptr->ComputePerDimensionConfigurationDistance(config1,
                                                                   config2);
      };
      expectation_ = ComputeExpectation(average_fn);
      variance_ = ComputeVariance(expectation_, distance_fn);
      variances_ = ComputeDirectionalVariance(expectation_, dim_distance_fn);
      space_independent_variance_
          = ComputeSpaceIndependentVariance(
              expectation_, distance_fn, step_size_);
      space_independent_variances_
          = ComputeSpaceIndependentDirectionalVariance(
              expectation_, dim_distance_fn, step_size_);
  }

  inline UncertaintyPlannerState()
    : goal_Pfeasibility_(0.0), state_id_(0), transition_id_(0),
      reverse_transition_id_(0), split_id_(0u), initialized_(false),
      has_particles_(false), use_for_nearest_neighbors_(false),
      action_outcome_is_nominally_independent_(false) {}

  bool IsInitialized() const { return initialized_; }

  bool HasParticles() const { return has_particles_; }

  bool UseForNearestNeighbors() const { return use_for_nearest_neighbors_; }

  bool IsActionOutcomeNominallyIndependent() const
  {
    return action_outcome_is_nominally_independent_;
  }

  void EnableForNearestNeighbors() { use_for_nearest_neighbors_ = true; }

  void DisableForNearestNeighbors() { use_for_nearest_neighbors_ = false; }

  double GetStepSize() const { return step_size_; }

  double GetRawEdgePfeasibility() const { return raw_edge_Pfeasibility_; }

  double GetEffectiveEdgePfeasibility() const
  {
    return effective_edge_Pfeasibility_;
  }

  double GetReverseEdgePfeasibility() const
  {
    return reverse_edge_Pfeasibility_;
  }

  double GetMotionPfeasibility() const { return motion_Pfeasibility_; }

  const Configuration& GetExpectation() const { return expectation_; }

  double GetGoalPfeasibility() const { return goal_Pfeasibility_; }

  std::pair<uint32_t, uint32_t> GetAttemptAndReachedCounts() const
  {
    return std::make_pair(attempt_count_, reached_count_);
  }

  std::pair<uint32_t, uint32_t> GetReverseAttemptAndReachedCounts() const
  {
    return std::make_pair(reverse_attempt_count_, reverse_reached_count_);
  }

  void UpdateAttemptAndReachedCounts(
      const uint32_t attempt_count, const uint32_t reached_count)
  {
    attempt_count_ = attempt_count;
    reached_count_ = reached_count;
    raw_edge_Pfeasibility_
        = static_cast<double>(reached_count_)
            / static_cast<double>(attempt_count_);
  }

  void UpdateReverseAttemptAndReachedCounts(
      const uint32_t reverse_attempt_count,
      const uint32_t reverse_reached_count)
  {
    reverse_attempt_count_ = reverse_attempt_count;
    reverse_reached_count_ = reverse_reached_count;
    reverse_edge_Pfeasibility_
        = static_cast<double>(reverse_reached_count_)
            / static_cast<double>(reverse_attempt_count_);
  }

  double SetEffectiveEdgePfeasibility(const double effective_edge_Pfeasibility)
  {
    effective_edge_Pfeasibility_ = effective_edge_Pfeasibility;
    motion_Pfeasibility_
        = effective_edge_Pfeasibility_ * parent_motion_Pfeasibility_;
    return motion_Pfeasibility_;
  }

  void SetGoalPfeasibility(const double goal_Pfeasibility)
  {
    // We allow negative values to signify reverse edges!
    if ((goal_Pfeasibility <= 1.0) && (goal_Pfeasibility >= -1.0))
    {
      goal_Pfeasibility_ = goal_Pfeasibility;
    }
    else
    {
      throw std::invalid_argument("goal_Pfeasibility out of range [-1, 1]");
    }
  }

  void SetReverseEdgePfeasibility(const double reverse_edge_Pfeasibility)
  {
    if ((reverse_edge_Pfeasibility <= 1.0)
        && (reverse_edge_Pfeasibility >= 0.0))
    {
      reverse_edge_Pfeasibility_ = reverse_edge_Pfeasibility;
    }
    else
    {
      throw std::invalid_argument
          ("reverse_edge_Pfeasibility out of range [0, 1]");
    }
  }

  uint64_t GetStateId() const { return state_id_; }

  uint64_t GetTransitionId() const { return transition_id_; }

  uint64_t GetReverseTransitionId() const { return reverse_transition_id_; }

  uint64_t GetSplitId() const { return split_id_; }

  const Configuration& GetCommand() const { return command_; }

  void SetCommand(const Configuration& command) { command_ = command; }

  size_t GetNumParticles() const { return particles_.size(); }

  common_robotics_utilities::ReferencingMaybe<
      const std::vector<Configuration, ConfigAlloc>>
  GetParticlePositionsImmutable() const
  {
    using common_robotics_utilities::ReferencingMaybe;
    if (has_particles_)
    {
      return ReferencingMaybe<const std::vector<Configuration, ConfigAlloc>>(
          particles_);
    }
    else
    {
      return ReferencingMaybe<const std::vector<Configuration, ConfigAlloc>>();
    }
  }

  common_robotics_utilities::ReferencingMaybe<
      std::vector<Configuration, ConfigAlloc>> GetParticlePositionsMutable()
  {
    using common_robotics_utilities::ReferencingMaybe;
    if (has_particles_)
    {
      return ReferencingMaybe<std::vector<Configuration, ConfigAlloc>>(
          particles_);
    }
    else
    {
      return ReferencingMaybe<std::vector<Configuration, ConfigAlloc>>();
    }
  }

  std::vector<Configuration, ConfigAlloc> CollectParticles(
      const size_t num_particles) const
  {
    if (particles_.size() == 0)
    {
      return std::vector<Configuration, ConfigAlloc>(
          num_particles, expectation_);
    }
    else if (particles_.size() == 1)
    {
      return std::vector<Configuration, ConfigAlloc>(
          num_particles, particles_[0]);
    }
    else
    {
      if (num_particles == particles_.size())
      {
        return particles_;
      }
      else
      {
        throw std::invalid_argument(
            "CollectParticles() called with particles_.size() > 1, and"
            " num_particles != particles_.size(). You must use"
            " ResampleParticles() instead.");
      }
    }
  }

  template<typename RNG>
  std::vector<Configuration, ConfigAlloc> ResampleParticles(
      const size_t num_particles, RNG& rng) const
  {
    if (particles_.size() == 0)
    {
      return std::vector<Configuration, ConfigAlloc>(
            num_particles, expectation_);
    }
    else if (particles_.size() == 1)
    {
      return std::vector<Configuration, ConfigAlloc>(
            num_particles, particles_[0]);
    }
    else
    {
      std::vector<Configuration, ConfigAlloc> resampled_particles(
          num_particles);
      double particle_probability
          = 1.0 / static_cast<double>(particles_.size());
      std::uniform_int_distribution<size_t> resampling_distribution(
          0, particles_.size() - 1);
      std::uniform_real_distribution<double> importance_sampling_distribution(
          0.0, 1.0);
      size_t resampled = 0;
      while (resampled < num_particles)
      {
        size_t random_index = resampling_distribution(rng);
        const Configuration& random_particle = particles_[random_index];
        if (importance_sampling_distribution(rng) < particle_probability)
        {
          resampled_particles[resampled] = random_particle;
          resampled++;
        }
      }
      return resampled_particles;
    }
  }

  double GetVariance() const { return variance_; }

  const Eigen::VectorXd& GetVariances() const { return variances_; }

  double GetSpaceIndependentVariance() const
  {
    return space_independent_variance_;
  }

  const Eigen::VectorXd& GetSpaceIndependentVariances() const
  {
    return space_independent_variances_;
  }

  Configuration ComputeExpectation(
      const std::function<Configuration(
          const std::vector<Configuration, ConfigAlloc>&)>& average_fn) const
  {
    if (particles_.size() == 0)
    {
      return expectation_;
    }
    else if (particles_.size() == 1)
    {
      return particles_[0];
    }
    else
    {
      return average_fn(particles_);
    }
  }

  double ComputeVariance(
      const Configuration& expectation,
      const std::function<double(
          const Configuration&, const Configuration&)>& distance_fn) const
  {
    if (particles_.size() == 0)
    {
      return 0.0;
    }
    else if (particles_.size() == 1)
    {
      return 0.0;
    }
    else
    {
      const double weight = 1.0 / static_cast<double>(particles_.size());
      double var_sum = 0.0;
      for (size_t idx = 0; idx < particles_.size(); idx++)
      {
        const double raw_distance = distance_fn(expectation, particles_[idx]);
        const double squared_distance = pow(raw_distance, 2.0);
        var_sum += (squared_distance * weight);
      }
      return var_sum;
    }
  }

  double ComputeSpaceIndependentVariance(
      const Configuration& expectation,
      const std::function<double(
          const Configuration&, const Configuration&)>& distance_fn,
      const double step_size) const
  {
    if (particles_.size() == 0)
    {
      return 0.0;
    }
    else if (particles_.size() == 1)
    {
      return 0.0;
    }
    else
    {
      const double weight = 1.0 / static_cast<double>(particles_.size());
      double var_sum = 0.0;
      for (size_t idx = 0; idx < particles_.size(); idx++)
      {
        const double raw_distance = distance_fn(expectation, particles_[idx]);
        const double space_independent_distance = raw_distance / step_size;
        const double squared_distance = pow(space_independent_distance, 2.0);
        var_sum += (squared_distance * weight);
      }
      return var_sum;
    }
  }

  Eigen::VectorXd ComputeDirectionalVariance(
      const Configuration& expectation,
      const std::function<Eigen::VectorXd(
          const Configuration&, const Configuration&)>& dim_distance_fn) const
  {
    if (particles_.size() == 0)
    {
      return dim_distance_fn(expectation, expectation);
    }
    else if (particles_.size() == 1)
    {
      return dim_distance_fn(particles_[0], particles_[0]);
    }
    else
    {
      const double weight = 1.0 / static_cast<double>(particles_.size());
      Eigen::VectorXd variances;
      for (size_t idx = 0; idx < particles_.size(); idx++)
      {
        const Eigen::VectorXd error
            = dim_distance_fn(expectation, particles_[idx]);
        const Eigen::VectorXd squared_error = error.cwiseProduct(error);
        const Eigen::VectorXd weighted_squared_error = squared_error * weight;
        if (variances.size() != weighted_squared_error.size())
        {
          variances.setZero(weighted_squared_error.size());
        }
        variances += weighted_squared_error;
      }
      return variances;
    }
  }

  Eigen::VectorXd ComputeSpaceIndependentDirectionalVariance(
      const Configuration& expectation,
      const std::function<Eigen::VectorXd(
          const Configuration&, const Configuration&)>& dim_distance_fn,
      const double step_size) const
  {
    if (particles_.size() == 0)
    {
      return dim_distance_fn(expectation, expectation);
    }
    else if (particles_.size() == 1)
    {
      return dim_distance_fn(particles_[0], particles_[0]);
    }
    else
    {
      const double weight = 1.0 / static_cast<double>(particles_.size());
      Eigen::VectorXd variances;
      for (size_t idx = 0; idx < particles_.size(); idx++)
      {
        const Eigen::VectorXd error
            = dim_distance_fn(expectation, particles_[idx]);
        const Eigen::VectorXd space_independent_error = error / step_size;
        const Eigen::VectorXd squared_error
            = space_independent_error.cwiseProduct(space_independent_error);
        const Eigen::VectorXd weighted_squared_error = squared_error * weight;
        if (variances.size() != weighted_squared_error.size())
        {
          variances.setZero(weighted_squared_error.size());
        }
        variances += weighted_squared_error;
      }
      return variances;
    }
  }

  std::string Print() const
  {
    std::ostringstream strm;
    strm << "Uncertainty Planner State (Configuration Type: "
         << GetConfigurationType() << ") - Expectation: "
         << common_robotics_utilities::print::Print(GetExpectation())
         << " Command: "
         << common_robotics_utilities::print::Print(GetCommand())
         << " Action outcome is nominally independent ["
         << common_robotics_utilities::print::Print(
              action_outcome_is_nominally_independent_)
         << "] Variance: " << GetVariance() << " Space-independent Variance: "
         << GetSpaceIndependentVariance() << " Raw Pfeasibility(parent->this): "
         << GetRawEdgePfeasibility()
         << " Effective Pfeasibility(parent->this): "
         << GetEffectiveEdgePfeasibility()
         << " Raw Pfeasibility(this->parent): " << GetReverseEdgePfeasibility()
         << " Pfeasibility(start->this): " << GetMotionPfeasibility();
    return strm.str();
  }
};
}

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc=std::allocator<Configuration>>
std::ostream& operator<<(
    std::ostream& strm,
    const uncertainty_planning_core::UncertaintyPlannerState<
        Configuration, ConfigSerializer, ConfigAlloc>& state)
{
    strm << state.Print();
    return strm;
}
