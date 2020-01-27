#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <queue>
#include <common_robotics_utilities/openmp_helpers.hpp>
#include <common_robotics_utilities/print.hpp>
#include <common_robotics_utilities/serialization.hpp>
#include <common_robotics_utilities/simple_rrt_planner.hpp>
#include <common_robotics_utilities/simple_graph.hpp>
#include <common_robotics_utilities/simple_graph_search.hpp>
#include <common_robotics_utilities/simple_knearest_neighbors.hpp>
#include <uncertainty_planning_core/uncertainty_planner_state.hpp>
#include <visualization_msgs/MarkerArray.h>

namespace uncertainty_planning_core
{
using DisplayFunction
    = std::function<void(const visualization_msgs::MarkerArray&)>;
using LoggingFunction
    = std::function<void(const std::string&, const int32_t)>;

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc=std::allocator<Configuration>>
class PolicyGraphBuilder
{
private:
  // Typedef so we don't hate ourselves
  using UncertaintyPlanningState
      = UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>;
  using UncertaintyPlanningTreeState
      = common_robotics_utilities::simple_rrt_planner::SimpleRRTPlannerState<
          UncertaintyPlanningState>;
  using UncertaintyPlanningTree
      = common_robotics_utilities::simple_rrt_planner::PlanningTree<
          UncertaintyPlanningState>;
  using PolicyGraphNode
      = common_robotics_utilities::simple_graph::GraphNode<
          UncertaintyPlanningState>;
  using PolicyGraph
      = common_robotics_utilities::simple_graph::Graph<
          UncertaintyPlanningState>;

  PolicyGraphBuilder() {}

public:
  static PolicyGraph BuildPolicyGraphFromPlannerTree(
      const UncertaintyPlanningTree& planner_tree,
      const UncertaintyPlanningState& goal_state)
  {
    if (planner_tree.empty())
    {
      throw std::invalid_argument(
          "Cannot build policy graph from empty planner tree");
    }
    if (!common_robotics_utilities::simple_rrt_planner
        ::CheckTreeLinkage(planner_tree))
    {
      throw std::invalid_argument("Provided planner tree has invalid linkage");
    }
    PolicyGraph policy_graph(planner_tree.size() + 1);
    // First pass, add all the nodes to the graph
    for (const auto& current_tree_state : planner_tree)
    {
      const UncertaintyPlanningState& current_planner_state
          = current_tree_state.GetValueImmutable();
      policy_graph.AddNode(PolicyGraphNode(current_planner_state));
    }
    const int64_t goal_idx = policy_graph.AddNode(PolicyGraphNode(goal_state));
    policy_graph.ShrinkToFit();
    // Second pass, add all the edges
    for (size_t idx = 0; idx < planner_tree.size(); idx++)
    {
      const UncertaintyPlanningTreeState& current_tree_state
          = planner_tree.at(idx);
      const UncertaintyPlanningState& current_planner_state
          = current_tree_state.GetValueImmutable();
      const int64_t parent_index = current_tree_state.GetParentIndex();
      const std::vector<int64_t>& child_indices
          = current_tree_state.GetChildIndices();
      if (parent_index >= 0)
      {
        const double edge_weight
            = current_planner_state.GetReverseEdgePfeasibility();
        policy_graph.AddEdgeBetweenNodes(
            static_cast<int64_t>(idx), parent_index, edge_weight);
      }
      for (const int64_t child_index : child_indices)
      {
        const double edge_weight
            = planner_tree.at(static_cast<size_t>(child_index))
                .GetValueImmutable().GetEffectiveEdgePfeasibility();
        policy_graph.AddEdgeBetweenNodes(
            static_cast<int64_t>(idx), child_index, edge_weight);
      }
      // Detect if we are a goal state and add edges to the goal
      if (child_indices.size() == 0
          && current_planner_state.GetGoalPfeasibility() > 0.0)
      {
        const double edge_weight = current_planner_state.GetGoalPfeasibility();
        policy_graph.AddEdgesBetweenNodes(
            static_cast<int64_t>(idx), goal_idx, edge_weight);
      }
    }
    if (policy_graph.CheckGraphLinkage() == false)
    {
      throw std::runtime_error("Generated policy graph has invalid linkage");
    }
    return policy_graph;
  }

  static uint32_t ComputeEstimatedEdgeAttemptCount(
      const PolicyGraph& graph,
      const common_robotics_utilities::simple_graph::GraphEdge& current_edge,
      const double conformant_planning_threshold,
      const uint32_t edge_repeat_threshold)
  {
    // First, safety checks to make sure the graph + edge are valid
    const int64_t from_index = current_edge.GetFromIndex();
    const int64_t to_index = current_edge.GetToIndex();
    if (graph.IndexInRange(from_index) == false)
    {
      throw std::invalid_argument("from_index out of range");
    }
    if (graph.IndexInRange(to_index) == false)
    {
      throw std::invalid_argument("to_index out of range");
    }
    if (from_index == to_index)
    {
      throw std::invalid_argument("from_index and to_index cannot be the same");
    }
    // Now, we estimate the number of executions of the edge necessary to reach
    // (1) the conformant planning threshold
    // OR
    // (2) we reach the edge repeat threshold
    // If we're going forwards
    if (from_index < to_index)
    {
      const PolicyGraphNode& from_node = graph.GetNodeImmutable(from_index);
      const PolicyGraphNode& to_node = graph.GetNodeImmutable(to_index);
      const UncertaintyPlanningState& to_node_value
          = to_node.GetValueImmutable();
      // Get the other child states for our action (if there are any)
      const auto& all_child_edges = from_node.GetOutEdgesImmutable();
      std::vector<common_robotics_utilities::simple_graph::GraphEdge>
          same_action_other_child_edges;
      same_action_other_child_edges.reserve(all_child_edges.size());
      for (const auto& other_child_edge : all_child_edges)
      {
        const int64_t child_index = other_child_edge.GetToIndex();
        const PolicyGraphNode& child_node = graph.GetNodeImmutable(child_index);
        const UncertaintyPlanningState& child_node_value
            = child_node.GetValueImmutable();
        // Only other child nodes with the same transition ID, but different
        // state IDs are kept
        if (child_node_value.GetTransitionId()
            == to_node_value.GetTransitionId()
            && child_node_value.GetStateId() != to_node_value.GetStateId())
        {
          same_action_other_child_edges.push_back(other_child_edge);
        }
      }
      same_action_other_child_edges.shrink_to_fit();
      // If we aren't a child of a split, we're done
      if (same_action_other_child_edges.size() == 0)
      {
        return 1u;
      }
      // Compute the retry count
      double percent_active = 1.0;
      double p_reached = 0.0;
      for (uint32_t try_attempt = 1; try_attempt <= edge_repeat_threshold;
           try_attempt++)
      {
        // How many particles got to our state on this attempt?
        p_reached += (percent_active * to_node_value.GetRawEdgePfeasibility());
        // See if we've reached our threshold
        if (p_reached >= conformant_planning_threshold)
        {
          return try_attempt;
        }
        // Update the percent of particles that are still usefully active
        double updated_percent_active = 0.0;
        for (const auto& other_child_edge : same_action_other_child_edges)
        {
          const int64_t child_index = other_child_edge.GetToIndex();
          const PolicyGraphNode& child_node
              = graph.GetNodeImmutable(child_index);
          const UncertaintyPlanningState& child_node_value
              = child_node.GetValueImmutable();
          if (child_node_value.IsActionOutcomeNominallyIndependent())
          {
            const double p_reached_other
                = percent_active * child_node_value.GetRawEdgePfeasibility();
            const double p_returned_to_parent
                = p_reached_other
                  * child_node_value.GetReverseEdgePfeasibility();
            updated_percent_active += p_returned_to_parent;
          }
        }
        percent_active = updated_percent_active;
      }
      return edge_repeat_threshold;
    }
    // If we're going backwards
    else if (from_index > to_index)
    {
      // We don't yet have a meaningful way to estimate retries of reverse
      // actions
      return 1u;
    }
    // Can't happen
    else
    {
      throw std::runtime_error("from_index cannot equal to_index");
    }
  }

  static PolicyGraph ComputeTrueEdgeWeights(
      const PolicyGraph& initial_graph, const double marginal_edge_weight,
      const double conformant_planning_threshold,
      const uint32_t edge_attempt_threshold)
  {
    PolicyGraph updated_graph = initial_graph;
    #pragma omp parallel for
    for (size_t idx = 0; idx < updated_graph.GetNodesImmutable().size(); idx++)
    {
      PolicyGraphNode& current_node
          = updated_graph.GetNodeMutable(static_cast<int64_t>(idx));
      // Update all edges going out of the node
      for (auto& current_out_edge : current_node.GetOutEdgesMutable())
      {
        // The current edge weight is the probability of that edge
        const double current_edge_weight = current_out_edge.GetWeight();
        // If the edge has positive probability, we need to consider the
        // estimated retry count of the edge
        if (current_edge_weight > 0.0)
        {
          const uint32_t estimated_attempt_count
              = ComputeEstimatedEdgeAttemptCount(
                  updated_graph, current_out_edge,
                  conformant_planning_threshold, edge_attempt_threshold);
          const double edge_probability_weight
              = (current_edge_weight >= std::numeric_limits<double>::epsilon())
                ? 1.0 / current_edge_weight
                : std::numeric_limits<double>::infinity();
          const double edge_attempt_weight
              = marginal_edge_weight
                  * static_cast<double>(estimated_attempt_count);
          const double new_edge_weight
              = edge_probability_weight * edge_attempt_weight;
          current_out_edge.SetWeight(new_edge_weight);
        }
        // If the edge is zero probability (here for linkage only)
        else
        {
          // We set the weight to infinity to remove it from consideration
          const double new_edge_weight
              = std::numeric_limits<double>::infinity();
          current_out_edge.SetWeight(new_edge_weight);
        }
      }
      // Update all edges going into the node
      for (auto& current_in_edge : current_node.GetInEdgesMutable())
      {
        // The current edge weight is the probability of that edge
        const double current_edge_weight = current_in_edge.GetWeight();
        // If the edge has positive probability, we need to consider the
        // estimated retry count of the edge
        if (current_edge_weight > 0.0)
        {
          const uint32_t estimated_attempt_count
              = ComputeEstimatedEdgeAttemptCount(
                  updated_graph, current_in_edge, conformant_planning_threshold,
                  edge_attempt_threshold);
          const double edge_probability_weight
              = (current_edge_weight >= std::numeric_limits<double>::epsilon())
                ? 1.0 / current_edge_weight
                : std::numeric_limits<double>::infinity();
          const double edge_attempt_weight
              = marginal_edge_weight
                  * static_cast<double>(estimated_attempt_count);
          const double new_edge_weight
              = edge_probability_weight * edge_attempt_weight;
          current_in_edge.SetWeight(new_edge_weight);
        }
        // If the edge is zero probability (here for linkage only)
        else
        {
          // We set the weight to infinity to remove it from consideration
          const double new_edge_weight
              = std::numeric_limits<double>::infinity();
          current_in_edge.SetWeight(new_edge_weight);
        }
      }
    }
    return updated_graph;
  }

  static common_robotics_utilities::simple_graph_search::DijkstrasResult
  ComputeNodeDistances(
      const PolicyGraph& initial_graph, const int64_t start_index)
  {
    const auto complete_search_results
        = common_robotics_utilities::simple_graph_search
            ::PerformDijkstrasAlgorithm<UncertaintyPlanningState>(
                initial_graph, start_index);
    for (int64_t idx = 0;
         idx < static_cast<int64_t>(complete_search_results.Size()); idx++)
    {
      const int64_t previous_index
          = complete_search_results.GetPreviousIndex(idx);
      if (!initial_graph.IndexInRange(previous_index))
      {
        throw std::runtime_error(
            "previous_index out of range, graph is no longer connected");
      }
    }
    return complete_search_results;
  }
};

template<typename Configuration>
class PolicyQueryResult
{
private:
  int64_t previous_state_index_ = -1;
  uint64_t desired_transition_id_ = 0;
  Configuration action_;
  Configuration expected_result_;
  double expected_cost_to_goal_ = std::numeric_limits<double>::infinity();
  bool is_reverse_action_ = false;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PolicyQueryResult() = delete;

  PolicyQueryResult(const int64_t previous_state_index,
                    const uint64_t desired_transition_id,
                    const Configuration& action,
                    const Configuration& expected_result,
                    const double expected_cost_to_goal,
                    const bool is_reverse_action)
      : previous_state_index_(previous_state_index),
        desired_transition_id_(desired_transition_id),
        action_(action),
        expected_result_(expected_result),
        expected_cost_to_goal_(expected_cost_to_goal),
        is_reverse_action_(is_reverse_action) {}

  int64_t PreviousStateIndex() const { return previous_state_index_; }

  uint64_t DesiredTransitionId() const { return desired_transition_id_; }

  const Configuration& Action() const { return action_; }

  const Configuration& ExpectedResult() const { return expected_result_; }

  double ExpectedCostToGoal() const { return expected_cost_to_goal_; }

  bool IsReverseAction() const { return is_reverse_action_; }
};

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc=std::allocator<Configuration>>
class ExecutionPolicy
{
private:
  // Typedef so we don't hate ourselves
  using UncertaintyPlanningState
      = UncertaintyPlannerState<Configuration, ConfigSerializer, ConfigAlloc>;
  using UncertaintyPlanningStateAllocator
      = Eigen::aligned_allocator<UncertaintyPlanningState>;
  using UncertaintyPlanningStateVector
      = std::vector<UncertaintyPlanningState,
                    UncertaintyPlanningStateAllocator>;
  using UncertaintyPlanningTreeState
      = common_robotics_utilities::simple_rrt_planner::SimpleRRTPlannerState<
          UncertaintyPlanningState>;
  using UncertaintyPlanningTree
      = common_robotics_utilities::simple_rrt_planner::PlanningTree<
          UncertaintyPlanningState>;
  using PolicyGraphNode
      = common_robotics_utilities::simple_graph::GraphNode<
          UncertaintyPlanningState>;
  using PolicyGraph
      = common_robotics_utilities::simple_graph::Graph<
          UncertaintyPlanningState>;
  using ExecutionPolicyGraphBuilder
      = PolicyGraphBuilder<Configuration, ConfigSerializer, ConfigAlloc>;
  using ExecutionPolicyType
      = ExecutionPolicy<Configuration, ConfigSerializer, ConfigAlloc>;

  bool initialized_ = false;
  // Raw data used to rebuild the policy graph
  UncertaintyPlanningTree planner_tree_;
  Configuration goal_;
  double marginal_edge_weight_ = 0.0;
  double conformant_planning_threshold_ = 0.0;
  uint32_t edge_attempt_threshold_ = 0u;
  uint32_t policy_action_attempt_count_ = 0u;
  // Actual policy graph
  PolicyGraph policy_graph_;
  common_robotics_utilities::simple_graph_search::DijkstrasResult
      policy_dijkstras_result_;
  // Logging function
  LoggingFunction logging_fn_;

public:
  static uint32_t AddWithOverflowClamp(
      const uint32_t original, const uint32_t additional)
  {
    if (additional == 0u)
    {
      return original;
    }
    if ((original + additional) <= original)
    {
      std::cerr << "@@@ WARNING - CLAMPING ON OVERFLOW OF UINT32_T @@@"
                << std::endl;
      return std::numeric_limits<uint32_t>::max();
    }
    else
    {
      return original + additional;
    }
  }

  static uint64_t Serialize(
      const ExecutionPolicyType& policy, std::vector<uint8_t>& buffer)
  {
    return policy.SerializeSelf(buffer);
  }

  static common_robotics_utilities::serialization
      ::Deserialized<ExecutionPolicyType> Deserialize(
          const std::vector<uint8_t>& buffer, const uint64_t current)
  {
    ExecutionPolicyType temp_policy;
    const uint64_t bytes_read = temp_policy.DeserializeSelf(buffer, current);
    return common_robotics_utilities::serialization::MakeDeserialized(
        temp_policy, bytes_read);
  }

  ExecutionPolicy(
      const UncertaintyPlanningTree& planner_tree, const Configuration& goal,
      const double marginal_edge_weight,
      const double conformant_planning_threshold,
      const uint32_t edge_attempt_threshold,
      const uint32_t policy_action_attempt_count,
      const LoggingFunction& logging_fn)
      : initialized_(true), planner_tree_(planner_tree), goal_(goal),
        marginal_edge_weight_(marginal_edge_weight),
        conformant_planning_threshold_(conformant_planning_threshold),
        edge_attempt_threshold_(edge_attempt_threshold),
        policy_action_attempt_count_(policy_action_attempt_count),
        logging_fn_(logging_fn)
  {
    RebuildPolicyGraph();
  }

  ExecutionPolicy()
    : initialized_(false),
      logging_fn_([] (const std::string& msg, const int32_t level)
          { std::cout << "Log [" << level << "] : " << msg << std::endl; }) {}

  void RegisterLoggingFunction(const LoggingFunction& logging_fn)
  {
    logging_fn_ = logging_fn;
  }

  void Log(const std::string& msg, const int32_t level) const
  {
    logging_fn_(msg, level);
  }

  std::pair<PolicyGraph,
            common_robotics_utilities::simple_graph_search::DijkstrasResult>
  BuildPolicyGraphComponentsFromTree(
      const UncertaintyPlanningTree& planner_tree, const Configuration& goal,
      const double marginal_edge_weight,
      const double conformant_planning_threshold,
      const uint32_t edge_attempt_threshold) const
  {
    const PolicyGraph preliminary_policy_graph
        = ExecutionPolicyGraphBuilder::BuildPolicyGraphFromPlannerTree(
            planner_tree, UncertaintyPlanningState(goal));
    const PolicyGraph intermediate_policy_graph
        = ExecutionPolicyGraphBuilder::ComputeTrueEdgeWeights(
            preliminary_policy_graph, marginal_edge_weight,
            conformant_planning_threshold, edge_attempt_threshold);
    const auto distances = ExecutionPolicyGraphBuilder::ComputeNodeDistances(
          intermediate_policy_graph,
          static_cast<int64_t>(
              intermediate_policy_graph.GetNodesImmutable().size()) - 1);
    return std::make_pair(intermediate_policy_graph, distances);
  }

  std::string PrintTree(
      const UncertaintyPlanningTree& planning_tree,
      const common_robotics_utilities::simple_graph_search
          ::DijkstrasResult& dijkstras_result) const
  {
    std::ostringstream strm;
    strm << "Planning tree with " << planning_tree.size() << " states:";
    for (size_t idx = 0; idx < planning_tree.size(); idx++)
    {
      const int64_t previous_index
          = dijkstras_result.GetPreviousIndex(static_cast<int64_t>(idx));
      const double distance
          = dijkstras_result.GetNodeDistance(static_cast<int64_t>(idx));
      const UncertaintyPlanningTreeState& current_tree_state
          = planning_tree.at(idx);
      const int64_t parent_index = current_tree_state.GetParentIndex();
      const UncertaintyPlanningState& current_state
          = current_tree_state.GetValueImmutable();
      const double raw_edge_probability
          = current_state.GetRawEdgePfeasibility();
      const double effective_edge_probability
          = current_state.GetEffectiveEdgePfeasibility();
      const double reverse_edge_probability
          = current_state.GetReverseEdgePfeasibility();
      const double goal_proability = current_state.GetGoalPfeasibility();
      strm << "\nState " << idx << " with P(" << parent_index << "->" << idx
           << ") = " << raw_edge_probability << "/"
           << effective_edge_probability << " [raw/effective] and P(" << idx
           << "->" << parent_index << ") = " << reverse_edge_probability
           << " and P(->goal) = " << goal_proability << " and Previous = ";
      if (previous_index == planning_tree.size())
      {
        strm << "(goal) with distance = " << distance;
      }
      else
      {
        strm << previous_index << " with distance = " << distance;
      }
    }
    return strm.str();
  }

  void RebuildPolicyGraph()
  {
    const auto processed_policy_graph_components
        = BuildPolicyGraphComponentsFromTree(
            planner_tree_, goal_, marginal_edge_weight_,
            conformant_planning_threshold_, edge_attempt_threshold_);
    policy_graph_ = processed_policy_graph_components.first;
    policy_dijkstras_result_ = processed_policy_graph_components.second;
  }

  uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const
  {
    using common_robotics_utilities::serialization::Serializer;
    using common_robotics_utilities::serialization::SerializeMemcpyable;
    using common_robotics_utilities::serialization::SerializeVectorLike;
    const uint64_t start_buffer_size = buffer.size();
    // Serialize the initialized
    SerializeMemcpyable<uint8_t>(static_cast<uint8_t>(initialized_), buffer);
    // Serialize the planner tree
    const Serializer<UncertaintyPlanningTreeState>
        planning_tree_state_serializer_fn
            = [] (const UncertaintyPlanningTreeState& state,
                  std::vector<uint8_t>& ser_buffer)
    {
      return UncertaintyPlanningTreeState::Serialize(
          state, ser_buffer, UncertaintyPlanningState::Serialize);
    };
    SerializeVectorLike(
        planner_tree_, buffer, planning_tree_state_serializer_fn);
    // Serialize the goal
    ConfigSerializer::Serialize(goal_, buffer);
    // Serialize the marginal edge weight
    SerializeMemcpyable<double>(marginal_edge_weight_, buffer);
    // Serialize the conformant planning threshold
    SerializeMemcpyable<double>(conformant_planning_threshold_, buffer);
    // Serialize the edge attempt threshold
    SerializeMemcpyable<uint32_t>(edge_attempt_threshold_, buffer);
    // Serialize the policy action attempt count
    SerializeMemcpyable<uint32_t>(policy_action_attempt_count_, buffer);
    // Figure out how many bytes were written
    const uint64_t end_buffer_size = buffer.size();
    const uint64_t bytes_written = end_buffer_size - start_buffer_size;
    return bytes_written;
  }

  uint64_t DeserializeSelf(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
  {
    using common_robotics_utilities::serialization::Deserializer;
    using common_robotics_utilities::serialization::DeserializeMemcpyable;
    using common_robotics_utilities::serialization::DeserializeVectorLike;
    uint64_t current_position = starting_offset;
    // Deserialize the initialized
    const auto initialized_deserialized
        = DeserializeMemcpyable<uint8_t>(buffer, current_position);
    initialized_ = static_cast<bool>(initialized_deserialized.Value());
    current_position += initialized_deserialized.BytesRead();
    // Deserialize the planner tree
    const Deserializer<UncertaintyPlanningTreeState>
        planning_tree_state_deserializer_fn
            = [] (const std::vector<uint8_t>& deser_buffer,
                  const uint64_t deser_current)
    {
      return UncertaintyPlanningTreeState::Deserialize(
          deser_buffer, deser_current, UncertaintyPlanningState::Deserialize);
    };
    const auto planner_tree_deserialized
        = DeserializeVectorLike<
            UncertaintyPlanningTreeState, UncertaintyPlanningTree>(
                buffer, current_position, planning_tree_state_deserializer_fn);
    planner_tree_ = planner_tree_deserialized.Value();
    current_position += planner_tree_deserialized.BytesRead();
    // Deserialize the goal
    const auto goal_deserialized
        = ConfigSerializer::Deserialize(buffer, current_position);
    goal_ = goal_deserialized.Value();
    current_position += goal_deserialized.BytesRead();
    // Deserialize the marginal edge weight
    const auto marginal_edge_weight_deserialized
        = DeserializeMemcpyable<double>(buffer, current_position);
    marginal_edge_weight_ = marginal_edge_weight_deserialized.Value();
    current_position += marginal_edge_weight_deserialized.BytesRead();
    // Deserialize the conformant planning threshold
    const auto conformant_planning_threshold_deserialized
        = DeserializeMemcpyable<double>(buffer, current_position);
    conformant_planning_threshold_
        = conformant_planning_threshold_deserialized.Value();
    current_position += conformant_planning_threshold_deserialized.BytesRead();
    // Deserialize the edge attempt threshold
    const auto edge_attempt_threshold_deserialized
        = DeserializeMemcpyable<uint32_t>(buffer, current_position);
    edge_attempt_threshold_ = edge_attempt_threshold_deserialized.Value();
    current_position += edge_attempt_threshold_deserialized.BytesRead();
    // Deserialize the policy action attempt count
    const auto policy_action_attempt_count_deserialized
        = DeserializeMemcpyable<uint32_t>(buffer, current_position);
    policy_action_attempt_count_
        = policy_action_attempt_count_deserialized.Value();
    current_position += policy_action_attempt_count_deserialized.BytesRead();
    // Rebuild the policy graph
    RebuildPolicyGraph();
    // Figure out how many bytes were read
    const uint64_t bytes_read = current_position - starting_offset;
    return bytes_read;
  }

  std::vector<std::string> PrintHumanReadablePolicyTreeNode(
      const int64_t node_index,
      const std::function<std::vector<std::string>(
          const UncertaintyPlanningState&)>& state_print_fn) const
  {
    const UncertaintyPlanningTreeState& policy_tree_state
        = planner_tree_.at(node_index);
    std::vector<std::string> state_string_rep;
    state_string_rep.push_back(
          "<state index=\"" + std::to_string(node_index)
          + "\" state_id=\""
          + std::to_string(policy_tree_state.GetValueImmutable().GetStateId())
          + "\" transition_id=\""
          + std::to_string(
              policy_tree_state.GetValueImmutable().GetTransitionId())
          + "\">");
    state_string_rep.push_back("  <value>");
    const std::vector<std::string> value_string_rep
        = state_print_fn(policy_tree_state.GetValueImmutable());
    for (const auto& value_string : value_string_rep)
    {
      state_string_rep.push_back("    " + value_string);
    }
    state_string_rep.push_back("  </value>");
    state_string_rep.push_back("  <children>");
    const std::vector<int64_t>& child_indices
        = policy_tree_state.GetChildIndices();

    for (const int64_t child_index : child_indices)
    {
      const std::vector<std::string> child_string_rep
          = PrintHumanReadablePolicyTreeNode(child_index, state_print_fn);
      for (const auto& child_string : child_string_rep)
      {
        state_string_rep.push_back("    " + child_string);
      }
    }
    state_string_rep.push_back("  </children>");
    state_string_rep.push_back("</state>");
    return state_string_rep;
  }

  std::string PrintHumanReadablePolicyTree(
      const std::function<std::vector<std::string>(
          const UncertaintyPlanningState&)>& state_print_fn) const
  {
    std::vector<std::string> string_lines
        = PrintHumanReadablePolicyTreeNode(0, state_print_fn);
    return common_robotics_utilities::print::Print(string_lines, false, "\n");
  }

  std::string PrintHumanReadablePolicyTree() const
  {
    const std::function<std::vector<std::string>(
        const UncertaintyPlanningState&)> state_print_fn
            = [] (const UncertaintyPlanningState& state)
    {
      return std::vector<std::string>{state.Print()};
    };
    return PrintHumanReadablePolicyTree(state_print_fn);
  }

  bool IsInitialized() const { return initialized_; }

  const UncertaintyPlanningTree& GetPlannerTreeImmutable() const
  {
    if (initialized_)
    {
      return planner_tree_;
    }
    else
    {
      throw std::runtime_error("PolicyGraph is not initialized");
    }
  }

  UncertaintyPlanningTree& GetPlannerTreeMutable()
  {
    if (initialized_)
    {
      return planner_tree_;
    }
    else
    {
      throw std::runtime_error("PolicyGraph is not initialized");
    }
  }

  const Configuration& GetRawGoalConfiguration() const
  {
    if (initialized_)
    {
      return goal_;
    }
    else
    {
      throw std::runtime_error("PolicyGraph is not initialized");
    }
  }

  double GetMarginalEdgeWeight() const { return marginal_edge_weight_; }

  double GetConformantPlanningThreshold() const
  {
    return conformant_planning_threshold_;
  }

  uint32_t GetEdgeAttemptThreshold() const
  {
    return edge_attempt_threshold_;
  }

  const UncertaintyPlanningTree& GetRawPolicyTree() const
  {
    if (initialized_)
    {
      return planner_tree_;
    }
    else
    {
      throw std::runtime_error("PolicyGraph is not initialized");
    }
  }

  const PolicyGraph& GetRawPolicy() const
  {
    if (initialized_)
    {
      return policy_graph_;
    }
    else
    {
      throw std::runtime_error("PolicyGraph is not initialized");
    }
  }

  const common_robotics_utilities::simple_graph_search::DijkstrasResult&
  GetRawPolicyDijkstrasResult() const
  {
    if (initialized_)
    {
      return policy_dijkstras_result_;
    }
    else
    {
      throw std::runtime_error("PolicyGraph is not initialized");
    }
  }

  uint32_t GetPolicyActionAttemptCount() const
  {
    return policy_action_attempt_count_;
  }

  void SetPolicyActionAttemptCount(const uint32_t new_count)
  {
    policy_action_attempt_count_ = new_count;
  }

private:
  PolicyQueryResult<Configuration> QueryNextAction(
      const int64_t current_state_index) const
  {
    if (!policy_graph_.IndexInRange(current_state_index))
    {
      throw std::invalid_argument("current_state_index is out of range");
    }
    const PolicyGraphNode& result_state_policy_node
        = policy_graph_.GetNodeImmutable(current_state_index);
    const UncertaintyPlanningState& result_state
        = result_state_policy_node.GetValueImmutable();
    // Get the action to take
    // Get the previous node, as indicated by Dijkstra's algorithm
    const int64_t target_state_index
        = policy_dijkstras_result_.GetPreviousIndex(current_state_index);
    const double expected_cost_to_goal
          = policy_dijkstras_result_.GetNodeDistance(current_state_index);
    if (target_state_index < 0)
    {
      throw std::runtime_error("Policy no longer has a solution");
    }
    // If we are at a goal state, we do not command to the "virtual goal node"
    // that links the graph together since this node has no meaningful value in
    // cases of goal regions
    else if (target_state_index
             == static_cast<int64_t>(policy_graph_.Size()) - 1)
    {
      Log("Already at a goal state " + std::to_string(current_state_index)
          + " - cannot proceed to virtual goal state - repeating transition "
          + std::to_string(result_state.GetTransitionId())
          + " to command to our expectation", 3);
      return PolicyQueryResult<Configuration>(
            current_state_index, result_state.GetTransitionId(),
            result_state.GetExpectation(), result_state.GetExpectation(),
            expected_cost_to_goal, false);
    }
    else
    {
      const PolicyGraphNode& target_state_policy_node
          = policy_graph_.GetNodeImmutable(target_state_index);
      const UncertaintyPlanningState& target_state
          = target_state_policy_node.GetValueImmutable();
      // Figure out the correct action to take
      const uint64_t result_state_id = result_state.GetStateId();
      const uint64_t target_state_id = target_state.GetStateId();
      // If the "previous" node that we want to go to is a downstream state, we
      // get the action of the downstream state
      if (result_state_id < target_state_id)
      {
        Log("Returning forward action for current state "
            + std::to_string(current_state_index) + ", transition ID "
            + std::to_string(target_state.GetTransitionId()), 2);
        return PolicyQueryResult<Configuration>(
              current_state_index, target_state.GetTransitionId(),
              target_state.GetCommand(), target_state.GetExpectation(),
              expected_cost_to_goal, false);
      }
      // If the "previous" node that we want to go to is an upstream state, we
      // get the expectation of the upstream state
      else if (target_state_id < result_state_id)
      {
        Log("Returning reverse action for current state "
            + std::to_string(current_state_index) + ", transition ID "
            + std::to_string(result_state.GetReverseTransitionId()), 2);
        return PolicyQueryResult<Configuration>(
              current_state_index, result_state.GetReverseTransitionId(),
              target_state.GetExpectation(), target_state.GetExpectation(),
              expected_cost_to_goal, true);
      }
      else
      {
        throw std::runtime_error(
              "target_state_id cannot equal result_state_id");
      }
    }
  }

public:
  /*
   * If your particle clustering function is not thread safe, you will have a
   * bad time!
   */
  PolicyQueryResult<Configuration> QueryBestAction(
      const uint64_t performed_transition_id,
      const Configuration& current_config, const bool allow_branch_jumping,
      const bool link_runtime_states_to_planned_parent,
      const std::function<bool(
          const std::vector<Configuration, ConfigAlloc>&,
          const Configuration&)>& particle_clustering_fn)
  {
    if (initialized_)
    {
      // If we're just starting out
      if (performed_transition_id == 0)
      {
        return QueryStartBestAction(current_config, particle_clustering_fn);
      }
      else
      {
        return QueryNormalBestAction(
              performed_transition_id, current_config, allow_branch_jumping,
              link_runtime_states_to_planned_parent, particle_clustering_fn);
      }
    }
    else
    {
      throw std::runtime_error("PolicyGraph is not initialized");
    }
  }

private:
  int64_t FindBestMatchingStateInPolicy(
      const Configuration& current_config,
      const std::function<bool(
          const std::vector<Configuration, ConfigAlloc>&,
          const Configuration&)>& particle_clustering_fn) const
  {
    using common_robotics_utilities::simple_knearest_neighbors
        ::IndexAndDistance;
    // Get the starting state - NOTE, we ignore the last node in the policy
    // graph, which is the virtual goal node
    std::vector<IndexAndDistance> per_thread_best_node(
        common_robotics_utilities::openmp_helpers::GetNumOmpThreads());
    #pragma omp parallel for
    for (int64_t node_idx = 0;
         node_idx < static_cast<int64_t>(
             policy_graph_.GetNodesImmutable().size()) - 1;
         node_idx++)
    {
      const PolicyGraphNode& current_node
          = policy_graph_.GetNodeImmutable(node_idx);
      const UncertaintyPlanningState& current_node_state
          = current_node.GetValueImmutable();
      // Are we a member of this cluster?
      // Make sure we are close enough to the start state
      const bool is_cluster_member
          = particle_clustering_fn(
              current_node_state.GetParticlePositionsImmutable().Value(),
              current_config);
      if (is_cluster_member)
      {
        const int32_t thread_id
            = common_robotics_utilities::openmp_helpers
                ::GetContextOmpThreadNum();
        const double expected_cost_to_goal
            = policy_dijkstras_result_.GetNodeDistance(node_idx);
        if (expected_cost_to_goal
            < per_thread_best_node.at(thread_id).Distance())
        {
          per_thread_best_node.at(thread_id).SetIndexAndDistance(
              node_idx, expected_cost_to_goal);
        }
      }
    }
    IndexAndDistance best_node;
    for (const auto& thread_best : per_thread_best_node)
    {
      if (thread_best.Distance() < best_node.Distance())
      {
        best_node.SetFromOther(thread_best);
      }
    }
    return best_node.Index();
  }

  PolicyQueryResult<Configuration> QueryStartBestAction(
      const Configuration& current_config,
      const std::function<bool(
          const std::vector<Configuration, ConfigAlloc>&,
          const Configuration&)>& particle_clustering_fn) const
  {
    const int64_t best_node_index
        = FindBestMatchingStateInPolicy(current_config, particle_clustering_fn);
    if (best_node_index >= 0)
    {
      Log("Starting configuration best matches node "
          + std::to_string(best_node_index), 2);
      return QueryNextAction(best_node_index);
    }
    else
    {
      throw std::runtime_error(
            "Starting at current_config is not covered by partial policy");
    }
  }

  PolicyQueryResult<Configuration> QueryNormalBestAction(
      const uint64_t performed_transition_id,
      const Configuration& current_config, const bool allow_branch_jumping,
      const bool link_runtime_states_to_planned_parent,
      const std::function<bool(
          const std::vector<Configuration, ConfigAlloc>&,
          const Configuration&)>& particle_clustering_fn)
  {
    Log("++++++++++\nQuerying the policy with performed transition ID "
        + std::to_string(performed_transition_id) + "...", 2);
    if (performed_transition_id <= 0)
    {
      throw std::invalid_argument(
            "performed_transition_id must be greater than zero");
    }
    // Collect the possible states that could have resulted from the transition
    // we just performed
    // TODO(calderpg) Can std::map be replaced here to allow this loop to be
    // parallelized?
    std::map<int64_t, std::vector<std::pair<int64_t, bool>>>
        expected_possibility_result_states;
    std::map<int64_t, uint64_t> previous_state_index_possibilities;
    // Go through the entire tree and retrieve all states with matching
    // transition IDs
    for (int64_t idx = 0; idx < static_cast<int64_t>(planner_tree_.size());
         idx++)
    {
      const UncertaintyPlanningTreeState& candidate_tree_state
          = planner_tree_.at(static_cast<size_t>(idx));
      const UncertaintyPlanningState& candidate_state
          = candidate_tree_state.GetValueImmutable();
      if (candidate_state.GetTransitionId() == performed_transition_id)
      {
        const int64_t parent_state_idx = candidate_tree_state.GetParentIndex();
        const UncertaintyPlanningTreeState& candidate_parent_tree_state
            = planner_tree_.at(static_cast<size_t>(parent_state_idx));
        const UncertaintyPlanningState& candidate_parent_state
            = candidate_parent_tree_state.GetValueImmutable();
        expected_possibility_result_states[parent_state_idx].push_back(
            std::make_pair(idx, false));
        previous_state_index_possibilities[parent_state_idx]
            = candidate_parent_state.GetStateId();
      }
      else if (candidate_state.GetReverseTransitionId()
               == performed_transition_id)
      {
        expected_possibility_result_states[idx].push_back(
            std::make_pair(idx, true));
        previous_state_index_possibilities[idx] = candidate_state.GetStateId();
      }
    }
    int64_t previous_state_index = -1;
    if (previous_state_index_possibilities.size() > 1)
    {
      Log("Multiple previous state index possibilities "
          + common_robotics_utilities::print::Print(
              previous_state_index_possibilities), 1);
      Log("Multiple sets of possible result states "
          + common_robotics_utilities::print::Print(
              expected_possibility_result_states), 1);
      // Prefer planned states
      for (auto itr = previous_state_index_possibilities.begin();
           itr != previous_state_index_possibilities.end(); ++itr)
      {
        const int64_t previous_state_index_possibility = itr->first;
        const uint64_t previous_state_id = itr->second;
        if (previous_state_id < INT64_C(1000000000))
        {
          previous_state_index = previous_state_index_possibility;
        }
      }
      Log("Selected " + std::to_string(previous_state_index)
          + " as previous state index", 2);
    }
    else if (previous_state_index_possibilities.size() == 1)
    {
      Log("Single previous state index possibility "
          + common_robotics_utilities::print::Print(
              previous_state_index_possibilities), 1);
      previous_state_index = previous_state_index_possibilities.begin()->first;
      Log("Selected " + std::to_string(previous_state_index)
          + " as previous state index", 2);
    }
    else
    {
      throw std::runtime_error(
          "previous_state_index_possibilities cannot be empty");
    }
    const std::vector<std::pair<int64_t, bool>>& expected_possible_result_states
        = expected_possibility_result_states.at(previous_state_index);
    if (expected_possible_result_states.empty())
    {
      throw std::runtime_error(
          "expected_possible_result_states cannot be empty");
    }
    Log("Result state could match "
        + std::to_string(expected_possible_result_states.size())
        + " states", 2);
    ////////////////////////////////////////////////////////////////////////////
    // Check if the current config matches one or more of the expected result
    // states
    std::vector<std::pair<int64_t, bool>> expected_result_state_matches;
    for (const auto& possible_match : expected_possible_result_states)
    {
      const int64_t possible_match_state_idx
          = (possible_match.second)
            ? planner_tree_.at(static_cast<size_t>(possible_match.first))
                .GetParentIndex()
            : possible_match.first;
      const UncertaintyPlanningTreeState& possible_match_tree_state
          = planner_tree_.at(static_cast<size_t>(possible_match_state_idx));
      const UncertaintyPlanningState& possible_match_state
          = possible_match_tree_state.GetValueImmutable();
      const std::vector<Configuration, ConfigAlloc>&
          possible_match_node_particles
              = possible_match_state.GetParticlePositionsImmutable().Value();
      const bool is_cluster_member
          = particle_clustering_fn(
              possible_match_node_particles, current_config);
      // If the current config is part of the cluster
      if (is_cluster_member)
      {
        const Configuration possible_match_state_expectation
            = possible_match_state.GetExpectation();
        Log("Possible result state matches with expectation "
            + common_robotics_utilities::print::Print(
                possible_match_state_expectation), 1);
        expected_result_state_matches.push_back(possible_match);
      }
    }
    // If any child states matched
    if (expected_result_state_matches.size() > 0)
    {
      const int64_t result_state_index
          = UpdateNodeCountsAndTree(
              expected_possible_result_states, expected_result_state_matches);
      ////////////////////////////////////////////////////////////////////////
      // Now that we've updated the tree, we can rebuild and query for the
      // action to take
      // The rebuild and action query process is the same in all cases
      RebuildPolicyGraph();
      return QueryNextAction(result_state_index);
    }
    // If none match, we add a new node
    else
    {
      Log("Result state matched none of the "
          + std::to_string(expected_possible_result_states.size())
          + " expected results, checking if it matches a child of the expected "
            "results", 2);
      std::vector<std::pair<int64_t, bool>>
          expected_possible_result_child_states;
      // Get all 1st-tier child states of the expected result states

      for (const auto& possible_match : expected_possible_result_states)
      {
        const int64_t possible_match_state_idx
            = (possible_match.second)
              ? planner_tree_.at(static_cast<size_t>(possible_match.first))
                  .GetParentIndex()
              : possible_match.first;
        const UncertaintyPlanningTreeState& possible_match_tree_state
            = planner_tree_.at(static_cast<size_t>(possible_match_state_idx));
        for (const int64_t child_state_index
                : possible_match_tree_state.GetChildIndices())
        {
          expected_possible_result_child_states.push_back(
              std::make_pair(child_state_index, false));
        }
      }
      Log("Result state could match "
          + std::to_string(expected_possible_result_states.size())
          + " child states", 1);
      // Check if the current config matches one or more of the expected result
      // states
      std::vector<std::pair<int64_t, bool>> expected_result_child_state_matches;
      for (const auto& possible_match : expected_possible_result_child_states)
      {
        const int64_t possible_match_state_idx
            = (possible_match.second)
              ? planner_tree_.at(static_cast<size_t>(possible_match.first))
                  .GetParentIndex()
              : possible_match.first;
        const UncertaintyPlanningTreeState& possible_match_tree_state
            = planner_tree_.at(static_cast<size_t>(possible_match_state_idx));
        const UncertaintyPlanningState& possible_match_state
            = possible_match_tree_state.GetValueImmutable();
        const std::vector<Configuration, ConfigAlloc>&
            possible_match_node_particles
                = possible_match_state.GetParticlePositionsImmutable().Value();
        const bool is_cluster_member
            = particle_clustering_fn(
                possible_match_node_particles, current_config);
        // If the current config is part of the cluster
        if (is_cluster_member)
        {
          const Configuration possible_match_state_expectation
              = possible_match_state.GetExpectation();
          Log("Possible result child state matches with expectation "
              + common_robotics_utilities::print::Print(
                  possible_match_state_expectation), 1);
          expected_result_child_state_matches.push_back(possible_match);
        }
      }
      if (expected_result_child_state_matches.size() > 0)
      {
        Log("Result state matched "
            + std::to_string(expected_result_child_state_matches.size())
            + " of the "
            + std::to_string(expected_possible_result_child_states.size())
            + " expected results child states", 1);
        // WE CANNOT LEARN ACROSS PARENT->CHILD BRANCHES
        // Select the current best-distance result state as THE result state
        std::pair<int64_t, bool> best_result_state(-1, false);
        double best_distance = std::numeric_limits<double>::infinity();

        for (const auto& result_match : expected_result_child_state_matches)
        {
          const int64_t result_match_state_idx
              = (result_match.second)
                ? planner_tree_.at(static_cast<size_t>(result_match.first))
                    .GetParentIndex()
                : result_match.first;
          const double result_match_distance
              = policy_dijkstras_result_.GetNodeDistance(
                  result_match_state_idx);
          if (result_match_distance < best_distance)
          {
            best_result_state = result_match;
            best_distance = result_match_distance;
          }
        }
        if (best_result_state.first < 0)
        {
          throw std::runtime_error("Could not identify best result state");
        }
        const int64_t result_state_index
            = (best_result_state.second)
              ? planner_tree_.at(static_cast<size_t>(best_result_state.first))
                  .GetParentIndex()
              : best_result_state.first;
        if (best_result_state.second == false)
        {
          Log("Selected best match result child state (forward movement): "
              + std::to_string(result_state_index), 2);
        }
        else
        {
          Log("Selected best match result child state (reverse movement): "
              + std::to_string(result_state_index), 2);
        }
        return QueryNextAction(result_state_index);
      }
      else
      {
        if (allow_branch_jumping)
        {
          // This loses any state->state linkage, but does provide more robust
          // error handling with fewer recovery steps in many cases. However,
          // your clustering must be precise! If it is not, the policy could
          // just switch to a totally different branch and you will get stuck.
          Log("Result state matched none of the "
              + std::to_string(expected_possible_result_states.size())
              + " expected results or their "
              + std::to_string(expected_possible_result_child_states.size())
              + " child states, trying to jump branches to find a matching "
                "state", 3);
          const int64_t best_matching_branch_jump_index
              = FindBestMatchingStateInPolicy(
                  current_config, particle_clustering_fn);
          if (best_matching_branch_jump_index >= 0)
          {
            Log("Branch jumping found a best-matching state with index "
                + std::to_string(best_matching_branch_jump_index), 2);
            return QueryNextAction(best_matching_branch_jump_index);
          }
          else
          {
            Log("Branch jumping failed to find a matching state", 3);
          }
        }
        // What if we made *all* runtime-added states point back to their parent
        // planning-time-added root node?
        // This loses any state->state linkage, but means that multi-state
        // returns would be handled properly
        Log("Result state matched none of the "
            + std::to_string(expected_possible_result_states.size())
            + " expected results or their "
            + std::to_string(expected_possible_result_child_states.size())
            + " child states, adding a new state", 3);
        // Compute the parameters of the new node
        const uint64_t new_child_state_id
            = planner_tree_.size() + UINT64_C(1000000000);
        // These will get updated in the recursive call
        // (that's why the reached counts are zero)
        const uint32_t reached_count = 0u;
        const double effective_edge_Pfeasibility = 0.0;
        const double parent_motion_Pfeasibility
            = planner_tree_.at(static_cast<size_t>(previous_state_index))
                .GetValueImmutable().GetMotionPfeasibility();
        const double step_size
            = planner_tree_.at(static_cast<size_t>(previous_state_index))
                .GetValueImmutable().GetStepSize();
        // Basic prior assuming that actions are reversible
        const uint32_t reverse_attempt_count = 1u;
        const uint32_t reverse_reached_count = 1u;
        // More fixed params
        const uint64_t transition_id = performed_transition_id;
        // Get a new transition ID for the reverse
        const uint64_t reverse_transition_id
            = planner_tree_.size() + UINT64_C(1000000000);
        // Get some params
        const uint64_t previous_state_reverse_transition_id
            = planner_tree_.at(static_cast<size_t>(previous_state_index))
                .GetValueImmutable().GetReverseTransitionId();
        const bool desired_transition_is_reversal
            = (performed_transition_id == previous_state_reverse_transition_id)
              ? true : false;
        Configuration command;
        uint32_t attempt_count = 0u;
        uint64_t split_id = 0;
        // If the action was a reversal, then grab the expectation of the
        // previous state's parent
        // TODO: I don't know what the right way to handle multiple reversals
        // is. This approach worked when used for motion-level planning, but it
        // may not work for task-level planning
        // If we walk back up the tree until we reach an "original" state with
        // state id < 1000000000 we can ignore the expectations of any "learned"
        // states and return to the "root parent" in the tree
        int64_t acting_parent_state_index = previous_state_index;
        if (desired_transition_is_reversal)
        {
          // This is new behavior - runtime-generated states always point to
          // their parent state in the tree, which allows multi-action returns
          // to the tree
          if (link_runtime_states_to_planned_parent)
          {
            // Find the parent state in the tree
            // (ignoring any previous runtime-added states)
            int64_t parent_index = -1;
            int64_t working_index = previous_state_index;
            while (parent_index < 0)
            {
              const UncertaintyPlanningTreeState& candidate_parent_tree_state
                  = planner_tree_.at(static_cast<size_t>(working_index));
              const UncertaintyPlanningState& candidate_parent_state
                  = candidate_parent_tree_state.GetValueImmutable();
              const uint64_t candidate_parent_state_id
                  = candidate_parent_state.GetStateId();
              // Is the candidate a planned state?
              if (candidate_parent_state_id < UINT64_C(1000000000))
              {
                parent_index = working_index;
              }
              // Is the candidate a runtime-added state?
              else
              {
                // Backtrack up the tree
                working_index = candidate_parent_tree_state.GetParentIndex();
              }
            }
            acting_parent_state_index = parent_index;
            const UncertaintyPlanningTreeState& parent_tree_state
                = planner_tree_.at(static_cast<size_t>(parent_index));
            const UncertaintyPlanningState& parent_state
                = parent_tree_state.GetValueImmutable();
            command = parent_state.GetExpectation();
            // This value doesn't really matter
            attempt_count
                = planner_tree_.at(static_cast<size_t>(previous_state_index))
                    .GetValueImmutable().GetReverseAttemptAndReachedCounts()
                        .first;
            // Split IDs aren't actually used, other than > 0 meaning children
            // of splits
            split_id = performed_transition_id;
            Log("Adding a new reversed state linked to planned parent", 2);
          }
          // This is the old behavior - runtime-generated states are linked to
          // each other, rahter than all pointing to the original parent state
          else
          {
            const int64_t parent_index
                = planner_tree_.at(static_cast<size_t>(previous_state_index))
                    .GetParentIndex();
            const UncertaintyPlanningTreeState& parent_tree_state
                = planner_tree_.at(static_cast<size_t>(parent_index));
            const UncertaintyPlanningState& parent_state
                = parent_tree_state.GetValueImmutable();
            command = parent_state.GetExpectation();
            // This value doesn't really matter
            attempt_count
                = planner_tree_.at(static_cast<size_t>(previous_state_index))
                    .GetValueImmutable().GetReverseAttemptAndReachedCounts()
                        .first;
            // Split IDs aren't actually used, other than > 0 meaning children
            // of splits
            split_id = performed_transition_id;
            Log("Adding a new reversed state linked to previous state", 2);
          }
        }
        // If not, then grab the expectation & split ID from one of the children
        else
        {
          const std::pair<int64_t, bool>& first_possible_match
              = expected_possible_result_states.front();
          if (first_possible_match.second)
          {
            throw std::runtime_error(
                  "Reversals cannot result in a parent index lookup");
          }
          const int64_t child_state_idx = first_possible_match.first;
          const UncertaintyPlanningTreeState& child_tree_state
              = planner_tree_.at(static_cast<size_t>(child_state_idx));
          const UncertaintyPlanningState& child_state
              = child_tree_state.GetValueImmutable();
          command = child_state.GetCommand();
          attempt_count = child_state.GetAttemptAndReachedCounts().first;
          split_id = child_state.GetSplitId();
          Log("Adding a new forward state linked to previous state", 2);
        }
        // Put together the new state
        // For now, we're going to assume that action outcomes are nominally
        // independent (this may be a terrible assumption, but we can change it
        // later)
        const UncertaintyPlanningState new_child_state(
            new_child_state_id, current_config, attempt_count, reached_count,
            effective_edge_Pfeasibility, reverse_attempt_count,
            reverse_reached_count, parent_motion_Pfeasibility, step_size,
            command, transition_id, reverse_transition_id, split_id, true);
        // We add a new child state to the graph
        const UncertaintyPlanningTreeState new_child_tree_state(
            new_child_state, acting_parent_state_index);
        planner_tree_.push_back(new_child_tree_state);
        // Add the linkage to the parent (link to the last state we just added)
        const int64_t new_state_index
            = static_cast<int64_t>(planner_tree_.size()) - 1;
        // NOTE - by adding to the tree, we have broken any references already
        // held so we can't use the previous_index_tree_state any more!
        planner_tree_.at(static_cast<size_t>(acting_parent_state_index))
            .AddChildIndex(new_state_index);
        // Update the policy graph with the new state
        RebuildPolicyGraph();
        // To get the action, we recursively call this function
        // (this time there will be an exact matching child state!)
        return QueryNormalBestAction(
            performed_transition_id, current_config, allow_branch_jumping,
            link_runtime_states_to_planned_parent, particle_clustering_fn);
      }
    }
  }

  int64_t UpdateNodeCountsAndTree(
      const std::vector<std::pair<int64_t, bool>>&
          expected_possible_result_states,
      const std::vector<std::pair<int64_t, bool>>&
          expected_result_state_matches)
  {
    // If there was one possible result state and it matches
    // This should be the most likely case, and requires the least editing of
    // the tree
    if (expected_possible_result_states.size() == 1
        && expected_result_state_matches.size() == 1)
    {
      Log("Result state matched single expected result", 2);
      const std::pair<int64_t, bool>& result_match
          = expected_result_state_matches.at(0);
      // Update the attempt/reached counts
      // If the transition was a forward transition, we update the result state
      if (result_match.second == false)
      {
        UncertaintyPlanningTreeState& result_tree_state
            = planner_tree_.at(static_cast<size_t>(result_match.first));
        UncertaintyPlanningState& result_state
            = result_tree_state.GetValueMutable();
        const std::pair<uint32_t, uint32_t> counts
            = result_state.GetAttemptAndReachedCounts();
        const uint32_t attempt_count
            = AddWithOverflowClamp(counts.first, policy_action_attempt_count_);
        const uint32_t reached_count
            = AddWithOverflowClamp(counts.second, policy_action_attempt_count_);
        result_state.UpdateAttemptAndReachedCounts(
            attempt_count, reached_count);
        return result_match.first;
      }
      else
      {
        UncertaintyPlanningTreeState& result_child_tree_state
            = planner_tree_.at(static_cast<size_t>(result_match.first));
        UncertaintyPlanningState& result_child_state
            = result_child_tree_state.GetValueMutable();
        const std::pair<uint32_t, uint32_t> counts
            = result_child_state.GetReverseAttemptAndReachedCounts();
        const uint32_t attempt_count
            = AddWithOverflowClamp(counts.first, policy_action_attempt_count_);
        const uint32_t reached_count
            = AddWithOverflowClamp(counts.second, policy_action_attempt_count_);
        result_child_state.UpdateReverseAttemptAndReachedCounts(
            attempt_count, reached_count);
        return result_child_tree_state.GetParentIndex();
      }
    }
    else
    {
      Log("Result state matched "
          + std::to_string(expected_result_state_matches.size()) + " of "
          + std::to_string(expected_possible_result_states.size())
          + " expected results", 2);
      //////////////////////////////////////////////////////////////////////////
      // Select the current best-distance result state as THE result state
      std::pair<int64_t, bool> best_result_state(-1, false);
      double best_distance = std::numeric_limits<double>::infinity();

      for (const auto& result_match : expected_result_state_matches)
      {
        const int64_t result_match_state_idx
            = (result_match.second)
              ? planner_tree_.at(static_cast<size_t>(result_match.first))
                  .GetParentIndex()
              : result_match.first;
        const double result_match_distance
            = policy_dijkstras_result_.GetNodeDistance(result_match_state_idx);
        if (result_match_distance < best_distance)
        {
          best_result_state = result_match;
          best_distance = result_match_distance;
        }
      }
      if (best_result_state.first < 0)
      {
        throw std::runtime_error("Could not identify best result state");
      }
      const int64_t result_state_index
          = (best_result_state.second)
            ? planner_tree_.at(static_cast<size_t>(best_result_state.first))
                .GetParentIndex()
            : best_result_state.first;
      if (best_result_state.second == false)
      {
        Log("Selected best match result state (forward movement): "
            + std::to_string(result_state_index), 1);
      }
      else
      {
        Log("Selected best match result state (reverse movement): "
            + std::to_string(result_state_index), 1);
      }
      //////////////////////////////////////////////////////////////////////////
      // Update the attempt/reached counts for all *POSSIBLE* result states
      for (const auto& possible_result_match : expected_possible_result_states)
      {
        UncertaintyPlanningTreeState& possible_result_tree_state
            = planner_tree_.at(static_cast<size_t>(
                possible_result_match.first));
        UncertaintyPlanningState& possible_result_state
            = possible_result_tree_state.GetValueMutable();
        if (possible_result_match.second == false)
        {
          const std::pair<uint32_t, uint32_t> counts
              = possible_result_state.GetAttemptAndReachedCounts();
          if ((possible_result_match.first == best_result_state.first)
              && (possible_result_match.second == best_result_state.second))
          {
            const uint32_t attempt_count
                = AddWithOverflowClamp(counts.first,
                                       policy_action_attempt_count_);
            const uint32_t reached_count
                = AddWithOverflowClamp(counts.second,
                                       policy_action_attempt_count_);
            possible_result_state.UpdateAttemptAndReachedCounts(
                attempt_count, reached_count);
          }
          else
          {
            const uint32_t attempt_count
                = AddWithOverflowClamp(counts.first,
                                       policy_action_attempt_count_);
            const uint32_t reached_count
                = AddWithOverflowClamp(counts.second, 0u);
            possible_result_state.UpdateAttemptAndReachedCounts(
                attempt_count, reached_count);
          }
        }
        else
        {
          const std::pair<uint32_t, uint32_t> reverse_counts
              = possible_result_state.GetReverseAttemptAndReachedCounts();
          if ((possible_result_match.first == best_result_state.first)
              && (possible_result_match.second == best_result_state.second))
          {
            const uint32_t attempt_count
                = AddWithOverflowClamp(reverse_counts.first,
                                       policy_action_attempt_count_);
            const uint32_t reached_count
                = AddWithOverflowClamp(reverse_counts.second,
                                       policy_action_attempt_count_);
            possible_result_state.UpdateReverseAttemptAndReachedCounts(
                attempt_count, reached_count);
          }
          else
          {
            const uint32_t attempt_count
                = AddWithOverflowClamp(reverse_counts.first,
                                       policy_action_attempt_count_);
            const uint32_t reached_count
                = AddWithOverflowClamp(reverse_counts.second, 0u);
            possible_result_state.UpdateReverseAttemptAndReachedCounts(
                attempt_count, reached_count);
          }
        }
      }
      //////////////////////////////////////////////////////////////////////////
      // Update the effective edge probabilities for the current transition
      UpdatePlannerTreeProbabilities();
      // Return the matching result state
      return result_state_index;
    }
  }

  void UpdatePlannerTreeProbabilities()
  {
    // Let's update the entire tree. This is slower than it could be, but I
    // don't want to miss anything
    UpdateChildTransitionProbabilities(0);
    // Backtrack up the tree and update P(->goal) probabilities
    for (int64_t idx = (static_cast<int64_t>(planner_tree_.size()) - 1);
         idx >= 0; idx--)
    {
      UpdateStateGoalReachedProbability(idx);
    }
    // Forward pass through the tree to update P(->goal) for leaf nodes
    for (auto& current_state : planner_tree_)
    {
      const int64_t parent_index = current_state.GetParentIndex();
      // This is only true for the root of the tree
      if (parent_index < 0)
      {
        continue;
      }
      // Get the parent state
      const UncertaintyPlanningTreeState& parent_state
          = planner_tree_.at(static_cast<size_t>(parent_index));
      // If the current state is on a goal branch
      if (current_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
      {
        continue;
      }
      // If we are a non-goal child of a goal branch state
      else if (parent_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
      {
        // Update P(goal reached) based on our ability to reverse to the goal
        // branch
        const double parent_pgoalreached
            = parent_state.GetValueImmutable().GetGoalPfeasibility();
        // We use negative goal reached probabilities to signal probability due
        // to reversing
        const double new_pgoalreached
            = -(parent_pgoalreached * current_state.GetValueImmutable()
                  .GetReverseEdgePfeasibility());
        current_state.GetValueMutable().SetGoalPfeasibility(new_pgoalreached);
      }
    }
  }

  void UpdateChildTransitionProbabilities(const int64_t current_state_index)
  {
    // Gather all the children, split them by transition, and recompute the
    // P(->)estimated edge probabilities
    const UncertaintyPlanningTreeState& current_tree_state
        = planner_tree_.at(static_cast<size_t>(current_state_index));
    const std::vector<int64_t>& child_state_indices
        = current_tree_state.GetChildIndices();
    // Split them by transition IDs
    std::map<uint64_t, std::vector<int64_t>> transition_children_map;
    for (const int64_t child_state_index : child_state_indices)
    {
      const UncertaintyPlanningTreeState& child_tree_state
          = planner_tree_.at(static_cast<size_t>(child_state_index));
      const UncertaintyPlanningState& child_state
          = child_tree_state.GetValueImmutable();
      const uint64_t child_state_transition_id
          = child_state.GetTransitionId();
      transition_children_map[child_state_transition_id]
          .push_back(child_state_index);
    }
    // Compute updated probabilites for each group
    for (auto itr = transition_children_map.begin();
         itr != transition_children_map.end(); ++itr)
    {
      const std::vector<int64_t> transition_child_indices = itr->second;
      // I don't know why this needs the return type declared, but it does.
      const std::function<UncertaintyPlanningState&(const int64_t)>
          get_planning_state_fn
              = [&] (const int64_t state_index) -> UncertaintyPlanningState&
      {
        return planner_tree_.at(static_cast<size_t>(state_index))
            .GetValueMutable();
      };
      UpdateEstimatedEffectiveProbabilities(
          get_planning_state_fn, transition_child_indices,
          edge_attempt_threshold_, logging_fn_);
    }
    // Perform the same update on all of our children
    for (const int64_t child_state_index : child_state_indices)
    {
      UpdateChildTransitionProbabilities(child_state_index);
    }
  }

  void UpdateStateGoalReachedProbability(const int64_t current_state_index)
  {
    UncertaintyPlanningTreeState& current_tree_state
        = planner_tree_.at(static_cast<size_t>(current_state_index));
    // Check all the children of the current node, and update the node's goal
    // reached probability accordingly
    //
    // Naively, the goal reached probability of a node is the maximum of the
    // child goal reached probabilities; intuitively, the probability of
    // reaching the goal is that of reaching the goal if we follow the best
    // child.
    //
    // HOWEVER - the existence of "split" child states, where multiple states
    // result from a single control input, makes this more compilcated. For
    // split child states, the goal reached probability of the split is the sum
    // over every split option of
    // (split goal probability * probability of split)
    //
    // We can identify split nodes as children which share a transition id
    // First, we go through the children and separate them based on transition
    // id (this puts all the children of a split together in one place)
    std::map<uint64_t, std::vector<int64_t>> effective_child_branches;
    for (const int64_t current_child_index
            : current_tree_state.GetChildIndices())
    {
      const uint64_t& child_transition_id
          = planner_tree_.at(static_cast<size_t>(current_child_index))
              .GetValueImmutable().GetTransitionId();
      effective_child_branches[child_transition_id]
          .push_back(current_child_index);
    }
    // Now that we have the transitions separated out, compute the goal
    // probability of each transition
    std::vector<double> effective_child_branch_probabilities;
    for (auto itr = effective_child_branches.begin();
         itr != effective_child_branches.end(); ++itr)
    {
      const double transtion_goal_probability
          = ComputeTransitionGoalProbability(
              itr->second, edge_attempt_threshold_);
      effective_child_branch_probabilities.push_back(
          transtion_goal_probability);
    }
    // Now, get the highest transtion probability
    if (effective_child_branch_probabilities.size() > 0)
    {
      const double max_transition_probability
          = *std::max_element(effective_child_branch_probabilities.begin(),
                              effective_child_branch_probabilities.end());
      if ((max_transition_probability < 0.0)
          || (max_transition_probability > 1.0))
      {
        throw std::runtime_error(
              "max_transition_probability out of range [0, 1]");
      }
      // Update the current state
      current_tree_state.GetValueMutable().SetGoalPfeasibility(
          max_transition_probability);
    }
    else
    {
      if (current_tree_state.GetValueMutable().GetGoalPfeasibility() > 0.0)
      {
        // Don't update P(goal reached) for an assumed goal state
      }
      else
      {
        // Don't P(goal reached) for a state with no children
      }
    }
  }

  double ComputeTransitionGoalProbability(
      const std::vector<int64_t>& child_node_indices,
      const uint32_t edge_attempt_threshold) const
  {
    UncertaintyPlanningStateVector child_states(
        child_node_indices.size());
    for (size_t idx = 0; idx < child_node_indices.size(); idx++)
    {
      // Get the current child
      const int64_t& current_child_index = child_node_indices.at(idx);
      const UncertaintyPlanningState& current_child
          = planner_tree_.at(static_cast<size_t>(current_child_index))
              .GetValueImmutable();
      child_states.at(idx) = current_child;
    }
    return ComputeTransitionGoalProbability(
        child_states, edge_attempt_threshold, logging_fn_);
  }

public:

  static void UpdateEstimatedEffectiveProbabilities(
      const std::function<UncertaintyPlanningState&(const int64_t)>&
          get_planning_state_fn,
      const std::vector<int64_t>& transition_state_indices,
      const uint32_t edge_attempt_threshold,
      const LoggingFunction& logging_fn)
  {
    // Now that we have the forward-propagated states, we go back and update
    // their effective edge P(feasibility)
    for (const int64_t current_state_index : transition_state_indices)
    {
      UncertaintyPlanningState& current_state
          = get_planning_state_fn(current_state_index);
      double percent_active = 1.0;
      double p_reached = 0.0;
      for (uint32_t try_attempt = 0; try_attempt < edge_attempt_threshold;
           try_attempt++)
      {
        // How many particles got to our state on this attempt?
        p_reached += (percent_active * current_state.GetRawEdgePfeasibility());
        // Update the percent of particles that are still usefully active
        double updated_percent_active = 0.0;
        for (const int64_t other_state_index : transition_state_indices)
        {
          if (other_state_index != current_state_index)
          {
            const UncertaintyPlanningState& other_state
                = get_planning_state_fn(other_state_index);
            // Only if this state has nominally independent outcomes can we
            // expect particles that return to the parent to actually reach a
            // different outcome in future repeats
            if (other_state.IsActionOutcomeNominallyIndependent())
            {
              const double p_reached_other
                  = percent_active * other_state.GetRawEdgePfeasibility();
              const double p_returned_to_parent
                  = p_reached_other * other_state.GetReverseEdgePfeasibility();
              updated_percent_active += p_returned_to_parent;
            }
          }
        }
        percent_active = updated_percent_active;
      }
      if ((p_reached >= 0.0) && (p_reached <= 1.0))
      {
        current_state.SetEffectiveEdgePfeasibility(p_reached);
      }
      else if ((p_reached >= 0.0) && (p_reached <= 1.001))
      {
        logging_fn(
            "WARNING - P(reached) = " + std::to_string(p_reached)
            + " > 1.0 (probably numerical error)", 1);
        p_reached = 1.0;
        current_state.SetEffectiveEdgePfeasibility(p_reached);
      }
      else
      {
        throw std::runtime_error("p_reached out of range [0, 1]");
      }
    }
  }

  static double ComputeTransitionGoalProbability(
      const UncertaintyPlanningStateVector& child_nodes,
      const uint32_t planner_action_try_attempts,
      const LoggingFunction& logging_fn)
  {
    // Let's handle the special cases first
    // The most common case - a non-split transition
    if (child_nodes.size() == 1)
    {
      const UncertaintyPlanningState& current_child = child_nodes.front();
      return (current_child.GetGoalPfeasibility()
              * current_child.GetEffectiveEdgePfeasibility());
    }
    // IMPOSSIBLE (but we handle it just to be sure)
    else if (child_nodes.size() == 0)
    {
      return 0.0;
    }
    // Let's handle the split case(s)
    else
    {
      double percent_active = 1.0;
      double total_p_goal_reached = 0.0;
      for (uint32_t try_attempt = 0;
           try_attempt < planner_action_try_attempts; try_attempt++)
      {
        const double attempt_percent_active = percent_active;
        for (const UncertaintyPlanningState& current_child : child_nodes)
        {
          const double percent_reached_child =
              attempt_percent_active * current_child.GetRawEdgePfeasibility();
          const double raw_child_goal_Pfeasibility =
              current_child.GetGoalPfeasibility();
          const double child_goal_Pfeasibility =
              (raw_child_goal_Pfeasibility > 0.0) ?
                  raw_child_goal_Pfeasibility : 0.0;
          const double percent_reached_goal =
              percent_reached_child * child_goal_Pfeasibility;
          total_p_goal_reached += percent_reached_goal;
          // Update percent active
          // Anything that reached the goal is no longer active.
          percent_active -= percent_reached_goal;
          // Do any return to the parent?
          const double percent_not_reached_goal =
              percent_reached_child * (1.0 - child_goal_Pfeasibility);
          const double p_return_to_parent =
              (current_child.IsActionOutcomeNominallyIndependent()) ?
                  current_child.GetReverseEdgePfeasibility() : 0.0;
          const double p_stuck_here = 1.0 - p_return_to_parent;
          const double percent_stuck_here =
                  percent_not_reached_goal * p_stuck_here;
          // Anything stuck at the child state is no longer active.
          percent_active -= percent_stuck_here;
        }
      }
      if ((total_p_goal_reached >= 0.0) && (total_p_goal_reached <= 1.0))
      {
        logging_fn(
            "Computed total P(goal reached) = "
            + std::to_string(total_p_goal_reached), 1);
        return total_p_goal_reached;
      }
      else if ((total_p_goal_reached >= 0.0) && (total_p_goal_reached <= 1.001))
      {
        logging_fn(
            "WARNING - total P(goal reached) = "
            + std::to_string(total_p_goal_reached)
            + " > 1.0 (probably numerical error)", 1);
        return 1.0;
      }
      else
      {
        throw std::runtime_error("total_p_goal_reached out of range [0, 1]");
      }
    }
  }
};
}  // namespace uncertainty_planning_core

template<typename Configuration, typename ConfigSerializer,
         typename ConfigAlloc=std::allocator<Configuration>>
std::ostream& operator<<(
    std::ostream& strm,
    const uncertainty_planning_core::ExecutionPolicy<
        Configuration, ConfigSerializer, ConfigAlloc>& policy)
{
  const auto& raw_policy_tree = policy.GetRawPolicyTree();
  strm << "Execution Policy - Policy: ";
  for (size_t idx = 0; idx < raw_policy_tree.size(); idx++)
  {
    const auto& policy_tree_state = raw_policy_tree.at(idx);
    const int64_t parent_index = policy_tree_state.GetParentIndex();
    const std::vector<int64_t>& child_indices
        = policy_tree_state.GetChildIndices();
    const auto& policy_state = policy_tree_state.GetValueImmutable();
    strm << "\nState # " << idx << " with parent " << parent_index
         << " and children "
         << common_robotics_utilities::print::Print(child_indices, true)
         << " - value: "
         << common_robotics_utilities::print::Print(policy_state);
  }
  return strm;
}
