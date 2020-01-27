#include <common_robotics_utilities/simple_prngs.hpp>
#include <uncertainty_planning_core/task_planner_adapter.hpp>

class PutInBoxState
{
private:

  int32_t objects_available_;
  bool object_put_away_;
  bool box_open_;

public:

  PutInBoxState()
    : objects_available_(-1), object_put_away_(false), box_open_(false) {}

  PutInBoxState(const int32_t objects_available,
                const bool object_put_away,
                const bool box_open)
    : objects_available_(objects_available),
      object_put_away_(object_put_away),
      box_open_(box_open) {}

  int32_t ObjectsAvailable() const { return objects_available_; }

  bool ObjectPutAway() const { return object_put_away_; }

  bool BoxOpen() const { return box_open_; }

  uint64_t GetStateReadiness() const
  {
    return GetStateReadiness(*this);
  }

  static uint64_t GetStateReadiness(const PutInBoxState& state)
  {
    // Readiness for the state of the box.
    const uint8_t BOX_READY_OPEN = 0x01;
    const uint8_t BOX_READY_CLOSED = 0x02;
    uint8_t box_readiness = 0x00;
    if (state.ObjectsAvailable() > 0)
    {
      if (state.BoxOpen())
      {
        box_readiness |= BOX_READY_OPEN;
      }
    }
    else if (state.ObjectsAvailable() == 0)
    {
      if (!state.BoxOpen())
      {
        box_readiness |= BOX_READY_CLOSED;
      }
    }
    // Readiness for where we are in completing the task.
    const uint8_t NUM_OBJECTS_KNOWN = 0x01;
    const uint8_t OBJECTS_ALL_PUT_AWAY = 0x02;
    uint8_t task_readiness = 0x00;
    if (state.ObjectsAvailable() >= 0)
    {
      task_readiness |= NUM_OBJECTS_KNOWN;
      if (state.ObjectsAvailable() == 0)
      {
        task_readiness |= OBJECTS_ALL_PUT_AWAY;
      }
    }
    // Readiness for where we are in handling the active object.
    const uint8_t ACTIVE_OBJECT_PUT_AWAY = 0x01;
    uint8_t active_object_readiness = 0x00;
    if (state.ObjectPutAway())
    {
      active_object_readiness |= ACTIVE_OBJECT_PUT_AWAY;
    }
    // Combine into state readiness.
    uint64_t state_readiness =
        static_cast<uint64_t>(task_readiness) |
        static_cast<uint64_t>(static_cast<uint64_t>(box_readiness) << 8) |
        static_cast<uint64_t>(
            static_cast<uint64_t>(active_object_readiness) << 16);
    return state_readiness;
  }

  std::string Print() const
  {
    std::string rep
        = "Objects available: " + std::to_string(objects_available_)
          + " Object put away: "
          + common_robotics_utilities::print::Print(object_put_away_)
          + " Box open: "
          + common_robotics_utilities::print::Print(box_open_);
    return rep;
  }

  static bool IsSingleExecutionComplete(const PutInBoxState& state)
  {
    if (IsTaskComplete(state))
    {
      return true;
    }
    else if ((state.ObjectsAvailable() > 0) && (state.ObjectPutAway() == true))
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  static bool IsTaskComplete(const PutInBoxState& state)
  {
    if ((state.ObjectsAvailable() == 0) && (state.BoxOpen() == false))
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  static std::string TypeName()
  {
    return std::string("PutInBoxState");
  }

  static uint64_t
  Serialize(const PutInBoxState& value, std::vector<uint8_t>& buffer)
  {
    using common_robotics_utilities::serialization::SerializeMemcpyable;
    const uint64_t start_buffer_size = buffer.size();
    SerializeMemcpyable<int32_t>(value.ObjectsAvailable(), buffer);
    SerializeMemcpyable<uint8_t>(
        static_cast<uint8_t>(value.ObjectPutAway()), buffer);
    SerializeMemcpyable<uint8_t>(
        static_cast<uint8_t>(value.BoxOpen()), buffer);
    const uint64_t end_buffer_size = buffer.size();
    const uint64_t bytes_written = end_buffer_size - start_buffer_size;
    return bytes_written;
  }

  static common_robotics_utilities::serialization::Deserialized<PutInBoxState>
  Deserialize(
      const std::vector<uint8_t>& buffer, const uint64_t starting_offset)
  {
    using common_robotics_utilities::serialization::DeserializeMemcpyable;
    uint64_t current_position = starting_offset;
    const auto deser_objects_available
        = DeserializeMemcpyable<int32_t>(buffer, current_position);
    const int32_t objects_available = deser_objects_available.Value();
    current_position += deser_objects_available.BytesRead();
    const auto deser_object_put_away
        = DeserializeMemcpyable<uint8_t>(buffer, current_position);
    const bool object_put_away
        = static_cast<bool>(deser_object_put_away.Value());
    current_position += deser_object_put_away.BytesRead();
    const auto deser_box_open
        = DeserializeMemcpyable<uint8_t>(buffer, current_position);
    const bool box_open = static_cast<bool>(deser_box_open.Value());
    current_position += deser_box_open.BytesRead();
    // How much did we read?
    const uint64_t bytes_read = current_position - starting_offset;
    return common_robotics_utilities::serialization::MakeDeserialized(
        PutInBoxState(objects_available, object_put_away, box_open),
        bytes_read);
  }
};

std::ostream& operator<<(std::ostream& strm, const PutInBoxState& state)
{
  strm << state.Print();
  return strm;
}

class OpenBoxPrimitive
    : public uncertainty_planning_core::task_planner_adapter
        ::ActionPrimitiveInterface<PutInBoxState>
{
public:

  OpenBoxPrimitive()
    : uncertainty_planning_core::task_planner_adapter
        ::ActionPrimitiveInterface<PutInBoxState>() {}

  virtual bool IsCandidate(const PutInBoxState& state) const
  {
    if (state.BoxOpen())
    {
      std::cout << "State " << state.Print() << " is not a candidate for "
                << Name() << " since box is already open" << std::endl;
      return false;
    }
    else
    {
      return true;
    }
  }

  virtual std::string Name() const
  {
    return "OpenBoxPrimitive";
  }

  virtual double Ranking() const
  {
    return 1.0;
  }

  virtual std::vector<std::pair<PutInBoxState, bool>>
  GetOutcomes(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      auto box_open
          = std::make_pair(PutInBoxState(state.ObjectsAvailable(),
                                         state.ObjectPutAway(), true),
                           false);
      return std::vector<std::pair<PutInBoxState, bool>>{box_open};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }

  virtual std::vector<PutInBoxState>
  Execute(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      return std::vector<PutInBoxState>{
        PutInBoxState(state.ObjectsAvailable(), state.ObjectPutAway(), true)};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }
};

class CloseBoxPrimitive
    : public uncertainty_planning_core::task_planner_adapter
        ::ActionPrimitiveInterface<PutInBoxState>
{
public:

  CloseBoxPrimitive()
    : uncertainty_planning_core::task_planner_adapter
        ::ActionPrimitiveInterface<PutInBoxState>() {}

  virtual bool IsCandidate(const PutInBoxState& state) const
  {
    if (state.BoxOpen())
    {
      return true;
    }
    else
    {
      std::cout << "State " << state.Print() << " is not a candidate for "
                << Name() << " since box is already closed" << std::endl;
      return false;
    }
  }

  virtual std::string Name() const
  {
    return "CloseBoxPrimitive";
  }

  virtual double Ranking() const
  {
    return 2.0;
  }

  virtual std::vector<std::pair<PutInBoxState, bool>>
  GetOutcomes(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      auto box_closed
          = std::make_pair(PutInBoxState(state.ObjectsAvailable(),
                                         state.ObjectPutAway(), false),
                           false);
      return std::vector<std::pair<PutInBoxState, bool>>{box_closed};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }

  virtual std::vector<PutInBoxState>
  Execute(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      return std::vector<PutInBoxState>{
        PutInBoxState(state.ObjectsAvailable(), state.ObjectPutAway(), false)};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }
};

class CheckIfAvailableObjectPrimitive
    : public uncertainty_planning_core::task_planner_adapter
        ::ActionPrimitiveInterface<PutInBoxState>
{
public:

  CheckIfAvailableObjectPrimitive()
    : uncertainty_planning_core::task_planner_adapter
        ::ActionPrimitiveInterface<PutInBoxState>() {}

  virtual bool IsCandidate(const PutInBoxState& state) const
  {
    if (state.ObjectsAvailable() < 0)
    {
      return true;
    }
    else
    {
      std::cout << "State " << state.Print() << " is not a candidate for "
                << Name() << " since num objects is already known" << std::endl;
      return false;
    }
  }

  virtual std::string Name() const
  {
    return "CheckIfAvailableObjectPrimitive";
  }

  virtual double Ranking() const
  {
    return 3.0;
  }

  virtual std::vector<std::pair<PutInBoxState, bool>>
  GetOutcomes(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      auto object_available
          = std::make_pair(PutInBoxState(1, state.ObjectPutAway(),
                                         state.BoxOpen()),
                           false);
      auto none_available
          = std::make_pair(PutInBoxState(0, state.ObjectPutAway(),
                                         state.BoxOpen()),
                           false);
      return std::vector<std::pair<PutInBoxState, bool>>{object_available,
                                                         none_available};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }

  virtual std::vector<PutInBoxState>
  Execute(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      return std::vector<PutInBoxState>{
        PutInBoxState(std::abs(state.ObjectsAvailable()),
                      state.ObjectPutAway(), state.BoxOpen())};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }
};

class PutObjectInBoxPrimitive
    : public uncertainty_planning_core::task_planner_adapter
        ::ActionPrimitiveInterface<PutInBoxState>
{
public:

  PutObjectInBoxPrimitive()
    : uncertainty_planning_core::task_planner_adapter
        ::ActionPrimitiveInterface<PutInBoxState>() {}

  virtual bool IsCandidate(const PutInBoxState& state) const
  {
    if ((state.ObjectsAvailable() > 0) && (state.ObjectPutAway() == false)
        && state.BoxOpen())
    {
      return true;
    }
    else
    {
      std::cout << "State " << state.Print() << " is not a candidate for "
                << Name() << std::endl;
      return false;
    }
  }

  virtual std::string Name() const
  {
    return "PutObjectInBoxPrimitive";
  }

  virtual double Ranking() const
  {
    return 4.0;
  }

  virtual std::vector<std::pair<PutInBoxState, bool>>
  GetOutcomes(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      auto object_remaining
          = std::make_pair(PutInBoxState(1, true, true), false);
      auto task_done
          = std::make_pair(PutInBoxState(0, true, true), false);
      return std::vector<std::pair<PutInBoxState, bool>>{object_remaining,
                                                         task_done};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }

  virtual std::vector<PutInBoxState>
  Execute(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      return std::vector<PutInBoxState>{
        PutInBoxState(state.ObjectsAvailable() - 1, true, true)};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "task_planner_adapter_test");
  ros::NodeHandle nh;
  ros::NodeHandle nhp;
  // Get debug level
  const int32_t debug_level = nhp.param(std::string("debug_level"), 0);
  // Make display function
  ros::Publisher display_debug_publisher =
      nh.advertise<visualization_msgs::MarkerArray>(
        "task_planner_debug_display_markers", 1, true);
  const uncertainty_planning_core::DisplayFunction display_fn
      = [&] (const visualization_msgs::MarkerArray& markers)
  {
    display_debug_publisher.publish(markers);
  };
  // Make logging function
  const uncertainty_planning_core::LoggingFunction logging_fn
      = [&] (const std::string& msg, const int32_t level)
  {
    ROS_INFO_NAMED(ros::this_node::getName(), "[%d] %s", level, msg.c_str());
  };
  // Get seed for PRNG
  int32_t prng_seed_init
      = static_cast<int32_t>(nhp.param(std::string("prng_seed_init"), -1));
  if (prng_seed_init == -1)
  {
    prng_seed_init = static_cast<int32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    logging_fn("No PRNG seed provided, initializing from clock to ["
               + std::to_string(prng_seed_init) + "]", 1);
  }
  const int64_t prng_seed
      = static_cast<int64_t>(
          common_robotics_utilities::simple_prngs
              ::SplitMix64PRNG(static_cast<uint64_t>(prng_seed_init))());
  // Make the planner interface
  const std::function<uint64_t(const PutInBoxState&)> state_readiness_fn
      = [] (const PutInBoxState& state)
  {
    return PutInBoxState::GetStateReadiness(state);
  };

  const std::function<bool(const PutInBoxState&)> single_execution_complete_fn
      = [] (const PutInBoxState& state)
  {
    return PutInBoxState::IsSingleExecutionComplete(state);
  };

  const std::function<bool(const PutInBoxState&)> task_complete_fn
      = [] (const PutInBoxState& state)
  {
    return PutInBoxState::IsTaskComplete(state);
  };

  uncertainty_planning_core::task_planner_adapter
      ::TaskPlannerAdapter<PutInBoxState, PutInBoxState>
          planner(state_readiness_fn, single_execution_complete_fn,
                  task_complete_fn, logging_fn, display_fn, prng_seed,
                  debug_level);
  // Add primitives
  uncertainty_planning_core::task_planner_adapter
      ::ActionPrimitivePtr<PutInBoxState> open_box_primitive_ptr(
          new OpenBoxPrimitive());
  uncertainty_planning_core::task_planner_adapter
      ::ActionPrimitivePtr<PutInBoxState> close_box_primitive_ptr(
          new CloseBoxPrimitive());
  uncertainty_planning_core::task_planner_adapter
      ::ActionPrimitivePtr<PutInBoxState> check_if_available_primitive_ptr(
          new CheckIfAvailableObjectPrimitive());
  uncertainty_planning_core::task_planner_adapter
      ::ActionPrimitivePtr<PutInBoxState> put_object_in_box_primitive_ptr(
          new PutObjectInBoxPrimitive());
  planner.RegisterPrimitive(open_box_primitive_ptr);
  planner.RegisterPrimitive(close_box_primitive_ptr);
  planner.RegisterPrimitive(check_if_available_primitive_ptr);
  planner.RegisterPrimitive(put_object_in_box_primitive_ptr);
  // Plan
  const auto plan_result
      = planner.PlanPolicy(PutInBoxState(), 10.0, 1.0, 0.01, 50u, 50u);
  logging_fn(
      "Task planning statistics: "
      + common_robotics_utilities::print::Print(plan_result.Statistics()), 1);
  logging_fn(
      "Planned policy:\n"
      + common_robotics_utilities::print::Print(plan_result.Policy()), 1);
  int32_t objects_to_put_away = 5;
  const std::function<PutInBoxState(void)> single_execution_initialization_fn
      = [&] (void)
  {
    const int32_t objects_to_put_away_old = objects_to_put_away;
    objects_to_put_away--;
    return PutInBoxState(-objects_to_put_away_old, false, false);
  };
  const std::function<void(const PutInBoxState&, const PutInBoxState&)>
      pre_action_callback_fn
          = [] (const PutInBoxState&, const PutInBoxState&) {};
  const std::function<void(const std::vector<PutInBoxState>&, const int64_t)>
      post_outcome_callback_fn
          = [] (const std::vector<PutInBoxState>&, const int64_t) {};
  auto exec_result
      = planner.ExecutePolicy(plan_result.Policy(),
                              single_execution_initialization_fn,
                              pre_action_callback_fn,
                              post_outcome_callback_fn,
                              100u, 100u, true, false);
  logging_fn(
      "Task execution statistics: "
      + common_robotics_utilities::print::Print(
          exec_result.ExecutionStatistics()), 1);
  return 0;
}
