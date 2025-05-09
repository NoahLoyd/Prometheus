import unittest
from unittest.mock import MagicMock, patch
from tools.tool_manager import ToolManager  # Replace with your actual ToolManager class
from core.brain import StrategicBrain
from core.memory import Memory  # Replace with your actual Memory implementation
from llm.llm_factory import build_llm_router


class TestStrategicBrain(unittest.TestCase):
    def setUp(self):
        """
        Set up dependencies for StrategicBrain and related components.
        """
        # Mock LLM Configuration for local model
        self.llm_config = {
            "mistral-7b": {
                "type": "local",
                "path": "/mock/models/mistral-7b",
                "device": "cpu",
                "tags": ["strategy", "planning"]
            }
        }

        # Initialize memory (replace with actual implementation)
        self.memory = Memory()

        # Initialize tool manager (replace with your actual tool manager implementation)
        self.tool_manager = ToolManager()

        # Build the LLMRouter using the factory
        self.llm_router = build_llm_router(self.llm_config)

        # Initialize StrategicBrain with the required dependencies
        self.brain = StrategicBrain(
            llm=self.llm_router,
            tool_manager=self.tool_manager,
            memory=self.memory
        )

    def test_step_logging_with_timestamp(self):
        """
        Verify that StrategicBrain logs each step with a timestamp and stores it in memory.
        """
        goal = "make $100 this week"
        self.brain.set_goal(goal)

        # Mock the tool execution to simulate steps
        mock_steps = [
            ("TaskRabbit", "Sign up for TaskRabbit", True),
            ("eBay", "Sell unused items", True),
            ("Fiverr", "Offer freelance services", False),
        ]
        self.brain.plan = [step[:2] for step in mock_steps]  # Set the plan manually for this test

        # Mock memory to track log calls
        self.memory.log_event = MagicMock()

        # Simulate achieving the goal
        self.brain.achieve_goal()

        # Verify that each step was logged with a timestamp
        for step in mock_steps:
            tool_name, query, success = step
            self.memory.log_event.assert_any_call(
                "step_execution",
                {
                    "tool_name": tool_name,
                    "query": query,
                    "success": success,
                    "timestamp": unittest.mock.ANY,  # Ensure the timestamp is present
                }
            )

    def test_daily_summary_generation(self):
        """
        Check if daily summary generation works based on mock memory state.
        """
        # Mock memory state
        self.memory.get_daily_logs = MagicMock(return_value=[
            {"goal": "make $100 this week", "steps": 3, "success": 2, "failure": 1},
            {"goal": "clean the house", "steps": 5, "success": 5, "failure": 0},
        ])

        # Simulate daily summary generation
        summary = self.brain.generate_daily_summary()

        # Verify the summary format and content
        self.assertIn("make $100 this week", summary)
        self.assertIn("clean the house", summary)
        self.assertIn("Steps completed: 3 (Success: 2, Failure: 1)", summary)
        self.assertIn("Steps completed: 5 (Success: 5, Failure: 0)", summary)

        # Print the summary for visual inspection
        print("\nDaily Summary:")
        print(summary)

    @patch("tools.tool_manager.ToolManager.execute_tool")
    def test_tool_execution_behavior(self, mock_execute_tool):
        """
        Add mocks or stubs for tool execution to simulate different tools returning results, failing, or timing out.
        """
        goal = "make $100 this week"
        self.brain.set_goal(goal)

        # Mock tool execution behavior
        mock_execute_tool.side_effect = [
            {"tool_name": "TaskRabbit", "query": "Sign up for TaskRabbit", "success": True},
            {"tool_name": "eBay", "query": "Sell unused items", "success": False},  # Simulate failure
            TimeoutError("Fiverr tool timed out"),  # Simulate timeout
        ]

        # Simulate achieving the goal
        with self.assertLogs("core.brain", level="ERROR") as cm:
            self.brain.achieve_goal()

        # Verify tool execution behavior
        mock_execute_tool.assert_called()  # Ensure tools were called
        self.assertIn("TaskRabbit", mock_execute_tool.call_args_list[0][0][0]["tool_name"])
        self.assertIn("eBay", mock_execute_tool.call_args_list[1][0][0]["tool_name"])

        # Verify timeout error handling
        self.assertTrue(any("Fiverr tool timed out" in log for log in cm.output))

    def test_fallback_behavior_on_failure(self):
        """
        Ensure the system calls fallback_strategy.refine_plan() only when all model results fail.
        """
        goal = "an impossible task"
        self.brain.set_goal(goal)

        # Mock the fallback strategy
        self.brain.fallback_strategy.refine_plan = MagicMock(return_value=[
            ("FallbackTool", "Fallback plan step 1")
        ])

        # Simulate all models failing
        self.brain.llm_router.generate_plan = MagicMock(return_value=[])

        # Achieve the goal (should trigger fallback behavior)
        output = self.brain.achieve_goal()

        # Verify fallback strategy was called
        self.brain.fallback_strategy.refine_plan.assert_called_once_with(goal, None, None)

        # Verify the fallback plan was used
        self.assertIn("FallbackTool", [step[0] for step in self.brain.plan])

    def test_logging_for_errors_and_insights(self):
        """
        Use assertLogs or memory log inspection to confirm specific logging behavior for errors and insights.
        """
        goal = "make $100 this week"
        self.brain.set_goal(goal)

        # Mock logging in memory
        self.memory.log_event = MagicMock()

        # Mock a failure scenario
        self.brain.llm_router.generate_plan = MagicMock(return_value=[])
        self.brain.fallback_strategy.refine_plan = MagicMock(return_value=[])

        # Simulate achieving the goal
        self.brain.achieve_goal()

        # Verify that error and insight logs were recorded
        self.memory.log_event.assert_any_call("error", unittest.mock.ANY)
        self.memory.log_event.assert_any_call("insight", unittest.mock.ANY)


if __name__ == "__main__":
    unittest.main()
    
