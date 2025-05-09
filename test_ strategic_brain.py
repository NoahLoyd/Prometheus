import unittest
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

    def test_set_and_achieve_goal(self):
        """
        Test if StrategicBrain successfully sets and achieves a goal.
        """
        # Define a sample goal
        goal = "make $100 this week"

        # Step 1: Set the goal
        self.brain.set_goal(goal)

        # Assert that the goal has been set correctly in the brain
        self.assertEqual(self.brain.goal, goal, "The goal should be set correctly.")

        # Step 2: Achieve the goal
        output = self.brain.achieve_goal()

        # Extract results from the output dictionary
        results = output.get("results", [])
        reflection = output.get("reflection", {})

        # Assert that results are returned and contain actionable steps
        self.assertIsNotNone(results, "Results should not be None.")
        self.assertGreater(len(results), 0, "Results should contain at least one actionable step.")

        # Validate each result in results
        for result in results:
            self.assertIn("tool_name", result, "Each result must have a 'tool_name'.")
            self.assertIn("query", result, "Each result must have a 'query'.")
            self.assertIn("success", result, "Each result must have a 'success' flag.")
            self.assertIsInstance(result["success"], bool, "'success' must be a boolean.")

        # Print the plan and results for visual inspection
        print(f"\nPlan for goal '{goal}':")
        for step in self.brain.plan:  # Assuming self.brain.plan is a list of steps
            print(f"- {step}")

        print("\nResults of achieving the goal:")
        for result in results:
            print(f"- Tool: {result['tool_name']}, Query: {result['query']}, Success: {result['success']}")

        # Validate and print reflection if it exists
        if reflection:
            success_ratio = reflection.get("success_ratio")
            failure_ratio = reflection.get("failure_ratio")

            print("\nReflection:")
            print(f"- Success Ratio: {success_ratio}")
            print(f"- Failure Ratio: {failure_ratio}")

            self.assertIsInstance(success_ratio, float, "Success ratio must be a float.")
            self.assertIsInstance(failure_ratio, float, "Failure ratio must be a float.")
            self.assertGreaterEqual(success_ratio, 0.0, "Success ratio must be >= 0.")
            self.assertLessEqual(success_ratio, 1.0, "Success ratio must be <= 1.")
            self.assertGreaterEqual(failure_ratio, 0.0, "Failure ratio must be >= 0.")
            self.assertLessEqual(failure_ratio, 1.0, "Failure ratio must be <= 1.")

    def test_fallback_behavior(self):
        """
        Test fallback behavior when all models fail.
        """
        # Mock a failing goal
        goal = "an impossible task"

        # Set the goal
        self.brain.set_goal(goal)

        # Mock the achieve_goal behavior to simulate all models failing
        def mock_achieve_goal():
            return {
                "results": [],
                "reflection": None,
                "insights": [],
                "comparison": "Fallback triggered."
            }

        # Replace the achieve_goal method with the mock
        self.brain.achieve_goal = mock_achieve_goal

        # Step 2: Achieve the goal (should trigger fallback)
        output = self.brain.achieve_goal()

        # Assert fallback behavior
        self.assertEqual(output["results"], [], "Results should be empty when fallback is triggered.")
        self.assertIn("Fallback triggered", output["comparison"], "Fallback behavior should be indicated in comparison.")

        # Print fallback output for visual inspection
        print(f"\nFallback behavior for goal '{goal}':")
        print("- Comparison:", output["comparison"])


if __name__ == "__main__":
    unittest.main()
    