# core/brain.py

class StrategicBrain:
    """
    A class to manage high-level goal execution for the AI agent Promethyn.
    It breaks down goals into actionable steps, executes them, and logs progress.
    """

    def __init__(self, tool_manager, memory):
        """
        Initialize the StrategicBrain with ToolManager and ShortTermMemory.

        Args:
            tool_manager (ToolManager): The ToolManager instance for executing tools.
            memory (ShortTermMemory): The memory instance for logging progress.
        """
        self.tool_manager = tool_manager
        self.memory = memory

    def break_down_goal(self, goal):
        """
        Break down a high-level goal into actionable (tool_name, query) steps.

        Args:
            goal (str): The high-level goal to be achieved.

        Returns:
            list: A list of (tool_name, query) tuples representing the steps.
        """
        # Placeholder logic for breaking down goals
        # Extend this function to handle diverse goals using keyword matching or a planner
        steps = []
        if "make" in goal and "$" in goal:
            steps.append(("calculator", f"Calculate revenue needed for {goal}"))
            steps.append(("internet_tool", f"Research methods to achieve {goal}"))
        elif "learn" in goal:
            steps.append(("note_tool", f"Create a learning plan for {goal}"))
            steps.append(("internet_tool", f"Find resources to accomplish {goal}"))
        else:
            steps.append(("note_tool", f"Document plan for {goal}"))
        return steps

    def execute_step(self, tool_name, query):
        """
        Execute a single action step and log progress.

        Args:
            tool_name (str): The name of the tool to be used.
            query (str): The query to be executed by the tool.

        Returns:
            dict: A dictionary containing the result or error of the execution.
        """
        try:
            result = self.tool_manager.call_tool(tool_name, query)
            self.memory.save(f"{tool_name}_result", result)
            return {"success": True, "result": result}
        except Exception as e:
            error_message = str(e)
            self.memory.save(f"{tool_name}_error", error_message)
            return {"success": False, "error": error_message}

    def execute_steps(self, steps):
        """
        Execute the action steps and log progress.

        Args:
            steps (list): A list of (tool_name, query) tuples.

        Returns:
            list: A list of results for each step, including success or error information.
        """
        results = []
        for tool_name, query in steps:
            result = self.execute_step(tool_name, query)
            results.append({"tool_name": tool_name, "query": query, **result})
        return results

    def retry_or_replan(self, goal, failed_steps):
        """
        Retry or re-plan for failed steps.

        Args:
            goal (str): The original high-level goal.
            failed_steps (list): The steps that failed.

        Returns:
            list: A new list of (tool_name, query) steps.
        """
        # Placeholder logic for retrying or replanning
        # Extend this function to handle retries or alternative plans intelligently
        new_steps = []
        for step in failed_steps:
            tool_name = step["tool_name"]
            query = step["query"]
            if "retry" in query.lower():
                new_steps.append((tool_name, query))  # Retry the same step
            else:
                new_steps.append(("note_tool", f"Log failure for {query}"))
        return new_steps

    def achieve_goal(self, goal):
        """
        Achieve a high-level goal by breaking it into steps, executing them, and handling failures.

        Args:
            goal (str): The high-level goal to be achieved.

        Returns:
            dict: The final result of the goal execution, including successes and failures.
        """
        self.memory.save("current_goal", goal)
        steps = self.break_down_goal(goal)
        step_results = self.execute_steps(steps)

        failed_steps = [step for step in step_results if not step["success"]]
        if failed_steps:
            new_steps = self.retry_or_replan(goal, failed_steps)
            retry_results = self.execute_steps(new_steps)
            step_results.extend(retry_results)

        # Log the final outcome
        self.memory.save("goal_result", step_results)
        return {
            "goal": goal,
            "results": step_results,
            "success": all(step["success"] for step in step_results),
        }