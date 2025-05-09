# core/brain.py

import time


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
        self.goal = None
        self.plan = None

    def set_goal(self, goal):
        """
        Set a high-level goal and plan steps to achieve it.

        Args:
            goal (str): The high-level goal to be achieved.
        """
        self.goal = goal
        self.plan = self.plan_goal(goal)
        self.memory.save("current_goal", goal)
        self.memory.save("current_plan", self.plan)

    def plan_goal(self, goal):
        """
        Plan steps to achieve a high-level goal based on keywords.

        Args:
            goal (str): The high-level goal to be achieved.

        Returns:
            list: A list of (tool_name, query) tuples representing the steps.
        """
        steps = []
        if "make money" in goal.lower():
            steps.append(("calculator", "Calculate revenue needed to achieve goal"))
            steps.append(("internet", "Research methods to make money"))
        elif "learn" in goal.lower():
            steps.append(("note", f"Create a learning plan for {goal}"))
            steps.append(("internet", f"Find resources to accomplish {goal}"))
        elif "grow audience" in goal.lower():
            steps.append(("summarizer", "Summarize effective audience growth techniques"))
            steps.append(("internet", "Research audience growth strategies"))
        else:
            steps.append(("note", f"Document plan for {goal}"))
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

    def execute_plan(self):
        """
        Execute the planned steps and log progress.

        Returns:
            list: A list of results for each step, including success or error information.
        """
        if not self.plan:
            raise ValueError("No plan set. Use set_goal() to define a plan.")

        results = []
        for tool_name, query in self.plan:
            result = self.execute_step(tool_name, query)
            results.append({"tool_name": tool_name, "query": query, **result})
            self.memory.save("step_result", result)
            time.sleep(1)  # Simulate real-time processing delay
        return results

    def retry_or_replan(self, failed_steps):
        """
        Retry or re-plan for failed steps.

        Args:
            failed_steps (list): The steps that failed.

        Returns:
            list: A new list of (tool_name, query) steps.
        """
        new_steps = []
        for step in failed_steps:
            tool_name = step["tool_name"]
            query = step["query"]
            if "retry" not in query.lower():
                new_steps.append((tool_name, f"Retry: {query}"))
            else:
                new_steps.append(("note", f"Log failure for {query}"))
            self.memory.save("replan_step", {"tool_name": tool_name, "query": query})
        return new_steps

    def achieve_goal(self):
        """
        Execute the current plan, handle failures, and log the results.

        Returns:
            dict: The final result of the goal execution, including successes and failures.
        """
        if not self.goal or not self.plan:
            raise ValueError("Goal and plan must be set before execution.")

        step_results = self.execute_plan()
        failed_steps = [step for step in step_results if not step["success"]]

        if failed_steps:
            new_steps = self.retry_or_replan(failed_steps)
            retry_results = self.execute_plan(new_steps)
            step_results.extend(retry_results)

        self.memory.save("goal_result", step_results)
        return {
            "goal": self.goal,
            "results": step_results,
            "success": all(step["success"] for step in step_results),
        }