from typing import Optional, List, Tuple


class FallbackStrategy:
    """
    Defines how to retry or adjust if no models succeed.
    """

    def refine_plan(self, goal: str, context: Optional[str], task_type: Optional[str]) -> List[Tuple[str, str]]:
        """
        Retry or adjust the plan if all models fail.

        :param goal: The original goal or task.
        :param context: Additional context for the task.
        :param task_type: The type of task (e.g., 'reasoning', 'coding').
        :return: A refined plan as a list of (tool_name, query) steps.
        """
        # Example fallback logic: adjust task constraints
        adjusted_goal = f"Refined goal: {goal}"
        return [("fallback_tool", adjusted_goal)]