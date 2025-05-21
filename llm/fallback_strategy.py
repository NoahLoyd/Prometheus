from typing import Optional, List, Tuple, Any

class FallbackStrategy:
    """
    Defines how to retry or adjust if no models succeed.
    """
    def __init__(self, logger: Any = None):
        self.logger = logger

    def refine_plan(self, goal: str, context: Optional[str], task_type: Optional[str]) -> List[Tuple[str, str]]:
        """
        Retry or adjust the plan if all models fail.

        :param goal: The original goal or task.
        :param context: Additional context for the task.
        :param task_type: The type of task (e.g., 'reasoning', 'coding').
        :return: A refined plan as a list of (tool_name, query) steps.
        """
        adjusted_goal = f"Refined goal: {goal}"
        if self.logger:
            self.logger.warning(f"FallbackStrategy: All models failed for goal: {goal}")
        return [("fallback_tool", adjusted_goal)]

class ChainOfThoughtFallbackStrategy(FallbackStrategy):
    """
    Implements a chain-of-thought fallback strategy for AGI systems.
    Reflects on the original task and previous failures, decomposes the task,
    reconstructs a solution path, and proposes a smarter retry plan.
    """
    def fallback(self, task: str, attempts: List[str]) -> str:
        failure_analysis = self._analyze_failures(attempts)
        subgoals = self._break_down_task(task)
        solution_path = self._reconstruct_solution(subgoals, failure_analysis)
        improved_action = self._suggest_improved_action(task, solution_path, failure_analysis)

        plan = []
        plan.append(f"Step 1: Analyze failure reason\n{failure_analysis}")
        plan.append(f"Step 2: Break down task into subgoals\n{subgoals}")
        plan.append(f"Step 3: Reconstruct solution path\n{solution_path}")
        plan.append(f"Step 4: Attempt improved action\n{improved_action}")

        if self.logger:
            self.logger.info("ChainOfThoughtFallbackStrategy generated a fallback plan.")
        return "\n\n".join(plan)

    def _analyze_failures(self, attempts: List[str]) -> str:
        if not attempts:
            return "No previous attempts detected. Proceeding with initial breakdown."
        summary = "After reviewing previous failures:\n"
        for idx, attempt in enumerate(attempts, 1):
            summary += f"  {idx}. {attempt}\n"
        summary += "Key insight: Identify consistent failure points and knowledge gaps."
        return summary

    def _break_down_task(self, task: str) -> str:
        subgoals = [
            "Clarify the objective and define precise success criteria.",
            "Enumerate all resources, dependencies, or preconditions required.",
            "Segment the task into atomic, sequential sub-tasks.",
            "Anticipate potential bottlenecks or failure points.",
            "Establish validation or monitoring at each key step."
        ]
        return "\n".join(f"- {sg}" for sg in subgoals)

    def _reconstruct_solution(self, subgoals: str, failure_analysis: str) -> str:
        return (
            "Integrate insights from failure analysis with the subgoals.\n"
            "For each subgoal:\n"
            "  - Apply lessons learned from past attempts.\n"
            "  - Adjust resource allocation, sequencing, or methods as needed.\n"
            "  - Validate progress at each step before proceeding.\n"
            "  - Document intermediate outcomes and anomalies for rapid feedback."
        )

    def _suggest_improved_action(self, task: str, solution_path: str, failure_analysis: str) -> str:
        return (
            "Initiate a new attempt with the following protocol:\n"
            "- Implement the revised, stepwise plan using insights from failure analysis.\n"
            "- Prioritize execution on subgoals that were most problematic previously.\n"
            "- Continuously monitor for new points of failure or deviation.\n"
            "- Adapt dynamically and document all results for compounding system intelligence."
        )
