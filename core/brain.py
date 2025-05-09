from typing import Dict
from core.goal_planning import GoalPlanning
from core.logging import Logging


class StrategicBrain:
    """
    StrategicBrain orchestrates long-term goal planning and execution.
    """

    def __init__(self, memory: Logging, llm_config: Dict[str, Dict]) -> None:
        """
        Initialize StrategicBrain with memory and LLM configuration.

        :param memory: A Logging instance for memory/context access
        :param llm_config: Configuration dictionary for LLM models
        """
        self.memory = memory
        self.planner = GoalPlanning(memory, llm_config)

    def think(self, goal: str) -> None:
        """
        Use the planner to set a goal and generate a plan.

        :param goal: The goal to achieve
        """
        self.planner.set_goal(goal)