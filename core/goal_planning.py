from typing import Optional, Dict
from llm.llm_factory import build_llm_router
from core.logging import Logging


class GoalPlanning:
    """
    Handles goal planning using an LLMRouter for strategy generation.
    """

    def __init__(self, memory: Logging, llm_config: Dict[str, Dict]) -> None:
        """
        Initialize GoalPlanning with memory and LLM configuration.

        :param memory: A Logging instance for memory/context access
        :param llm_config: Configuration dictionary for LLM models
        """
        self.memory = memory
        self.router = build_llm_router(llm_config)

    def set_goal(self, goal: str) -> None:
        """
        Set a goal and generate a plan to achieve it using the LLMRouter.

        :param goal: The goal to achieve
        """
        # Extract tags and context from memory
        tags = self.memory.retrieve_tags(goal)
        context = self.memory.retrieve_context(goal)

        # Generate a plan using the LLMRouter
        plan = self.router.generate_plan(goal, context=context, task_type=tags)
        print(f"Generated Plan: {plan}")