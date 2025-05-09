# brain.py
from goal_planning import GoalPlanning
from execution import Execution
from evaluation import Evaluation
from logging import Logging

class StrategicBrain:
    """
    The core orchestrator of Promethyn, responsible for goal planning, execution, evaluation, and logging.
    """
    def __init__(self, tool_manager, memory, llm):
        """
        Initialize StrategicBrain with injected dependencies.

        Parameters:
        - tool_manager: Manages tools for execution.
        - memory: Handles short- and long-term memory.
        - llm: A LocalLLM instance for generating plans.
        """
        self.goal_planning = GoalPlanning(llm, memory)
        self.execution = Execution(tool_manager, memory)
        self.evaluation = Evaluation(memory)
        self.logging = Logging(memory)