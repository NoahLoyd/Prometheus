# brain.py
from goal_planning import GoalPlanning
from execution import Execution
from evaluation import Evaluation
from logging import Logging
from local_llm import LocalLLM

class StrategicBrain:
    def __init__(self, tool_manager, memory):
        # Replace OpenAILLM with LocalLLM
        llm = LocalLLM(model_path="mistral-7b", device="cpu", quantized=True)
        self.goal_planning = GoalPlanning(llm, memory)
        self.execution = Execution(tool_manager, memory)
        self.evaluation = Evaluation(memory)
        self.logging = Logging(memory)