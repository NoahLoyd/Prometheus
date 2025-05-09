# promethyn_shell.py
from local_llm import LocalLLM

class PromethynShell:
    def __init__(self):
        # Initialize StrategicBrain with LocalLLM and dependencies
        memory = Memory()
        tool_manager = ToolManager()
        llm = LocalLLM(model_path="mistral-7b", device="cpu", quantized=True)
        self.brain = StrategicBrain(tool_manager, memory)
        self.todo_queue = []
     