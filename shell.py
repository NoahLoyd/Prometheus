# promethyn_shell.py
import argparse
from local_llm import LocalLLM
from core.brain import StrategicBrain
from core.memory import Memory
from core.tool_manager import ToolManager

class PromethynShell:
    """
    The command-line interface for Promethyn, enabling natural language interaction and offline goal execution.
    """
    def __init__(self, batch_mode=False):
        """
        Initialize PromethynShell with optional batch mode.

        Parameters:
        - batch_mode: If True, enables batch execution of goals.
        """
        memory = Memory()
        tool_manager = ToolManager()
        llm = LocalLLM(model_path="mistral-7b", device="cpu", quantized=True)
        self.brain = StrategicBrain(tool_manager, memory, llm)
        self.batch_mode = batch_mode
        self.todo_queue = []

    def start(self):
        self.print_header()
        while True:
            user_input = self.get_user_input()
            if user_input.lower() in {"exit", "quit"}:
                self.print_goodbye()
                break
            elif user_input.lower() == "daily":
                self.daily_mode()
            else:
                self.process_goal(user_input)

    def process_goal(self, goal):
        print("\nSetting the goal...")
        self.brain.set_goal(goal)
        print("Goal set! Starting execution...\n")
        results = self.brain.achieve_goal(batch_mode=self.batch_mode)
        self.stream_execution_results(results["results"])

    def stream_execution_results(self, results):
        for step in results:
            tool = step["tool_name"]
            query = step["query"]
            success = step["success"]
            color = "green" if success else "red"
            print(f"[{tool}] {query} -> {success}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Promethyn Shell.")
    parser.add_argument("--batch", action="store_true", help="Enable batch execution mode.")
    args = parser.parse_args()
    
    shell = PromethynShell(batch_mode=args.batch)
    shell.start()
     