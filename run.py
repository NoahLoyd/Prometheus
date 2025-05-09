# run.py

from core.brain import StrategicBrain
from tools.tool_manager import ToolManager
from memory.short_term import ShortTermMemory
from tools.calculator import CalculatorTool
from tools.internet_tool import InternetTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool
from datetime import datetime


def initialize_tools(tool_manager):
    """
    Initialize and register tools with the ToolManager.
    Add safeguards to detect and handle initialization failures.
    """
    tools = [
        ("calculator", CalculatorTool),
        ("internet", InternetTool),
        ("note", NoteTool),
        ("file", FileTool),
        ("summarizer", SummarizerTool),
    ]
    for name, tool_cls in tools:
        try:
            tool_manager.register_tool(name, tool_cls())
        except Exception as e:
            print(f"[ERROR] Failed to initialize tool '{name}': {str(e)}")


def log_session_start(memory):
    """
    Log the start of a session with a timestamp.
    """
    start_timestamp = datetime.now().isoformat()
    memory.save("session_start", start_timestamp)
    print(f"Session started at {start_timestamp}")


def log_session_end(memory):
    """
    Log the end of a session with a timestamp.
    """
    end_timestamp = datetime.now().isoformat()
    memory.save("session_end", end_timestamp)
    print(f"Session ended at {end_timestamp}")


def main():
    # Initialize memory and tool manager
    memory = ShortTermMemory()
    tool_manager = ToolManager()

    # Log session start
    log_session_start(memory)

    # Safely initialize tools
    initialize_tools(tool_manager)

    # Start Promethyn's brain
    brain = StrategicBrain(tool_manager, memory)

    try:
        # Set a high-level, compound test goal
        goal = "Make $1000 and grow an audience"
        brain.set_goal(goal)

        print("\n--- EXECUTING PLAN ---")
        result = brain.achieve_goal()

        print("\n--- FINAL REPORT ---")
        print(f"Goal: {result['goal']}")
        print(f"Success: {result['success']}")
        print("\n--- Step Results ---")
        for step in result["results"]:
            print(f"\n[{step['tool_name']}] {step['query']}")
            if step["success"]:
                print(f"  → SUCCESS: {step['result']}")
            else:
                print(f"  → FAILURE: {step['error']}")

        print("\n--- Evaluation Summary ---")
        evaluation = memory.load("evaluation_summary")
        print(evaluation)

    except Exception as e:
        print(f"[ERROR] Execution failed: {str(e)}")

    # Log session end
    log_session_end(memory)


if __name__ == "__main__":
    main()