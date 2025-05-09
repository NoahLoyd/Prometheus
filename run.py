# run.py

from core.brain import StrategicBrain
from tools.tool_manager import ToolManager
from memory.short_term import ShortTermMemory
from tools.calculator import CalculatorTool
from tools.internet_tool import InternetTool
from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.summarizer_tool import SummarizerTool

def main():
    # Initialize ToolManager and ShortTermMemory
    tool_manager = ToolManager()
    memory = ShortTermMemory()

    # Register tools with ToolManager using correct names
    tool_manager.register_tool("calculator", CalculatorTool())
    tool_manager.register_tool("internet", InternetTool())
    tool_manager.register_tool("note", NoteTool())
    tool_manager.register_tool("file", FileTool())
    tool_manager.register_tool("summarizer", SummarizerTool())

    # Initialize StrategicBrain
    brain = StrategicBrain(tool_manager, memory)

    # Set and execute a test goal
    goal = "Make $1000 this month"
    brain.set_goal(goal)

    print("\n--- EXECUTING PLAN ---")
    result = brain.achieve_goal()

    print("\n--- FINAL RESULTS ---")
    print(f"Goal: {result['goal']}")
    for step in result["results"]:
        print(f"\n[{step['tool_name']}] {step['query']}")
        if step["success"]:
            print(f"→ SUCCESS: {step['result']}")
        else:
            print(f"→ FAILURE: {step['error']}")

if __name__ == "__main__":
    main()