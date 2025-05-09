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

    # Register tools with ToolManager
    tool_manager.register_tool("calculator", CalculatorTool())
    tool_manager.register_tool("internet_tool", InternetTool())
    tool_manager.register_tool("note_tool", NoteTool())
    tool_manager.register_tool("file_tool", FileTool())
    tool_manager.register_tool("summarizer_tool", SummarizerTool())

    # Initialize StrategicBrain
    brain = StrategicBrain(tool_manager, memory)

    # Set a test goal
    test_goal = "Make $1000 this month"
    brain.memory.save("current_goal", test_goal)

    # Execute the goal plan
    result = brain.achieve_goal(test_goal)

    # Print results with clear formatting
    print("\nGoal Execution Results:")
    print(f"Goal: {result['goal']}")
    for step_result in result["results"]:
        tool_name = step_result["tool_name"]
        query = step_result["query"]
        if step_result["success"]:
            print(f"[SUCCESS] Tool: {tool_name}, Query: {query}, Result: {step_result['result']}")
        else:
            print(f"[FAILURE] Tool: {tool_name}, Query: {query}, Error: {step_result['error']}")

if __name__ == "__main__":
    main()