from tools.note_tool import NoteTool
from tools.calculator import calculate
from tools.internet_tool import fetch_summary
from core.agent import PrometheusAgent
from core.router import Router

if __name__ == "__main__":
    agent = PrometheusAgent()
    router = Router(agent)

    # Tool instances
    note_tool = NoteTool()

    # Register tools (functions, not classes)
    agent.register_tool("calculator", calculate)
    agent.register_tool("internet", fetch_summary)
    agent.register_tool("notepad", note_tool.add_note)
    agent.register_tool("list_notes", note_tool.list_notes)

    # Test cases
    print("Command: Calculate 100 - 44")
    print("Response:", router.interpret("Calculate 100 - 44"))

    print("\nCommand: Summarize https://example.com")
    print("Response:", router.interpret("Summarize https://example.com"))

    print("\nCommand: Write down a note")
    print("Response:", router.interpret("Write down a note"))

    print("\nCommand: List notes")
    print("Response:", router.interpret("List notes"))
