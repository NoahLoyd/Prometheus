from tools.note_tool import NoteTool
from core.agent import PrometheusAgent

if __name__ == "__main__":
    agent = PrometheusAgent()
    note_tool = NoteTool()
    agent.register_tool("notepad", note_tool)

    print(agent.think("What is 10 * 3?"))
    print(agent.act("calculator", "10 * 3"))

    print("\nMemory contents:")
    print(agent.recall())

    print("\nUsing notepad tool:")
    print(agent.act("notepad"))
    print("Saved notes:", note_tool.list_notes())
