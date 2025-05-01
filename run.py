from core.agent import PrometheusAgent
from core.router import CommandRouter
from tools.note_tool import NoteTool

if __name__ == "__main__":
    agent = PrometheusAgent()
    note_tool = NoteTool()
    agent.register_tool("notepad", note_tool)

    router = CommandRouter(agent)

    # Sample natural commands
    commands = [
        "Calculate 25 * 4",
        "Summarize https://example.com",
        "Write down a note"
    ]

    for cmd in commands:
        print(f"\nCommand: {cmd}")
        print("Response:", router.interpret(cmd))

    print("\nMemory contents:")
    print(agent.recall())
