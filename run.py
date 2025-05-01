from core.agent import PrometheusAgent
from core.router import CommandRouter
from tools.note_tool import NoteTool

if __name__ == "__main__":
    agent = PrometheusAgent()
    note_tool = NoteTool()
    agent.register_tool("notepad", note_tool)

    router = CommandRouter(agent)

    # Test a range of natural language commands
    commands = [
        "Calculate 64 / 8",
        "Summarize https://example.com",
        "Write down a note",
        "Remember that my GPU is a 3090",
        "What do you remember?",
        "Clear memory",
        "What do you remember?",
        "Do something unexpected",
        "what can you do?"
    ]

    for cmd in commands:
        print(f"\nCommand: {cmd}")
        print("Response:", router.interpret(cmd))
