from core.agent import PrometheusAgent
from core.router import CommandRouter
from tools.note_tool import NoteTool

if __name__ == "__main__":
    agent = PrometheusAgent()
    note_tool = NoteTool()
    agent.register_tool("notepad", note_tool)

    router = CommandRouter(agent)

    # Natural language commands to test
    commands = [
        "Calculate 100 - 44",
        "Summarize https://example.com",
        "Write down a note",
        "Remember that Prometheus is my AI",
        "What do you remember?",
        "What can you do?",
        "What should I do to research GPUs?",
        "Forget everything",
        "What do you remember?",
        "Analyze the best way to learn Python"
    ]

    for cmd in commands:
        print(f"\nCommand: {cmd}")
        print("Response:", router.interpret(cmd))
