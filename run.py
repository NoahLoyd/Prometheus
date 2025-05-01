from tools.note_tool import NoteTool
from core.agent import PrometheusAgent
from core.router import CommandRouter

# Initialize agent and tools
agent = PrometheusAgent()
note_tool = NoteTool()
agent.tool_manager.register_tool("notepad", note_tool)

# Initialize router
router = CommandRouter(agent)

# Example test commands
print("Command: Calculate 100 - 44")
print("Response:", router.interpret("Calculate 100 - 44"))

print("\nCommand: Summarize https://example.com")
print("Response:", router.interpret("Summarize https://example.com"))

print("\nCommand: Write down a note")
print("Response:", router.interpret("Write down a note"))

print("\nCommand: Remember Alpha Protocol")
print("Response:", router.interpret("Remember Alpha Protocol"))

print("\nCommand: What do you remember?")
print("Response:", router.interpret("What do you remember?"))

print("\nCommand: List tools")
print("Response:", router.interpret("List tools"))
