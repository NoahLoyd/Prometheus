# run.py

from tools.note_tool import NoteTool
from tools.file_tool import FileTool
from tools.internet_tool import fetch_summary
from tools.calculator import calculate
from core.agent import PrometheusAgent
from core.router import CommandRouter

# Initialize agent and tools
agent = PrometheusAgent()
agent.tool_manager.register_tool("notepad", NoteTool())
agent.tool_manager.register_tool("file", FileTool())

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

print("\nCommand: Write to file saying Hello World to notes.txt")
print("Response:", router.interpret("Write to file saying Hello World to notes.txt"))

print("\nCommand: Read file notes.txt")
print("Response:", router.interpret("Read file notes.txt"))

print("\nCommand: List all files")
print("Response:", router.interpret("List all files"))
