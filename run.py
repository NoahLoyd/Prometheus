from core.agent import PrometheusAgent
from tools.calculator import calculator_tool
from tools.notepad import note_tool
from tools.file_tool import file_tool
from tools.summarizer import summarizer_tool
from tools.memory_tool import memory_tool
from tools.internet_tool import internet_tool
from router import CommandRouter

# Initialize the agent
agent = PrometheusAgent()

# Register tools
agent.tool_manager.register_tool("calculator", calculator_tool)
agent.tool_manager.register_tool("notepad", note_tool)
agent.tool_manager.register_tool("file", file_tool)
agent.tool_manager.register_tool("summarizer", summarizer_tool)
agent.tool_manager.register_tool("memory", memory_tool)
agent.tool_manager.register_tool("internet", internet_tool)

# Initialize router
router = CommandRouter(agent)

# === TEST PROMETHEUS TOOLS ===
print(router.route("Calculate 2 + 2"))
print(router.route("Write to file saying Hello World to test file tool"))
print(router.route("Read from file output.txt"))
print(router.route("Remember: Prometheus is starting fresh"))
print(router.route("Recall memory"))
print(router.route("Summarize: This is a long paragraph meant to test summarization. It should return a concise version."))
print(router.route("Search internet for OpenAI news"))
