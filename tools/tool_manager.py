# Test script to verify ToolManager
from tools.tool_manager import ToolManager

class MockTool:
    """
    A mock tool for testing purposes.
    """
    def __init__(self, name):
        self.name = name

    def run(self, query):
        return f"{self.name} executed with query: {query}"

# Initialize ToolManager
tool_manager = ToolManager()

# Register mock tools
tool_manager.register_tool("mock1", MockTool("MockTool1"))
tool_manager.register_tool("mock2", MockTool("MockTool2"))

# Run the mock tools
print(tool_manager.run_tool("mock1: Test query 1"))  # Output: MockTool1 executed with query: Test query 1
print(tool_manager.run_tool("mock2: Test query 2"))  # Output: MockTool2 executed with query: Test query 2

# Test invalid tool
print(tool_manager.run_tool("mock3: Test query 3"))  # Output: Error: Tool 'mock3' not found.
