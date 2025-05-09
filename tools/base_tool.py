# tools/base_tool.py

class BaseTool:
    def __init__(self, name="base_tool", description="A base tool with no functionality"):
        self.name = name
        self.description = description

    def run(self, query: str) -> str:
        raise NotImplementedError("Each tool must implement a run method.") 
