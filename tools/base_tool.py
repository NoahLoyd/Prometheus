# tools/base_tool.py

class BaseTool:
    def run(self, command: str) -> str:
        raise NotImplementedError("Tool must implement the run method.")
