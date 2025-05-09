# tools/calculator.py
from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(name="calculator", description="Perform basic mathematical calculations.")

    def run(self, query: str) -> str:
        try:
            allowed_names = {"__builtins__": {}}  # Prevent unsafe code execution
            result = eval(query, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
