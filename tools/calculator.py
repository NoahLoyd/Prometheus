# tools/calculator.py
from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):
    def run(self, query: str) -> str:
        try:
            allowed_names = {"__builtins__": {}}
            result = eval(query, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
