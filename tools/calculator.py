# tools/calculator.py

from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "A tool that performs basic arithmetic calculations."

    def run(self, query: str) -> str:
        try:
            # WARNING: eval is dangerous; only use with trusted input.
            result = eval(query, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Calculator error: {e}"
