from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):
    def run(self, query: str) -> str:
        try:
            result = eval(query, {"__builtins__": {}})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
