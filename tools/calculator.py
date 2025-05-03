from tools.base_tool import BaseTool

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__(name="calculator", description="Performs basic arithmetic calculations.")

    def run(self, query: str) -> str:
        return self.calculate(query)

    def calculate(self, query: str) -> str:
        try:
            # Very basic and safe eval environment
            allowed_names = {"__builtins__": {}}
            result = eval(query, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
