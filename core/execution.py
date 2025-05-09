# execution.py
class Execution:
    def __init__(self, tool_manager):
        self.tool_manager = tool_manager

    def execute_plan(self, plan):
        results = []
        for step in plan:
            try:
                result = self.tool_manager.execute(step)
                results.append(result)
            except Exception as e:
                print(f"Retrying step due to error: {e}")
                result = self.tool_manager.retry(step)
                results.append(result)
        return results