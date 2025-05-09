# execution.py
from concurrent.futures import ThreadPoolExecutor

class Execution:
    def __init__(self, tool_manager, memory):
        self.tool_manager = tool_manager
        self.memory = memory

    def execute_plan(self, plan, batch_mode=False):
        results = []
        if batch_mode:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._execute_step, step) for step in plan]
                results = [future.result() for future in futures]
        else:
            for step in plan:
                results.append(self._execute_step(step))
        return results

    def _execute_step(self, step):
        tool_name, query = step
        try:
            result = self.tool_manager.call_tool(tool_name, query)
            self.memory.log_step(tool_name, query, result)
            return {"success": True, "result": result}
        except Exception as e:
            return self._handle_error(tool_name, query, str(e))

    def _handle_error(self, tool_name, query, error_message):
        # Retry logic with tool substitution
        substitute_tool = self.memory.get_alternative_tool(tool_name)
        if substitute_tool:
            result = self.tool_manager.call_tool(substitute_tool, query)
            return {"success": True, "result": result, "substituted_tool": substitute_tool}
        else:
            return {"success": False, "error": error_message}