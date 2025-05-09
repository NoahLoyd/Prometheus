# logging.py
class Logging:
    def __init__(self, memory):
        self.memory = memory

    def log_step(self, tool_name, query, result):
        self.memory.store_short_term({"tool": tool_name, "query": query, "result": result})

    def log_goal_lifecycle(self, goal, results):
        lifecycle = {
            "goal": goal,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.store_long_term(lifecycle)

    def generate_daily_summary(self):
        # Summarize daily activities
        entries = self.memory.retrieve_short_term()
        summary = {
            "date": datetime.now().date().isoformat(),
            "entries": entries
        }
        self.memory.store_daily_summary(summary)
        return summary