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

    def log_reflection(self, reflection):
        # Store structured self-analysis and lessons learned
        reflection_entry = {
            "timestamp": datetime.now().isoformat(),
            "reflection": reflection
        }
        self.memory.store_long_term(reflection_entry)

    def log_tool_performance(self, tool_name, success):
        # Track win/loss stats per tool
        tool_stats = self.memory.retrieve_tool_stats(tool_name) or {"wins": 0, "losses": 0}
        if success:
            tool_stats["wins"] += 1
        else:
            tool_stats["losses"] += 1
        self.memory.store_tool_stats(tool_name, tool_stats)

    def generate_daily_review(self):
        # Generate a daily review report
        goals = self.memory.retrieve_goals_for_today()
        tools_used = self.memory.retrieve_tool_usage()
        failures = self.memory.retrieve_failures_for_today()
        insights = self.memory.retrieve_insights()

        review = {
            "date": datetime.now().date().isoformat(),
            "goals_achieved": len(goals),
            "tools_used": tools_used,
            "failures": failures,
            "meta_lessons": insights
        }
        self.memory.store_daily_summary(review)
        return review