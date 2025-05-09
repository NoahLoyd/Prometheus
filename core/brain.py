# brain.py
from goal_planning import GoalPlanning
from execution import Execution
from evaluation import Evaluation
from logging import Logging

class StrategicBrain:
    def __init__(self, llm, tool_manager, memory):
        self.goal_planning = GoalPlanning(llm, memory)
        self.execution = Execution(tool_manager, memory)
        self.evaluation = Evaluation(memory)
        self.logging = Logging(memory)

    def set_goal(self, goal):
        self.goal, self.tags, self.plan = self.goal_planning.set_goal(goal)
        return self.goal, self.tags, self.plan

    def achieve_goal(self, batch_mode=False):
        if not self.goal or not self.plan:
            raise ValueError("Goal and plan must be set before execution.")
        results = self.execution.execute_plan(self.plan, batch_mode=batch_mode)
        self.logging.log_goal_lifecycle(self.goal, results)
        
        # Reflection and Comparison
        reflection = self.evaluation.reflect_on_performance(results)
        self.logging.log_reflection(reflection)
        comparison = self.evaluation.compare_with_past_goals(self.goal, results)
        insights = self.evaluation.generate_summary(results)

        # Return detailed report
        return {
            "results": results,
            "reflection": reflection,
            "comparison": comparison,
            "insights": insights,
        }

    def generate_goal_summary(self):
        results = self.logging.memory.retrieve_short_term()
        return self.evaluation.generate_summary(results)

    def summarize_top_tools(self):
        tool_usage = self.logging.memory.get_tool_usage()
        return sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:3]

    def generate_daily_review(self):
        return self.logging.generate_daily_review()