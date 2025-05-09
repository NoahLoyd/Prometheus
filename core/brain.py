# brain.py
from goal_planning import GoalPlanning
from execution import Execution
from evaluation import Evaluation
from logging import Logging

class StrategicBrain:
    def __init__(self, llm, tag_extractor, tool_manager, memory):
        self.goal_planning = GoalPlanning(llm, tag_extractor)
        self.execution = Execution(tool_manager)
        self.evaluation = Evaluation()
        self.logging = Logging(memory)

    def set_goal(self, goal):
        self.goal, self.tags, self.plan = self.goal_planning.set_goal(goal)
        return self.goal, self.tags, self.plan

    def achieve_goal(self):
        results = self.execution.execute_plan(self.plan)
        self.logging.archive_goal(self.goal, results)
        return results

    def generate_goal_summary(self):
        results = self.logging.memory.retrieve_short_term()
        return self.evaluation.generate_summary(results)

    def summarize_top_tools(self):
        results = self.logging.memory.retrieve_short_term()
        ranked_steps = self.evaluation.rank_steps(results)
        return [step['tool'] for step in ranked_steps[:3]]
       