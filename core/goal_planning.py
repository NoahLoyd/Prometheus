# goal_planning.py
class GoalPlanning:
    def __init__(self, llm, tag_extractor):
        self.llm = llm
        self.tag_extractor = tag_extractor

    def set_goal(self, goal):
        self.goal = goal
        self.tags = self.tag_extractor.extract_tags(goal)
        self.plan = self.llm.generate_plan(goal)
        return self.goal, self.tags, self.plan