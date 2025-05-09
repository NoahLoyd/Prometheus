# goal_planning.py
class GoalPlanning:
    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory

    def set_goal(self, goal):
        self.goal = goal
        self.tags = self.extract_tags(goal)
        self.plan = self.generate_plan(goal)
        return self.goal, self.tags, self.plan

    def extract_tags(self, goal):
        # Extract tags from goal and past memory
        tags = set(self.memory.search_tags(goal))
        tags.update(self.llm.extract_tags(goal))
        return list(tags)

    def generate_plan(self, goal):
        # Context-aware planning with memory integration
        history = self.memory.retrieve_related_history(self.tags)
        plan = self.llm.generate_plan(goal, context=history)
        return plan