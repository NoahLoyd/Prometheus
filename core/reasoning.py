# core/reasoning.py

class ReasoningEngine:
    def __init__(self, agent):
        self.agent = agent

    def analyze_goal(self, goal):
        goal = goal.lower().strip()

        if "research" in goal:
            return "You should use the 'internet' tool to gather information."

        elif "calculate" in goal or "math" in goal:
            return "You should use the 'calculator' tool to compute something."

        elif "remember" in goal:
            return "Store it in memory using the 'remember' command."

        else:
            return "No direct strategy found. Try asking or clarifying the goal."
