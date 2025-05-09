from .llm_router import LLMRouter

class LLMPlanner:
    def __init__(self, config):
        self.router = LLMRouter(config)

    def plan(self, goal: str) -> List[Tuple[str, str]]:
        return self.router.generate_plan(goal)
