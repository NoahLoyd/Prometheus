def __init__(self, tool_manager, short_term_memory, long_term_memory):
    self.tool_manager = tool_manager
    self.short_term_memory = short_term_memory
    self.long_term_memory = long_term_memory
    self.goal = None
    self.plan = []
    self.tags = []
    self.llm_planner = LLMPlanner()
