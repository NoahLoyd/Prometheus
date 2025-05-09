import os
import openai
from tools.tool_manager import ToolManager
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory

class LLMPlanner:
    def __init__(self, model="gpt-4", tool_manager=None, long_term_memory=None, short_term_memory=None):
        self.model = model
        self.tool_manager = tool_manager or ToolManager()
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.short_term_memory = short_term_memory or ShortTermMemory()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def plan(self, goal):
        available_tools = self.tool_manager.get_available_tools()
        memory_tags = self.long_term_memory.get_memory().get("tags", [])
        prompt = self.build_prompt(goal, available_tools, memory_tags)

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a world-class AI strategist. Your task is to generate actionable plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            output = response['choices'][0]['message']['content']
            return self.parse_steps(output)
        except Exception as e:
            return [("note", f"save: Planning failed for goal '{goal}' with error: {str(e)}")]

    def build_prompt(self, goal, tools, memory_tags):
        tool_list = ", ".join(tools) if tools else "calculator, internet, note, file, summarizer"
        tag_info = f"Relevant memory tags: {', '.join(memory_tags)}.\n" if memory_tags else ""
        return (
            f"Goal: {goal}\n\n"
            f"Available tools: {tool_list}\n\n"
            f"{tag_info}"
            f"Break this goal into 3-6 tool-based steps. "
            f"Format each step as (tool_name, query). Ensure all steps are clear, actionable, and executable.\n\n"
            f"Example:\n"
            f"[(\"internet\", \"Search: how to build a YouTube channel\"),\n"
            f" (\"summarizer\", \"Summarize top strategies\"),\n"
            f" (\"note\", \"save: Growth strategy\")]\n\n"
            f"Your answer:"
        )

    def parse_steps(self, output_text):
        try:
            steps = eval(output_text.strip())
            if isinstance(steps, list) and all(isinstance(s, tuple) and len(s) == 2 for s in steps):
                return steps
        except Exception:
            pass
        return [("note", "save: Failed to parse LLM planner output. Fallback to static strategy.")]

    def get_available_tools(self):
        """
        Fetches the list of available tools from the ToolManager.
        """
        return list(self.tool_manager.tools.keys())

    def log_memory_usage(self, tool_name):
        """
        Logs tool usage in the long-term memory.
        """
        self.long_term_memory.log_tool_usage(tool_name)

    def tailor_plan_with_memory(self, steps):
        """
        Adjusts the plan based on past memory tags or tool usage.
        """
        adjusted_steps = []
        for step in steps:
            tool_name, query = step
            if tool_name in self.long_term_memory.get_memory().get("tool_usage", {}):
                query = f"Prioritize this: {query}"  # Example adjustment
            adjusted_steps.append((tool_name, query))
        return adjusted_steps