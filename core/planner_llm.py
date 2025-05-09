# core/planner_llm.py

import os
import openai  # or replace with Claude if needed

class LLMPlanner:
    def __init__(self, model="gpt-4"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def plan(self, goal, available_tools=None):
        prompt = self.build_prompt(goal, available_tools)
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a smart, structured AI planner that outputs a clean list of (tool_name, query) steps to achieve the user's goal."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            output = response['choices'][0]['message']['content']
            return self.parse_steps(output)
        except Exception as e:
            return [("note", f"save: Planning failed for goal '{goal}' with error: {str(e)}")]

    def build_prompt(self, goal, tools):
        tool_list = ", ".join(tools) if tools else "calculator, internet, note, file, summarizer"
        return (
            f"Goal: {goal}\n\n"
            f"Available tools: {tool_list}\n\n"
            f"Break this goal into 3–6 tool-based steps. "
            f"Format: (tool_name, query). Keep it clear, actionable, and compatible with an execution engine.\n\n"
            f"Example:\n"
            f"[('internet', 'Search: how to build a YouTube channel'),\n"
            f" ('summarizer', 'Summarize top strategies'),\n"
            f" ('note', 'save: Growth strategy')]\n\n"
            f"Your answer:"
        )

    def parse_steps(self, output_text):
        try:
            # Use eval safely — ensure the format matches expected list of tuples
            steps = eval(output_text.strip())
            if isinstance(steps, list) and all(isinstance(s, tuple) and len(s) == 2 for s in steps):
                return steps
        except Exception:
            pass
        return [("note", "save: Failed to parse LLM planner output. Fallback to static strategy.")]