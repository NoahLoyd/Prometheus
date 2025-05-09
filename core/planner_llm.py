import os
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tools.tool_manager import ToolManager
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory

class LLMPlanner:
    def __init__(self, model="gpt-4", backend="openai", model_path=None, tool_manager=None, long_term_memory=None, short_term_memory=None):
        self.model = model
        self.backend = backend
        self.model_path = model_path
        self.tool_manager = tool_manager or ToolManager()
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.short_term_memory = short_term_memory or ShortTermMemory()
        
        if backend == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif backend == "local" and model_path:
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.local_model = AutoModelForCausalLM.from_pretrained(model_path)
            self.local_pipeline = pipeline("text-generation", model=self.local_model, tokenizer=self.local_tokenizer)

    def plan(self, goal):
        available_tools = self.tool_manager.get_available_tools()
        memory_tags = self.long_term_memory.get_memory().get("tags", [])
        prompt = self.build_prompt(goal, available_tools, memory_tags)

        if self.backend == "openai":
            try:
                return self._plan_with_openai(prompt)
            except Exception as e:
                self.log_error("openai", str(e))

        if self.backend == "local":
            try:
                return self._plan_with_local(prompt)
            except Exception as e:
                self.log_error("local", str(e))
        
        # Fallback to OpenAI if local fails
        if self.backend == "local":
            try:
                return self._plan_with_openai(prompt)
            except Exception as e:
                self.log_error("fallback_openai", str(e))

        return [("note", f"save: Planning failed for goal '{goal}' with all backends.")]

    def _plan_with_openai(self, prompt):
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

    def _plan_with_local(self, prompt):
        response = self.local_pipeline(prompt, max_length=512, num_return_sequences=1)
        output = response[0]['generated_text']
        return self.parse_steps(output)

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

    def log_error(self, backend, error_message):
        self.short_term_memory.save(f"{backend}_error", error_message)
        self.long_term_memory.log_event(f"{backend}_planning_failure", error_message)