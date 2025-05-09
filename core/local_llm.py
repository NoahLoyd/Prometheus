# local_llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

class LocalLLM:
    def __init__(self, model_path="mistral-7b", device="cpu", quantized=False):
        """
        Initialize the local Hugging Face-based LLM interface.

        Parameters:
        - model_path: Path to the local model directory or model name (e.g., "mistral-7b").
        - device: Device to run the model on ("cpu" or "cuda").
        - quantized: If True, load quantized models for lower memory usage.
        """
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if quantized:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, torch_dtype="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True
            )
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if device == "cuda" else -1)

    def generate_plan(self, goal, context=None):
        """
        Generate a structured plan from the given goal and context.

        Parameters:
        - goal: The high-level goal to plan for.
        - context: Optional additional context (e.g., memory, tags, or past goals).

        Returns:
        - List of (tool_name, query) tuples representing the plan steps.
        """
        prompt = self._format_prompt(goal, context)
        response = self.pipeline(
            prompt,
            max_length=256,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )[0]["generated_text"]
        return self._parse_plan(response)

    def _format_prompt(self, goal, context=None):
        """
        Format the input prompt for the LLM.

        Parameters:
        - goal: The high-level goal.
        - context: Optional context.

        Returns:
        - A formatted string prompt.
        """
        context_string = f"Context: {context}" if context else "Context: None"
        prompt = (f"Goal: {goal}\n"
                  f"{context_string}\n"
                  "Generate a structured plan as a list of (tool_name, query) steps:")
        return prompt

    def _parse_plan(self, response):
        """
        Parse the LLM response into structured (tool_name, query) steps.

        Parameters:
        - response: The raw text response from the LLM.

        Returns:
        - A list of (tool_name, query) tuples.
        """
        steps = []
        for line in response.split("\n"):
            if "(" in line and "," in line:
                try:
                    tool_name, query = line.strip("()").split(", ")
                    steps.append((tool_name.strip(), query.strip()))
                except ValueError:
                    continue
        return steps