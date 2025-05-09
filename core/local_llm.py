# local_llm.py
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ast import literal_eval

class LocalLLM:
    """
    A local Hugging Face-based LLM interface for fully offline operation.
    Supports generating structured plans from high-level goals and context.
    """
    def __init__(self, model_path="mistral-7b", device="cpu", quantized=False):
        """
        Initialize the LocalLLM with a model and tokenizer loaded from disk.

        Parameters:
        - model_path: Path to the local model directory or model name (e.g., "mistral-7b").
        - device: Device to run the model on ("cpu" or "cuda").
        - quantized: If True, load a quantized model for reduced memory usage.
        """
        self.model_path = model_path
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            if quantized:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, local_files_only=True, torch_dtype="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
            self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if device == "cuda" else -1)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer from {model_path}: {e}")

    def generate_plan(self, goal, context=None):
        """
        Generate a structured plan based on the goal and optional context.

        Parameters:
        - goal: The high-level goal to plan for.
        - context: Additional context (e.g., memory or related goals).

        Returns:
        - A list of (tool_name, query) tuples representing the plan steps.
        """
        prompt = self._format_prompt(goal, context)
        try:
            response = self.pipeline(
                prompt,
                max_length=256,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )[0]["generated_text"]
            return self._parse_plan(response)
        except Exception as e:
            raise RuntimeError(f"Failed to generate plan: {e}")

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
        return f"Goal: {goal}\n{context_string}\nGenerate a structured plan as a list of (tool_name, query) steps:"

    def _parse_plan(self, response):
        """
        Parse the LLM response into structured (tool_name, query) steps.

        Parameters:
        - response: The raw text response from the LLM.

        Returns:
        - A list of (tool_name, query) tuples.
        """
        steps = []
        lines = response.split("\n")
        for line in lines:
            # Use regex or literal_eval for robust parsing
            match = re.match(r"\((.*?)\)", line)
            if match:
                try:
                    step = literal_eval(match.group(0))
                    if isinstance(step, tuple) and len(step) == 2:
                        steps.append(step)
                except (SyntaxError, ValueError):
                    continue
        return steps