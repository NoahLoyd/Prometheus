from tools.prompt_decomposer import PromptDecomposer
from tools.module_builder import ModuleBuilderTool
from typing import Dict, Any

class SelfCodingEngine:
    """
    SelfCodingEngine orchestrates the self-coding AGI workflow:
      - Takes in a natural language prompt.
      - Decomposes it into structured module plans.
      - Invokes ModuleBuilderTool to generate code files.
    """

    def __init__(self):
        self.decomposer = PromptDecomposer()
        self.builder = ModuleBuilderTool()

    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Processes a prompt, generates modules, and writes them to disk.
        Returns the structured plan for inspection.
        """
        plan = self.decomposer.decompose(prompt)
        self.builder.write_module(plan)
        return plan
