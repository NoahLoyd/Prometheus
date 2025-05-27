import json
from typing import Dict, Any
from llm.llm_router import LLMRouter

# Promethyn elite standards context for tool/code generation
STANDARD_CONTEXT = (
    "You are Promethyn's code generator. Enforce these elite standards at all times:\n"
    "- All tools must follow the BaseTool interface and Promethyn modular architecture.\n"
    "- Every tool must include production-grade docstrings, robust and safe fallback logic, and corresponding tests.\n"
    "- Never overwrite existing files unless explicitly allowed by the user.\n"
    "- After generation, validate tools using tool.run('test').\n"
    "- Log any validation failures to AddOnNotebook if available.\n"
    "- Always implement overwrite protection using os.path.exists().\n"
    "- Write clean, modular, future-proof Python 3.11+ code.\n"
    "- Never remove or break existing logic—only enhance and extend safely.\n"
    "- The goal is to make Promethyn code better than any human or team. Do not settle.\n"
)

class PromptDecomposer:
    """
    Decomposes natural language prompts into actionable coding plans using a local or placeholder LLM.
    The LLM call is modular and can be swapped for a production LLM later.
    """

    def __init__(self):
        # Initialize the LLMRouter for structured code generation
        self.llm = LLMRouter()

    def decompose(self, prompt: str) -> Dict[str, Any]:
        """
        Given a prompt, returns a structured plan dictionary.
        Validates LLM output and falls back to a safe default plan as needed.
        """
        plan = self._call_llm(prompt)
        validated_plan = self._validate_and_fallback(plan, prompt)
        return validated_plan

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call to LLMRouter to generate a structured build plan.
        Returns a dict with keys: file, class, code, test.
        Prepends STANDARD_CONTEXT to every prompt to enforce elite standards.
        """
        try:
            full_prompt = f"{STANDARD_CONTEXT}\n{prompt}"
            response = self.llm.generate(full_prompt, task_type="code")
            plan = json.loads(response)
            if not isinstance(plan, dict):
                print("[PromptDecomposer] LLMRouter returned non-dict JSON. Falling back.")
                return {}
            return plan
        except Exception as e:
            print(f"[PromptDecomposer] LLMRouter or JSON parsing failed: {e}")
            return {}

    def _validate_and_fallback(self, plan: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """
        Ensures the plan contains required keys and valid values.
        Falls back to a simple safe plan if LLM output is missing or invalid.
        """
        required_keys = {"file", "class", "code", "test"}
        if not (isinstance(plan, dict) and required_keys.issubset(plan) and all(plan.get(k) for k in required_keys)):
            # Fallback: very basic plan
            return {
                "file": "tools/undefined_module.py",
                "class": "UndefinedModule",
                "code": (
                    "# Auto-generated placeholder due to LLM error or incomplete output.\n"
                    "class UndefinedModule:\n"
                    "    def run(self, query: str) -> str:\n"
                    "        return 'No implementation. Prompt was: %s'\n" % prompt
                ),
                "test": (
                    "from tools.undefined_module import UndefinedModule\n"
                    "mod = UndefinedModule()\n"
                    "print(mod.run('test'))\n"
                )
            }
        return plan
