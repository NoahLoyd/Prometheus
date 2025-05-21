# tools/prompt_decomposer.py

from typing import Dict, Any
import json

class PromptDecomposer:
    """
    Decomposes natural language prompts into actionable coding plans using a local or placeholder LLM.
    The LLM call is modular and can be swapped for a production LLM later.
    """

    def __init__(self):
        # Placeholder for LLM router or API. Replace with your real LLM connection.
        self.llm_router = self._default_llm_router

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
        Call to a local or remote LLM to generate a structured build plan.
        Returns a dict with keys: file, class, code, test.
        """
        try:
            # Placeholder logic: simulate an LLM response with JSON string.
            # Replace this block with an actual LLM call, e.g., self.llm_router.generate(...)
            simulated_llm_output = self.llm_router(prompt)
            if isinstance(simulated_llm_output, dict):
                return simulated_llm_output
            # try to parse JSON if returned as string
            return json.loads(simulated_llm_output)
        except Exception:
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

    def _default_llm_router(self, prompt: str) -> Dict[str, Any]:
        """
        Placeholder LLM router.
        Returns a hardcoded plan for demonstration; replace this with real LLM inference.
        """
        # Example: if prompt contains "timer", create a timer tool; otherwise return empty
        if "timer" in prompt.lower():
            code = (
                "from tools.base_tool import BaseTool\n"
                "import time\n\n"
                "class TimerTool(BaseTool):\n"
                "    name = 'timer'\n"
                "    description = 'Simple timer tool.'\n"
                "    def __init__(self):\n"
                "        self.start = None\n"
                "    def run(self, query: str) -> str:\n"
                "        q = query.strip().lower()\n"
                "        if q == 'start':\n"
                "            self.start = time.time()\n"
                "            return 'Timer started.'\n"
                "        elif q == 'stop':\n"
                "            if self.start is None:\n"
                "                return 'Timer not started.'\n"
                "            elapsed = time.time() - self.start\n"
                "            self.start = None\n"
                "            return f'Timer stopped: {elapsed:.2f} sec.'\n"
                "        else:\n"
                "            return 'Commands: start, stop.'\n"
            )
            test_code = (
                "from tools.timer import TimerTool\n"
                "timer = TimerTool()\n"
                "print(timer.run('start'))\n"
                "import time; time.sleep(1)\n"
                "print(timer.run('stop'))\n"
            )
            return {
                "file": "tools/timer.py",
                "class": "TimerTool",
                "code": code,
                "test": test_code
            }
        # Else, return empty to trigger fallback
        return {}
