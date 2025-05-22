import json
import traceback
from typing import Any, Optional, Dict

class LLMRouter:
    """
    LLMRouter for Promethyn: future-ready, modular, and safe.
    Routes prompt generation via simulated or local models,
    and is easily extensible for real LLM backends.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Accepts an optional config dictionary for customization
        if config is None:
            config = {}
        self.model_name = config.get("model", "simulated-llm")  # Default model name
        self.use_hf = config.get("use_hf", False)
        self.hf_model_path = config.get("hf_model_path", None)
        self._hf_model = None
        self._hf_tokenizer = None
        if self.use_hf and self.hf_model_path is not None:
            try:
                print("[LLMRouter] Initializing Hugging Face model...")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path)
                self._hf_model = AutoModelForCausalLM.from_pretrained(self.hf_model_path)
                self._hf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self._hf_model = self._hf_model.to(self._hf_device)
                print(f"[LLMRouter] Hugging Face model '{self.hf_model_path}' loaded on {self._hf_device}.")
            except Exception as e:
                print(f"[LLMRouter] Failed to initialize Hugging Face model: {e}")
                print(traceback.format_exc())
                self.use_hf = False  # Fallback to simulated

    def generate(self, prompt: str, task_type: str = "code") -> str:
        """
        Generate code or text from prompt using the selected backend.
        Returns a string (usually JSON for code tasks).
        """
        try:
            if self.use_hf and self._hf_model is not None and self._hf_tokenizer is not None:
                print(f"[LLMRouter] Generating response with Hugging Face model '{self.hf_model_path}'.")
                response = self._generate_with_huggingface(prompt, task_type)
                print("[LLMRouter] Generation success (Hugging Face).")
                return response
            else:
                print(f"[LLMRouter] Generating response for task_type='{task_type}' using model '{self.model_name}' (simulated).")
                response = self._simulate_llm(prompt, task_type)
                print("[LLMRouter] Generation success (simulated).")
                return response
        except Exception as e:
            print(f"[LLMRouter] Generation error: {e}")
            print(traceback.format_exc())
            fallback = self._fallback_response(prompt, task_type, error=e)
            print("[LLMRouter] Fallback response returned.")
            return fallback

    def _generate_with_huggingface(self, prompt: str, task_type: str = "code") -> str:
        """
        Generate a response using a local Hugging Face model.
        Returns a JSON string containing 'file', 'class', 'code', and 'test'.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Build the input prompt for code generation
            # You may want to engineer this prompt for better results
            system_prompt = (
                "Given the following user prompt, generate a Python module plan as a JSON object with keys "
                "'file', 'class', 'code', and 'test'. "
                "Only output the JSON object. Prompt: "
            )
            full_prompt = system_prompt + prompt + "\nJSON:"

            # Tokenize and generate
            inputs = self._hf_tokenizer(full_prompt, return_tensors="pt").to(self._hf_device)
            with torch.no_grad():
                generated_ids = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    pad_token_id=self._hf_tokenizer.eos_token_id,
                    do_sample=False
                )
            output = self._hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Find the JSON substring (extract between first "{" and last "}")
            start = output.find('{')
            end = output.rfind('}')
            if start == -1 or end == -1 or end <= start:
                raise ValueError("Could not find JSON object in model output.")

            json_str = output[start:end+1]
            # Validate and pretty-print JSON
            plan = json.loads(json_str)
            # Ensure all required keys exist
            for key in ("file", "class", "code", "test"):
                if key not in plan:
                    raise ValueError(f"Missing key '{key}' in generated plan.")
            return json.dumps(plan, indent=2)

        except Exception as e:
            print(f"[LLMRouter] Hugging Face generation failed: {e}")
            print(traceback.format_exc())
            # Fallback to simulation if Hugging Face fails
            return self._simulate_llm(prompt, task_type)

    def _simulate_llm(self, prompt: str, task_type: str = "code") -> str:
        """
        Simulate LLM output for local/dev use.
        Returns a hardcoded JSON string for code, or a generic reply for text.
        """
        if task_type == "code":
            # Simulate a JSON code-generation response for tool creation
            if "time track" in prompt.lower() or "track time" in prompt.lower():
                code = (
                    "from tools.base_tool import BaseTool\n"
                    "import time\n\n"
                    "class TimeTrackerTool(BaseTool):\n"
                    "    name = 'time_tracker'\n"
                    "    description = 'Tracks time spent on tasks.'\n"
                    "    def __init__(self):\n"
                    "        self.tasks = {}\n"
                    "        self.active_task = None\n"
                    "        self.start_time = None\n"
                    "    def run(self, query: str) -> str:\n"
                    "        cmd = query.strip().lower()\n"
                    "        if cmd.startswith('start:'):\n"
                    "            task = cmd[len('start:'):].strip()\n"
                    "            if self.active_task:\n"
                    "                return f'Already tracking {self.active_task}'\n"
                    "            self.active_task = task\n"
                    "            self.start_time = time.time()\n"
                    "            return f'Started tracking {task}'\n"
                    "        elif cmd == 'stop':\n"
                    "            if not self.active_task:\n"
                    "                return 'No active task.'\n"
                    "            elapsed = time.time() - self.start_time\n"
                    "            self.tasks[self.active_task] = self.tasks.get(self.active_task, 0) + elapsed\n"
                    "            msg = f'Stopped {self.active_task}. Time: {elapsed:.2f} seconds.'\n"
                    "            self.active_task = None\n"
                    "            self.start_time = None\n"
                    "            return msg\n"
                    "        elif cmd == 'report':\n"
                    "            if not self.tasks:\n"
                    "                return 'No tasks tracked.'\n"
                    "            return '\\n'.join(f'{t}: {s:.2f} sec' for t, s in self.tasks.items())\n"
                    "        else:\n"
                    "            return 'Commands: start:<task>, stop, report.'\n"
                )
                test_code = (
                    "from tools.time_tracker import TimeTrackerTool\n"
                    "tt = TimeTrackerTool()\n"
                    "print(tt.run('start: coding'))\n"
                    "import time; time.sleep(1)\n"
                    "print(tt.run('stop'))\n"
                    "print(tt.run('report'))\n"
                )
                plan = {
                    "file": "tools/time_tracker.py",
                    "class": "TimeTrackerTool",
                    "code": code,
                    "test": test_code
                }
                return json.dumps(plan, indent=2)
            else:
                plan = {
                    "file": "tools/undefined_module.py",
                    "class": "UndefinedModule",
                    "code": (
                        "# Placeholder for undefined module\n"
                        "class UndefinedModule:\n"
                        "    def run(self, query: str) -> str:\n"
                        "        return 'No implementation for: ' + query\n"
                    ),
                    "test": "print('No test defined for undefined module.')"
                }
                return json.dumps(plan, indent=2)
        else:
            return f"[Simulated LLM] Response for task '{task_type}' to prompt: {prompt}"

    def _fallback_response(self, prompt: str, task_type: str, error: Any = None) -> str:
        """
        Return a safe fallback response (JSON for code; text for others) if generation fails.
        """
        msg = (
            "[LLMRouter Fallback] Generation failed.\n"
            f"Prompt: {prompt}\n"
            f"Task Type: {task_type}\n"
            f"Error: {str(error)}\n"
            "Please retry or check system logs."
        )
        if task_type == "code":
            plan = {
                "file": "tools/generation_failed.py",
                "class": "GenerationFailed",
                "code": (
                    "# Generation failed for your request.\n"
                    "class GenerationFailed:\n"
                    "    def run(self, query: str) -> str:\n"
                    "        return 'Tool generation failed. Please try again.'\n"
                ),
                "test": "# No test: generation failed."
            }
            return json.dumps(plan, indent=2)
        else:
            return msg
