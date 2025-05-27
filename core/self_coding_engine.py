import importlib
import sys
import traceback
import os
from typing import Dict, Any, Optional, List

from tools.prompt_decomposer import PromptDecomposer
from tools.module_builder import ModuleBuilderTool
from tools.base_tool import BaseTool
from tools.tool_manager import ToolManager
from addons.notebook import AddOnNotebook

class SelfCodingEngine:
    """
    SelfCodingEngine orchestrates the self-coding AGI workflow:
      - Takes in a natural language prompt (single or multiple tools).
      - Decomposes it into one or more structured module plans.
      - Generates, writes, validates, and registers each tool.
      - Logs all outcomes for strategic learning.
    """

    def __init__(self, notebook: Optional[AddOnNotebook] = None):
        self.decomposer = PromptDecomposer()
        self.builder = ModuleBuilderTool()
        self.notebook = notebook or AddOnNotebook()

    def process_prompt(
        self,
        prompt: str,
        tool_manager: Optional[ToolManager] = None,
        short_term_memory: Optional[dict] = None,
    ) -> dict:
        """
        Process a prompt that may request one or many tools:
        - Decomposes prompt into structured plans (multi-tool aware).
        - Enforces overwrite protection: skips or errors if code or test file exists.
        - Writes tool code to /tools/, test code to /test/.
        - Dynamically imports, instantiates, and validates each tool using run('test').
        - Logs all successes/failures.
        - Failed generations/validations are added to retry_later and logged.
        - Successes are saved to short_term_memory['generated_tools'] if provided.
        Returns a dict summarizing all operations.
        """
        results: List[dict] = []
        retry_later: List[dict] = []

        # 1. Decompose prompt (multi-tool support)
        try:
            plans = self.decomposer.decompose(prompt)
            if not isinstance(plans, list):
                plans = [plans]
        except Exception as e:
            tb = traceback.format_exc()
            log_msg = f"Prompt decomposition failed: {e}\n{tb}"
            if self.notebook:
                self.notebook.log("prompt_decomposition_failure", {"prompt": prompt, "error": log_msg})
            return {"success": False, "error": log_msg, "results": [], "retry_later": []}

        # 2. For each tool, run the generation/validation/registration pipeline
        for plan in plans:
            single_result = {"plan": plan, "registration": None, "validation": None}
            tool_file = plan.get("file")
            test_file = plan.get("test_file")
            class_name = plan.get("class")
            tool_code = plan.get("code")
            test_code = plan.get("test_code")

            # Compute output paths
            tool_path = os.path.join("tools", tool_file) if tool_file else None
            test_path = os.path.join("test", test_file) if test_file else None  # test/ not tests/
            try:
                # --- Overwrite protection: skip or error if exists ---
                if tool_path and os.path.exists(tool_path):
                    msg = f"Tool file exists, skipping: {tool_path}"
                    single_result["registration"] = {"success": False, "error": msg}
                    retry_later.append({"plan": plan, "reason": msg})
                    results.append(single_result)
                    continue
                if test_path and os.path.exists(test_path):
                    msg = f"Test file exists, skipping: {test_path}"
                    single_result["registration"] = {"success": False, "error": msg}
                    retry_later.append({"plan": plan, "reason": msg})
                    results.append(single_result)
                    continue

                # --- Write the main tool code file ---
                if tool_path and tool_code:
                    os.makedirs(os.path.dirname(tool_path), exist_ok=True)
                    with open(tool_path, "w", encoding="utf-8") as f:
                        f.write(tool_code)
                else:
                    raise ValueError("Missing tool_path or tool_code in plan.")

                # --- Write the test file ---
                if test_path and test_code:
                    os.makedirs(os.path.dirname(test_path), exist_ok=True)
                    with open(test_path, "w", encoding="utf-8") as f:
                        f.write(test_code)
                else:
                    raise ValueError("Missing test_path or test_code in plan.")

                # --- Dynamic import of tool module and class ---
                module_path = tool_path[:-3].replace("/", ".").replace("\\", ".") if tool_path.endswith(".py") else tool_path.replace("/", ".").replace("\\", ".")
                if module_path in sys.modules:
                    del sys.modules[module_path]
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name, None)
                if tool_class is None:
                    raise ImportError(f"Class '{class_name}' not found in '{module_path}'.")

                # --- Instantiate tool and validate by running test ---
                tool_instance = tool_class()
                if not hasattr(tool_instance, "run"):
                    raise AttributeError(f"Tool '{class_name}' does not implement a 'run' method.")

                # Validate using run("test")
                try:
                    validation_result = tool_instance.run("test")
                    single_result["validation"] = validation_result
                    if not (isinstance(validation_result, dict) and validation_result.get("success", False)):
                        fail_msg = f"Tool validation failed: {validation_result}"
                        single_result["registration"] = {"success": False, "error": fail_msg}
                        retry_later.append({"plan": plan, "reason": fail_msg})
                        if self.notebook:
                            self.notebook.log("tool_validation_failure", {
                                "plan": plan,
                                "validation": validation_result,
                                "error": fail_msg,
                            })
                        results.append(single_result)
                        continue

                except Exception as val_err:
                    tb = traceback.format_exc()
                    fail_msg = f"Tool validation raised exception: {val_err}\n{tb}"
                    single_result["registration"] = {"success": False, "error": fail_msg}
                    retry_later.append({"plan": plan, "reason": fail_msg})
                    if self.notebook:
                        self.notebook.log("tool_validation_failure", {
                            "plan": plan,
                            "error": fail_msg,
                        })
                    results.append(single_result)
                    continue

                # --- Register tool if requested ---
                reg_result = self._register_tool(plan, tool_manager)
                single_result["registration"] = reg_result

                # --- Log success to ShortTermMemory if available ---
                if short_term_memory is not None and reg_result.get("success", False):
                    if "generated_tools" not in short_term_memory:
                        short_term_memory["generated_tools"] = []
                    short_term_memory["generated_tools"].append({
                        "plan": plan,
                        "tool_file": tool_file,
                        "test_file": test_file,
                        "class": class_name,
                    })

            except Exception as e:
                tb = traceback.format_exc()
                fail_msg = f"Module build or validation error: {e}\n{tb}"
                single_result["registration"] = {"success": False, "error": fail_msg}
                retry_later.append({"plan": plan, "reason": fail_msg})
                if self.notebook:
                    self.notebook.log("module_build_or_validation_error", {
                        "plan": plan,
                        "error": str(e),
                        "traceback": tb,
                    })
            results.append(single_result)

        # --- Log retry_later to memory or AddOnNotebook ---
        if retry_later:
            retry_log = {"retry_later": retry_later, "prompt": prompt}
            if short_term_memory is not None:
                short_term_memory.setdefault("tool_retry_queue", []).extend(retry_later)
            if self.notebook:
                self.notebook.log("tool_retry_later", retry_log)

        return {
            "success": len(retry_later) == 0,
            "results": results,
            "retry_later": retry_later,
        }

    def _register_tool(
        self,
        plan: Dict[str, Any],
        tool_manager: Optional[ToolManager]
    ) -> Dict[str, Any]:
        """
        Dynamically imports, instantiates, and registers a tool class from a generated module.
        Returns a dict with success status and error message if any.
        Logs registration failures to AddOnNotebook if available.
        """
        try:
            file_path = plan.get("file")
            class_name = plan.get("class")
            if not file_path or not class_name:
                msg = "Missing 'file' or 'class' in plan."
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg})
                return {"success": False, "error": msg}

            # Convert file path to module path (e.g. tools/time_tracker.py -> tools.time_tracker)
            module_path = file_path[:-3].replace("/", ".").replace("\\", ".") if file_path.endswith(".py") else file_path.replace("/", ".").replace("\\", ".")

            # Remove module from sys.modules if it's already loaded (force reload)
            if module_path in sys.modules:
                del sys.modules[module_path]

            try:
                module = importlib.import_module(module_path)
            except Exception as imp_exc:
                msg = f"Failed to import module '{module_path}': {imp_exc}"
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg})
                return {"success": False, "error": msg}

            tool_class = getattr(module, class_name, None)
            if tool_class is None:
                msg = f"Class '{class_name}' not found in module '{module_path}'."
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg})
                return {"success": False, "error": msg}

            # Ensure the tool class inherits from BaseTool
            if not issubclass(tool_class, BaseTool):
                msg = f"Class '{class_name}' does not inherit from BaseTool."
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg})
                return {"success": False, "error": msg}

            tool_instance = tool_class()

            if tool_manager:
                tool_manager.register_tool(tool_instance)
                success_msg = f"Tool '{class_name}' registered successfully in ToolManager."
                print(f"[Tool Registration] {success_msg}")
                return {"success": True, "error": "", "tool": tool_instance}
            else:
                info_msg = (
                    f"Tool '{class_name}' instantiated, but no ToolManager provided.\n"
                    f"To register manually: tool_manager.register_tool(tool_instance)"
                )
                print(f"[Tool Registration] {info_msg}")
                return {"success": True, "warning": info_msg, "tool": tool_instance}

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Tool Registration] Exception: {e}\n{tb}")
            if self.notebook:
                self.notebook.log("tool_registration_failure", {"plan": plan, "error": str(e)})
            return {"success": False, "error": str(e), "traceback": tb}
