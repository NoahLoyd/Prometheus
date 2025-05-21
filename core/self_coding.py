import importlib
import sys
import traceback
from typing import Dict, Any, Optional

from tools.prompt_decomposer import PromptDecomposer
from tools.module_builder import ModuleBuilderTool
from tools.base_tool import BaseTool
from tools.tool_manager import ToolManager

class SelfCodingEngine:
    """
    SelfCodingEngine orchestrates the self-coding AGI workflow:
      - Takes in a natural language prompt.
      - Decomposes it into structured module plans.
      - Invokes ModuleBuilderTool to generate code files.
      - Optionally registers new tool modules into a ToolManager.
    """

    def __init__(self):
        self.decomposer = PromptDecomposer()
        self.builder = ModuleBuilderTool()

    def process_prompt(
        self, 
        prompt: str, 
        tool_manager: Optional[ToolManager] = None
    ) -> Dict[str, Any]:
        """
        Processes a prompt, generates modules, writes them to disk, and attempts to auto-register the tool.
        Returns the structured plan and a tool registration result.
        """
        plan = self.decomposer.decompose(prompt)
        result = {"plan": plan, "registration": None}

        try:
            self.builder.write_module(plan)
            reg_result = self._register_tool(plan, tool_manager)
            result["registration"] = reg_result
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Module build error: {e}\n{tb}")
            result["registration"] = {
                "success": False,
                "error": str(e),
                "traceback": tb
            }
        return result

    def _register_tool(
        self,
        plan: Dict[str, Any],
        tool_manager: Optional[ToolManager]
    ) -> Dict[str, Any]:
        """
        Dynamically imports, instantiates, and registers a tool class from a generated module.
        Returns a dict with success status and error message if any.
        """
        try:
            file_path = plan.get("file")
            class_name = plan.get("class")
            if not file_path or not class_name:
                msg = "Missing 'file' or 'class' in plan."
                print(f"[Tool Registration] {msg}")
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
                return {"success": False, "error": msg}

            tool_class = getattr(module, class_name, None)
            if tool_class is None:
                msg = f"Class '{class_name}' not found in module '{module_path}'."
                print(f"[Tool Registration] {msg}")
                return {"success": False, "error": msg}

            # Ensure the tool class inherits from BaseTool
            if not issubclass(tool_class, BaseTool):
                msg = f"Class '{class_name}' does not inherit from BaseTool."
                print(f"[Tool Registration] {msg}")
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
            return {"success": False, "error": str(e), "traceback": tb}
