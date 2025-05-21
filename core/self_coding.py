import importlib.util
import sys
import traceback
from datetime import datetime
from typing import Dict, Any

from tools.prompt_decomposer import PromptDecomposer
from tools.module_builder import ModuleBuilderTool
from tools.tool_manager import ToolManager
from addons.notebook import AddOnNotebook

class SelfCodingEngine:
    """
    SelfCodingEngine orchestrates the self-coding AGI workflow:
      - Takes in a natural language prompt.
      - Decomposes it into structured module plans.
      - Invokes ModuleBuilderTool to generate code files.
      - Dynamically loads and registers new tools.
      - Logs strategic intelligence and failures.
    """

    def __init__(self, tool_manager: ToolManager = None, notebook: AddOnNotebook = None):
        self.decomposer = PromptDecomposer()
        self.builder = ModuleBuilderTool()
        self.tool_manager = tool_manager
        self.notebook = notebook or AddOnNotebook()

    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Processes a prompt, generates modules, writes them to disk, and attempts to auto-register the tool.
        Returns the structured plan for inspection.
        """
        plan = self.decomposer.decompose(prompt)
        try:
            self.builder.write_module(plan)
            reg_result = self.auto_register_tool(plan)
            if reg_result["success"]:
                print(f"Tool '{plan.get('class')}' registered successfully.")
            else:
                print(f"Tool registration failed: {reg_result['error']}")
                self.notebook.log(
                    entry_type="user_prompt",
                    content={
                        "prompt": prompt,
                        "plan": plan,
                        "error": reg_result["error"]
                    }
                )
        except Exception as e:
            print(f"Module build error: {e}")
            self.notebook.log(
                entry_type="user_prompt",
                content={
                    "prompt": prompt,
                    "plan": plan,
                    "error": str(e)
                }
            )
        return plan

    def auto_register_tool(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        After writing a module, dynamically imports and registers the tool with ToolManager.
        Returns dict with success status and error message if any.
        """
        try:
            file_path = plan.get("file")
            class_name = plan.get("class")
            if not file_path or not class_name:
                return {"success": False, "error": "Missing file or class in plan."}
            # Remove .py and convert to module path
            module_path = file_path.replace("/", ".").rstrip(".py")
            # Dynamically import module
            spec = importlib.util.spec_from_file_location(module_path, file_path)
            if not spec or not spec.loader:
                return {"success": False, "error": f"Could not load spec for {file_path}"}
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            spec.loader.exec_module(module)
            # Get class and instantiate
            tool_class = getattr(module, class_name, None)
            if not tool_class:
                return {"success": False, "error": f"Class {class_name} not found in {file_path}"}
            tool_instance = tool_class()
            if self.tool_manager:
                self.tool_manager.register_tool(tool_instance)
            return {"success": True, "error": ""}
        except Exception as e:
            tb = traceback.format_exc()
            return {"success": False, "error": f"{e}\n{tb}"}
