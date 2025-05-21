import os
from typing import Dict, Any

class ModuleBuilderTool:
    """
    Accepts structured module plans and writes Python files to disk.
    """

    def write_module(self, plan: Dict[str, Any]) -> None:
        """
        Writes the code and test files as specified in the plan.
        """
        file_path = plan.get("file")
        code = plan.get("code", "")
        test_code = plan.get("test", "")
        test_file_path = file_path.replace(".py", "_test.py")

        # Write the main module
        if file_path and code:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Module written: {file_path}")

        # Write the test file
        if test_code:
            with open(test_file_path, "w", encoding="utf-8") as f:
                f.write(test_code)
            print(f"Test written: {test_file_path}")
