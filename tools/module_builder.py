import os
from typing import Dict, Any, List, Optional, Union

class ModuleBuilderTool:
    """
    Promethyn ModuleBuilderTool (Python 3.11+)
    Accepts structured build plans and writes one or multiple Python files to disk.
    Follows Promethyn standards: modular, safe, and extensible.
    """

    def write_module(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write modules (single or multiple files) as specified in the build plan.

        Args:
            plan (dict): Structured build plan. Supports legacy (single-file) and enhanced (multi-file) format.

        Returns:
            dict: Structured log of written and skipped files, and any errors encountered.
        """
        results = {"written": [], "skipped": [], "errors": []}

        # Legacy single-file plan support (assumptions validated below)
        if "file" in plan or ("code" in plan and isinstance(plan.get("code"), str)):
            file_path = plan.get("file")
            code = plan.get("code", "")
            test_code = plan.get("test", "") or plan.get("test_code", "")
            # Assumption: single-file legacy plans use 'file' and 'code' keys.
            if not file_path or not code:
                results["errors"].append(
                    "Legacy plan missing required 'file' or 'code' fields."
                )
            else:
                try:
                    self._write_file(file_path, code, results)
                except Exception as e:
                    results["errors"].append(f"Error writing {file_path}: {e}")
                # Write single-file test if provided
                if test_code:
                    test_file_path = file_path.replace(".py", "_test.py")
                    try:
                        self._write_file(test_file_path, test_code, results)
                    except Exception as e:
                        results["errors"].append(f"Error writing {test_file_path}: {e}")
            return results

        # Enhanced multi-file plan
        files = plan.get("files")
        if not files or not isinstance(files, list):
            results["errors"].append("No 'files' array found in plan.")
            return results

        for entry in files:
            # Assumption: each entry must be a dict with 'path' and 'code'
            if not isinstance(entry, dict):
                results["errors"].append(f"File entry not a dict: {entry}")
                continue
            path = entry.get("path")
            code = entry.get("code")
            if not path or not code:
                results["errors"].append(
                    f"File entry missing required 'path' or 'code': {entry}"
                )
                continue
            try:
                self._write_file(path, code, results)
            except Exception as e:
                results["errors"].append(f"Error writing {path}: {e}")

        # Handle test code or files
        if "test_code" in plan and isinstance(plan["test_code"], str):
            # Fallback: write main test file if specified
            test_file_path = "test_main.py"
            try:
                self._write_file(test_file_path, plan["test_code"], results)
            except Exception as e:
                results["errors"].append(f"Error writing {test_file_path}: {e}")
        if "test_files" in plan and isinstance(plan["test_files"], list):
            for tfile in plan["test_files"]:
                if not isinstance(tfile, dict):
                    results["errors"].append(f"Test file entry not a dict: {tfile}")
                    continue
                tpath = tfile.get("path")
                tcode = tfile.get("code")
                if not tpath or not tcode:
                    results["errors"].append(
                        f"Test file missing 'path' or 'code': {tfile}"
                    )
                    continue
                try:
                    self._write_file(tpath, tcode, results)
                except Exception as e:
                    results["errors"].append(f"Error writing {tpath}: {e}")

        return results

    def _write_file(self, file_path: str, code: str, results: Dict[str, List[str]]) -> None:
        """
        Safely writes a file to disk with overwrite protection.

        Args:
            file_path (str): Path to write the file.
            code (str): Source code to write.
            results (dict): Log dictionary to update.
        Raises:
            Exception: If file write fails.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            results["skipped"].append(file_path)
            return
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        results["written"].append(file_path)
