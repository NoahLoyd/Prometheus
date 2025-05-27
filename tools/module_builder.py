import os
from typing import Dict, Any, List, Optional

class ModuleBuilderTool:
    """
    Promethyn AGI ModuleBuilderTool (Python 3.11+)
    Accepts single-file (legacy) and multi-file (enhanced) structured build plans,
    writes code and test files to disk, and enforces world-class engineering standards.

    Features:
    - Legacy and enhanced multi-file plan support.
    - Overwrite protection (never overwrites unless explicitly allowed).
    - Robust schema validation and logging of structural inconsistencies.
    - Modular, extensible, and safe.
    - Comprehensive docstrings, comments, and result tracking.
    """

    def write_module(self, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Main entry: Writes modules and test files as specified in the build plan.

        Args:
            plan (dict): Structured build plan. Must contain either:
                - Legacy: "file" and "code" (both str)
                - Enhanced: "files" (list of dicts with "path" and "code" keys)
                Optional: "test", "test_code", "test_files", "overwrite_allowed" (bool)

        Returns:
            dict: Structured log with keys:
                - "written": files written
                - "skipped": files skipped due to overwrite protection or missing info
                - "errors": files or plan errors encountered
        """
        results = {"written": [], "skipped": [], "errors": []}
        overwrite_allowed = bool(plan.get("overwrite_allowed", False))

        # --- Legacy single-file plan support ---
        legacy_mode = "file" in plan and "code" in plan

        # --- Enhanced multi-file plan support ---
        enhanced_mode = "files" in plan and isinstance(plan["files"], list)

        # --- Schema Validation and Logging ---
        if not (legacy_mode or enhanced_mode):
            results["errors"].append(
                "Plan must contain either 'file' and 'code', or a 'files' list of dicts with 'path' and 'code'."
            )
            return results

        # --- Process Legacy Plan ---
        if legacy_mode:
            file_path = plan.get("file")
            code = plan.get("code", "")
            test_code = plan.get("test", "") or plan.get("test_code", "")
            test_file_path = file_path.replace(".py", "_test.py") if file_path else None

            # Check required fields
            if not file_path or not code:
                results["errors"].append("Legacy plan missing required 'file' or 'code' fields.")
            else:
                try:
                    self._safe_write_file(
                        file_path,
                        code,
                        results,
                        overwrite_allowed=overwrite_allowed
                    )
                except Exception as e:
                    results["errors"].append(f"Error writing {file_path}: {e}")

                # Test file (optional)
                if test_code and test_file_path:
                    try:
                        self._safe_write_file(
                            test_file_path,
                            test_code,
                            results,
                            overwrite_allowed=overwrite_allowed
                        )
                    except Exception as e:
                        results["errors"].append(f"Error writing {test_file_path}: {e}")

                # Optionally generate placeholder test if test file is missing
                if test_file_path and not os.path.exists(test_file_path):
                    try:
                        placeholder = self._generate_placeholder_test(file_path)
                        self._safe_write_file(
                            test_file_path,
                            placeholder,
                            results,
                            overwrite_allowed=overwrite_allowed
                        )
                    except Exception as e:
                        results["errors"].append(f"Error writing placeholder {test_file_path}: {e}")

        # --- Process Enhanced Plan ---
        if enhanced_mode:
            files = plan["files"]
            # Validate structure for each file entry
            for idx, entry in enumerate(files):
                if not isinstance(entry, dict):
                    results["errors"].append(f"File entry at index {idx} is not a dict: {entry}")
                    continue
                path = entry.get("path")
                code = entry.get("code")
                if not path or not code:
                    results["errors"].append(f"File entry missing 'path' or 'code': {entry}")
                    continue
                try:
                    self._safe_write_file(
                        path,
                        code,
                        results,
                        overwrite_allowed=overwrite_allowed
                    )
                except Exception as e:
                    results["errors"].append(f"Error writing {path}: {e}")

            # --- Handle test files: "test_code" (single), "test_files" (list) ---
            if "test_code" in plan and isinstance(plan["test_code"], str):
                # Write main test file at top level or as 'test_main.py'
                test_file_path = "test_main.py"
                try:
                    self._safe_write_file(
                        test_file_path,
                        plan["test_code"],
                        results,
                        overwrite_allowed=overwrite_allowed
                    )
                except Exception as e:
                    results["errors"].append(f"Error writing {test_file_path}: {e}")

            if "test_files" in plan and isinstance(plan["test_files"], list):
                for idx, tfile in enumerate(plan["test_files"]):
                    if not isinstance(tfile, dict):
                        results["errors"].append(f"Test file entry at index {idx} is not a dict: {tfile}")
                        continue
                    tpath = tfile.get("path")
                    tcode = tfile.get("code")
                    if not tpath or not tcode:
                        results["errors"].append(f"Test file missing 'path' or 'code': {tfile}")
                        continue
                    try:
                        self._safe_write_file(
                            tpath,
                            tcode,
                            results,
                            overwrite_allowed=overwrite_allowed
                        )
                    except Exception as e:
                        results["errors"].append(f"Error writing {tpath}: {e}")

            # --- Optionally generate placeholder test for each main file if missing ---
            for entry in files:
                path = entry.get("path")
                if path and path.endswith(".py"):
                    test_path = path.replace(".py", "_test.py")
                    if not os.path.exists(test_path):
                        try:
                            placeholder = self._generate_placeholder_test(path)
                            self._safe_write_file(
                                test_path,
                                placeholder,
                                results,
                                overwrite_allowed=overwrite_allowed
                            )
                        except Exception as e:
                            results["errors"].append(f"Error writing placeholder {test_path}: {e}")

        return results

    def _safe_write_file(self, file_path: str, code: str, results: Dict[str, List[str]], overwrite_allowed: bool = False) -> None:
        """
        Safely writes a file to disk, with overwrite protection and result logging.

        Args:
            file_path (str): File path for output.
            code (str): File content.
            results (dict): Tracking dict for written/skipped/errors.
            overwrite_allowed (bool): If True, will overwrite existing files.

        Raises:
            Exception: Propagates file system errors.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path) and not overwrite_allowed:
            results["skipped"].append(file_path)
            return
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        results["written"].append(file_path)

    def _generate_placeholder_test(self, source_path: str) -> str:
        """
        Generates a placeholder Python test file for a given source file.

        Args:
            source_path (str): Path to the source module.

        Returns:
            str: Minimal pytest-style placeholder for the given module.
        """
        mod = os.path.basename(source_path).replace(".py", "")
        return f'''"""
Placeholder test for {mod}.py.
Auto-generated by Promethyn AGI.
"""

def test_placeholder():
    assert True, "Placeholder test for {mod}.py"
'''
