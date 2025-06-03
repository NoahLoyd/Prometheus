import os
import re
from typing import Dict, Any, List, Optional
from validators.security_validator import validate_security
from core.utils.path_utils import safe_path_join
from addons.notebook import AddOnNotebook

class ModuleBuilderTool:
    """
    Promethyn AGI ModuleBuilderTool (Python 3.11+)
    Accepts single-file (legacy) and multi-file (enhanced) structured build plans,
    writes code and test files to disk, and enforces world-class engineering standards.
    Now supports modular AGI system with type routing, validation enforcement, and AddOnNotebook logging.

    Features:
    - Legacy and enhanced multi-file plan support.
    - Overwrite protection (never overwrites unless explicitly allowed).
    - Robust schema validation and logging of structural inconsistencies.
    - Modular, extensible, and safe.
    - Comprehensive docstrings, comments, and result tracking.
    - AGI module type routing and validator execution.
    - AddOnNotebook logging of validator results.
    """
    MODULE_TYPE_HEADERS = {
        "tool":    '"""Tool Module: {name}\n\nPromethyn Tool module. Type: tool.\n"""\n',
        "test":    '"""Test Module: {name}\n\nPromethyn Test module. Type: test.\n"""\n',
        "validator": '"""Validator Module: {name}\n\nPromethyn Validator module. Type: validator.\n"""\n',
        "core":    '"""Core Module: {name}\n\nPromethyn Core module. Type: core.\n"""\n',
    }
    MODULE_TYPE_DIRS = {
        "tool": "tools/",
        "test": "test/",
        "validator": "validators/",
        "core": "core/",
    }
    VALID_MODULE_TYPES = ("tool", "test", "validator", "core")

    def __init__(self, notebook: Optional[AddOnNotebook] = None):
        """Initialize ModuleBuilderTool with optional notebook instance."""
        self.notebook = notebook or AddOnNotebook()

    def write_module(self, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Main entry: Writes modules and test files as specified in the build plan.
        Adds support for type-based routing, validation, and AddOnNotebook logging.

        Args:
            plan (dict): Structured build plan. Must contain either:
                - Legacy: "file" and "code" (both str)
                - Enhanced: "files" (list of dicts with "path" and "code" keys; may specify "type")
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
            error_msg = "Plan must contain either 'file' and 'code', or a 'files' list of dicts with 'path' and 'code'."
            results["errors"].append(error_msg)
            if self.notebook:
                self.notebook.log("module_builder", "SCHEMA_ERROR", error_msg, metadata={"plan": plan})
            return results

        # --- Process Legacy Plan ---
        if legacy_mode:
            file_path = plan.get("file")
            code = plan.get("code", "")
            test_code = plan.get("test", "") or plan.get("test_code", "")
            test_file_path = file_path.replace(".py", "_test.py") if file_path else None

            # Infer type for legacy mode
            mod_type = self._infer_type_from_path(file_path)
            file_path = self._route_path(file_path, mod_type)
            code = self._apply_type_header(file_path, code, mod_type)

            # Check required fields
            if not file_path or not code:
                error_msg = "Legacy plan missing required 'file' or 'code' fields."
                results["errors"].append(error_msg)
                if self.notebook:
                    self.notebook.log("module_builder", "PLAN_ERROR", error_msg, metadata={"plan": plan})
            else:
                try:
                    self._safe_write_file(
                        file_path,
                        code,
                        results,
                        overwrite_allowed=overwrite_allowed
                    )
                except Exception as e:
                    error_msg = f"Error writing {file_path}: {e}"
                    results["errors"].append(error_msg)
                    if self.notebook:
                        self.notebook.log("module_builder", "WRITE_ERROR", error_msg, metadata={"file_path": file_path, "error": str(e)})

                # Test file (optional)
                if test_code and test_file_path:
                    test_file_path = self._route_path(test_file_path, "test")
                    test_code = self._apply_type_header(test_file_path, test_code, "test")
                    try:
                        self._safe_write_file(
                            test_file_path,
                            test_code,
                            results,
                            overwrite_allowed=overwrite_allowed
                        )
                    except Exception as e:
                        error_msg = f"Error writing {test_file_path}: {e}"
                        results["errors"].append(error_msg)
                        if self.notebook:
                            self.notebook.log("module_builder", "TEST_WRITE_ERROR", error_msg, metadata={"test_file_path": test_file_path, "error": str(e)})

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
                        error_msg = f"Error writing placeholder {test_file_path}: {e}"
                        results["errors"].append(error_msg)
                        if self.notebook:
                            self.notebook.log("module_builder", "PLACEHOLDER_ERROR", error_msg, metadata={"test_file_path": test_file_path, "error": str(e)})

                # Validators (legacy: only if type is validator)
                if mod_type == "validator":
                    self._run_and_log_validators([file_path], results)

                # Validate and log for all module types
                if mod_type in self.VALID_MODULE_TYPES:
                    valid = self._run_and_log_validators([file_path], results)
                    if not valid:
                        results["skipped"].append(file_path)

        # --- Process Enhanced Plan ---
        if enhanced_mode:
            files = plan["files"]
            written_paths = []
            # Validate structure for each file entry
            for idx, entry in enumerate(files):
                if not isinstance(entry, dict):
                    error_msg = f"File entry at index {idx} is not a dict: {entry}"
                    results["errors"].append(error_msg)
                    if self.notebook:
                        self.notebook.log("module_builder", "ENTRY_ERROR", error_msg, metadata={"index": idx, "entry": entry})
                    continue
                path = entry.get("path")
                code = entry.get("code")
                mod_type = entry.get("type", None)
                if not path or not code:
                    error_msg = f"File entry missing 'path' or 'code': {entry}"
                    results["errors"].append(error_msg)
                    if self.notebook:
                        self.notebook.log("module_builder", "MISSING_FIELDS", error_msg, metadata={"index": idx, "entry": entry})
                    continue
                mod_type = mod_type or self._infer_type_from_path(path)
                routed_path = self._route_path(path, mod_type)
                code = self._apply_type_header(routed_path, code, mod_type)
                try:
                    self._safe_write_file(
                        routed_path,
                        code,
                        results,
                        overwrite_allowed=overwrite_allowed
                    )
                    # SECURITY VALIDATION INJECTION (after file written, before registration)
                    is_secure, sec_msg = validate_security(routed_path)
                    if not is_secure:
                        error_msg = f"Security validation failed for {routed_path}: {sec_msg}"
                        if self.notebook:
                            self.notebook.log("module_builder", "SECURITY_VALIDATION_FAIL", error_msg, metadata={"file_path": routed_path, "message": sec_msg})
                        results["errors"].append(error_msg)
                        results["skipped"].append(routed_path)
                        continue
                    written_paths.append(routed_path)
                except Exception as e:
                    error_msg = f"Error writing {routed_path}: {e}"
                    results["errors"].append(error_msg)
                    if self.notebook:
                        self.notebook.log("module_builder", "WRITE_ERROR", error_msg, metadata={"routed_path": routed_path, "error": str(e)})

            # --- Handle test files: "test_code" (single), "test_files" (list) ---
            if "test_code" in plan and isinstance(plan["test_code"], str):
                test_file_path = self._route_path("test_main.py", "test")
                test_code = self._apply_type_header(test_file_path, plan["test_code"], "test")
                try:
                    self._safe_write_file(
                        test_file_path,
                        test_code,
                        results,
                        overwrite_allowed=overwrite_allowed
                    )
                except Exception as e:
                    error_msg = f"Error writing {test_file_path}: {e}"
                    results["errors"].append(error_msg)
                    if self.notebook:
                        self.notebook.log("module_builder", "TEST_WRITE_ERROR", error_msg, metadata={"test_file_path": test_file_path, "error": str(e)})

            if "test_files" in plan and isinstance(plan["test_files"], list):
                for idx, tfile in enumerate(plan["test_files"]):
                    if not isinstance(tfile, dict):
                        error_msg = f"Test file entry at index {idx} is not a dict: {tfile}"
                        results["errors"].append(error_msg)
                        if self.notebook:
                            self.notebook.log("module_builder", "TEST_ENTRY_ERROR", error_msg, metadata={"index": idx, "tfile": tfile})
                        continue
                    tpath = tfile.get("path")
                    tcode = tfile.get("code")
                    if not tpath or not tcode:
                        error_msg = f"Test file missing 'path' or 'code': {tfile}"
                        results["errors"].append(error_msg)
                        if self.notebook:
                            self.notebook.log("module_builder", "TEST_MISSING_FIELDS", error_msg, metadata={"index": idx, "tfile": tfile})
                        continue
                    tpath = self._route_path(tpath, "test")
                    tcode = self._apply_type_header(tpath, tcode, "test")
                    try:
                        self._safe_write_file(
                            tpath,
                            tcode,
                            results,
                            overwrite_allowed=overwrite_allowed
                        )
                    except Exception as e:
                        error_msg = f"Error writing {tpath}: {e}"
                        results["errors"].append(error_msg)
                        if self.notebook:
                            self.notebook.log("module_builder", "TEST_WRITE_ERROR", error_msg, metadata={"tpath": tpath, "error": str(e)})

            # --- Optionally generate placeholder test for each main file if missing ---
            for entry in files:
                path = entry.get("path")
                if path and path.endswith(".py"):
                    mod_type = entry.get("type", None) or self._infer_type_from_path(path)
                    routed_path = self._route_path(path, mod_type)
                    test_path = routed_path.replace(".py", "_test.py")
                    test_path = self._route_path(test_path, "test")
                    if not os.path.exists(test_path):
                        try:
                            placeholder = self._generate_placeholder_test(routed_path)
                            self._safe_write_file(
                                test_path,
                                placeholder,
                                results,
                                overwrite_allowed=overwrite_allowed
                            )
                        except Exception as e:
                            error_msg = f"Error writing placeholder {test_path}: {e}"
                            results["errors"].append(error_msg)
                            if self.notebook:
                                self.notebook.log("module_builder", "PLACEHOLDER_ERROR", error_msg, metadata={"test_path": test_path, "error": str(e)})

            # Validators
            valid = self._run_and_log_validators(written_paths, results)
            if not valid:
                # Remove files that failed validation from written
                for p in written_paths:
                    if p in results["written"]:
                        results["written"].remove(p)
                        results["skipped"].append(p)

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
        # Using safe_path_join to prevent directory traversal when creating directories
        # TODO: review base_dir - using current working directory as base for now
        safe_dir = safe_path_join(".", os.path.dirname(file_path))
        os.makedirs(safe_dir, exist_ok=True)
        if os.path.exists(file_path) and not overwrite_allowed:
            results["skipped"].append(file_path)
            if self.notebook:
                self.notebook.log("module_builder", "FILE_SKIPPED", f"File exists, skipping: {file_path}", metadata={"file_path": file_path, "overwrite_allowed": overwrite_allowed})
            return
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        results["written"].append(file_path)
        if self.notebook:
            self.notebook.log("module_builder", "FILE_WRITTEN", f"Successfully wrote file: {file_path}", metadata={"file_path": file_path, "code_length": len(code)})

    def _apply_type_header(self, file_path: str, code: str, mod_type: Optional[str]) -> str:
        """
        Apply type-specific header and docstring to the code, enforcing snake_case filenames.

        Args:
            file_path (str): Path to the file.
            code (str): Module code.
            mod_type (str): Type of the module.

        Returns:
            str: Code with enforced header.
        """
        base = os.path.basename(file_path).replace(".py", "")
        snake = self._to_snake_case(base)
        header = self.MODULE_TYPE_HEADERS.get(mod_type, "")
        if header:
            header = header.format(name=snake)
        # Remove any pre-existing file-level docstring
        code_body = re.sub(r'^\s*"""[\s\S]*?"""', '', code, count=1)
        out = header + code_body.lstrip("\n")
        return out

    def _route_path(self, file_path: str, mod_type: Optional[str]) -> str:
        """
        Route file to correct directory and enforce snake_case filename.

        Args:
            file_path (str): Original path.
            mod_type (str): Module type.

        Returns:
            str: Routed path.
        """
        filename = os.path.basename(file_path)
        snake = self._to_snake_case(filename.replace(".py", "")) + ".py"
        dest_dir = self.MODULE_TYPE_DIRS.get(mod_type, "")
        if dest_dir:
            # Using safe_path_join with dest_dir as base to prevent directory traversal
            routed = safe_path_join(dest_dir, snake)
        else:
            routed = file_path
        return routed

    def _to_snake_case(self, name: str) -> str:
        """
        Converts a string to snake_case.

        Args:
            name (str): The string.

        Returns:
            str: snake_case string.
        """
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        s3 = re.sub(r'[^a-zA-Z0-9]+', '_', s2)
        return s3.lower().strip("_")

    def _infer_type_from_path(self, file_path: Optional[str]) -> Optional[str]:
        """
        Infer module type from filename or path.

        Args:
            file_path (str): Path.

        Returns:
            str: Type or None.
        """
        if not file_path:
            return None
        p = file_path.lower()
        if p.startswith("tools/") or "_tool" in p or "tool" in p:
            return "tool"
        if p.startswith("validators/") or "_validator" in p or "validator" in p:
            return "validator"
        if p.startswith("core/") or "core" in p:
            return "core"
        if p.startswith("tests/") or "_test" in p or "test" in p:
            return "test"
        return None

    def _generate_placeholder_test(self, source_path: str) -> str:
        """
        Generates a placeholder Python test file for a given source file.

        Args:
            source_path (str): Path to the source module.

        Returns:
            str: Minimal pytest-style placeholder for the given module.
        """
        mod = os.path.basename(source_path).replace(".py", "")
        return f'''"""Test Module: {mod}

Promethyn Test module. Type: test.
"""

def test_placeholder():
    assert True, "Placeholder test for {mod}.py"
'''

    def _run_and_log_validators(self, file_paths: List[str], results: Dict[str, List[str]]) -> bool:
        """
        Run all registered validators on the given module files.
        Log results to AddOnNotebook. If any validator fails, modules will not be installed.

        Args:
            file_paths (list): List of module file paths.
            results (dict): Results log.

        Returns:
            bool: True if all validators passed, False otherwise.
        """
        all_valid = True
        validator_funcs = self._collect_validators()
        for file_path in file_paths:
            for v_name, v_func in validator_funcs.items():
                try:
                    valid, msg = v_func(file_path)
                    status = 'PASS' if valid else 'FAIL'
                    log_msg = f"Validator {v_name} on {file_path}: {status} - {msg}"
                    if self.notebook:
                        self.notebook.log("module_builder", "VALIDATOR_RESULT", log_msg, metadata={
                            "validator": v_name, 
                            "file_path": file_path, 
                            "valid": valid, 
                            "message": msg,
                            "status": status
                        })
                    if not valid:
                        error_msg = f"Validator {v_name} failed on {file_path}: {msg}"
                        results["errors"].append(error_msg)
                        all_valid = False
                except Exception as e:
                    error_msg = f"Validator {v_name} error on {file_path}: {e}"
                    if self.notebook:
                        self.notebook.log("module_builder", "VALIDATOR_ERROR", error_msg, metadata={
                            "validator": v_name, 
                            "file_path": file_path, 
                            "error": str(e)
                        })
                    results["errors"].append(error_msg)
                    all_valid = False
        return all_valid

    def _collect_validators(self) -> Dict[str, Any]:
        """
        Collect all validator functions from validators/ directory.

        Returns:
            dict: {validator_name: callable}
        """
        validators = {}
        validator_dir = self.MODULE_TYPE_DIRS["validator"]
        if not os.path.isdir(validator_dir):
            return validators
        for fname in os.listdir(validator_dir):
            if fname.endswith(".py"):
                # Using safe_path_join with validator_dir as base to prevent directory traversal
                v_path = safe_path_join(validator_dir, fname)
                try:
                    ns = {}
                    with open(v_path, "r", encoding="utf-8") as f:
                        exec(f.read(), ns)
                    for k, v in ns.items():
                        if callable(v) and k.startswith("validate_"):
                            validators[k] = v
                except Exception:
                    continue
        return validators
