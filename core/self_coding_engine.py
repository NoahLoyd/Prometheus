import importlib
import sys
import traceback
import os
import logging
from typing import Dict, Any, Optional, List, Callable  # <-- Added Callable for type hints

from tools.prompt_decomposer import PromptDecomposer
from tools.module_builder import ModuleBuilderTool
from tools.base_tool import BaseTool
from tools.tool_manager import ToolManager
from addons.notebook import AddOnNotebook
from tools.test_tool_runner import TestToolRunner  # <--- NEW IMPORT
from core.validators.extended_validators import register_validators

# --- Begin: Imports for new validator modules ---
try:
    from core.validators.code_quality_assessor import CodeQualityAssessor
except ImportError:
    CodeQualityAssessor = None
try:
    from core.validators.security_scanner import SecurityScanner
except ImportError:
    SecurityScanner = None
try:
    from core.validators.behavioral_simulator import BehavioralSimulator
except ImportError:
    BehavioralSimulator = None
# --- End: Imports for new validator modules ---

# --- Import Security Validator as required ---
try:
    from validators.security_validator import validate_security
except ImportError:
    validate_security = None
# --- End Security Validator import ---

class SelfCodingEngine:
    """
    SelfCodingEngine orchestrates the self-coding AGI workflow:
      - Takes in a natural language prompt (single or multiple tools).
      - Decomposes it into one or more structured module plans.
      - Generates, writes, validates, and registers each tool.
      - Logs all outcomes for strategic learning.
      - Enforces Promethyn standards and supports future extensibility.
    """

    # --- Validator hooks (extensible, for future use) ---
    VALIDATORS = ["MathEvaluator", "PlanVerifier", "CodeCritic"]  # Placeholder for plugging in

    def __init__(self, notebook: Optional[AddOnNotebook] = None):
        self.decomposer = PromptDecomposer()
        self.builder = ModuleBuilderTool()
        self.notebook = notebook or AddOnNotebook()
        self.logger = self._get_logger()
        self.validator_registry = {}  # For future validator plug-in

        # --- Register core validators safely on instantiation ---
        self.register_validator("MathEvaluator", MathEvaluator())
        self.register_validator("PlanVerifier", PlanVerifier())
        register_validators(self.validators)  # <-- Inject extended Promethyn validators here

        # --- Register TestToolRunner instance (instantiated ONCE here) ---
        self.test_runner = TestToolRunner()  # <--- Elite singleton placement

        # --- Inject new validators at the end of the pipeline in a defensive, modular way ---
        self._register_enhanced_validators()

        # --- Inject SecurityValidator into registry and VALIDATORS pipeline if not already present ---
        # Ensure PlanVerifier ➝ MathEvaluator ➝ CodeQualityAssessor ➝ SecurityValidator ➝ TestToolRunner
        # Do NOT remove or overwrite any existing validators
        if validate_security is not None:
            if "SecurityValidator" not in self.validator_registry:
                self.register_validator("SecurityValidator", validate_security)
            # Insert SecurityValidator after CodeQualityAssessor, before TestToolRunner
            # If CodeQualityAssessor exists, insert after; else after PlanVerifier/MathEvaluator as fallback
            vlist = self.VALIDATORS
            if "SecurityValidator" not in vlist:
                # Find index for CodeQualityAssessor, else PlanVerifier or MathEvaluator
                insert_after = None
                for name in ["CodeQualityAssessor", "MathEvaluator", "PlanVerifier"]:
                    if name in vlist:
                        insert_after = name
                        break
                if insert_after is not None:
                    idx = vlist.index(insert_after) + 1
                else:
                    idx = len(vlist)
                vlist.insert(idx, "SecurityValidator")

    def _register_enhanced_validators(self):
        """
        Dynamically and safely register enhanced validators at the end of the validator chain.
        Preserves all existing validation logic and order.
        """
        # Defensive: do not overwrite existing names, log if registration fails.
        enhanced_validators = [
            ("CodeQualityAssessor", CodeQualityAssessor),
            ("SecurityScanner", SecurityScanner),
            ("BehavioralSimulator", BehavioralSimulator)
        ]
        for name, validator_cls in enhanced_validators:
            if validator_cls is None:
                self.logger.warning(f"Validator module '{name}' is missing or failed to import; skipping registration.")
                continue
            try:
                instance = validator_cls()
            except Exception as e:
                self.logger.error(f"Could not instantiate validator '{name}': {e}")
                continue
            try:
                self.register_validator(name, instance)
                self.logger.info(f"Validator '{name}' registered and appended to VALIDATORS pipeline.")
            except ValueError as ve:
                self.logger.warning(f"Validator '{name}' could not be registered: {ve}")

        # Ensure order: append to VALIDATORS after MathEvaluator/TestToolRunner, never before.
        # Existing pipeline: ["MathEvaluator", "PlanVerifier", "CodeCritic"]
        for name, validator_cls in enhanced_validators:
            if name not in self.VALIDATORS and validator_cls is not None:
                self.VALIDATORS.append(name)

    def _get_logger(self):
        logger = logging.getLogger("Promethyn.SelfCodingEngine")
        logger.setLevel(logging.INFO)  # Change to DEBUG for more verbosity
        if not logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def register_validator(
        self,
        name: str,
        instance: Callable,
        allow_overwrite: bool = False
    ):
        """
        Register a validator at runtime in a modular, safe, and extensible way.

        :param name: Unique string identifier for the validator.
        :param instance: Callable with signature (plan, tool_code, test_code) -> dict.
        :param allow_overwrite: If False (default), protects existing validators from being overwritten.
        :raises ValueError: If instance is not callable or if attempting to overwrite without permission.
        """
        if not callable(instance):
            error_msg = f"Validator '{name}' must be callable."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if name in self.validator_registry and not allow_overwrite:
            error_msg = (
                f"Validator '{name}' is already registered. "
                f"Use allow_overwrite=True to replace."
            )
            self.logger.warning(error_msg)
            raise ValueError(error_msg)
        # Log if overwriting
        if name in self.validator_registry and allow_overwrite:
            self.logger.info(f"Validator '{name}' is being overwritten.")
        self.validator_registry[name] = instance
        self.logger.info(f"Validator '{name}' registered successfully.")

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
        - Performs Promethyn internal standards checks.
        - Hooks for future validators.
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
            self.logger.error(log_msg)
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
                    self.logger.warning(msg)
                    continue
                if test_path and os.path.exists(test_path):
                    msg = f"Test file exists, skipping: {test_path}"
                    single_result["registration"] = {"success": False, "error": msg}
                    retry_later.append({"plan": plan, "reason": msg})
                    results.append(single_result)
                    self.logger.warning(msg)
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

                # --- Promethyn internal standards check ---
                standards_errors = self._check_standards(plan, tool_code, test_code)
                if standards_errors:
                    err_msg = f"Promethyn standards not met: {standards_errors}"
                    single_result["registration"] = {"success": False, "error": err_msg}
                    retry_later.append({"plan": plan, "reason": err_msg})
                    if self.notebook:
                        self.notebook.log("standards_failure", {"plan": plan, "errors": standards_errors})
                    self.logger.error(err_msg)
                    results.append(single_result)
                    continue

                # --- BEGIN: Promethyn AGI Enhanced Validation Pipeline Injection ---
                validation_passed = True
                validator_results = []
                test_run_result = None

                # 1. Run all existing validators (PlanVerifier, MathEvaluator, CodeQualityAssessor)
                for validator_name in ["PlanVerifier", "MathEvaluator", "CodeQualityAssessor"]:
                    validator = self.validator_registry.get(validator_name)
                    if validator:
                        try:
                            result = validator(plan, tool_code, test_code)
                            validator_results.append((validator_name, result))
                            if not result.get("success", True):
                                validation_passed = False
                                if self.notebook:
                                    self.notebook.log("validator_failure", {
                                        "plan": plan,
                                        "validator": validator_name,
                                        "error": result.get("error")
                                    })
                        except Exception as ex:
                            validation_passed = False
                            if self.notebook:
                                self.notebook.log("validator_exception", {
                                    "plan": plan,
                                    "validator": validator_name,
                                    "exception": str(ex)
                                })

                # 2. Inject a security scan using validate_security(file_path)
                security_validation_result = None
                if validate_security is not None and tool_path:
                    try:
                        security_validation_result = validate_security(tool_path)
                        validator_results.append(("SecurityValidator", security_validation_result))
                        if not security_validation_result.get("success", True):
                            validation_passed = False
                            if self.notebook:
                                self.notebook.log("validator_failure", {
                                    "plan": plan,
                                    "validator": "SecurityValidator",
                                    "error": security_validation_result.get("error")
                                })
                    except Exception as ex:
                        validation_passed = False
                        if self.notebook:
                            self.notebook.log("validator_exception", {
                                "plan": plan,
                                "validator": "SecurityValidator",
                                "exception": str(ex)
                            })

                # 3. If the file is a test (ends with "_test.py"), run test_tool_runner.run_test_file()
                if tool_path and tool_path.endswith("_test.py"):
                    try:
                        test_run_result = self.test_runner.run_test_file(tool_path)
                        if not test_run_result.get("passed", False):
                            validation_passed = False
                            if self.notebook:
                                self.notebook.log("test_tool_runner_failure", {
                                    "plan": plan,
                                    "test_result": test_run_result,
                                    "error": test_run_result.get("error", test_run_result)
                                })
                    except Exception as ex:
                        validation_passed = False
                        if self.notebook:
                            self.notebook.log("test_tool_runner_exception", {
                                "plan": plan,
                                "exception": str(ex)
                            })

                # 4. If ANY validator or test fails, reject the tool/module and do NOT register it.
                if not validation_passed:
                    fail_msg = "Tool/module rejected due to failed validation or test."
                    single_result["registration"] = {"success": False, "error": fail_msg}
                    retry_later.append({"plan": plan, "reason": fail_msg})
                    if self.notebook:
                        self.notebook.log("tool_rejected", {
                            "plan": plan,
                            "validator_results": validator_results,
                            "test_run_result": test_run_result,
                            "error": fail_msg
                        })
                    results.append(single_result)
                    continue
                else:
                    if self.notebook:
                        self.notebook.log("tool_validated", {
                            "plan": plan,
                            "validator_results": validator_results,
                            "test_run_result": test_run_result,
                            "status": "success"
                        })
                # --- END: Promethyn AGI Enhanced Validation Pipeline Injection ---

                # --- Validator hooks (future extensibility) ---
                validators_passed = True
                for validator_name in self.VALIDATORS:
                    validator = self.validator_registry.get(validator_name)
                    if validator:
                        try:
                            validator_result = validator(plan, tool_code, test_code)
                            if not validator_result.get("success", True):
                                val_msg = f"Validator {validator_name} failed: {validator_result.get('error')}"
                                single_result["registration"] = {"success": False, "error": val_msg}
                                retry_later.append({"plan": plan, "reason": val_msg})
                                if self.notebook:
                                    self.notebook.log("validator_failure", {
                                        "plan": plan,
                                        "validator": validator_name,
                                        "error": val_msg,
                                    })
                                self.logger.error(val_msg)
                                results.append(single_result)
                                validators_passed = False
                                break  # Stop further validators and do not run test runner
                        except Exception as val_ex:
                            tb = traceback.format_exc()
                            val_msg = f"Validator {validator_name} raised: {val_ex}\n{tb}"
                            single_result["registration"] = {"success": False, "error": val_msg}
                            retry_later.append({"plan": plan, "reason": val_msg})
                            if self.notebook:
                                self.notebook.log("validator_exception", {
                                    "plan": plan,
                                    "validator": validator_name,
                                    "error": val_msg,
                                })
                            self.logger.error(val_msg)
                            results.append(single_result)
                            validators_passed = False
                            break

                if not validators_passed:
                    continue

                # --- Final test: TestToolRunner on generated test file ---
                if test_path:
                    test_result = self.test_runner.run_test_file(test_path)
                    if not test_result.get("passed", False):
                        fail_msg = f"TestToolRunner failed: {test_result.get('error', test_result)}"
                        single_result["registration"] = {"success": False, "error": fail_msg}
                        retry_later.append({"plan": plan, "reason": fail_msg})
                        if self.notebook:
                            self.notebook.log("test_tool_runner_failure", {
                                "plan": plan,
                                "test_result": test_result,
                                "error": fail_msg,
                            })
                            self._log_to_notebook()
                        self.logger.error(fail_msg)
                        results.append(single_result)
                        continue

                # --- AGI EXTENSION: Enhanced Validator Audit Logging ---
                for validator_name in self.VALIDATORS:
                    validator = self.validator_registry.get(validator_name)
                    if validator:
                        try:
                            validator_result = validator(plan, tool_code, test_code)
                            audit_log = {
                                "plan": plan,
                                "validator": validator_name,
                                "result": validator_result,
                            }
                            if self.notebook:
                                self.notebook.log("validator_audit", audit_log)
                            self.logger.debug(f"Validator '{validator_name}' outcome: {validator_result}")
                        except Exception as val_ex:
                            audit_log = {
                                "plan": plan,
                                "validator": validator_name,
                                "exception": str(val_ex),
                            }
                            if self.notebook:
                                self.notebook.log("validator_audit_exception", audit_log)
                            self.logger.debug(f"Validator '{validator_name}' exception during audit: {val_ex}")

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
                    if self.notebook:
                        self.notebook.log("test_run", {
                            "plan": plan,
                            "result": validation_result,
                            "status": "success"
                        })
                    self.logger.info(f"Test run for '{class_name}' succeeded: {validation_result}")
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
                        self.logger.error(fail_msg)
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
                        self.notebook.log("test_run", {
                            "plan": plan,
                            "error": fail_msg,
                            "status": "exception"
                        })
                    self.logger.error(fail_msg)
                    self.logger.error(f"Test run for '{class_name}' raised exception: {fail_msg}")
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
                self.logger.error(fail_msg)
            results.append(single_result)

        # --- Log retry_later to memory or AddOnNotebook ---
        if retry_later:
            retry_log = {"retry_later": retry_later, "prompt": prompt}
            if short_term_memory is not None:
                short_term_memory.setdefault("tool_retry_queue", []).extend(retry_later)
            if self.notebook:
                self.notebook.log("tool_retry_later", retry_log)
            for retry in retry_later:
                self._schedule_retry(retry["plan"], retry["reason"])

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
                self.logger.error(msg)
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
                self.logger.error(msg)
                return {"success": False, "error": msg}

            tool_class = getattr(module, class_name, None)
            if tool_class is None:
                msg = f"Class '{class_name}' not found in module '{module_path}'."
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg})
                self.logger.error(msg)
                return {"success": False, "error": msg}

            # Ensure the tool class inherits from BaseTool
            if not issubclass(tool_class, BaseTool):
                msg = f"Class '{class_name}' does not inherit from BaseTool."
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg})
                self.logger.error(msg)
                return {"success": False, "error": msg}

            tool_instance = tool_class()

            if tool_manager:
                tool_manager.register_tool(tool_instance)
                success_msg = f"Tool '{class_name}' registered successfully in ToolManager."
                print(f"[Tool Registration] {success_msg}")
                self.logger.info(success_msg)
                return {"success": True, "error": "", "tool": tool_instance}
            else:
                info_msg = (
                    f"Tool '{class_name}' instantiated, but no ToolManager provided.\n"
                    f"To register manually: tool_manager.register_tool(tool_instance)"
                )
                print(f"[Tool Registration] {info_msg}")
                self.logger.info(info_msg)
                return {"success": True, "warning": info_msg, "tool": tool_instance}

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[Tool Registration] Exception: {e}\n{tb}")
            if self.notebook:
                self.notebook.log("tool_registration_failure", {"plan": plan, "error": str(e)})
            self.logger.error(f"Tool registration exception: {e}")
            return {"success": False, "error": str(e), "traceback": tb}

    # --- Internal standards enforcement for Promethyn AGI tools ---
    def _check_standards(self, plan, tool_code, test_code) -> List[str]:
        """
        Run Promethyn standards checks on code and plan: safety, modularity, testability.
        Returns list of error strings, or empty list if all standards are met.
        """
        errors = []
        # Safety: Check for dangerous operations (basic, extendable to AST-based)
        unsafe_keywords = ["os.system", "eval(", "exec(", "subprocess.Popen", "open('/dev", "rm -rf"]
        for kw in unsafe_keywords:
            if kw in (tool_code or ""):
                errors.append(f"Unsafe operation detected: {kw}")

        # Modularity: Check for at least one class, and that the class matches plan
        if plan.get("class") not in (tool_code or ""):
            errors.append("Tool class does not match plan or is missing.")

        # Testability: Must have test code and 'run' method
        if not test_code or ("def test" not in test_code and "class Test" not in test_code):
            errors.append("No proper test defined or missing test function/class.")
        if "def run(" not in (tool_code or ""):
            errors.append("No 'run' method implemented in tool.")

        # TODO: Add more robust AST-based and pattern-based checks for safety, modularity, testability

        return errors

    # --- Placeholder for future caching mechanism ---
    def _cache_result(self, key, value):
        """
        TODO: Implement caching (e.g., Redis, in-memory, disk) for tool generation and validation results.
        """
        pass

    # --- AGI EXTENSION: Retry Scheduling Telemetry ---
    def _schedule_retry(self, plan, reason):
        """
        TODO: Implement exponential backoff retry system for failed tool generations/validations.
        """
        self.logger.warning(f"Retry scheduled for plan {plan} due to: {reason}")
        if self.notebook:
            self.notebook.log("retry_scheduled", {"plan": plan, "reason": reason})

    # --- Hook for future multi-phase planning ---
    def _multi_phase_plan(self, plan):
        """
        TODO: Implement multi-phase build/execution logic for complex agentic projects.
        """
        return plan

    def _log_to_notebook(self):
        if self.notebook:
            self.notebook.log("log", {"msg": "TestToolRunner failure or other critical event."})

class MathEvaluator:
    """
    MathEvaluator
    -------------
    Validator for Promethyn AGI that ensures generated tool code:
      - Handles mathematical reasoning or evaluation safely and accurately.
      - Detects unsafe or inappropriate math operations (e.g., direct use of `eval`).
      - Checks for presence of core math operations if the plan or code suggests math intent.
    Usage:
        validator = MathEvaluator()
        result = validator(plan, tool_code, test_code)
    """

    SAFE_MATH_KEYWORDS = [
        "+", "-", "*", "/", "//", "%", "**", "math.", "abs(", "round(", "sum(", "min(", "max("
    ]
    UNSAFE_MATH_PATTERNS = [
        "eval(", "exec(", "import os", "import subprocess"
    ]

    def __call__(self, plan: dict, tool_code: str, test_code: str) -> dict:
        """
        Validates the tool code for safe and correct mathematical reasoning.
        Returns:
            dict: { 'success': bool, 'error': Optional[str] }
        """
        errors = []

        # 1. Detect unsafe math patterns (e.g., eval, exec)
        for unsafe in self.UNSAFE_MATH_PATTERNS:
            if unsafe in (tool_code or ""):
                errors.append(f"Unsafe math operation detected: '{unsafe}'")

        # 2. If the plan or code mentions math, check for at least one math operation
        plan_str = str(plan).lower()
        code_str = (tool_code or "").lower()
        math_intent = any(word in plan_str for word in ["math", "arithmetic", "calculate", "sum", "multiply", "divide", "add", "subtract"])
        if math_intent:
            if not any(keyword in code_str for keyword in self.SAFE_MATH_KEYWORDS):
                errors.append("Math intent detected in plan, but no safe math operations found in code.")

        # 3. Optionally: check for prohibited direct user input to math functions
        if "input(" in code_str and ("eval(" in code_str or any(op in code_str for op in ["+", "-", "*", "/", "%", "**"])):
            errors.append("Direct user input used in math operation; consider sanitizing input.")

        return {
            "success": len(errors) == 0,
            "error": "; ".join(errors) if errors else None
        }


class PlanVerifier:
    """
    PlanVerifier
    -------------
    Validator for Promethyn AGI that ensures:
      - The tool code structure matches the provided plan.
      - Required class name and methods (e.g., 'run') are implemented as specified.
      - Detects structural mismatches between plan and implementation.
    Usage:
        validator = PlanVerifier()
        result = validator(plan, tool_code, test_code)
    """

    def __call__(self, plan: dict, tool_code: str, test_code: str) -> dict:
        """
        Validates that the code matches the plan structure.
        Returns:
            dict: { 'success': bool, 'error': Optional[str] }
        """
        errors = []
        plan_class = plan.get("class")
        plan_methods = plan.get("methods", ["run"])

        # 1. Check if the class is implemented in code
        if plan_class and f"class {plan_class}" not in (tool_code or ""):
            errors.append(f"Class '{plan_class}' not found in tool code.")

        # 2. Check for each required method in the class (default: 'run')
        for method in plan_methods:
            method_signature = f"def {method}("
            if method_signature not in (tool_code or ""):
                errors.append(f"Required method '{method}' not found in tool code.")

        # 3. Check that test code exists if expected
        if not test_code or (not ("def test" in test_code or "class Test" in test_code)):
            errors.append("Test code missing or does not define a test function/class.")

        return {
            "success": len(errors) == 0,
            "error": "; ".join(errors) if errors else None
        }

# Test-mode only validator check (delete after confirming)
if __name__ == "__main__":
    engine = SelfCodingEngine()
    print("Registered validators:", list(engine.validator_registry.keys()))
