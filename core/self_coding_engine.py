import importlib
import sys
import traceback
import os
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple # <-- Added Tuple

from tools.prompt_decomposer import PromptDecomposer
from tools.module_builder import ModuleBuilderTool
from tools.base_tool import BaseTool
from tools.tool_manager import ToolManager
from addons.notebook import AddOnNotebook
from tools.test_tool_runner import TestToolRunner  # <--- NEW IMPORT
from core.validators.extended_validators import register_validators
from core.sandbox_runner import SandboxRunner # <--- ADDED FOR SANDBOX INTEGRATION
from core.utils.path_utils import safe_path_join
from core.utils.validator_importer import import_validator

# --- BEGIN: Updated Validator Import System ---
def import_validator_fallback(name):
    """
    Fallback validator import function using the new import_validator utility.
    Issues warning if validator cannot be imported.
    """
    validator_module = import_validator(name)
    if validator_module is None:
        logging.getLogger("Promethyn.SelfCodingEngine").warning(f"Validator '{name}' could not be imported and will be unavailable.")
    return validator_module
# --- END: Updated Validator Import System ---

# --- Begin: Imports for new validator modules using import_validator ---
try:
    code_quality_module = import_validator("code_quality_assessor")
    CodeQualityAssessor = getattr(code_quality_module, "CodeQualityAssessor", None) if code_quality_module else None
except Exception:
    CodeQualityAssessor = None

try:
    security_scanner_module = import_validator("security_scanner")
    SecurityScanner = getattr(security_scanner_module, "SecurityScanner", None) if security_scanner_module else None
except Exception:
    SecurityScanner = None

try:
    behavioral_simulator_module = import_validator("behavioral_simulator")
    BehavioralSimulator = getattr(behavioral_simulator_module, "BehavioralSimulator", None) if behavioral_simulator_module else None
except Exception:
    BehavioralSimulator = None
# --- End: Imports for new validator modules ---

# --- Import Security Validator as required using import_validator ---
try:
    security_validator_module = import_validator("security_validator")
    validate_security = getattr(security_validator_module, "validate_security", None) if security_validator_module else None
except Exception:
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
        self.sandbox_runner = SandboxRunner() # <--- ADDED FOR SANDBOX INTEGRATION

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
            if name not in self.VALIDATORS and validator_cls is not None and name in self.validator_registry:
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
                                 Or for SecurityValidator: (file_path) -> dict
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

    def _run_validation_pipeline(self, plan: Dict[str, Any], tool_code: str, test_code: str, tool_path: Optional[str], test_path: Optional[str]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Runs a specific sequence of validators:
        PlanVerifier -> MathEvaluator -> CodeQualityAssessor -> SecurityValidator -> TestToolRunner.
        Uses self.validator_registry for most validators, and self.test_runner for TestToolRunner.
        Logs individual validator failures/exceptions to self.notebook.
        Returns:
            - bool: Overall validation success.
            - List[Dict[str, Any]]: A list of dictionaries, each containing 'validator' name and 'result'.
        """
        all_results_summary: List[Dict[str, Any]] = []
        overall_success = True

        validator_names_in_sequence = [
            "PlanVerifier",
            "MathEvaluator",
            "CodeQualityAssessor",
            "SecurityValidator", # Handled via self.validator_registry which holds `validate_security`
        ]

        for validator_name in validator_names_in_sequence:
            validator_instance = self.validator_registry.get(validator_name)
            current_validator_result = None # Define outside try block

            if validator_instance:
                try:
                    if validator_name == "SecurityValidator":
                        if tool_path and callable(validator_instance):
                             current_validator_result = validator_instance(tool_path)
                        elif not tool_path:
                            current_validator_result = {"success": True, "info": f"{validator_name} skipped: tool_path not available."}
                            self.logger.info(f"{validator_name} skipped as tool_path is not available for plan: {plan.get('name', 'N/A')}.")
                        else: # Not callable or other issue
                            current_validator_result = {"success": False, "error": f"{validator_name} is not configured correctly or tool_path missing for plan: {plan.get('name', 'N/A')}."}
                            self.logger.warning(current_validator_result["error"])
                            overall_success = False # A misconfigured critical validator is a failure
                    else:
                        current_validator_result = validator_instance(plan, tool_code, test_code)
                    
                    all_results_summary.append({"validator": validator_name, "result": current_validator_result})
                    
                    if not current_validator_result.get("success", True):
                        overall_success = False
                        if self.notebook:
                            self.notebook.log("validator_failure", {
                                "plan": plan,
                                "validator": validator_name,
                                "error": current_validator_result.get("error", "Unknown validation error")
                            })
                        self.logger.warning(f"Validator '{validator_name}' failed for plan: {plan.get('name', 'N/A')}. Error: {current_validator_result.get('error')}")

                except Exception as e:
                    overall_success = False
                    tb_str = traceback.format_exc()
                    error_result = {"success": False, "error": str(e), "exception": tb_str}
                    all_results_summary.append({"validator": validator_name, "result": error_result})
                    if self.notebook:
                        self.notebook.log("validator_exception", {
                            "plan": plan,
                            "validator": validator_name,
                            "exception": str(e),
                            "traceback": tb_str
                        })
                    self.logger.error(f"Exception in validator '{validator_name}' for plan: {plan.get('name', 'N/A')}. Exception: {str(e)}\n{tb_str}")
            else:
                skip_result = {"success": True, "info": f"Validator '{validator_name}' not found in registry, skipped."}
                all_results_summary.append({"validator": validator_name, "result": skip_result})
                self.logger.warning(f"Validator '{validator_name}' not found in registry for plan: {plan.get('name', 'N/A')}. Skipped.")
                # Depending on criticality, a missing validator might set overall_success = False
                # For now, it's logged and skipped as per "handle missing ones".

        # TestToolRunner (using self.test_runner, runs on test_path)
        if self.test_runner and test_path:
            current_test_runner_result = None # Define outside try block
            try:
                current_test_runner_result = self.test_runner.run_test_file(test_path)
                all_results_summary.append({"validator": "TestToolRunner", "result": current_test_runner_result})

                if not current_test_runner_result.get("passed", False): # TestToolRunner uses "passed"
                    overall_success = False
                    if self.notebook:
                        self.notebook.log("test_tool_runner_failure", {
                            "plan": plan,
                            "test_result": current_test_runner_result,
                            "error": current_test_runner_result.get("error", "TestToolRunner reported failure.")
                        })
                    self.logger.warning(f"TestToolRunner failed for test_path '{test_path}' related to plan {plan.get('name', 'N/A')}. Result: {current_test_runner_result}")
            
            except Exception as e:
                overall_success = False
                tb_str = traceback.format_exc()
                error_result = {"passed": False, "success": False, "error": str(e), "exception": tb_str}
                all_results_summary.append({"validator": "TestToolRunner", "result": error_result})
                if self.notebook:
                    self.notebook.log("test_tool_runner_exception", {
                        "plan": plan,
                        "test_path": test_path,
                        "exception": str(e),
                        "traceback": tb_str
                    })
                self.logger.error(f"Exception in TestToolRunner for test_path '{test_path}' (plan: {plan.get('name', 'N/A')}). Exception: {str(e)}\n{tb_str}")
        elif not self.test_runner:
             all_results_summary.append({"validator": "TestToolRunner", "result": {"passed": True, "success": True, "info": "TestToolRunner not available/configured."}}) # Assume success if not r[...]
             self.logger.info(f"TestToolRunner not available/configured, skipped for plan: {plan.get('name', 'N/A')}.")
        elif not test_path:
             all_results_summary.append({"validator": "TestToolRunner", "result": {"passed": True, "success": True, "info": "TestToolRunner skipped: test_path not available."}}) # Assume success [...]
             self.logger.info(f"TestToolRunner skipped as test_path is not available for plan: {plan.get('name', 'N/A')}.")

        return overall_success, all_results_summary

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
            tool_path = safe_path_join("tools", tool_file) if tool_file else None
            test_path = safe_path_join("test", test_file) if test_file else None  # test/ not tests/
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
                    # If tool_path or tool_code is missing, this is a critical plan failure.
                    # Log it and add to retry_later, then continue to the next plan.
                    missing_info = []
                    if not tool_path: missing_info.append("tool_path")
                    if not tool_code: missing_info.append("tool_code")
                    err_msg = f"Plan is missing critical information: {', '.join(missing_info)}."
                    self.logger.error(f"{err_msg} For plan: {plan}")
                    single_result["registration"] = {"success": False, "error": err_msg}
                    retry_later.append({"plan": plan, "reason": err_msg})
                    results.append(single_result)
                    if self.notebook:
                         self.notebook.log("plan_execution_failure", {"plan": plan, "error": err_msg, "missing": missing_info})
                    continue


                # --- Write the test file ---
                if test_path and test_code:
                    os.makedirs(os.path.dirname(test_path), exist_ok=True)
                    with open(test_path, "w", encoding="utf-8") as f:
                        f.write(test_code)
                # If test_path or test_code is missing, it's not necessarily a fatal error for the tool itself,
                # but validation related to tests will likely fail or be skipped.
                # This was previously raising ValueError, but now we log and proceed.
                elif not test_path and test_code:
                     self.logger.warning(f"Test code provided for plan {plan.get('name', 'N/A')} but no test_path specified. Test code will not be written.")
                elif test_path and not test_code:
                     self.logger.warning(f"Test path {test_path} specified for plan {plan.get('name', 'N/A')} but no test_code provided. Empty or no test file will be written.")


                # --- Promethyn internal standards check ---\
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

                # --- BEGIN: Promethyn AGI Validation Pipeline using _run_validation_pipeline ---
                validation_passed, detailed_validator_outputs = self._run_validation_pipeline(
                    plan, tool_code, test_code, tool_path, test_path
                )

                # Extract TestToolRunner result for logging, if it ran and produced a result
                final_test_run_log_entry = None
                for entry in detailed_validator_outputs:
                    if entry["validator"] == "TestToolRunner" and "result" in entry:
                        final_test_run_log_entry = entry["result"]
                        break
                
                if not validation_passed:
                    fail_msg = "Tool/module rejected due to failed validation or test during pipeline execution."
                    single_result["registration"] = {"success": False, "error": fail_msg}
                    # detailed_validator_outputs already contains specific error reasons from validators
                    # We can augment the reason for retry_later if needed
                    specific_errors = [
                        f"{item['validator']}: {item['result'].get('error', 'Failed')}" 
                        for item in detailed_validator_outputs 
                        if not item['result'].get('success', item['result'].get('passed', True)) # check success or passed
                    ]
                    augmented_fail_msg = f"{fail_msg} Details: {'; '.join(specific_errors)}"
                    retry_later.append({"plan": plan, "reason": augmented_fail_msg})
                    
                    if self.notebook:
                        self.notebook.log("tool_rejected", {
                            "plan": plan,
                            "validator_results": detailed_validator_outputs, 
                            "test_run_result": final_test_run_log_entry, 
                            "error": fail_msg,
                            "detailed_errors": specific_errors
                        })
                    results.append(single_result)
                    continue
                else:
                    if self.notebook:
                        self.notebook.log("tool_validated", { 
                            "plan": plan,
                            "validator_results": detailed_validator_outputs,
                            "test_run_result": final_test_run_log_entry,
                            "status": "success"
                        })
                # --- END: Promethyn AGI Validation Pipeline ---

                # --- Validator hooks (future extensibility) ---
                # This section remains as per instruction to preserve all old logic.
                # It might run some validators that were already part of _run_validation_pipeline.
                validators_passed_secondary_check = True # Renamed to avoid conflict
                for validator_name in self.VALIDATORS: # self.VALIDATORS can be different from the pipeline sequence
                    validator = self.validator_registry.get(validator_name)
                    if validator:
                        # Avoid re-running TestToolRunner if it was just done and part of self.VALIDATORS
                        # However, self.VALIDATORS doesn't typically include "TestToolRunner" by name.
                        # The primary TestToolRunner execution is now within _run_validation_pipeline.
                        try:
                            # SecurityValidator expects tool_path, others (plan, tool_code, test_code)
                            if validator_name == "SecurityValidator":
                                if tool_path and callable(validator):
                                    validator_result = validator(tool_path)
                                else: # Skip if no tool_path or validator misconfigured
                                    validator_result = {"success": True, "info": f"Secondary check for {validator_name} skipped."}
                            else:
                                validator_result = validator(plan, tool_code, test_code)

                            if not validator_result.get("success", True):
                                val_msg = f"Secondary validator hook {validator_name} failed: {validator_result.get('error')}"
                                single_result["registration"] = {"success": False, "error": val_msg}
                                retry_later.append({"plan": plan, "reason": val_msg})
                                if self.notebook:
                                    self.notebook.log("secondary_validator_failure", { # Distinct log key
                                        "plan": plan,
                                        "validator": validator_name,
                                        "error": val_msg,
                                    })
                                self.logger.error(val_msg)
                                results.append(single_result)
                                validators_passed_secondary_check = False
                                break 
                        except Exception as val_ex:
                            tb = traceback.format_exc()
                            val_msg = f"Secondary validator hook {validator_name} raised: {val_ex}\n{tb}"
                            single_result["registration"] = {"success": False, "error": val_msg}
                            retry_later.append({"plan": plan, "reason": val_msg})
                            if self.notebook:
                                self.notebook.log("secondary_validator_exception", { # Distinct log key
                                    "plan": plan,
                                    "validator": validator_name,
                                    "error": val_msg,
                                })
                            self.logger.error(val_msg)
                            results.append(single_result)
                            validators_passed_secondary_check = False
                            break

                if not validators_passed_secondary_check:
                    continue

                # --- Final test: TestToolRunner on generated test file ---
                # This section also remains. It's a specific call to TestToolRunner.
                # The _run_validation_pipeline also includes a TestToolRunner step.
                # This could be redundant if test_path is always present, but preserved for now.
                if test_path: # Ensure test_path exists before calling
                    # Check if TestToolRunner already ran successfully in the main pipeline
                    # to avoid redundant execution if not desired.
                    # For now, preserving original logic means it *may* run again.
                    test_result_secondary = self.test_runner.run_test_file(test_path) # Renamed to avoid conflict
                    if not test_result_secondary.get("passed", False):
                        fail_msg = f"Final TestToolRunner check failed: {test_result_secondary.get('error', test_result_secondary)}"
                        single_result["registration"] = {"success": False, "error": fail_msg}
                        retry_later.append({"plan": plan, "reason": fail_msg})
                        if self.notebook:
                            self.notebook.log("final_test_tool_runner_failure", { # Distinct log key
                                "plan": plan,
                                "test_result": test_result_secondary,
                                "error": fail_msg,
                            })
                            self._log_to_notebook() # Original call
                        self.logger.error(fail_msg)
                        results.append(single_result)
                        continue
                
                # --- BEGIN SANDBOX INTEGRATION ---
                # After the tool is generated and all previous validators pass, run in sandbox.
                sandbox_run_successful = True # Assume true unless sandbox fails or is not applicable
                if tool_path and hasattr(self, 'sandbox_runner') and self.sandbox_runner:
                    self.logger.info(f"Attempting sandbox execution for tool: {tool_path} from plan: {plan.get('name', 'N/A')}")
                    try:
                        # Ensure the argument name matches the SandboxRunner's method signature
                        sandbox_result = self.sandbox_runner.run_python_file_in_sandbox(python_file_path=tool_path) 
                        
                        # Log the full sandbox result to AddOnNotebook
                        if self.notebook:
                            self.notebook.log("sandbox_result", {
                                "plan_name": plan.get('name', 'N/A'), 
                                "tool_path": tool_path, 
                                "result": sandbox_result  # This is the full sandbox result
                            })
                        
                        # Check if sandbox run was successful
                        if not sandbox_result.get("success", False):
                            sandbox_run_successful = False
                            error_details = sandbox_result.get("error", "Sandbox execution indicated failure.")
                            stdout_log = sandbox_result.get("stdout", "")
                            stderr_log = sandbox_result.get("stderr", "")
                            if stdout_log: error_details += f" | stdout: {stdout_log}"
                            if stderr_log: error_details += f" | stderr: {stderr_log}"
                            
                            fail_msg = f"Sandbox execution failed for tool '{tool_path}'. Details: {error_details}"
                            self.logger.error(fail_msg)
                            single_result["registration"] = {"success": False, "error": fail_msg, "sandbox_output": sandbox_result}
                            retry_later.append({"plan": plan, "reason": fail_msg, "details": "sandbox_failure"})
                            results.append(single_result) 
                            if self.notebook: # Log specific failure event for sandbox
                                self.notebook.log("sandbox_execution_failure", {
                                    "plan": plan, 
                                    "tool_path": tool_path, 
                                    "error": fail_msg, 
                                    "sandbox_result_details": sandbox_result # Full result for this specific failure log
                                })
                            continue # Skip to next plan, do not register
                        else:
                            self.logger.info(f"Sandbox execution successful for tool '{tool_path}'. stdout: {sandbox_result.get('stdout', 'N/A')}")
                            # Optionally log specific success if needed for sandbox execution
                            # if self.notebook:
                            #    self.notebook.log("sandbox_execution_success", {"plan": plan, "tool_path": tool_path, "sandbox_result_details": sandbox_result})

                    except Exception as e_sandbox:
                        sandbox_run_successful = False
                        tb_sandbox = traceback.format_exc()
                        fail_msg = f"Exception during sandbox execution of tool '{tool_path}': {str(e_sandbox)}\n{tb_sandbox}"
                        self.logger.error(fail_msg)
                        single_result["registration"] = {"success": False, "error": fail_msg, "exception_details": tb_sandbox}
                        retry_later.append({"plan": plan, "reason": fail_msg, "details": "sandbox_exception"})
                        results.append(single_result)
                        if self.notebook: # Log specific exception event for sandbox
                            self.notebook.log("sandbox_execution_exception", {
                                "plan": plan, 
                                "tool_path": tool_path, 
                                "error": str(e_sandbox), 
                                "traceback": tb_sandbox
                            })
                        continue # Skip to next plan, do not register
                elif not tool_path and hasattr(self, 'sandbox_runner') and self.sandbox_runner:
                    # Log if sandbox was skipped due to no tool_path, but don't mark as failure for this reason alone.
                    self.logger.info(f"Sandbox execution skipped for plan {plan.get('name', 'N/A')} as tool_path is not available.")
                
                # If sandbox_run_successful is False due to an actual failure (handled by 'continue' above), 
                # subsequent steps including registration will be skipped.
                # --- END SANDBOX INTEGRATION ---

                # --- AGI EXTENSION: Enhanced Validator Audit Logging ---\
                # This audit log remains, it iterates self.VALIDATORS.
                for validator_name in self.VALIDATORS:
                    validator = self.validator_registry.get(validator_name)
                    if validator:
                        try:
                            # Handle SecurityValidator signature
                            if validator_name == "SecurityValidator":
                                if tool_path and callable(validator):
                                    validator_result = validator(tool_path)
                                else:
                                    validator_result = {"success": True, "info": f"Audit for {validator_name} skipped."}
                            else:
                                validator_result = validator(plan, tool_code, test_code)
                            audit_log = {
                                "plan": plan,
                                "validator": validator_name,
                                "result": validator_result,
                            }
                            if self.notebook:
                                self.notebook.log("validator_audit", audit_log)
                            self.logger.debug(f"Validator '{validator_name}' audit outcome: {validator_result}")
                        except Exception as val_ex:
                            audit_log = {
                                "plan": plan,
                                "validator": validator_name,
                                "exception": str(val_ex),
                            }
                            if self.notebook:
                                self.notebook.log("validator_audit_exception", audit_log)
                            self.logger.debug(f"Validator '{validator_name}' exception during audit: {val_ex}")

                # --- Dynamic import of tool module and class ---\
                module_path = tool_path[:-3].replace("/", ".").replace("\\\\", ".") if tool_path and tool_path.endswith(".py") else (tool_path.replace("/", ".").replace("\\\\", ".") if tool_path else [...]
                if not module_path:
                    raise ValueError(f"Cannot determine module path from tool_path: {tool_path}")

                if module_path in sys.modules:
                    del sys.modules[module_path] # Force reload
                module = importlib.import_module(module_path)
                tool_class = getattr(module, class_name, None)
                if tool_class is None:
                    raise ImportError(f"Class '{class_name}' not found in '{module_path}'.")

                # --- Instantiate tool and validate by running test ---\
                tool_instance = tool_class()
                if not hasattr(tool_instance, "run"):
                    raise AttributeError(f"Tool '{class_name}' does not implement a 'run' method.")

                # Validate using run("test")
                try:
                    validation_result = tool_instance.run("test") # This is the tool's own test method
                    single_result["validation"] = validation_result # Store this specific validation
                    if self.notebook:
                        self.notebook.log("tool_run_test_method", { # Changed log key for clarity
                            "plan": plan,
                            "result": validation_result,
                            "status": "success"
                        })
                    self.logger.info(f"Tool's run('test') method for '{class_name}' succeeded: {validation_result}")
                    if not (isinstance(validation_result, dict) and validation_result.get("success", False)):
                        fail_msg = f"Tool's run('test') method validation failed: {validation_result}"
                        single_result["registration"] = {"success": False, "error": fail_msg}
                        retry_later.append({"plan": plan, "reason": fail_msg})
                        if self.notebook:
                            self.notebook.log("tool_run_test_method_failure", { # Changed log key
                                "plan": plan,
                                "validation": validation_result,
                                "error": fail_msg,
                            })
                        self.logger.error(fail_msg)
                        results.append(single_result)
                        continue

                except Exception as val_err:
                    tb = traceback.format_exc()
                    fail_msg = f"Tool's run('test') method raised exception: {val_err}\n{tb}"
                    single_result["registration"] = {"success": False, "error": fail_msg}
                    retry_later.append({"plan": plan, "reason": fail_msg})
                    if self.notebook:
                        self.notebook.log("tool_run_test_method_exception", { # Changed log key
                            "plan": plan,
                            "error": fail_msg,
                        })
                    self.logger.error(f"Tool's run('test') for '{class_name}' raised exception: {fail_msg}")
                    results.append(single_result)
                    continue

                # --- Register tool if requested ---\
                # This is reached only if all prior checks, including sandbox, passed.
                reg_result = self._register_tool(plan, tool_manager)
                single_result["registration"] = reg_result

                # --- Log success to ShortTermMemory if available ---\
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
                fail_msg = f"Module build or validation error for plan {plan.get('name', 'N/A')}: {e}\n{tb}"
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

        # --- Log retry_later to memory or AddOnNotebook ---\
        if retry_later:
            retry_log = {"retry_later": retry_later, "prompt": prompt}
            if short_term_memory is not None:
                short_term_memory.setdefault("tool_retry_queue", []).extend(retry_later)
            if self.notebook:
                self.notebook.log("tool_retry_later", retry_log)
            for retry_item in retry_later: # Renamed to avoid conflict
                self._schedule_retry(retry_item["plan"], retry_item["reason"])

        return {
            "success": len(retry_later) == 0 and any(r.get("registration", {}).get("success") for r in results if "registration" in r), # More robust success check
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
            file_path = plan.get("file") # This is tool_file, effectively tool_path basis
            class_name = plan.get("class")
            if not file_path or not class_name:
                msg = "Missing 'file' or 'class' in plan for tool registration."
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg})
                self.logger.error(msg)
                return {"success": False, "error": msg}

            # Convert file path to module path (e.g. tools/time_tracker.py -> tools.time_tracker)
            # Assuming file_path is relative to repo root, e.g., "tools/my_tool.py"
            module_path = file_path[:-3].replace("/", ".").replace("\\\\", ".") if file_path.endswith(".py") else file_path.replace("/", ".").replace("\\\\", ".")
            
            # Remove module from sys.modules if it's already loaded (force reload)
            if module_path in sys.modules:
                del sys.modules[module_path]

            try:
                module = importlib.import_module(module_path)
            except Exception as imp_exc:
                tb_imp = traceback.format_exc()
                msg = f"Failed to import module '{module_path}': {imp_exc}\n{tb_imp}"
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg, "module_path": module_path})
                self.logger.error(msg)
                return {"success": False, "error": msg}

            tool_class = getattr(module, class_name, None)
            if tool_class is None:
                msg = f"Class '{class_name}' not found in module '{module_path}'."
                print(f"[Tool Registration] {msg}")
                if self.notebook:
                    self.notebook.log("tool_registration_failure", {"plan": plan, "error": msg, "module_path": module_path, "class_name": class_name})
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
            err_msg = f"Tool registration exception for plan {plan.get('name', 'N/A')}: {e}\n{tb}"
            print(f"[Tool Registration] Exception: {err_msg}")
            if self.notebook:
                self.notebook.log("tool_registration_failure", {"plan": plan, "error": str(e), "traceback": tb})
            self.logger.error(err_msg)
            return {"success": False, "error": str(e), "traceback": tb}

    # --- Internal standards enforcement for Promethyn AGI tools ---\
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
        if plan.get("class") and plan.get("class") not in (tool_code or ""): # Check if class exists in plan first
            errors.append("Tool class does not match plan or is missing.")
        elif not plan.get("class") and "class " not in (tool_code or ""): # If no class in plan, expect one in code
             errors.append("No class definition found in tool code.")


        # Testability: Must have test code and 'run' method
        if not test_code or ("def test" not in test_code and "class Test" not in test_code):
            errors.append("No proper test defined or missing test function/class.")
        if "def run(" not in (tool_code or ""):
            errors.append("No 'run' method implemented in tool.")

        # TODO: Add more robust AST-based and pattern-based checks for safety, modularity, testability

        return errors

    # --- Placeholder for future caching mechanism ---\
    def _cache_result(self, key, value):
        """
        TODO: Implement caching (e.g., Redis, in-memory, disk) for tool generation and validation results.
        """
        pass

    # --- AGI EXTENSION: Retry Scheduling Telemetry ---\
    def _schedule_retry(self, plan, reason):
        """
        TODO: Implement exponential backoff retry system for failed tool generations/validations.
        """
        self.logger.warning(f"Retry scheduled for plan {plan.get('name','Unnamed Plan')} due to: {reason}")
        if self.notebook:
            self.notebook.log("retry_scheduled", {"plan": plan, "reason": reason})

    # --- Hook for future multi-phase planning ---\
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
        plan_methods = plan.get("methods", ["run"]) # Default to checking for "run"

        # 1. Check if the class is implemented in code
        if plan_class and f"class {plan_class}" not in (tool_code or ""):
            errors.append(f"Class '{plan_class}' from plan not found in tool code.")
        elif not plan_class and "class " not in (tool_code or ""): # If no class in plan, still expect some class
             errors.append("No class definition found in tool code, and no class specified in plan.")


        # 2. Check for each required method in the class (default: 'run')
        # This check is basic; it doesn't ensure method is in the correct class if multiple classes exist.
        for method_name in plan_methods:
            method_signature = f"def {method_name}("
            if method_signature not in (tool_code or ""):
                errors.append(f"Required method '{method_name}' from plan not found in tool code.")

        # 3. Check that test code exists if expected by plan or convention
        # Plan might specify test_code presence or TestToolRunner might expect it.
        # This is a basic check for test code content.
        if not test_code or (not ("def test" in test_code or "class Test" in test_code)):
            errors.append("Test code missing or does not define a recognizable test function/class.")

        return {
            "success": len(errors) == 0,
            "error": "; ".join(errors) if errors else None
        }

# Test-mode only validator check (delete after confirming)
if __name__ == "__main__":
    # This is a basic test stub, needs more elaborate setup for full testing
    print("SelfCodingEngine module loaded. Constructing instance for basic check...")
    
    # Mock notebook for testing
    class MockNotebook:
        def __init__(self):
            self.logs = []
        def log(self, event_type, data):
            self.logs.append({"type": event_type, "data": data})
            print(f"[MockNotebook] Event: {event_type}, Data: {data}")

    mock_notebook_instance = MockNotebook()
    engine = SelfCodingEngine(notebook=mock_notebook_instance)
    print("Registered validators:", list(engine.validator_registry.keys()))
    print("VALIDATORS list:", engine.VALIDATORS)

    # Example of how one might test process_prompt (requires more setup for files etc.)
    # test_prompt = "Create a simple calculator tool that can add two numbers."
    # print(f"\nTesting process_prompt with: '{test_prompt}'")
    # results = engine.process_prompt(test_prompt)
    # print("process_prompt results:", results)
    # print("\nMock Notebook Logs:")
    # for log_entry in mock_notebook_instance.logs:
    #     print(log_entry)
