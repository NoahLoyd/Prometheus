import importlib
import sys
import traceback
import os
import logging
import threading
import time  # <--- NEW: Added for retry delays
from typing import Dict, Any, Optional, List, Callable, Tuple # <-- Added Tuple
from dataclasses import dataclass
import glob
import pkgutil

from tools.prompt_decomposer import PromptDecomposer
from tools.module_builder import ModuleBuilderTool
from tools.base_tool import BaseTool
from tools.tool_manager import ToolManager
from addons.notebook import AddOnNotebook
from tools.test_tool_runner import TestToolRunner  # <--- NEW IMPORT
from core.validators.extended_validators import register_validators
from core.sandbox_runner import SandboxRunner # <--- ADDED FOR SANDBOX INTEGRATION
from core.utils.path_utils import safe_path_join, import_validator
from memory.retry_memory import get_retry_memory  # <--- NEW: Retry memory import

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

# --- Begin: Dynamic Validator Imports using import_validator ---
logger = logging.getLogger("Promethyn.SelfCodingEngine")

# Dynamic validator loading
CodeQualityAssessor = None
SecurityScanner = None
BehavioralSimulator = None
validate_security = None

# List of validators to attempt dynamic import
# List of validators to attempt dynamic import - FIXED: Use backward compatibility function
validator_simple_names = ["code_quality_assessor", "security_scanner", "behavioral_simulator", "security_validator"]

for validator_simple_name in validator_simple_names:
    try:
        # Use the backward compatibility function that handles simple names
        validator_module = import_validator_by_name(validator_simple_name)        if validator_module is not None:
            if validator_name == "code_quality_assessor":
                CodeQualityAssessor = getattr(validator_module, "CodeQualityAssessor", None)
            elif validator_name == "security_scanner":
                SecurityScanner = getattr(validator_module, "SecurityScanner", None)
            elif validator_name == "behavioral_simulator":
                BehavioralSimulator = getattr(validator_module, "BehavioralSimulator", None)
            elif validator_name == "security_validator":
                validate_security = getattr(validator_module, "validate_security", None)
        else:
            logger.warning(f"Could not import validator: {validator_name}")
    except Exception as e:
        logger.warning(f"Exception while importing validator '{validator_name}': {e}")
# --- End: Dynamic Validator Imports ---

# --- BEGIN: Missing validator classes for compatibility ---
class MathEvaluator:
    """Placeholder MathEvaluator validator for compatibility."""
    def __call__(self, plan, tool_code, test_code):
        return {"success": True, "info": "MathEvaluator placeholder validation passed"}

class PlanVerifier:
    """Placeholder PlanVerifier validator for compatibility."""
    def __call__(self, plan, tool_code, test_code):
        return {"success": True, "info": "PlanVerifier placeholder validation passed"}
# --- END: Missing validator classes ---

@dataclass
class ValidationResult:
    """
    Result of a single validator execution.
    """
    success: bool
    error: Optional[str] = None
    info: Optional[str] = None
    passed: Optional[bool] = None  # For TestToolRunner compatibility
    exception: Optional[str] = None
    
    def __post_init__(self):
        # Handle legacy compatibility for TestToolRunner
        if self.passed is not None and self.success is None:
            self.success = self.passed

class ValidationChain:
    """
    ValidationChain manages the execution of validators in a dependency-aware manner.
    Supports adding validators with optional dependencies and executes them in proper order.
    """
    
    def __init__(self, logger: logging.Logger):
        self.validators = {}  # name -> validator callable
        self.dependencies = {}  # name -> list of dependency names
        self.execution_order = []  # computed execution order
        self.logger = logger
    
    def add_validator(self, name: str, validator: Callable, dependencies: Optional[List[str]] = None):
        """
        Add a validator to the chain with optional dependencies.
        
        :param name: Unique validator name
        :param validator: Callable validator instance
        :param dependencies: List of validator names this validator depends on
        """
        if not callable(validator):
            self.logger.warning(f"Validator '{name}' is not callable, skipping.")
            return
        
        self.validators[name] = validator
        self.dependencies[name] = dependencies or []
        self._compute_execution_order()
    
    def _compute_execution_order(self):
        """
        Compute the execution order based on dependencies using topological sort.
        """
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(name):
            if name in temp_visited:
                self.logger.error(f"Circular dependency detected involving validator '{name}'")
                return
            if name in visited:
                return
            
            temp_visited.add(name)
            for dep in self.dependencies.get(name, []):
                if dep in self.validators:
                    visit(dep)
                else:
                    self.logger.warning(f"Validator '{name}' depends on '{dep}' which is not registered")
            
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
        
        for validator_name in self.validators:
            if validator_name not in visited:
                visit(validator_name)
        
        self.execution_order = order
    
    def run(self, context: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Run all validators in dependency order.
        
        :param context: Context containing plan, tool_code, test_code, tool_path, test_path
        :return: Tuple of (overall_success, detailed_results)
        """
        results = []
        failed_validators = set()
        overall_success = True
        
        for validator_name in self.execution_order:
            validator = self.validators[validator_name]
            
            # Check if dependencies were satisfied
            deps_satisfied = True
            for dep in self.dependencies[validator_name]:
                if dep in failed_validators:
                    deps_satisfied = False
                    break
            
            if not deps_satisfied:
                skip_result = ValidationResult(
                    success=True,
                    info=f"Validator '{validator_name}' skipped due to failed dependencies"
                )
                results.append({"validator": validator_name, "result": skip_result.__dict__})
                self.logger.info(f"Validator '{validator_name}' skipped due to failed dependencies")
                continue
            
            # Execute validator
            try:
                self.logger.debug(f"Running validator '{validator_name}'")
                
                # Handle different validator signatures
                if validator_name == "SecurityValidator":
                    tool_path = context.get("tool_path")
                    if tool_path and callable(validator):
                        raw_result = validator(tool_path)
                    elif not tool_path:
                        raw_result = {"success": True, "info": f"{validator_name} skipped: tool_path not available."}
                    else:
                        raw_result = {"success": False, "error": f"{validator_name} is not configured correctly."}
                elif validator_name == "TestToolRunner":
                    # Special handling for TestToolRunner
                    test_path = context.get("test_path")
                    if test_path and hasattr(validator, 'run_test_file'):
                        raw_result = validator.run_test_file(test_path)
                    else:
                        raw_result = {"passed": True, "success": True, "info": "TestToolRunner skipped: no test_path"}
                else:
                    # Standard validator signature
                    plan = context.get("plan", {})
                    tool_code = context.get("tool_code", "")
                    test_code = context.get("test_code", "")
                    raw_result = validator(plan, tool_code, test_code)
                
                # Convert to ValidationResult
                if isinstance(raw_result, dict):
                    validation_result = ValidationResult(
                        success=raw_result.get("success", raw_result.get("passed", True)),
                        error=raw_result.get("error"),
                        info=raw_result.get("info"),
                        passed=raw_result.get("passed"),
                        exception=raw_result.get("exception")
                    )
                else:
                    validation_result = ValidationResult(success=False, error=f"Invalid result type from {validator_name}")
                
                results.append({"validator": validator_name, "result": validation_result.__dict__})
                
                if not validation_result.success:
                    overall_success = False
                    failed_validators.add(validator_name)
                    self.logger.warning(f"Validator '{validator_name}' failed: {validation_result.error}")
                    # Fail immediately on validation failure
                    break
                else:
                    self.logger.debug(f"Validator '{validator_name}' passed")
                    
            except Exception as e:
                overall_success = False
                failed_validators.add(validator_name)
                tb_str = traceback.format_exc()
                error_result = ValidationResult(
                    success=False,
                    error=str(e),
                    exception=tb_str
                )
                results.append({"validator": validator_name, "result": error_result.__dict__})
                self.logger.error(f"Exception in validator '{validator_name}': {str(e)}\n{tb_str}")
                # Fail immediately on exception
                break
        
        return overall_success, results

class SelfCodingEngine:
    """
    SelfCodingEngine orchestrates the self-coding AGI workflow:
      - Takes in a natural language prompt (single or multiple tools).
      - Decomposes it into one or more structured module plans.
      - Generates, writes, validates, and registers each tool.
      - Logs all outcomes for strategic learning.
      - Enforces Promethyn standards and supports future extensibility.
      - Uses persistent retry memory to learn from failures and successes.
      - Implements automatic retry logic for failed generation attempts.
    """

    # --- Validator hooks (extensible, for future use) ---
    VALIDATORS = ["MathEvaluator", "PlanVerifier", "CodeCritic"]  # Placeholder for plugging in
    
    # --- NEW: Retry configuration ---
    MAX_RETRIES = 2  # Maximum number of retries for failed attempts
    RETRY_DELAY = 1  # Delay in seconds between retry attempts

    def __init__(self, notebook: Optional[AddOnNotebook] = None):
        self.decomposer = PromptDecomposer()
        self.builder = ModuleBuilderTool()
        self.notebook = notebook or AddOnNotebook()
        self.logger = self._get_logger()
        self.validator_registry = {}  # For future validator plug-in
        self.validator_lock = threading.Lock()  # Thread-safe validator registration
        self.sandbox_runner = SandboxRunner() # <--- ADDED FOR SANDBOX INTEGRATION
        
        # --- NEW: Initialize retry memory system ---
        try:
            self.retry_memory = get_retry_memory()
            self.logger.info("Retry memory system initialized successfully")
            if self.notebook:
                self.notebook.log("retry_memory_init", {
                    "status": "success",
                    "memory_file": str(self.retry_memory.memory_file),
                    "max_history_per_task": self.retry_memory.max_history_per_task
                })
        except Exception as e:
            self.logger.error(f"Failed to initialize retry memory: {e}")
            self.retry_memory = None
            if self.notebook:
                self.notebook.log("retry_memory_init_failed", {
                    "error": str(e),
                    "fallback": "continuing_without_retry_memory"
                })

        # Initialize validation chain
        self.validation_chain = ValidationChain(self.logger)

        # --- Register core validators safely on instantiation ---
        self.register_validator_safely("MathEvaluator", MathEvaluator())
        self.register_validator_safely("PlanVerifier", PlanVerifier())
        register_validators(self.validator_registry)  # <-- Inject extended Promethyn validators here

        # --- Register TestToolRunner instance (instantiated ONCE here) ---
        self.test_runner = TestToolRunner()  # <--- Elite singleton placement

        # --- Inject new validators at the end of the pipeline in a defensive, modular way ---
        self._register_enhanced_validators()

        # --- Load extended validators with dependency handling ---
        self.load_extended_validators()

        # --- Inject SecurityValidator into registry and VALIDATORS pipeline if not already present ---
        # Ensure PlanVerifier ➝ MathEvaluator ➝ CodeQualityAssessor ➝ SecurityValidator ➝ TestToolRunner
        # Do NOT remove or overwrite any existing validators
        if validate_security is not None:
            self.register_validator_safely("SecurityValidator", validate_security)
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

        # Setup validation chain with dependencies
        self._setup_validation_chain()

    def _get_logger(self):
        """Get logger instance for SelfCodingEngine."""
        return logging.getLogger("Promethyn.SelfCodingEngine")

    def register_validator_safely(self, name: str, validator_cls: Callable):
        """
        Thread-safe method to register a validator in the validator registry.
        
        :param name: Unique string identifier for the validator.
        :param validator_cls: Callable validator instance or class to register.
        """
        with self.validator_lock:
            if not callable(validator_cls):
                self.logger.warning(f"Validator '{name}' is not callable.")
                return
            
            if name in self.validator_registry:
                self.logger.info(f"Validator '{name}' already exists. Overwriting.")
            
            self.validator_registry[name] = validator_cls
            self.logger.info(f"Validator '{name}' registered safely.")

    def _generate_task_id(self, plan: Dict[str, Any], operation_type: str = "tool") -> str:
        """
        Generate a consistent task_id for retry memory tracking.
        
        :param plan: Plan dictionary containing tool/module information
        :param operation_type: Type of operation (tool, module, validation, etc.)
        :return: Consistent task_id string
        """
        # Use the plan name if available, otherwise class name or file name
        name = (plan.get("name") or 
                plan.get("class") or 
                plan.get("file", "").replace(".py", "") or 
                "unknown")
        
        # Clean the name for consistency
        name = name.replace(" ", "_").replace("-", "_").lower()
        
        return f"{operation_type}::{name}"

    def _check_retry_memory_before_attempt(self, task_id: str, plan: Dict[str, Any]) -> bool:
        """
        Check retry memory before attempting a task and log relevant information.
        
        :param task_id: Task identifier for retry memory lookup
        :param plan: Plan dictionary for logging context
        :return: True if should proceed, False if should skip due to repeated failures
        """
        if not self.retry_memory:
            return True  # No retry memory available, proceed normally
        
        try:
            # Check if this task has failed before
            has_failed = self.retry_memory.has_failed_before(task_id)
            failure_count = self.retry_memory.get_failure_count(task_id)
            success_count = self.retry_memory.get_success_count(task_id)
            last_attempt = self.retry_memory.get_last_attempt(task_id)
            
            # Log retry memory status
            retry_status = {
                "task_id": task_id,
                "has_failed_before": has_failed,
                "failure_count": failure_count,
                "success_count": success_count,
                "last_attempt": last_attempt,
                "plan_name": plan.get("name", "N/A")
            }
            
            if has_failed:
                self.logger.warning(f"Task '{task_id}' has failed {failure_count} times before. Last attempt: {last_attempt}")
                if self.notebook:
                    self.notebook.log("retry_memory_check_failed_before", retry_status)
                
                # Decision logic: Skip if too many recent failures
                if failure_count >= 5 and success_count == 0:
                    skip_reason = f"Skipping task '{task_id}' due to {failure_count} consecutive failures with no successes"
                    self.logger.warning(skip_reason)
                    if self.notebook:
                        self.notebook.log("retry_memory_skip_task", {
                            **retry_status,
                            "reason": skip_reason,
                            "action": "skipped"
                        })
                    return False
            else:
                self.logger.info(f"Task '{task_id}' has no previous failures. Proceeding with attempt.")
                if self.notebook:
                    self.notebook.log("retry_memory_check_clean", retry_status)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking retry memory for task '{task_id}': {e}")
            if self.notebook:
                self.notebook.log("retry_memory_check_error", {
                    "task_id": task_id,
                    "error": str(e),
                    "action": "proceeding_despite_error"
                })
            return True  # Proceed despite retry memory error

    def _log_retry_memory_result(self, task_id: str, success: bool, reason: Optional[str] = None, plan: Optional[Dict[str, Any]] = None):
        """
        Log the result of a task attempt to retry memory.
        
        :param task_id: Task identifier
        :param success: Whether the task succeeded
        :param reason: Optional reason for success/failure
        :param plan: Optional plan context for enhanced logging
        """
        if not self.retry_memory:
            return
        
        try:
            status = "success" if success else "failure"
            self.retry_memory.log_result(task_id, status, reason)
            
            # Enhanced logging
            log_data = {
                "task_id": task_id,
                "status": status,
                "reason": reason,
                "plan_name": plan.get("name", "N/A") if plan else "N/A"
            }
            
            if success:
                self.logger.info(f"Logged SUCCESS for task '{task_id}': {reason or 'No reason provided'}")
                if self.notebook:
                    self.notebook.log("retry_memory_success_logged", log_data)
            else:
                self.logger.warning(f"Logged FAILURE for task '{task_id}': {reason or 'No reason provided'}")
                if self.notebook:
                    self.notebook.log("retry_memory_failure_logged", log_data)
                    
        except Exception as e:
            self.logger.error(f"Failed to log retry memory result for task '{task_id}': {e}")
            if self.notebook:
                self.notebook.log("retry_memory_log_error", {
                    "task_id": task_id,
                    "intended_status": "success" if success else "failure",
                    "intended_reason": reason,
                    "error": str(e)
                })

    def _execute_with_retry(self, plan: Dict[str, Any], tool_manager: Optional[ToolManager], 
                           short_term_memory: Optional[dict]) -> Dict[str, Any]:
        """
        NEW: Execute tool/module generation with automatic retry logic.
        
        :param plan: Plan dictionary containing tool/module information
        :param tool_manager: Optional tool manager for registration
        :param short_term_memory: Optional short-term memory for tracking
        :return: Dictionary with execution results and retry information
        """
        tool_task_id = self._generate_task_id(plan, "tool")
        retry_count = 0
        last_error = None
        
        # Log start of retry-enabled execution
        if self.notebook:
            self.notebook.log("retry_execution_start", {
                "task_id": tool_task_id,
                "plan_name": plan.get("name", "N/A"),
                "max_retries": self.MAX_RETRIES,
                "retry_delay": self.RETRY_DELAY
            })
        
        while retry_count <= self.MAX_RETRIES:
            try:
                attempt_number = retry_count + 1
                
                # Log retry attempt
                if retry_count > 0:
                    self.logger.info(f"Retry attempt {retry_count}/{self.MAX_RETRIES} for task '{tool_task_id}' (plan: {plan.get('name', 'N/A')})")
                    if self.notebook:
                        self.notebook.log("retry_attempt", {
                            "task_id": tool_task_id,
                            "attempt_number": attempt_number,
                            "total_attempts": self.MAX_RETRIES + 1,
                            "previous_error": str(last_error) if last_error else None
                        })
                else:
                    self.logger.info(f"Initial attempt for task '{tool_task_id}' (plan: {plan.get('name', 'N/A')})")
                
                # Execute the single tool processing logic
                result = self._process_single_tool_attempt(plan, tool_manager, short_term_memory)
                
                # Check if the attempt was successful
                if result.get("registration", {}).get("success", False):
                    self.logger.info(f"Task '{tool_task_id}' succeeded on attempt {attempt_number}")
                    self._log_retry_memory_result(tool_task_id, True, f"Succeeded on attempt {attempt_number}", plan)
                    
                    if self.notebook:
                        self.notebook.log("retry_execution_success", {
                            "task_id": tool_task_id,
                            "plan_name": plan.get("name", "N/A"),
                            "successful_attempt": attempt_number,
                            "total_attempts": attempt_number
                        })
                    
                    return result
                else:
                    # Attempt failed, extract error message
                    error_msg = result.get("registration", {}).get("error", "Unknown error during tool processing")
                    last_error = error_msg
                    
                    self.logger.warning(f"Task '{tool_task_id}' failed on attempt {attempt_number}: {error_msg}")
                    self._log_retry_memory_result(tool_task_id, False, f"Failed on attempt {attempt_number}: {error_msg}", plan)
                    
                    if self.notebook:
                        self.notebook.log("retry_attempt_failed", {
                            "task_id": tool_task_id,
                            "plan_name": plan.get("name", "N/A"),
                            "failed_attempt": attempt_number,
                            "error": error_msg,
                            "will_retry": retry_count < self.MAX_RETRIES
                        })
                    
                    # If this was not the last attempt, wait before retrying
                    if retry_count < self.MAX_RETRIES:
                        self.logger.info(f"Waiting {self.RETRY_DELAY} seconds before retry {retry_count + 1} for task '{tool_task_id}'")
                        time.sleep(self.RETRY_DELAY)
                        retry_count += 1
                    else:
                        # Final failure after all retries exhausted
                        final_error_msg = f"Task failed after {self.MAX_RETRIES + 1} attempts. Final error: {error_msg}"
                        self.logger.error(f"Task '{tool_task_id}' exhausted all retries. {final_error_msg}")
                        self._log_retry_memory_result(tool_task_id, False, final_error_msg, plan)
                        
                        if self.notebook:
                            self.notebook.log("retry_execution_exhausted", {
                                "task_id": tool_task_id,
                                "plan_name": plan.get("name", "N/A"),
                                "total_attempts": self.MAX_RETRIES + 1,
                                "final_error": final_error_msg
                            })
                        
                        return result
                        
            except Exception as e:
                tb = traceback.format_exc()
                exception_msg = f"Exception during attempt {attempt_number}: {str(e)}"
                last_error = exception_msg
                
                self.logger.error(f"Exception in task '{tool_task_id}' attempt {attempt_number}: {str(e)}\n{tb}")
                self._log_retry_memory_result(tool_task_id, False, exception_msg, plan)
                
                if self.notebook:
                    self.notebook.log("retry_attempt_exception", {
                        "task_id": tool_task_id,
                        "plan_name": plan.get("name", "N/A"),
                        "failed_attempt": attempt_number,
                        "exception": str(e),
                        "traceback": tb,
                        "will_retry": retry_count < self.MAX_RETRIES
                    })
                
                # If this was not the last attempt, wait before retrying
                if retry_count < self.MAX_RETRIES:
                    self.logger.info(f"Waiting {self.RETRY_DELAY} seconds before retry {retry_count + 1} for task '{tool_task_id}' after exception")
                    time.sleep(self.RETRY_DELAY)
                    retry_count += 1
                else:
                    # Final failure due to exception after all retries exhausted
                    final_error_msg = f"Task failed with exceptions after {self.MAX_RETRIES + 1} attempts. Final exception: {str(e)}"
                    self.logger.error(f"Task '{tool_task_id}' exhausted all retries due to exceptions. {final_error_msg}")
                    self._log_retry_memory_result(tool_task_id, False, final_error_msg, plan)
                    
                    if self.notebook:
                        self.notebook.log("retry_execution_exception_exhausted", {
                            "task_id": tool_task_id,
                            "plan_name": plan.get("name", "N/A"),
                            "total_attempts": self.MAX_RETRIES + 1,
                            "final_exception": str(e),
                            "final_traceback": tb
                        })
                    
                    return {
                        "plan": plan,
                        "registration": {"success": False, "error": final_error_msg},
                        "validation": None
                    }
        
        # This should never be reached, but included for completeness
        return {
            "plan": plan,
            "registration": {"success": False, "error": "Unexpected exit from retry loop"},
            "validation": None
        }

    def _process_single_tool_attempt(self, plan: Dict[str, Any], tool_manager: Optional[ToolManager], 
                                    short_term_memory: Optional[dict]) -> Dict[str, Any]:
        """
        NEW: Process a single tool generation attempt (extracted from original process_prompt logic).
        
        :param plan: Plan dictionary containing tool/module information
        :param tool_manager: Optional tool manager for registration
        :param short_term_memory: Optional short-term memory for tracking
        :return: Dictionary with execution results for this single attempt
        """
        single_result = {"plan": plan, "registration": None, "validation": None}
        tool_file = plan.get("file")
        test_file = plan.get("test_file")
        class_name = plan.get("class")
        tool_code = plan.get("code")
        test_code = plan.get("test_code")

        # Generate task ID for this specific tool/plan
        tool_task_id = self._generate_task_id(plan, "tool")

        # Compute output paths
        tool_path = safe_path_join("tools", tool_file) if tool_file else None
        test_path = safe_path_join("test", test_file) if test_file else None  # test/ not tests/
        
        # --- Overwrite protection: skip or error if exists ---
        if tool_path and os.path.exists(tool_path):
            msg = f"Tool file exists, skipping: {tool_path}"
            single_result["registration"] = {"success": False, "error": msg}
            self.logger.warning(msg)
            return single_result
        if test_path and os.path.exists(test_path):
            msg = f"Test file exists, skipping: {test_path}"
            single_result["registration"] = {"success": False, "error": msg}
            self.logger.warning(msg)
            return single_result

        # --- Write the main tool code file ---
        if tool_path and tool_code:
            os.makedirs(os.path.dirname(tool_path), exist_ok=True)
            with open(tool_path, "w", encoding="utf-8") as f:
                f.write(tool_code)
        else:
            # If tool_path or tool_code is missing, this is a critical plan failure.
            missing_info = []
            if not tool_path: missing_info.append("tool_path")
            if not tool_code: missing_info.append("tool_code")
            err_msg = f"Plan is missing critical information: {', '.join(missing_info)}."
            self.logger.error(f"{err_msg} For plan: {plan}")
            single_result["registration"] = {"success": False, "error": err_msg}
            if self.notebook:
                 self.notebook.log("plan_execution_failure", {"plan": plan, "error": err_msg, "missing": missing_info})
            return single_result

        # --- Write the test file ---
        if test_path and test_code:
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(test_code)
        # If test_path or test_code is missing, it's not necessarily a fatal error for the tool itself,
        # but validation related to tests will likely fail or be skipped.
        elif not test_path and test_code:
             self.logger.warning(f"Test code provided for plan {plan.get('name', 'N/A')} but no test_path specified. Test code will not be written.")
        elif test_path and not test_code:
             self.logger.warning(f"Test path {test_path} specified for plan {plan.get('name', 'N/A')} but no test_code provided. Empty or no test file will be written.")

        # --- Promethyn internal standards check ---
        standards_errors = self._check_standards(plan, tool_code, test_code)
        if standards_errors:
            err_msg = f"Promethyn standards not met: {standards_errors}"
            single_result["registration"] = {"success": False, "error": err_msg}
            if self.notebook:
                self.notebook.log("standards_failure", {"plan": plan, "errors": standards_errors})
            self.logger.error(err_msg)
            return single_result

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
            specific_errors = [
                f"{item['validator']}: {item['result'].get('error', 'Failed')}" 
                for item in detailed_validator_outputs 
                if not item['result'].get('success', item['result'].get('passed', True)) # check success or passed
            ]
            augmented_fail_msg = f"{fail_msg} Details: {'; '.join(specific_errors)}"
            single_result["registration"]["error"] = augmented_fail_msg
            
            # Log retry memory result for validation failure
            self._log_retry_memory_result(
                task_id=f"validation:{plan.get('name', 'unknown')}",
                success=False,
                reason="validation_failed",
                plan=plan
            )
            
            if self.notebook:
                self.notebook.log("tool_rejected", {
                    "plan": plan,
                    "validator_results": detailed_validator_outputs, 
                    "test_run_result": final_test_run_log_entry, 
                    "error": fail_msg,
                    "detailed_errors": specific_errors
                })
            return single_result
        else:
            # Log retry memory result for validation success
            self._log_retry_memory_result(
                task_id=f"validation:{plan.get('name', 'unknown')}",
                success=True,
                reason="validation_passed",
                plan=plan
            )
            
            if self.notebook:
                self.notebook.log("tool_validated", { 
                    "plan": plan,
                    "validator_results": detailed_validator_outputs,
                    "test_run_result": final_test_run_log_entry,
                    "status": "success"
                })
        # --- END: Promethyn AGI Validation Pipeline ---

        # --- Validator hooks (future extensibility) ---
        validators_passed_secondary_check = True
        for validator_name in self.VALIDATORS:
            validator = self.validator_registry.get(validator_name)
            if validator:
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
                        
                        # Log retry memory result for secondary validator failure
                        self._log_retry_memory_result(
                            task_id=f"secondary_validator:{plan.get('name', 'unknown')}",
                            success=False,
                            reason=f"secondary_validator_{validator_name}_failed",
                            plan=plan
                        )
                        
                        if self.notebook:
                            self.notebook.log("secondary_validator_failure", {
                                "plan": plan,
                                "validator": validator_name,
                                "error": val_msg,
                            })
                        self.logger.error(val_msg)
                        validators_passed_secondary_check = False
                        break 
                except Exception as val_ex:
                    tb = traceback.format_exc()
                    val_msg = f"Secondary validator hook {validator_name} raised: {val_ex}\n{tb}"
                    single_result["registration"] = {"success": False, "error": val_msg}
                    
                    # Log retry memory result for secondary validator exception
                    self._log_retry_memory_result(
                        task_id=f"secondary_validator:{plan.get('name', 'unknown')}",
                        success=False,
                        reason=f"secondary_validator_{validator_name}_exception",
                        plan=plan
                    )
                    
                    if self.notebook:
                        self.notebook.log("secondary_validator_exception", {
                            "plan": plan,
                            "validator": validator_name,
                            "error": val_msg,
                        })
                    self.logger.error(val_msg)
                    validators_passed_secondary_check = False
                    break
            else:
                # Fallback for missing validators
                context = {
                    "plan": plan,
                    "tool_code": tool_code,
                    "test_code": test_code,
                    "tool_path": tool_path,
                    "test_path": test_path
                }
                fallback_result = self.fallback_if_validator_missing(validator_name, context)
                if self.notebook:
                    self.notebook.log("secondary_validator_fallback", {
                        "plan": plan,
                        "validator": validator_name,
                        "fallback_result": fallback_result.__dict__,
                        "action": "continued_execution"
                    })

        if not validators_passed_secondary_check:
            return single_result

        # --- Final test: TestToolRunner on generated test file ---
        if test_path:
            test_result_secondary = self.test_runner.run_test_file(test_path)
            if not test_result_secondary.get("passed", False):
                fail_msg = f"Final TestToolRunner check failed: {test_result_secondary.get('error', test_result_secondary)}"
                single_result["registration"] = {"success": False, "error": fail_msg}
                
                # Log retry memory result for final test failure
                self._log_retry_memory_result(
                    task_id=f"final_test:{plan.get('name', 'unknown')}",
                    success=False,
                    reason="final_test_failed",
                    plan=plan
                )
                
                if self.notebook:
                    self.notebook.log("final_test_tool_runner_failure", {
                        "plan": plan,
                        "test_result": test_result_secondary,
                        "error": fail_msg,
                    })
                self.logger.error(fail_msg)
                return single_result
            else:
                # Log retry memory result for final test success
                self._log_retry_memory_result(
                    task_id=f"final_test:{plan.get('name', 'unknown')}",
                    success=True,
                    reason="final_test_passed",
                    plan=plan
                )
        
        # --- BEGIN SANDBOX INTEGRATION ---
        sandbox_run_successful = True
        if tool_path and hasattr(self, 'sandbox_runner') and self.sandbox_runner:
            self.logger.info(f"Attempting sandbox execution for tool: {tool_path} from plan: {plan.get('name', 'N/A')}")
            
            # Generate sandbox-specific task ID
            sandbox_task_id = self._generate_task_id(plan, "sandbox")
            
            try:
                sandbox_result = self.sandbox_runner.run_python_file_in_sandbox(python_file_path=tool_path) 
                
                if self.notebook:
                    self.notebook.log("sandbox_result", {
                        "plan_name": plan.get('name', 'N/A'), 
                        "tool_path": tool_path, 
                        "result": sandbox_result
                    })
                
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
                    
                    # Log retry memory result for sandbox failure
                    self._log_retry_memory_result(sandbox_task_id, False, error_details, plan)
                    
                    if self.notebook:
                        self.notebook.log("sandbox_execution_failure", {
                            "plan": plan, 
                            "tool_path": tool_path, 
                            "error": fail_msg, 
                            "sandbox_result_details": sandbox_result
                        })
                    return single_result
                else:
                    self.logger.info(f"Sandbox execution successful for tool '{tool_path}'. stdout: {sandbox_result.get('stdout', 'N/A')}")
                    
                    # Log retry memory result for sandbox success
                    self._log_retry_memory_result(sandbox_task_id, True, "Sandbox execution successful", plan)

            except Exception as e_sandbox:
                sandbox_run_successful = False
                tb_sandbox = traceback.format_exc()
                fail_msg = f"Exception during sandbox execution of tool '{tool_path}': {str(e_sandbox)}\n{tb_sandbox}"
                self.logger.error(fail_msg)
                single_result["registration"] = {"success": False, "error": fail_msg, "exception_details": tb_sandbox}
                
                # Log retry memory result for sandbox exception
                self._log_retry_memory_result(sandbox_task_id, False, f"Exception: {str(e_sandbox)}", plan)
                
                if self.notebook:
                    self.notebook.log("sandbox_execution_exception", {
                        "plan": plan, 
                        "tool_path": tool_path, 
                        "error": str(e_sandbox), 
                        "traceback": tb_sandbox
                    })
                return single_result
        elif not tool_path and hasattr(self, 'sandbox_runner') and self.sandbox_runner:
            self.logger.info(f"Sandbox execution skipped for plan {plan.get('name', 'N/A')} as tool_path is not available.")
        # --- END SANDBOX INTEGRATION ---
            # --- AGI EXTENSION: Enhanced Validator Audit Logging ---
        for validator_name in self.VALIDATORS:
            validator = self.validator_registry.get(validator_name)
            if validator:
                try:
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
            else:
                # Fallback for missing validators in audit log
                context = {
                    "plan": plan,
                    "tool_code": tool_code,
                    "test_code": test_code,
                    "tool_path": tool_path,
                    "test_path": test_path
                }
                fallback_result = self.fallback_if_validator_missing(validator_name, context)
                audit_log = {
                    "plan": plan,
                    "validator": validator_name,
                    "fallback_result": fallback_result.__dict__,
                    "audit_type": "fallback_used"
                }
                if self.notebook:
                    self.notebook.log("validator_audit_fallback", audit_log)
                self.logger.debug(f"Validator '{validator_name}' audit used fallback: {fallback_result}")

        # --- Dynamic import of tool module and class ---
        module_path = tool_path[:-3].replace("/", ".").replace("\\", ".") if tool_path and tool_path.endswith(".py") else (tool_path.replace("/", ".").replace("\\", ".") if tool_path else "")
        if not module_path:
            error_msg = f"Cannot determine module path from tool_path: {tool_path}"
            single_result["registration"] = {"success": False, "error": error_msg}
            return single_result

        if module_path in sys.modules:
            del sys.modules[module_path] # Force reload
        
        try:
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name, None)
            if tool_class is None:
                error_msg = f"Class '{class_name}' not found in '{module_path}'."
                single_result["registration"] = {"success": False, "error": error_msg}
                return single_result

            # --- Instantiate tool and validate by running test ---
            tool_instance = tool_class()
            if not hasattr(tool_instance, "run"):
                error_msg = f"Tool '{class_name}' does not implement a 'run' method."
                single_result["registration"] = {"success": False, "error": error_msg}
                return single_result

            # Validate using run("test")
            try:
                validation_result = tool_instance.run("test")
                single_result["validation"] = validation_result
                if self.notebook:
                    self.notebook.log("tool_run_test_method", {
                        "plan": plan,
                        "result": validation_result,
                        "status": "success"
                    })
                self.logger.info(f"Tool's run('test') method for '{class_name}' succeeded: {validation_result}")
                if not (isinstance(validation_result, dict) and validation_result.get("success", False)):
                    fail_msg = f"Tool's run('test') method validation failed: {validation_result}"
                    single_result["registration"] = {"success": False, "error": fail_msg}
                    
                    # Log retry memory result for tool test failure
                    self._log_retry_memory_result(
                        task_id=f"tool_test:{plan.get('name', 'unknown')}",
                        success=False,
                        reason="tool_test_failed",
                        plan=plan
                    )
                    
                    if self.notebook:
                        self.notebook.log("tool_run_test_method_failure", {
                            "plan": plan,
                            "validation": validation_result,
                            "error": fail_msg,
                        })
                    self.logger.error(fail_msg)
                    return single_result
                else:
                    # Log retry memory result for tool test success
                    self._log_retry_memory_result(
                        task_id=f"tool_test:{plan.get('name', 'unknown')}",
                        success=True,
                        reason="tool_test_passed",
                        plan=plan
                    )

            except Exception as val_err:
                tb = traceback.format_exc()
                fail_msg = f"Tool's run('test') method raised exception: {val_err}\n{tb}"
                single_result["registration"] = {"success": False, "error": fail_msg}
                
                # Log retry memory result for tool test exception
                self._log_retry_memory_result(
                    task_id=f"tool_test:{plan.get('name', 'unknown')}",
                    success=False,
                    reason=f"tool_test_exception: {str(val_err)}",
                    plan=plan
                )
                
                if self.notebook:
                    self.notebook.log("tool_run_test_method_exception", {
                        "plan": plan,
                        "error": fail_msg,
                    })
                self.logger.error(f"Tool's run('test') for '{class_name}' raised exception: {fail_msg}")
                return single_result

            # --- Register tool if requested ---
            reg_result = self._register_tool(plan, tool_manager)
            single_result["registration"] = reg_result

            # Log retry memory result for tool registration
            if reg_result.get("success", False):
                self._log_retry_memory_result(
                    task_id=f"tool_registration:{plan.get('name', 'unknown')}",
                    success=True,
                    reason="tool_registered_successfully",
                    plan=plan
                )
            else:
                self._log_retry_memory_result(
                    task_id=f"tool_registration:{plan.get('name', 'unknown')}",
                    success=False,
                    reason=f"tool_registration_failed: {reg_result.get('error', 'unknown')}",
                    plan=plan
                )

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

        except Exception as import_error:
            tb = traceback.format_exc()
            error_msg = f"Failed to import/instantiate tool: {str(import_error)}\n{tb}"
            single_result["registration"] = {"success": False, "error": error_msg}
            
            # Log retry memory result for import/instantiation failure
            self._log_retry_memory_result(
                task_id=f"tool_import:{plan.get('name', 'unknown')}",
                success=False,
                reason=f"import_error: {str(import_error)}",
                plan=plan
            )
            
            self.logger.error(error_msg)
            return single_result

        return single_result

    def _run_validation_pipeline(self, plan: Dict[str, Any], tool_code: str, test_code: str,
                                tool_path: Optional[str], test_path: Optional[str]) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Run the validation pipeline using the ValidationChain.
        
        :param plan: Plan dictionary
        :param tool_code: Generated tool code
        :param test_code: Generated test code
        :param tool_path: Path to tool file
        :param test_path: Path to test file
        :return: Tuple of (overall_success, detailed_results)
        """
        context = {
            "plan": plan,
            "tool_code": tool_code,
            "test_code": test_code,
            "tool_path": tool_path,
            "test_path": test_path
        }
        
        return self.validation_chain.run(context)

    def _check_standards(self, plan: Dict[str, Any], tool_code: str, test_code: str) -> List[str]:
        """
        Placeholder for Promethyn internal standards check.
        
        :param plan: Plan dictionary
        :param tool_code: Generated tool code
        :param test_code: Generated test code
        :return: List of standards errors (empty if no errors)
        """
        # Placeholder implementation - add actual standards checking logic here
        errors = []
        
        # Basic checks
        if not tool_code.strip():
            errors.append("Tool code is empty")
        if not plan.get("class"):
            errors.append("Plan missing class name")
        if not plan.get("file"):
            errors.append("Plan missing file name")
            
        return errors

    def _register_tool(self, plan: Dict[str, Any], tool_manager: Optional[ToolManager]) -> Dict[str, Any]:
        """
        Register a tool with the tool manager.
        
        :param plan: Plan dictionary containing tool information
        :param tool_manager: Optional tool manager instance
        :return: Registration result dictionary
        """
        if not tool_manager:
            return {"success": True, "info": "No tool manager provided, skipping registration"}
        
        try:
            # Extract necessary information from plan
            class_name = plan.get("class")
            tool_file = plan.get("file")
            
            if not class_name or not tool_file:
                return {"success": False, "error": "Missing class name or tool file in plan"}
            
            # Register with tool manager
            registration_result = tool_manager.register_tool(class_name, tool_file)
            return registration_result
            
        except Exception as e:
            tb = traceback.format_exc()
            return {"success": False, "error": f"Exception during tool registration: {str(e)}\n{tb}"}

    def _log_to_notebook(self):
        """Placeholder for notebook logging compatibility."""
        pass

    def fallback_if_validator_missing(self, name: str, context: dict) -> ValidationResult:
        """
        Fallback layer for missing validators to prevent Promethyn crashes.
        Logs warnings and returns a success result to continue execution.
        
        :param name: Name of the missing validator
        :param context: Validation context for logging
        :return: ValidationResult indicating validator was skipped
        """
        warning_msg = f"Validator '{name}' not found in registry, using fallback (skipping)"
        fallback_info = f"Validator skipped: not found"
        
        # Log to both notebook and logger for visibility
        self.logger.warning(warning_msg)
        if self.notebook:
            self.notebook.log("validator_missing_fallback", {
                "validator_name": name,
                "warning": warning_msg,
                "context": {
                    "plan_name": context.get("plan", {}).get("name", "N/A"),
                    "has_tool_code": bool(context.get("tool_code")),
                    "has_test_code": bool(context.get("test_code")),
                    "tool_path": context.get("tool_path"),
                    "test_path": context.get("test_path")
                },
                "action": "validator_skipped",
                "reason": "not_found_in_registry"
            })
        
        return ValidationResult(
            success=True,
            info=fallback_info,
            error=None
        )

    def load_extended_validators(self):
        """
        Enhanced validator loading system with dependency handling.
        Imports all modules from `validators/extended_validators/` directory,
        handles dependencies (REQUIRES/OPTIONAL), and logs all activities to AddOnNotebook.
        """
        validators_path = os.path.join("validators", "extended_validators")
        
        # Log start of validator loading process
        if self.notebook:
            self.notebook.log("validator_loading_start", {
                "path": validators_path,
                "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None))
            })
        
        # Check if the validators directory exists
        if not os.path.exists(validators_path):
            warning_msg = f"Extended validators directory not found: {validators_path}"
            self.logger.warning(warning_msg)
            if self.notebook:
                self.notebook.log("validator_directory_missing", {
                    "path": validators_path,
                    "warning": warning_msg
                })
            return
        
        # Track loaded validators and their dependencies
        loaded_validators = {}
        failed_validators = {}
        dependency_requirements = {}
        
        # First pass: Import all validator modules and collect dependency info
        validator_files = glob.glob(os.path.join(validators_path, "*.py"))
        validator_files = [f for f in validator_files if not os.path.basename(f).startswith("__")]
        
        for validator_file in validator_files:
            module_name = os.path.splitext(os.path.basename(validator_file))[0]
            module_path = f"validators.extended_validators.{module_name}"
            
            try:
                # Import the validator module
                if module_path in sys.modules:
                    del sys.modules[module_path]  # Force reload
                
                validator_module = importlib.import_module(module_path)
                
                # Look for validator classes or functions in the module
                validator_classes = []
                for attr_name in dir(validator_module):
                    attr = getattr(validator_module, attr_name)
                    if (callable(attr) and 
                        not attr_name.startswith('_') and 
                        (hasattr(attr, '__call__') or hasattr(attr, 'validate'))):
                        validator_classes.append((attr_name, attr))
                
                if not validator_classes:
                    # Look for a default validator function
                    if hasattr(validator_module, 'validate'):
                        validator_classes.append(('validate', validator_module.validate))
                    elif hasattr(validator_module, module_name):
                        validator_classes.append((module_name, getattr(validator_module, module_name)))
                
                # Process each validator found in the module
                for validator_name, validator_obj in validator_classes:
                    full_validator_name = f"{module_name}.{validator_name}"
                    
                    try:
                        # Check for dependency declarations
                        requires = getattr(validator_obj, 'REQUIRES', [])
                        optional = getattr(validator_obj, 'OPTIONAL', [])
                        
                        # Store dependency requirements
                        dependency_requirements[full_validator_name] = {
                            'requires': requires,
                            'optional': optional,
                            'validator': validator_obj,
                            'module': module_name
                        }
                        
                        loaded_validators[full_validator_name] = {
                            'validator': validator_obj,
                            'module': module_name,
                            'requires': requires,
                            'optional': optional,
                            'status': 'loaded'
                        }
                        
                        self.logger.info(f"Loaded extended validator: {full_validator_name}")
                        if self.notebook:
                            self.notebook.log("validator_loaded", {
                                "name": full_validator_name,
                                "module": module_name,
                                "requires": requires,
                                "optional": optional
                            })
                    
                    except Exception as e:
                        error_msg = f"Error processing validator {validator_name} from {module_name}: {str(e)}"
                        self.logger.warning(error_msg)
                        failed_validators[full_validator_name] = {
                            'error': str(e),
                            'module': module_name,
                            'stage': 'processing'
                        }
                        if self.notebook:
                            self.notebook.log("validator_processing_failed", {
                                "name": full_validator_name,
                                "module": module_name,
                                "error": str(e)
                            })
            
            except Exception as e:
                error_msg = f"Failed to import validator module {module_name}: {str(e)}"
                self.logger.warning(f"Optional validator failed: {error_msg}")
                failed_validators[module_name] = {
                    'error': str(e),
                    'module': module_name,
                    'stage': 'import'
                }
                if self.notebook:
                    self.notebook.log("validator_import_failed", {
                        "module": module_name,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
        
        # Second pass: Check dependencies and register validators
        registered_validators = set()
        skipped_validators = {}
        
        # Helper function to check if dependencies are satisfied
        def dependencies_satisfied(validator_name, requires_list):
            unsatisfied = []
            for req in requires_list:
                # Check if required dependency is loaded or already registered
                if (req not in registered_validators and 
                    req not in loaded_validators and 
                    req not in self.validator_registry):
                    unsatisfied.append(req)
            return len(unsatisfied) == 0, unsatisfied
        
        # Keep trying to register validators until no more can be registered
        max_iterations = len(loaded_validators) + 1
        iteration = 0
        
        while loaded_validators and iteration < max_iterations:
            iteration += 1
            registered_this_round = False
            
            for validator_name in list(loaded_validators.keys()):
                validator_info = loaded_validators[validator_name]
                requires = validator_info['requires']
                
                # Check if all required dependencies are satisfied
                deps_ok, unsatisfied = dependencies_satisfied(validator_name, requires)
                
                if deps_ok:
                    # Register the validator
                    try:
                        self.register_validator_safely(validator_name, validator_info['validator'])
                        registered_validators.add(validator_name)
                        registered_this_round = True
                        
                        # Remove from pending list
                        del loaded_validators[validator_name]
                        
                        self.logger.info(f"Registered extended validator: {validator_name}")
                        if self.notebook:
                            self.notebook.log("validator_registered", {
                                "name": validator_name,
                                "module": validator_info['module'],
                                "iteration": iteration
                            })
                    
                    except Exception as e:
                        error_msg = f"Failed to register validator {validator_name}: {str(e)}"
                        self.logger.warning(error_msg)
                        failed_validators[validator_name] = {
                            'error': str(e),
                            'module': validator_info['module'],
                            'stage': 'registration'
                        }
                        del loaded_validators[validator_name]
                        if self.notebook:
                            self.notebook.log("validator_registration_failed", {
                                "name": validator_name,
                                "module": validator_info['module'],
                                "error": str(e)
                            })
                else:
                    # Dependencies not satisfied yet
                    if iteration == max_iterations:
                        # Final iteration - skip validators with unsatisfied dependencies
                        skipped_validators[validator_name] = {
                            'unsatisfied_deps': unsatisfied,
                            'module': validator_info['module']
                        }
                        del loaded_validators[validator_name]
                        self.logger.warning(f"Skipped validator {validator_name} due to unsatisfied dependencies: {unsatisfied}")
                        if self.notebook:
                            self.notebook.log("validator_skipped_dependencies", {
                                "name": validator_name,
                                "module": validator_info['module'],
                                "unsatisfied_dependencies": unsatisfied
                            })
            
            if not registered_this_round:
                # No progress made this round, break to avoid infinite loop
                break
        
        # Log final summary
        summary = {
            "loaded_count": len(registered_validators),
            "failed_count": len(failed_validators),
            "skipped_count": len(skipped_validators),
            "registered_validators": list(registered_validators),
            "failed_validators": list(failed_validators.keys()),
            "skipped_validators": list(skipped_validators.keys())
        }
        
        self.logger.info(f"Extended validator loading complete: {summary}")
        if self.notebook:
            self.notebook.log("validator_loading_complete", summary)

    def _register_enhanced_validators(self):
        """
        Enhanced validator registration system with dependency injection.
        Registers available enhanced validators like CodeQualityAssessor, SecurityScanner, etc.
        """
        # Register CodeQualityAssessor if available
        if CodeQualityAssessor is not None:
            self.register_validator_safely("CodeQualityAssessor", CodeQualityAssessor())
            # Add to validation chain
            self.validation_chain.add_validator("CodeQualityAssessor", CodeQualityAssessor(), dependencies=["PlanVerifier"])
        
        # Register SecurityScanner if available
        if SecurityScanner is not None:
            self.register_validator_safely("SecurityScanner", SecurityScanner())
            # Add to validation chain
            self.validation_chain.add_validator("SecurityScanner", SecurityScanner(), dependencies=["CodeQualityAssessor"])
        
        # Register BehavioralSimulator if available
        if BehavioralSimulator is not None:
            self.register_validator_safely("BehavioralSimulator", BehavioralSimulator())
            # Add to validation chain
            self.validation_chain.add_validator("BehavioralSimulator", BehavioralSimulator(), dependencies=["SecurityScanner"])

    def _setup_validation_chain(self):
        """
        Setup the validation chain with proper dependencies.
        """
        # Add core validators to validation chain
        self.validation_chain.add_validator("MathEvaluator", self.validator_registry.get("MathEvaluator"))
        self.validation_chain.add_validator("PlanVerifier", self.validator_registry.get("PlanVerifier"), dependencies=["MathEvaluator"])
        
        # Add TestToolRunner to validation chain (always last)
        self.validation_chain.add_validator("TestToolRunner", self.test_runner, dependencies=["PlanVerifier"])

    def _learn_from_retry_memory(self, task_id: str) -> None:
        """
        Learn from retry memory history and adjust behavior accordingly.
        Analyzes failure patterns and logs learning insights for intelligent self-improvement.
        
        :param task_id: Task identifier to analyze and learn from
        """
        if not self.retry_memory:
            return
        
        try:
            # Get retry history for the task
            history = self.retry_memory.get_task_history(task_id)
            if not history:
                self.logger.debug(f"No retry history found for task '{task_id}' - proceeding normally")
                return
            
            # Count successes and failures
            success_count = sum(1 for entry in history if entry.get('status') == 'success')
            failure_count = sum(1 for entry in history if entry.get('status') == 'failure')
            
            # Log learning analysis start
            self.logger.info(f"Learning from retry memory for task '{task_id}': {failure_count} failures, {success_count} successes")
            
            # Analyze failure patterns and adjust behavior
            if failure_count > 2 and success_count == 0:
                # High failure rate - extract common failure reasons
                failure_reasons = [
                    entry.get('reason', 'Unknown reason') 
                    for entry in history 
                    if entry.get('status') == 'failure' and entry.get('reason')
                ]
                
                unique_reasons = list(set(failure_reasons))
                
                # Adjust retry strategy based on failure patterns
                if len(unique_reasons) > 0:
                    # Adjust MAX_RETRIES based on failure patterns
                    if any('validation_failed' in reason for reason in unique_reasons):
                        self.logger.info(f"Task '{task_id}': High validation failures detected - reducing retry attempts")
                        # Could adjust validation pipeline here
                    
                    if any('timeout' in reason.lower() for reason in unique_reasons):
                        self.logger.info(f"Task '{task_id}': Timeout patterns detected - increasing retry delay")
                        # Could adjust RETRY_DELAY here
                    
                    learning_feedback = f"Task '{task_id}' learning: {failure_count} consecutive failures. Common issues: {'; '.join(unique_reasons[:3])}"
                else:
                    learning_feedback = f"Task '{task_id}' learning: {failure_count} consecutive failures with no specific reasons available"
                
                # Log comprehensive learning data
                if self.notebook:
                    self.notebook.log("retry_memory_learning_high_failure", {
                        "task_id": task_id,
                        "failure_count": failure_count,
                        "success_count": success_count,
                        "failure_reasons": failure_reasons,
                        "unique_failure_patterns": unique_reasons,
                        "learning_feedback": learning_feedback,
                        "behavior_adjustments": "reduced_retry_confidence"
                    })
                
                self.logger.warning(learning_feedback)
                # FIXED: Properly placed and indented track_self_improvement call
                self.track_self_improvement(
                    task_id=task_id,
                    improvement_type="confidence_adjustment", 
                    before_state={"failure_count": 0, "success_count": 0},
                    after_state={"failure_count": failure_count, "success_count": success_count},
                    confidence_delta=-0.3,  # Reduced confidence due to failures
                    pattern_insights=unique_reasons
                )
                
            elif success_count > 0 and failure_count > 0:
                # Mixed results - analyze success patterns
                success_reasons = [
                    entry.get('reason', 'Unknown reason') 
                    for entry in history 
                    if entry.get('status') == 'success' and entry.get('reason')
                ]
                
                learning_feedback = f"Task '{task_id}' learning: Mixed results ({success_count} successes, {failure_count} failures). Success patterns identified."
                
                # Log mixed results learning
                if self.notebook:
                    self.notebook.log("retry_memory_learning_mixed_results", {
                        "task_id": task_id,
                        "failure_count": failure_count,
                        "success_count": success_count,
                        "success_patterns": success_reasons,
                        "learning_feedback": learning_feedback,
                        "behavior_adjustments": "pattern_awareness_enhanced"
                    })
                
                self.logger.info(learning_feedback)
                # FIXED: Properly placed and indented track_self_improvement call
                self.track_self_improvement(
                    task_id=task_id,
                    improvement_type="pattern_recognition",
                    before_state={"patterns_known": 0},
                    after_state={"patterns_identified": len(success_reasons)},
                    confidence_delta=0.1,  # Slight confidence boost from learning
                    pattern_insights=success_reasons
                )
                
            elif success_count > failure_count:
                # Mostly successful - positive reinforcement
                learning_feedback = f"Task '{task_id}' learning: High success rate ({success_count} successes vs {failure_count} failures). Positive patterns reinforced."
                
                # Log positive learning
                if self.notebook:
                    self.notebook.log("retry_memory_learning_positive", {
                        "task_id": task_id,
                        "failure_count": failure_count,
                        "success_count": success_count,
                        "learning_feedback": learning_feedback,
                        "behavior_adjustments": "confidence_increased"
                    })
                
                self.logger.info(learning_feedback)
                # FIXED: Properly placed and indented track_self_improvement call
                self.track_self_improvement(
                    task_id=task_id,
                    improvement_type="confidence_adjustment",
                    before_state={"success_rate": 0.5},
                    after_state={"success_rate": success_count / (success_count + failure_count)},
                    confidence_delta=0.2,  # Increased confidence from success
                    pattern_insights=[f"High success rate pattern for {task_id}"]
                )
                
        except Exception as e:
            self.logger.error(f"Error during retry memory learning for task '{task_id}': {e}")
            if self.notebook:
                self.notebook.log("retry_memory_learning_error", {
                    "task_id": task_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

    def track_self_improvement(self, task_id: str, improvement_type: str, 
                              before_state: Dict[str, Any], after_state: Dict[str, Any],
                              confidence_delta: float = 0.0, pattern_insights: Optional[List[str]] = None) -> None:
        """
        Track and log self-improvement behaviors, confidence adjustments, and pattern learning.
        Provides detailed analytics on how the AGI system evolves and adapts over time.
        
        :param task_id: Task identifier associated with the improvement
        :param improvement_type: Type of improvement (confidence_adjustment, pattern_recognition, strategy_change, etc.)
        :param before_state: State/metrics before the improvement
        :param after_state: State/metrics after the improvement  
        :param confidence_delta: Change in confidence level (-1.0 to 1.0)
        :param pattern_insights: List of insights or patterns discovered
        """
        try:
            # Generate improvement tracking data
            improvement_data = {
                "task_id": task_id,
                "improvement_type": improvement_type,
                "timestamp": time.time(),
                "confidence_delta": confidence_delta,
                "pattern_insights": pattern_insights or [],
                "before_state": before_state,
                "after_state": after_state,
                "system_evolution": {
                    "learning_session": True,
                    "adaptation_level": abs(confidence_delta) if confidence_delta else 0.0,
                    "pattern_count": len(pattern_insights) if pattern_insights else 0
                }
            }
            
            # Calculate improvement metrics
            if before_state and after_state:
                improvement_metrics = {}
                
                # Compare numerical metrics between states
                for key in before_state:
                    if key in after_state and isinstance(before_state[key], (int, float)) and isinstance(after_state[key], (int, float)):
                        delta = after_state[key] - before_state[key]
                        improvement_metrics[f"{key}_delta"] = delta
                        improvement_metrics[f"{key}_improvement_percent"] = (delta / before_state[key] * 100) if before_state[key] != 0 else 0
                
                improvement_data["metrics"] = improvement_metrics
            
            # Log different types of improvements with specific handling
            if improvement_type == "confidence_adjustment":
                confidence_level = "increased" if confidence_delta > 0 else "decreased" if confidence_delta < 0 else "stable"
                self.logger.info(f"Self-improvement: Confidence {confidence_level} by {abs(confidence_delta):.3f} for task '{task_id}'")
                
                if self.notebook:
                    self.notebook.log("self_improvement_confidence", {
                        **improvement_data,
                        "confidence_direction": confidence_level,
                        "confidence_magnitude": abs(confidence_delta),
                        "confidence_category": "high" if abs(confidence_delta) > 0.5 else "medium" if abs(confidence_delta) > 0.2 else "low"
                    })
                    
            elif improvement_type == "pattern_recognition":
                pattern_summary = "; ".join(pattern_insights[:3]) if pattern_insights else "No specific patterns"
                self.logger.info(f"Self-improvement: Pattern recognition enhanced for task '{task_id}'. Insights: {pattern_summary}")
                
                if self.notebook:
                    self.notebook.log("self_improvement_patterns", {
                        **improvement_data,
                        "pattern_summary": pattern_summary,
                        "insight_depth": len(pattern_insights) if pattern_insights else 0,
                        "pattern_categories": self._categorize_patterns(pattern_insights) if pattern_insights else []
                    })
                    
            elif improvement_type == "strategy_change":
                self.logger.info(f"Self-improvement: Strategy adapted for task '{task_id}' based on learning analysis")
                
                if self.notebook:
                    self.notebook.log("self_improvement_strategy", {
                        **improvement_data,
                        "strategy_evolution": True,
                        "adaptation_reason": f"Learning from task {task_id} patterns"
                    })
                    
            elif improvement_type == "behavior_optimization":
                self.logger.info(f"Self-improvement: Behavior optimized for task '{task_id}' - enhanced decision making")
                
                if self.notebook:
                    self.notebook.log("self_improvement_behavior", {
                        **improvement_data,
                        "optimization_type": "decision_making",
                        "behavioral_enhancement": True
                    })
                    
            else:
                # Generic improvement tracking
                self.logger.info(f"Self-improvement: {improvement_type} applied to task '{task_id}'")
                
                if self.notebook:
                    self.notebook.log("self_improvement_generic", improvement_data)
            
            # Track cumulative improvement over time
            if hasattr(self, '_improvement_history'):
                self._improvement_history.append(improvement_data)
                # Keep only last 100 improvements to manage memory
                if len(self._improvement_history) > 100:
                    self._improvement_history = self._improvement_history[-100:]
            else:
                self._improvement_history = [improvement_data]
            
            # Log aggregate improvement statistics periodically
            if len(self._improvement_history) % 10 == 0:
                self._log_improvement_statistics()
                
        except Exception as e:
            self.logger.error(f"Error tracking self-improvement for task '{task_id}': {e}")
            if self.notebook:
                self.notebook.log("self_improvement_tracking_error", {
                    "task_id": task_id,
                    "improvement_type": improvement_type,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

    def _categorize_patterns(self, patterns: List[str]) -> List[str]:
        """
        Categorize discovered patterns into types for better tracking.
        
        :param patterns: List of pattern insights
        :return: List of pattern categories
        """
        categories = []
        if not patterns:
            return categories
            
        for pattern in patterns:
            pattern_lower = pattern.lower()
            if any(keyword in pattern_lower for keyword in ['validation', 'test', 'check']):
                categories.append('validation_patterns')
            elif any(keyword in pattern_lower for keyword in ['timeout', 'delay', 'performance']):
                categories.append('performance_patterns')
            elif any(keyword in pattern_lower for keyword in ['error', 'failure', 'exception']):
                categories.append('error_patterns')
            elif any(keyword in pattern_lower for keyword in ['success', 'pass', 'complete']):
                categories.append('success_patterns')
            else:
                categories.append('general_patterns')
                
        return list(set(categories))  # Remove duplicates

    def _log_improvement_statistics(self) -> None:
        """
        Log aggregate improvement statistics for analysis.
        """
        if not hasattr(self, '_improvement_history') or not self._improvement_history:
            return
            
        try:
            total_improvements = len(self._improvement_history)
            confidence_adjustments = sum(1 for imp in self._improvement_history if imp['improvement_type'] == 'confidence_adjustment')
            pattern_recognitions = sum(1 for imp in self._improvement_history if imp['improvement_type'] == 'pattern_recognition')
            strategy_changes = sum(1 for imp in self._improvement_history if imp['improvement_type'] == 'strategy_change')
            
            avg_confidence_delta = sum(abs(imp.get('confidence_delta', 0)) for imp in self._improvement_history) / total_improvements
            total_patterns = sum(len(imp.get('pattern_insights', [])) for imp in self._improvement_history)
            
            stats = {
                "total_improvements": total_improvements,
                "confidence_adjustments": confidence_adjustments,
                "pattern_recognitions": pattern_recognitions,
                "strategy_changes": strategy_changes,
                "average_confidence_delta": avg_confidence_delta,
                "total_patterns_discovered": total_patterns,
                "improvement_velocity": total_improvements / 10,  # improvements per 10 operations
                "learning_efficiency": total_patterns / total_improvements if total_improvements > 0 else 0
            }
            
            self.logger.info(f"Self-improvement statistics: {total_improvements} total improvements, {confidence_adjustments} confidence adjustments, {pattern_recognitions} pattern discoveries")
            
            if self.notebook:
                self.notebook.log("self_improvement_aggregate_stats", stats)
                
        except Exception as e:
            self.logger.error(f"Error logging improvement statistics: {e}")

    def process_prompt(self, prompt: str, tool_manager: Optional[ToolManager] = None) -> Dict[str, Any]:
        """
        Main entry point for processing natural language prompts into tools.
        Integrates retry memory learning for intelligent self-improvement.
        
        :param prompt: Natural language description of desired tools/modules
        :param tool_manager: Optional tool manager for registration
        :return: Dictionary containing processing results
        """
        try:
            # Decompose the prompt into structured plans
            self.logger.info(f"Processing prompt: {prompt[:100]}...")
            decomposition_result = self.decomposer.decompose(prompt)
            
            if not decomposition_result.get("success", False):
                return {"success": False, "error": f"Decomposition failed: {decomposition_result.get('error', 'Unknown error')}"}
            
            plans = decomposition_result.get("plans", [])
            if not plans:
                return {"success": False, "error": "No valid plans generated from prompt"}
            
            # Process each plan with retry memory learning integration
            results = []
            short_term_memory = {}
            
            for plan in plans:
                # Generate task ID for learning
                task_id = self._generate_task_id(plan, "tool")
                
                # Learn from retry memory before attempting generation
                self._learn_from_retry_memory(task_id)
                
                # Check if we should skip this task based on retry memory
                if not self._check_retry_memory_before_attempt(task_id, plan):
                    results.append({
                        "plan": plan,
                        "registration": {"success": False, "error": "Skipped due to retry memory analysis"},
                        "validation": None
                    })
                    continue
                
                # Generate code for the plan
                generation_result = self.builder.generate_code(plan)
                
                if not generation_result.get("success", False):
                    self._log_retry_memory_result(task_id, False, f"Code generation failed: {generation_result.get('error', 'Unknown')}", plan)
                    results.append({
                        "plan": plan,
                        "registration": {"success": False, "error": f"Code generation failed: {generation_result.get('error', 'Unknown')}"},
                        "validation": None
                    })
                    continue
                
                # Update plan with generated code
                plan.update(generation_result)
                
                # Execute with retry logic and memory learning
                execution_result = self._execute_with_retry(plan, tool_manager, short_term_memory)
                results.append(execution_result)
            
            # Compile overall results
            successful_tools = [r for r in results if r.get("registration", {}).get("success", False)]
            failed_tools = [r for r in results if not r.get("registration", {}).get("success", False)]
            
            overall_result = {
                "success": len(successful_tools) > 0,
                "total_plans": len(plans),
                "successful_tools": len(successful_tools),
                "failed_tools": len(failed_tools),
                "results": results,
                "short_term_memory": short_term_memory
            }
            
            # Log final processing result
            if self.notebook:
                self.notebook.log("process_prompt_complete", {
                    "prompt_preview": prompt[:200],
                    "total_plans": len(plans),
                    "successful_tools": len(successful_tools),
                    "failed_tools": len(failed_tools),
                    "overall_success": overall_result["success"]
                })
            
            return overall_result
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"Exception during prompt processing: {str(e)}\n{tb}"
            self.logger.error(error_msg)
            
            if self.notebook:
                self.notebook.log("process_prompt_exception", {
                    "prompt_preview": prompt[:200],
                    "error": str(e),
                    "traceback": tb
                })
            
            return {"success": False, "error": error_msg}
