import importlib
import sys
import traceback
import os
import logging
import threading
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
validator_names = ["code_quality_assessor", "security_scanner", "behavioral_simulator", "security_validator"]

for validator_name in validator_names:
    try:
        validator_module = import_validator(validator_name)
        if validator_module is not None:
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
    """

    # --- Validator hooks (extensible, for future use) ---
    VALIDATORS = ["MathEvaluator", "PlanVerifier", "CodeCritic"]  # Placeholder for plugging in

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
                if failure_count >= 3 and success_count == 0:
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
            
            for validator_name, validator_info in list(loaded_validators.items()):
                requires = validator_info['requires']
                validator_obj = validator_info['validator']
                
                # Check if dependencies are satisfied
                deps_satisfied, unsatisfied_deps = dependencies_satisfied(validator_name, requires)
                
                if deps_satisfied:
                    try:
                        # Instantiate if it's a class
                        if hasattr(validator_obj, '__init__') and not callable(validator_obj):
                            validator_instance = validator_obj()
                        else:
                            validator_instance = validator_obj
                        
                        # Register the validator
                        self.register_validator_safely(validator_name, validator_instance)
                        registered_validators.add(validator_name)
                        registered_this_round = True
                        
                        # Remove from pending list
                        del loaded_validators[validator_name]
                        
                        success_msg = f"Successfully registered extended validator: {validator_name}"
                        self.logger.info(success_msg)
                        if self.notebook:
                            self.notebook.log("validator_registered", {
                                "name": validator_name,
                                "module": validator_info['module'],
                                "requires": requires,
                                "optional": validator_info['optional'],
                                "status": "success"
                            })
                    
                    except Exception as e:
                        error_msg = f"Failed to instantiate/register validator {validator_name}: {str(e)}"
                        self.logger.error(error_msg)
                        failed_validators[validator_name] = {
                            'error': str(e),
                            'module': validator_info['module'],
                            'stage': 'registration'
                        }
                        if self.notebook:
                            self.notebook.log("validator_registration_failed", {
                                "name": validator_name,
                                "module": validator_info['module'],
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            })
                        # Remove from pending list
                        del loaded_validators[validator_name]
                        registered_this_round = True
                
                else:
                    # Dependencies not satisfied, will try again in next iteration
                    if iteration == max_iterations - 1:  # Last iteration, log as skipped
                        skip_msg = f"Skipping validator {validator_name} due to unsatisfied dependencies: {unsatisfied_deps}"
                        self.logger.warning(skip_msg)
                        skipped_validators[validator_name] = {
                            'reason': 'unsatisfied_dependencies',
                            'missing_deps': unsatisfied_deps,
                            'module': validator_info['module']
                        }
                        if self.notebook:
                            self.notebook.log("validator_skipped", {
                                "name": validator_name,
                                "module": validator_info['module'],
                                "reason": "unsatisfied_dependencies",
                                "missing_dependencies": unsatisfied_deps,
                                "requires": requires
                            })
            
            # If no validators were registered this round, break to avoid infinite loop
            if not registered_this_round:
                break
        
        # Log any remaining unregistered validators
        for validator_name, validator_info in loaded_validators.items():
            skip_msg = f"Could not register validator {validator_name} after {iteration} iterations"
            self.logger.warning(skip_msg)
            skipped_validators[validator_name] = {
                'reason': 'dependency_resolution_failed',
                'module': validator_info['module'],
                'requires': validator_info['requires']
            }
            if self.notebook:
                self.notebook.log("validator_dependency_resolution_failed", {
                    "name": validator_name,
                    "module": validator_info['module'],
                    "requires": validator_info['requires'],
                    "iterations": iteration
                })
        
        # Final summary log
        summary = {
            "total_attempted": len(validator_files),
            "successfully_registered": len(registered_validators),
            "failed_imports": len([v for v in failed_validators.values() if v['stage'] == 'import']),
            "failed_registrations": len([v for v in failed_validators.values() if v['stage'] in ['processing', 'registration']]),
            "skipped_dependencies": len(skipped_validators),
            "registered_validators": list(registered_validators),
            "failed_validators": list(failed_validators.keys()),
            "skipped_validators": list(skipped_validators.keys())
        }
        
        self.logger.info(f"Extended validator loading complete: {summary}")
        if self.notebook:
            self.notebook.log("validator_loading_complete", summary)

    def _setup_validation_chain(self):
        """
        Setup the validation chain with proper dependencies.
        """
        # Add validators to chain with dependencies
        self.validation_chain.add_validator("PlanVerifier", self.validator_registry.get("PlanVerifier"))
        self.validation_chain.add_validator("MathEvaluator", self.validator_registry.get("MathEvaluator"), 
                                           dependencies=["PlanVerifier"])
        
        if "CodeQualityAssessor" in self.validator_registry:
            self.validation_chain.add_validator("CodeQualityAssessor", self.validator_registry.get("CodeQualityAssessor"),
                                               dependencies=["MathEvaluator"])
        
        if "SecurityValidator" in self.validator_registry:
            deps = ["CodeQualityAssessor"] if "CodeQualityAssessor" in self.validator_registry else ["MathEvaluator"]
            self.validation_chain.add_validator("SecurityValidator", self.validator_registry.get("SecurityValidator"),
                                               dependencies=deps)
        
        # Add TestToolRunner last
        if self.test_runner:
            last_deps = []
            if "SecurityValidator" in self.validator_registry:
                last_deps = ["SecurityValidator"]
            elif "CodeQualityAssessor" in self.validator_registry:
                last_deps = ["CodeQualityAssessor"]
            else:
                last_deps = ["MathEvaluator"]
            self.validation_chain.add_validator("TestToolRunner", self.test_runner, dependencies=last_deps)

    def register_validator_safely(self, name: str, validator_cls: Callable):
        """
        Thread-safe method to register a validator in the validator registry.
        
        :param name: Unique string identifier for the validator.
        :param validator_cls: Callable validator instance or class to register.
        """
        with self.validator_lock:
            if not callable(validator_cls):
                self.logger.warning(f"Validator '{name}' is not callable, skipping registration.")
                return
            
            if name in self.validator_registry:
                self.logger.info(f"Validator '{name}' already exists in registry, overwriting.")
            
            self.validator_registry[name] = validator_cls
            self.logger.info(f"Validator '{name}' registered successfully in thread-safe manner.")

    def _register_enhanced_validators(self):
        """
        Dynamically and safely register enhanced validators at the end of the validator chain.
        Preserves all existing validation logic and order.
        Uses the new import_validator function from core.utils.path_utils for fallback-safe importing.
        """
        # Enhanced validator configurations with module names and class names
        enhanced_validator_configs = [
            ("code_quality_assessor", "CodeQualityAssessor"),
            ("security_scanner", "SecurityScanner"), 
            ("behavioral_simulator", "BehavioralSimulator")
        ]
        
        for validator_module_name, validator_class_name in enhanced_validator_configs:
            # Use dynamic import for each validator using the new import_validator function
            validator_module = import_validator(validator_module_name)
            if validator_module is None:
                self.logger.warning(f"Validator module '{validator_module_name}' is missing or failed to import; skipping registration.")
                continue
                
            validator_cls = getattr(validator_module, validator_class_name, None)
            
            if validator_cls is None:
                self.logger.warning(f"Validator class '{validator_class_name}' not found in module '{validator_module_name}'; skipping registration.")
                continue
                
            try:
                instance = validator_cls()
            except Exception as e:
                self.logger.error(f"Could not instantiate validator '{validator_class_name}': {e}")
                continue
            
            self.register_validator_safely(validator_class_name, instance)

        # Ensure order: append to VALIDATORS after MathEvaluator/TestToolRunner, never before.
        # Existing pipeline: ["MathEvaluator", "PlanVerifier", "CodeCritic"]
        for _, validator_class_name in enhanced_validator_configs:
            if validator_class_name not in self.VALIDATORS and validator_class_name in self.validator_registry:
                self.VALIDATORS.append(validator_class_name)

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

    def _run_validation_pipeline(self, plan: Dict[str, Any], tool_code: str, test_code: str, tool_path: Optional[str] = None, test_path: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Runs validation pipeline using ValidationChain with retry memory integration.
        Uses self.validator_registry for validators, executed through ValidationChain.
        Logs individual validator failures/exceptions to self.notebook and retry memory.
        Returns:
            - bool: Overall validation success.
            - List[Dict[str, Any]]: A list of dictionaries, each containing 'validator' name and 'result'.
        """
        # Generate task ID for validation pipeline
        validation_task_id = self._generate_task_id(plan, "validation")
        
        # Check retry memory for validation attempts
        if not self._check_retry_memory_before_attempt(validation_task_id, plan):
            self.logger.warning(f"Skipping validation for plan '{plan.get('name', 'N/A')}' due to retry memory policy")
            return False, [{"validator": "retry_memory_skip", "result": {"success": False, "error": "Skipped due to retry memory policy"}}]
        
        # Prepare context for validators
        context = {
            "plan": plan,
            "tool_code": tool_code,
            "test_code": test_code,
            "tool_path": tool_path,
            "test_path": test_path
        }
        
        try:
            # Run validation chain
            overall_success, detailed_results = self.validation_chain.run(context)
            
            # Log results to notebook
            for result_entry in detailed_results:
                validator_name = result_entry["validator"]
                result = result_entry["result"]
                
                if not result.get("success", True):
                    if self.notebook:
                        self.notebook.log("validator_failure", {
                            "plan": plan,
                            "validator": validator_name,
                            "error": result.get("error", "Unknown validation error")
                        })
                    self.logger.warning(f"Validator '{validator_name}' failed for plan: {plan.get('name', 'N/A')}. Error: {result.get('error')}")
                
                if result.get("exception"):
                    if self.notebook:
                        self.notebook.log("validator_exception", {
                            "plan": plan,
                            "validator": validator_name,
                            "exception": result.get("error"),
                            "traceback": result.get("exception")
                        })
            
            # Log validation result to retry memory
            validation_reason = None
            if not overall_success:
                failed_validators = [
                    f"{item['validator']}: {item['result'].get('error', 'Failed')}"
                    for item in detailed_results
                    if not item['result'].get('success', item['result'].get('passed', True))
                ]
                validation_reason = f"Validation failed: {'; '.join(failed_validators)}"
            else:
                validation_reason = "All validators passed successfully"
            
            self._log_retry_memory_result(validation_task_id, overall_success, validation_reason, plan)
            
            return overall_success, detailed_results
            
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"Validation pipeline exception: {str(e)}"
            self.logger.error(f"{error_msg}\n{tb_str}")
            
            # Log validation pipeline failure to retry memory
            self._log_retry_memory_result(validation_task_id, False, f"Pipeline exception: {str(e)}", plan)
            
            return False, [{"validator": "validation_pipeline", "result": {"success": False, "error": error_msg, "exception": tb_str}}]

    def process_prompt(
        self,
        prompt: str,
        tool_manager: Optional[ToolManager] = None,
        short_term_memory: Optional[dict] = None,
    ) -> dict:
        """
        Process a prompt that may request one or many tools with retry memory integration:
        - Decomposes prompt into structured plans (multi-tool aware).
        - Checks retry memory before attempting each tool generation.
        - Enforces overwrite protection: skips or errors if code or test file exists.
        - Writes tool code to /tools/, test code to /test/.
        - Dynamically imports, instantiates, and validates each tool using run('test').
        - Logs all successes/failures to retry memory system.
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
            
            # Log decomposition failure to retry memory
            decomp_task_id = f"decomposition::{hash(prompt) % 10000}"  # Simple hash for prompt-based task ID
            self._log_retry_memory_result(decomp_task_id, False, f"Decomposition failed: {str(e)}")
            
            return {"success": False, "error": log_msg, "results": [], "retry_later": []}

        # 2. For each tool, run the generation/validation/registration pipeline
        for plan in plans:
            single_result = {"plan": plan, "registration": None, "validation": None}
            tool_file = plan.get("file")
            test_file = plan.get("test_file")
            class_name = plan.get("class")
            tool_code = plan.get("code")
            test_code = plan.get("test_code")

            # Generate task ID for this specific tool/plan
            tool_task_id = self._generate_task_id(plan, "tool")
            
            # Check retry memory before attempting this tool
            if not self._check_retry_memory_before_attempt(tool_task_id, plan):
                skip_msg = f"Skipping tool generation for '{plan.get('name', 'N/A')}' due to retry memory policy"
                single_result["registration"] = {"success": False, "error": skip_msg}
                retry_later.append({"plan": plan, "reason": skip_msg})
                results.append(single_result)
                self.logger.warning(skip_msg)
                continue

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
                    # Log to retry memory as a failure due to file conflict
                    self._log_retry_memory_result(tool_task_id, False, "File already exists", plan)
                    continue
                if test_path and os.path.exists(test_path):
                    msg = f"Test file exists, skipping: {test_path}"
                    single_result["registration"] = {"success": False, "error": msg}
                    retry_later.append({"plan": plan, "reason": msg})
                    results.append(single_result)
                    self.logger.warning(msg)
                    # Log to retry memory as a failure due to file conflict
                    self._log_retry_memory_result(tool_task_id, False, "Test file already exists", plan)
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
                    # Log to retry memory as a failure due to missing plan information
                    self._log_retry_memory_result(tool_task_id, False, err_msg, plan)
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
                    # Log to retry memory as a failure due to standards violation
                    self._log_retry_memory_result(tool_task_id, False, f"Standards check failed: {standards_errors}", plan)
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
                    # Log to retry memory as a failure due to validation failure
                    self._log_retry_memory_result(tool_task_id, False, augmented_fail_msg, plan)
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
                                # Log to retry memory as a failure due to secondary validator failure
                                self._log_retry_memory_result(tool_task_id, False, val_msg, plan)
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
                            # Log to retry memory as a failure due to secondary validator exception
                            self._log_retry_memory_result(tool_task_id, False, val_msg, plan)
                            break
                    else:
                        # --- NEW: Fallback for missing validators ---
                        context = {
                            "plan": plan,
                            "tool_code": tool_code,
                            "test_code": test_code,
                            "tool_path": tool_path,
                            "test_path": test_path
                        }
                        fallback_result = self.fallback_if_validator_missing(validator_name, context)
                        # Continue execution with fallback result (which is success=True)
                        # Log the fallback for audit trail
                        if self.notebook:
                            self.notebook.log("secondary_validator_fallback", {
                                "plan": plan,
                                "validator": validator_name,
                                "fallback_result": fallback_result.__dict__,
                                "action": "continued_execution"
                            })

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
                        # Log to retry memory as a failure due to final test failure
                        self._log_retry_memory_result(tool_task_id, False, fail_msg, plan)
                        continue
                
                # --- BEGIN SANDBOX INTEGRATION ---
                # After the tool is generated and all previous validators pass, run in sandbox.
                sandbox_run_successful = True # Assume true unless sandbox fails or is not applicable
                if tool_path and hasattr(self, 'sandbox_runner') and self.sandbox_runner:
                    self.logger.info(f"Attempting sandbox execution for tool: {tool_path} from plan: {plan.get('name', 'N/A')}")
                    
                    # Generate sandbox-specific task ID
                    sandbox_task_id = self._generate_task_id(plan, "sandbox")
                    
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
                            # Log to retry memory as a failure due to sandbox execution failure
                            self._log_retry_memory_result(tool_task_id, False, fail_msg, plan)
                            self._log_retry_memory_result(sandbox_task_id, False, error_details, plan)
                            continue # Skip to next plan, do not register
                        else:
                            self.logger.info(f"Sandbox execution successful for tool '{tool_path}'. stdout: {sandbox_result.get('stdout', 'N/A')}")
                            # Log sandbox success to retry memory
                            self._log_retry_memory_result(sandbox_task_id, True, "Sandbox execution successful", plan)
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
                        # Log to retry memory as a failure due to sandbox exception
                        self._log_retry_memory_result(tool_task_id, False, fail_msg, plan)
                        self._log_retry_memory_result(sandbox_task_id, False, f"Exception: {str(e_sandbox)}", plan)
                        continue # Skip to next plan, do not register
                elif not tool_path and hasattr(self, 'sandbox_runner') and self.sandbox_runner:
                    # Log if sandbox was skipped due to no tool_path, but don't mark as failure for this reason alone.
                    self.logger.info(f"Sandbox execution skipped for plan {plan.get('name', 'N/A')} as tool_path is not available.")
                
                # If sandbox_run_successful is False due to an actual failure (handled by 'continue' above), 
                # subsequent steps including registration will be skipped.
                # --- END SANDBOX INTEGRATION ---

                # --- AGI EXTENSION: Enhanced Validator Audit Logging ---
                # This audit log remains, it iterates self.VALIDATORS.
                for validator_name in self.VALIDATORS:
                    validator = self.validator_registry.get(
