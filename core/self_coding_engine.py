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
        Runs validation pipeline using ValidationChain.
        Uses self.validator_registry for validators, executed through ValidationChain.
        Logs individual validator failures/exceptions to self.notebook.
        Returns:
            - bool: Overall validation success.
            - List[Dict[str, Any]]: A list of dictionaries, each containing 'validator' name and 'result'.
        """
        # Prepare context for validators
        context = {
            "plan": plan,
            "tool_code": tool_code,
            "test_code": test_code,
            "tool_path": tool_path,
            "test_path": test_path
        }
        
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
        
        return overall_success, detailed_results

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
                if self.notebook:
                    self.notebook.log("self_coding_engine", "TOOL_REGISTRATION_ERROR", msg, metadata={"plan": plan, "error": msg})
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
                if self.notebook:
                    self.notebook.log("self_coding_engine", "TOOL_IMPORT_ERROR", msg, metadata={"plan": plan, "error": msg, "module_path": module_path})
                self.logger.error(msg)
                return {"success": False, "error": msg}

            tool_class = getattr(module, class_name, None)
            if tool_class is None:
                msg = f"Class '{class_name}' not found in module '{module_path}'."
                if self.notebook:
                    self.notebook.log("self_coding_engine", "TOOL_CLASS_ERROR", msg, metadata={"plan": plan, "error": msg, "module_path": module_path, "class_name": class_name})
                self.logger.error(msg)
                return {"success": False, "error": msg}

            # Ensure the tool class inherits from BaseTool
            if not issubclass(tool_class, BaseTool):
                msg = f"Class '{class_name}' does not inherit from BaseTool."
                if self.notebook:
                    self.notebook.log("self_coding_engine", "TOOL_INHERITANCE_ERROR", msg, metadata={"plan": plan, "error": msg})
                self.logger.error(msg)
                return {"success": False, "error": msg}

            tool_instance = tool_class()

            if tool_manager:
                tool_manager.register_tool(tool_instance)
                success_msg = f"Tool '{class_name}' registered successfully in ToolManager."
                if self.notebook:
                    self.notebook.log("self_coding_engine", "TOOL_REGISTRATION_SUCCESS", success_msg, metadata={"class_name": class_name, "tool": tool_instance})
                self.logger.info(success_msg)
                return {"success": True, "error": "", "tool": tool_instance}
            else:
                info_msg = (
                    f"Tool '{class_name}' instantiated, but no ToolManager provided.\n"
                    f"To register manually: tool_manager.register_tool(tool_instance)"
                )
                if self.notebook:
                    self.notebook.log("self_coding_engine", "TOOL_INSTANTIATED", info_msg, metadata={"class_name": class_name, "tool": tool_instance})
                self.logger.info(info_msg)
                return {"success": True, "warning": info_msg, "tool": tool_instance}

        except Exception as e:
            tb = traceback.format_exc()
            err_msg = f"Tool registration exception for plan {plan.get('name', 'N/A')}: {e}\n{tb}"
            if self.notebook:
                self.notebook.log("self_coding_engine", "TOOL_REGISTRATION_EXCEPTION", err_msg, metadata={"plan": plan, "error": str(e), "traceback": tb})
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
