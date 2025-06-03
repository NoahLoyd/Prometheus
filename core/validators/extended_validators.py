import ast
import logging
import re
import sys
import importlib.util
import tempfile
from typing import Optional, Dict, Any
from core.utils.path_utils import safe_path_join
from core.utils.validator_importer import import_validator
from addons.notebook import AddOnNotebook

logger = logging.getLogger("promethyn.validators")
logging.basicConfig(level=logging.INFO)


class CodeQualityAssessor:
    """
    Validates code for cleanliness, maintainability, readability, and adherence to standards.
    Flags poor formatting, long functions, duplicate code, and non-standard practices.
    Optionally integrates pylint or flake8 if available.
    """

    MAX_FUNCTION_LENGTH = 40  # lines
    DUPLICATE_THRESHOLD = 10  # lines

    def __init__(self, notebook: Optional[AddOnNotebook] = None):
        self.notebook = notebook
        self.pylint = self._import_linter('pylint.lint')
        self.flake8 = self._import_linter('flake8.main.application')

    def _import_linter(self, module_name: str):
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return importlib.import_module(module_name)
        return None

    def validate(self, code: str) -> Dict[str, Optional[str]]:
        if self.notebook:
            self.notebook.log("code_quality_assessor", "VALIDATION_START", "Starting code quality assessment", metadata={"code_length": len(code)})
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            error_msg = f"Syntax error: {e}"
            logger.error(error_msg)
            if self.notebook:
                self.notebook.log("code_quality_assessor", "SYNTAX_ERROR", error_msg, metadata={"error": str(e)})
            return {'success': False, 'error': error_msg}

        issues = []

        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start = node.lineno
                end = max(getattr(child, 'end_lineno', start) for child in ast.walk(node))
                length = end - start + 1
                if length > self.MAX_FUNCTION_LENGTH:
                    issue = f"Function '{node.name}' is too long ({length} lines)."
                    issues.append(issue)
                    if self.notebook:
                        self.notebook.log("code_quality_assessor", "LONG_FUNCTION", issue, metadata={
                            "function_name": node.name,
                            "length": length,
                            "max_allowed": self.MAX_FUNCTION_LENGTH
                        })

        # Check for duplicate code (simple substring method)
        code_lines = code.splitlines()
        for i in range(len(code_lines) - self.DUPLICATE_THRESHOLD):
            snippet = '\n'.join(code_lines[i:i+self.DUPLICATE_THRESHOLD])
            if code.count(snippet) > 1:
                issue = "Duplicate code detected."
                issues.append(issue)
                if self.notebook:
                    self.notebook.log("code_quality_assessor", "DUPLICATE_CODE", issue, metadata={
                        "start_line": i + 1,
                        "snippet_length": self.DUPLICATE_THRESHOLD
                    })
                break

        # Optionally run linter
        linter_report = None
        if self.pylint:
            try:
                from pylint.lint import Run
                # Save code to a temp file
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
                    # Get temp directory safely
                    temp_dir = tempfile.gettempdir()
                    safe_temp_path = safe_path_join(temp_dir, tf.name)
                    tf.write(code)
                    tf.flush()
                    results = Run([tf.name], do_exit=False)
                    if results.linter.msg_status != 0:
                        linter_report = "Pylint flagged issues."
                        if self.notebook:
                            self.notebook.log("code_quality_assessor", "PYLINT_ISSUES", linter_report, metadata={
                                "temp_file": tf.name,
                                "msg_status": results.linter.msg_status
                            })
            except Exception as e:
                error_msg = f"Pylint integration failed: {e}"
                logger.warning(error_msg)
                if self.notebook:
                    self.notebook.log("code_quality_assessor", "PYLINT_ERROR", error_msg, metadata={"error": str(e)})
        elif self.flake8:
            try:
                from flake8.main.application import Application
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
                    # Get temp directory safely
                    temp_dir = tempfile.gettempdir()
                    safe_temp_path = safe_path_join(temp_dir, tf.name)
                    tf.write(code)
                    tf.flush()
                    app = Application()
                    app.run([tf.name])
                    if app.result_count > 0:
                        linter_report = "Flake8 flagged issues."
                        if self.notebook:
                            self.notebook.log("code_quality_assessor", "FLAKE8_ISSUES", linter_report, metadata={
                                "temp_file": tf.name,
                                "result_count": app.result_count
                            })
            except Exception as e:
                error_msg = f"Flake8 integration failed: {e}"
                logger.warning(error_msg)
                if self.notebook:
                    self.notebook.log("code_quality_assessor", "FLAKE8_ERROR", error_msg, metadata={"error": str(e)})

        if linter_report:
            issues.append(linter_report)

        if issues:
            error_msg = "; ".join(issues)
            logger.info(f"Code quality failed: {issues}")
            if self.notebook:
                self.notebook.log("code_quality_assessor", "VALIDATION_FAILED", f"Code quality assessment failed", metadata={
                    "issues": issues,
                    "issue_count": len(issues),
                    "error": error_msg
                })
            return {'success': False, 'error': error_msg}
        
        success_msg = "Code quality check passed."
        logger.info(success_msg)
        if self.notebook:
            self.notebook.log("code_quality_assessor", "VALIDATION_PASSED", success_msg, metadata={"code_length": len(code)})
        return {'success': True, 'error': None}


class SecurityScanner:
    """
    Scans code for insecure imports, risky logic, hardcoded credentials, open ports, and known CVE patterns.
    """

    RISKY_BUILTINS = {'eval', 'exec', 'compile', 'input', 'os.system', 'subprocess', 'pickle', 'yaml.load', 'open'}
    CREDENTIAL_PATTERNS = [
        r'password\s*=\s*[\'"].+[\'"]',
        r'api[_-]?key\s*=\s*[\'"].+[\'"]',
        r'(?:AWS|SECRET|TOKEN)[_\w]*\s*=\s*[\'"].+[\'"]',
    ]
    OPEN_PORT_PATTERN = r'(?:bind\s*\(|listen\s*\().*[,=]\s*(\d{2,5})'
    CVE_PATTERNS = [
        r'\bos\.system\(',
        r'\bsubprocess\.(?:Popen|call|run)\(',
        r'eval\s*\(',
    ]

    def __init__(self, notebook: Optional[AddOnNotebook] = None):
        self.notebook = notebook

    def validate(self, code: str) -> Dict[str, Optional[str]]:
        if self.notebook:
            self.notebook.log("security_scanner", "VALIDATION_START", "Starting security scan", metadata={"code_length": len(code)})
        
        issues = []

        # Insecure imports and risky built-ins
        for risky in self.RISKY_BUILTINS:
            if risky in code:
                issue = f"Risky builtin or import detected: {risky}"
                issues.append(issue)
                if self.notebook:
                    self.notebook.log("security_scanner", "RISKY_BUILTIN", issue, metadata={"risky_item": risky})

        # Hardcoded credentials
        for pattern in self.CREDENTIAL_PATTERNS:
            if re.search(pattern, code, flags=re.IGNORECASE):
                issue = "Hardcoded credentials detected."
                issues.append(issue)
                if self.notebook:
                    self.notebook.log("security_scanner", "HARDCODED_CREDENTIALS", issue, metadata={"pattern": pattern})

        # Open ports
        for match in re.finditer(self.OPEN_PORT_PATTERN, code):
            port = int(match.group(1))
            if port > 0 and port < 65536:
                issue = f"Open port detected: {port}"
                issues.append(issue)
                if self.notebook:
                    self.notebook.log("security_scanner", "OPEN_PORT", issue, metadata={"port": port})

        # Known CVE patterns
        for pattern in self.CVE_PATTERNS:
            if re.search(pattern, code):
                issue = f"Known risky pattern detected: {pattern}"
                issues.append(issue)
                if self.notebook:
                    self.notebook.log("security_scanner", "CVE_PATTERN", issue, metadata={"pattern": pattern})

        if issues:
            error_msg = "; ".join(issues)
            logger.warning(f"Security issues found: {issues}")
            if self.notebook:
                self.notebook.log("security_scanner", "VALIDATION_FAILED", "Security scan failed", metadata={
                    "issues": issues,
                    "issue_count": len(issues),
                    "error": error_msg
                })
            return {'success': False, 'error': error_msg}
        
        success_msg = "Security scan passed."
        logger.info(success_msg)
        if self.notebook:
            self.notebook.log("security_scanner", "VALIDATION_PASSED", success_msg, metadata={"code_length": len(code)})
        return {'success': True, 'error': None}


class BehavioralSimulator:
    """
    Simulates code execution in a restricted environment to detect risky or unexpected runtime behaviors.
    """

    def __init__(self, notebook: Optional[AddOnNotebook] = None):
        self.notebook = notebook

    def validate(self, code: str) -> Dict[str, Optional[str]]:
        import builtins

        if self.notebook:
            self.notebook.log("behavioral_simulator", "VALIDATION_START", "Starting behavioral simulation", metadata={"code_length": len(code)})

        # Restricted builtins
        SAFE_BUILTINS = {
            "abs", "all", "any", "bin", "bool", "bytes", "chr", "dict", "divmod", "enumerate",
            "filter", "float", "format", "hash", "hex", "int", "isinstance", "len", "list", "map",
            "max", "min", "next", "object", "oct", "ord", "pow", "range", "repr", "reversed",
            "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip"
        }
        restricted_globals = {k: getattr(builtins, k) for k in SAFE_BUILTINS}
        restricted_globals['__builtins__'] = restricted_globals

        try:
            exec_env = {}
            exec(compile(code, "<string>", "exec"), restricted_globals, exec_env)
        except Exception as e:
            error_msg = f"Simulation flagged a runtime error or risky behavior: {e}"
            logger.warning(error_msg)
            if self.notebook:
                self.notebook.log("behavioral_simulator", "VALIDATION_FAILED", error_msg, metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "code_length": len(code)
                })
            return {'success': False, 'error': error_msg}

        success_msg = "Behavioral simulation passed."
        logger.info(success_msg)
        if self.notebook:
            self.notebook.log("behavioral_simulator", "VALIDATION_PASSED", success_msg, metadata={
                "code_length": len(code),
                "safe_builtins_count": len(SAFE_BUILTINS)
            })
        return {'success': True, 'error': None}


# Registration for Promethyn's validator chain (Do NOT overwrite existing logic; inject safely)
def register_validators(chain: list, notebook: Optional[AddOnNotebook] = None):
    """
    Inject Promethyn validators into the provided chain using dynamic imports.
    """
    if notebook:
        notebook.log("extended_validators", "REGISTRATION_START", "Starting validator registration", metadata={"chain_length": len(chain)})
    
    # List of validator names to dynamically load
    validator_names = [
        "CodeQualityAssessor",
        "SecurityScanner", 
        "BehavioralSimulator"
    ]
    
    # Mapping of validator names to local classes for fallback
    local_validator_classes = {
        "CodeQualityAssessor": CodeQualityAssessor,
        "SecurityScanner": SecurityScanner,
        "BehavioralSimulator": BehavioralSimulator
    }
    
    registered_count = 0
    for validator_name in validator_names:
        # Check if validator is already in chain
        if not any(type(v).__name__ == validator_name for v in chain):
            # Try to import validator dynamically
            validator_module = import_validator(validator_name.lower().replace("assessor", "_assessor").replace("scanner", "_scanner").replace("simulator", "_simulator"))
            
            validator_cls = None
            if validator_module is not None:
                # Try to get the class from the imported module
                validator_cls = getattr(validator_module, validator_name, None)
                if validator_cls is not None:
                    logger.info(f"Dynamically loaded validator: {validator_name}")
                    if notebook:
                        notebook.log("extended_validators", "DYNAMIC_LOAD_SUCCESS", f"Dynamically loaded validator: {validator_name}", metadata={"validator_name": validator_name})
                else:
                    logger.warning(f"Validator class {validator_name} not found in imported module — falling back to local class.")
                    if notebook:
                        notebook.log("extended_validators", "DYNAMIC_LOAD_FALLBACK", f"Validator class {validator_name} not found in imported module", metadata={"validator_name": validator_name})
            else:
                logger.warning(f"Validator module for {validator_name} not found or failed to load — falling back to local class.")
                if notebook:
                    notebook.log("extended_validators", "MODULE_LOAD_FAILED", f"Validator module for {validator_name} not found", metadata={"validator_name": validator_name})
            
            # Fallback to local class if dynamic import failed
            if validator_cls is None:
                validator_cls = local_validator_classes.get(validator_name)
                if validator_cls is not None:
                    logger.info(f"Using local fallback for validator: {validator_name}")
                    if notebook:
                        notebook.log("extended_validators", "LOCAL_FALLBACK", f"Using local fallback for validator: {validator_name}", metadata={"validator_name": validator_name})
                else:
                    logger.warning(f"Validator {validator_name} not found or failed to load — skipping.")
                    if notebook:
                        notebook.log("extended_validators", "VALIDATOR_SKIP", f"Validator {validator_name} not found", metadata={"validator_name": validator_name})
                    continue
            
            # Instantiate and add to chain
            try:
                validator_instance = validator_cls(notebook=notebook)
                chain.append(validator_instance)
                registered_count += 1
                logger.info(f"Validator {validator_name} registered successfully.")
                if notebook:
                    notebook.log("extended_validators", "VALIDATOR_REGISTERED", f"Validator {validator_name} registered successfully", metadata={"validator_name": validator_name})
            except Exception as e:
                error_msg = f"Failed to instantiate validator {validator_name}: {e}"
                logger.error(error_msg)
                if notebook:
                    notebook.log("extended_validators", "INSTANTIATION_ERROR", error_msg, metadata={"validator_name": validator_name, "error": str(e)})
    
    completion_msg = "Promethyn validators registration complete."
    logger.info(completion_msg)
    if notebook:
        notebook.log("extended_validators", "REGISTRATION_COMPLETE", completion_msg, metadata={
            "total_validators": len(validator_names),
            "registered_count": registered_count,
            "final_chain_length": len(chain)
        })
