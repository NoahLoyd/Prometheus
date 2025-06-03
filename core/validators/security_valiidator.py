"""
Security Validator for Promethyn AGI System
Scans Python files for potentially dangerous operations and patterns.

Author: Promethyn AGI
Created: 2025-05-30
"""

import ast
import logging
import os
from typing import Tuple, Set, List, Optional
from core.utils.path_utils import safe_path_join
from addons.notebook import AddOnNotebook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityVisitor(ast.NodeVisitor):
    """AST Visitor that checks for security violations in Python code."""
    
    def __init__(self):
        self.violations: List[str] = []
        self.imported_modules: Set[str] = set()
    
    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for dangerous operations."""
        # Check for eval/exec calls
        if isinstance(node.func, ast.Name) and node.func.id in {'eval', 'exec'}:
            self.violations.append(f"Dangerous built-in function call: {node.func.id}")
        
        # Check for open() with write/append modes
        if isinstance(node.func, ast.Name) and node.func.id == 'open':
            if len(node.args) >= 2:  # Has mode parameter
                if isinstance(node.args[1], ast.Str):
                    mode = node.args[1].s
                    if 'w' in mode or 'a' in mode or '+' in mode:
                        self.violations.append(f"File open with write/append mode: {mode}")
        
        # Visit child nodes
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Check for dangerous module imports."""
        for alias in node.names:
            if alias.name in {'os', 'subprocess'}:
                self.violations.append(f"Dangerous module import: {alias.name}")
            self.imported_modules.add(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for dangerous from-imports."""
        if node.module in {'os', 'subprocess'}:
            self.violations.append(f"Dangerous module import: {node.module}")
        self.generic_visit(node)

def validate_security(file_path: str, notebook: Optional[AddOnNotebook] = None) -> Tuple[bool, str]:
    """
    Validate a Python file for security concerns.
    
    Args:
        file_path: Path to the Python file to validate
        notebook: Optional AddOnNotebook instance for enhanced logging
        
    Returns:
        Tuple[bool, str]: (is_safe, message)
            - is_safe: True if no security violations found, False otherwise
            - message: Detailed explanation of validation results
    """
    try:
        # Use safe_path_join to ensure path is secure
        # Get the directory and filename to construct safe path
        dir_path = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Use safe_path_join with directory as base to prevent directory traversal
        # TODO: verify base_dir for path safety - currently using extracted directory
        safe_file_path = safe_path_join(dir_path if dir_path else ".", filename)
        
        if notebook:
            notebook.log("security_validator", "VALIDATION_START", f"Starting security validation for: {file_path}", metadata={"file_path": file_path, "safe_file_path": safe_file_path})
        
        with open(safe_file_path, 'r', encoding='utf-8') as file:
            source = file.read()
        
        # Parse the source code into an AST
        tree = ast.parse(source)
        
        # Run our security visitor
        visitor = SecurityVisitor()
        visitor.visit(tree)
        
        if visitor.violations:
            message = "Security validation failed.\nViolations found:\n"
            message += "\n".join(f"- {violation}" for violation in visitor.violations)
            
            if notebook:
                notebook.log("security_validator", "VALIDATION_FAILED", message, metadata={
                    "file_path": file_path,
                    "violations": visitor.violations,
                    "violation_count": len(visitor.violations),
                    "imported_modules": list(visitor.imported_modules)
                })
            
            logger.warning(f"Security validation failed for {file_path}: {len(visitor.violations)} violations found")
            return False, message
        
        success_message = "Security validation passed. No dangerous operations detected."
        
        if notebook:
            notebook.log("security_validator", "VALIDATION_PASSED", success_message, metadata={
                "file_path": file_path,
                "imported_modules": list(visitor.imported_modules),
                "source_length": len(source)
            })
        
        logger.info(f"Security validation passed for {file_path}")
        return True, success_message
        
    except FileNotFoundError:
        error_message = f"Error: File not found: {file_path}"
        if notebook:
            notebook.log("security_validator", "FILE_NOT_FOUND", error_message, metadata={"file_path": file_path})
        logger.error(error_message)
        return False, error_message
    except SyntaxError as e:
        error_message = f"Error: Invalid Python syntax in file: {str(e)}"
        if notebook:
            notebook.log("security_validator", "SYNTAX_ERROR", error_message, metadata={"file_path": file_path, "error": str(e)})
        logger.error(error_message)
        return False, error_message
    except Exception as e:
        error_message = f"Error: Validation failed due to unexpected error: {str(e)}"
        logger.error(f"Unexpected error during security validation: {str(e)}")
        if notebook:
            notebook.log("security_validator", "UNEXPECTED_ERROR", error_message, metadata={"file_path": file_path, "error": str(e), "error_type": type(e).__name__})
        return False, error_message
