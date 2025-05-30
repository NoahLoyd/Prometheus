"""
Security Validator for Promethyn AGI System
Scans Python files for potentially dangerous operations and patterns.

Author: Promethyn AGI
Created: 2025-05-30
"""

import ast
import logging
from typing import Tuple, Set, List

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

def validate_security(file_path: str) -> Tuple[bool, str]:
    """
    Validate a Python file for security concerns.
    
    Args:
        file_path: Path to the Python file to validate
        
    Returns:
        Tuple[bool, str]: (is_safe, message)
            - is_safe: True if no security violations found, False otherwise
            - message: Detailed explanation of validation results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
        
        # Parse the source code into an AST
        tree = ast.parse(source)
        
        # Run our security visitor
        visitor = SecurityVisitor()
        visitor.visit(tree)
        
        if visitor.violations:
            message = "Security validation failed.\nViolations found:\n"
            message += "\n".join(f"- {violation}" for violation in visitor.violations)
            return False, message
        
        return True, "Security validation passed. No dangerous operations detected."
        
    except FileNotFoundError:
        return False, f"Error: File not found: {file_path}"
    except SyntaxError as e:
        return False, f"Error: Invalid Python syntax in file: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during security validation: {str(e)}")
        return False, f"Error: Validation failed due to unexpected error: {str(e)}"
