"""
Security Scanner for Prometheus AGI System

This module provides comprehensive security analysis for generated Python code,
identifying potential vulnerabilities, dangerous function calls, and security risks.
"""

import ast
import re
import hashlib
from typing import Dict, Any, List, Set, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum


class SecurityLevel(Enum):
    """Security risk levels for identified issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """Represents a security issue found in code."""
    level: SecurityLevel
    category: str
    description: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


class SecurityScanner:
    """
    Comprehensive security scanner for Python code analysis.
    
    Identifies various security vulnerabilities including:
    - Dangerous function calls
    - Unsafe imports
    - Code injection risks
    - Hardcoded secrets
    - Path traversal vulnerabilities
    - Insecure random usage
    - SQL injection patterns
    """
    
    def __init__(self):
        """Initialize the security scanner with predefined rules."""
        self.logger = logging.getLogger(__name__)
        
        # Critical dangerous functions that execute arbitrary code
        self.critical_functions = {
            'eval': 'Executes arbitrary Python expressions',
            'exec': 'Executes arbitrary Python code',
            'compile': 'Compiles code that can be executed',
            '__import__': 'Dynamic imports can execute arbitrary code',
        }
        
        # High-risk functions that can access system resources
        self.high_risk_functions = {
            'open': 'File operations can be dangerous without validation',
            'input': 'User input without validation can be exploited',
            'raw_input': 'User input without validation can be exploited',
            'getattr': 'Dynamic attribute access can be dangerous',
            'setattr': 'Dynamic attribute modification can be dangerous',
            'delattr': 'Dynamic attribute deletion can be dangerous',
            'hasattr': 'Attribute checking can leak information',
            'globals': 'Access to global namespace can be dangerous',
            'locals': 'Access to local namespace can be dangerous',
            'vars': 'Access to variable namespace can be dangerous',
            'dir': 'Directory listing can leak information',
        }
        
        # Medium-risk system and OS functions
        self.system_functions = {
            'os.system': 'Executes shell commands directly',
            'os.popen': 'Opens pipe to shell command',
            'os.spawn': 'Spawns new processes',
            'os.execl': 'Executes programs',
            'os.execle': 'Executes programs with environment',
            'os.execlp': 'Executes programs in PATH',
            'os.execv': 'Executes programs with arguments',
            'os.execve': 'Executes programs with arguments and environment',
            'os.execvp': 'Executes programs in PATH with arguments',
            'os.execvpe': 'Executes programs in PATH with arguments and environment',
            'subprocess.call': 'Executes shell commands',
            'subprocess.run': 'Executes shell commands',
            'subprocess.Popen': 'Creates subprocess',
            'subprocess.check_call': 'Executes and checks shell commands',
            'subprocess.check_output': 'Executes shell commands and returns output',
            'subprocess.getstatusoutput': 'Executes shell commands',
            'subprocess.getoutput': 'Executes shell commands',
        }
        
        # Dangerous modules that should be carefully reviewed
        self.dangerous_modules = {
            'os': 'Operating system interface - can access files and execute commands',
            'sys': 'System-specific parameters and functions',
            'subprocess': 'Subprocess management - can execute shell commands',
            'pickle': 'Can execute arbitrary code during deserialization',
            'dill': 'Extended pickle - can execute arbitrary code',
            'marshal': 'Can execute code during deserialization',
            'shelve': 'Uses pickle internally',
            'multiprocessing': 'Process management',
            'threading': 'Thread management',
            'ctypes': 'Foreign function library',
            'importlib': 'Dynamic import utilities',
            'runpy': 'Module execution utilities',
            'code': 'Interactive interpreter utilities',
            'codeop': 'Code compilation utilities',
            'imp': 'Import utilities (deprecated)',
            'pkgutil': 'Package utilities',
            'modulefinder': 'Module finder utilities',
            'ast': 'Abstract syntax trees - can be used for code generation',
        }
        
        # SQL injection patterns
        self.sql_patterns = [
            r'SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*%s',
            r'INSERT\s+INTO\s+.*\s+VALUES\s+.*%s',
            r'UPDATE\s+.*\s+SET\s+.*\s+WHERE\s+.*%s',
            r'DELETE\s+FROM\s+.*\s+WHERE\s+.*%s',
            r'DROP\s+TABLE\s+.*%s',
            r'CREATE\s+TABLE\s+.*%s',
            r'ALTER\s+TABLE\s+.*%s',
        ]
        
        # Common secret patterns
        self.secret_patterns = {
            r'password\s*=\s*["\'][^"\']+["\']': 'Hardcoded password',
            r'secret\s*=\s*["\'][^"\']+["\']': 'Hardcoded secret',
            r'api_key\s*=\s*["\'][^"\']+["\']': 'Hardcoded API key',
            r'token\s*=\s*["\'][^"\']+["\']': 'Hardcoded token',
            r'key\s*=\s*["\'][^"\']+["\']': 'Hardcoded key',
            r'private_key\s*=\s*["\'][^"\']+["\']': 'Hardcoded private key',
            r'secret_key\s*=\s*["\'][^"\']+["\']': 'Hardcoded secret key',
            r'access_token\s*=\s*["\'][^"\']+["\']': 'Hardcoded access token',
            r'auth_token\s*=\s*["\'][^"\']+["\']': 'Hardcoded auth token',
            r'database_url\s*=\s*["\'][^"\']+["\']': 'Hardcoded database URL',
        }
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r'\.\./.*',
            r'\.\.\\.*',
            r'/etc/passwd',
            r'/etc/shadow',
            r'\.\.%2F',
            r'\.\.%5C',
        ]
        
    def scan_for_security_issues(self, code: str) -> Dict[str, Any]:
        """
        Scan Python code for security issues and vulnerabilities.
        
        Args:
            code (str): The Python code to scan
            
        Returns:
            Dict[str, Any]: Security scan results containing:
                - is_safe: Boolean indicating if code is considered safe
                - issues: List of security issues found
                - risk_level: Overall risk assessment
                - summary: Summary of findings
        """
        if not isinstance(code, str):
            return {
                "is_safe": False,
                "issues": ["Invalid input: code must be a string"],
                "risk_level": "critical",
                "summary": {"total_issues": 1, "critical": 1, "high": 0, "medium": 0, "low": 0}
            }
        
        if not code.strip():
            return {
                "is_safe": True,
                "issues": [],
                "risk_level": "safe",
                "summary": {"total_issues": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
            }
        
        issues = []
        
        # 1. Parse AST for structural analysis
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "is_safe": False,
                "issues": [f"Syntax error prevents security analysis: {str(e)}"],
                "risk_level": "high",
                "summary": {"total_issues": 1, "critical": 0, "high": 1, "medium": 0, "low": 0}
            }
        except Exception as e:
            return {
                "is_safe": False,
                "issues": [f"Failed to parse code for security analysis: {str(e)}"],
                "risk_level": "high",
                "summary": {"total_issues": 1, "critical": 0, "high": 1, "medium": 0, "low": 0}
            }
        
        # 2. Scan for dangerous function calls
        function_issues = self._scan_dangerous_functions(tree)
        issues.extend(function_issues)
        
        # 3. Scan for dangerous imports
        import_issues = self._scan_dangerous_imports(tree)
        issues.extend(import_issues)
        
        # 4. Scan for hardcoded secrets
        secret_issues = self._scan_hardcoded_secrets(code)
        issues.extend(secret_issues)
        
        # 5. Scan for SQL injection patterns
        sql_issues = self._scan_sql_injection(code)
        issues.extend(sql_issues)
        
        # 6. Scan for path traversal vulnerabilities
        path_issues = self._scan_path_traversal(code)
        issues.extend(path_issues)
        
        # 7. Scan for insecure random usage
        random_issues = self._scan_insecure_random(tree)
        issues.extend(random_issues)
        
        # 8. Scan for code injection patterns
        injection_issues = self._scan_code_injection(tree, code)
        issues.extend(injection_issues)
        
        # 9. Scan for information disclosure
        disclosure_issues = self._scan_information_disclosure(tree)
        issues.extend(disclosure_issues)
        
        # Calculate risk level and safety
        risk_level, is_safe, summary = self._calculate_risk_level(issues)
        
        # Convert issues to string format for backward compatibility
        issue_strings = [self._format_issue(issue) for issue in issues]
        
        return {
            "is_safe": is_safe,
            "issues": issue_strings,
            "risk_level": risk_level,
            "summary": summary,
            "detailed_issues": issues  # Include detailed issue objects
        }
    
    def _scan_dangerous_functions(self, tree: ast.AST) -> List[SecurityIssue]:
        """Scan for dangerous function calls."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                line_num = getattr(node, 'lineno', None)
                
                # Check critical functions
                if func_name in self.critical_functions:
                    issues.append(SecurityIssue(
                        level=SecurityLevel.CRITICAL,
                        category="Dangerous Function Call",
                        description=f"Critical security risk: {func_name}() - {self.critical_functions[func_name]}",
                        line_number=line_num,
                        suggestion=f"Avoid using {func_name}(). Consider safer alternatives."
                    ))
                
                # Check high-risk functions
                elif func_name in self.high_risk_functions:
                    issues.append(SecurityIssue(
                        level=SecurityLevel.HIGH,
                        category="High-Risk Function Call",
                        description=f"High security risk: {func_name}() - {self.high_risk_functions[func_name]}",
                        line_number=line_num,
                        suggestion=f"Validate all inputs to {func_name}() and consider security implications."
                    ))
                
                # Check system functions (including attribute access like os.system)
                elif func_name in self.system_functions or self._is_system_function(node.func):
                    system_func = func_name if func_name in self.system_functions else self._get_full_function_name(node.func)
                    issues.append(SecurityIssue(
                        level=SecurityLevel.MEDIUM,
                        category="System Function Call",
                        description=f"System access risk: {system_func} - Executes system commands",
                        line_number=line_num,
                        suggestion=f"Ensure {system_func} inputs are sanitized and validated."
                    ))
        
        return issues
    
    def _scan_dangerous_imports(self, tree: ast.AST) -> List[SecurityIssue]:
        """Scan for dangerous module imports."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.dangerous_modules:
                        issues.append(SecurityIssue(
                            level=SecurityLevel.MEDIUM,
                            category="Dangerous Import",
                            description=f"Potentially dangerous import: {alias.name} - {self.dangerous_modules[alias.name]}",
                            line_number=getattr(node, 'lineno', None),
                            suggestion=f"Review usage of {alias.name} module for security implications."
                        ))
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in self.dangerous_modules:
                    issues.append(SecurityIssue(
                        level=SecurityLevel.MEDIUM,
                        category="Dangerous Import",
                        description=f"Import from dangerous module: {node.module} - {self.dangerous_modules[node.module]}",
                        line_number=getattr(node, 'lineno', None),
                        suggestion=f"Review imported functions from {node.module} for security risks."
                    ))
        
        return issues
    
    def _scan_hardcoded_secrets(self, code: str) -> List[SecurityIssue]:
        """Scan for hardcoded secrets and credentials."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            for pattern, description in self.secret_patterns.items():
                if re.search(pattern, line_lower, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        level=SecurityLevel.HIGH,
                        category="Hardcoded Secret",
                        description=f"{description} detected in code",
                        line_number=i,
                        suggestion="Use environment variables or secure configuration files for sensitive data."
                    ))
        
        return issues
    
    def _scan_sql_injection(self, code: str) -> List[SecurityIssue]:
        """Scan for potential SQL injection vulnerabilities."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern in self.sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        level=SecurityLevel.HIGH,
                        category="SQL Injection Risk",
                        description="Potential SQL injection vulnerability detected",
                        line_number=i,
                        suggestion="Use parameterized queries or prepared statements instead of string formatting."
                    ))
        
        return issues
    
    def _scan_path_traversal(self, code: str) -> List[SecurityIssue]:
        """Scan for path traversal vulnerabilities."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for pattern in self.path_traversal_patterns:
                if re.search(pattern, line):
                    issues.append(SecurityIssue(
                        level=SecurityLevel.MEDIUM,
                        category="Path Traversal Risk",
                        description="Potential path traversal vulnerability detected",
                        line_number=i,
                        suggestion="Validate and sanitize file paths, use os.path.normpath() and check against allowed directories."
                    ))
        
        return issues
    
    def _scan_insecure_random(self, tree: ast.AST) -> List[SecurityIssue]:
        """Scan for insecure random number generation."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                
                if func_name in ['random.random', 'random.randint', 'random.choice', 'random.shuffle']:
                    issues.append(SecurityIssue(
                        level=SecurityLevel.LOW,
                        category="Insecure Random",
                        description=f"Insecure random function: {func_name}()",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Use secrets module for cryptographically secure random numbers."
                    ))
        
        return issues
    
    def _scan_code_injection(self, tree: ast.AST, code: str) -> List[SecurityIssue]:
        """Scan for code injection patterns."""
        issues = []
        
        # Check for dynamic code construction patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                
                # String concatenation with exec/eval
                if func_name in ['exec', 'eval'] and node.args:
                    if isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Add):
                        issues.append(SecurityIssue(
                            level=SecurityLevel.CRITICAL,
                            category="Code Injection",
                            description=f"Dynamic code construction with {func_name}()",
                            line_number=getattr(node, 'lineno', None),
                            suggestion=f"Never construct code dynamically with user input for {func_name}()."
                        ))
        
        # Check for format string code construction
        if re.search(r'exec\s*\(\s*["\'].*\{\}.*["\']\.format\s*\(', code) or \
           re.search(r'eval\s*\(\s*["\'].*\{\}.*["\']\.format\s*\(', code):
            issues.append(SecurityIssue(
                level=SecurityLevel.CRITICAL,
                category="Code Injection",
                description="Dynamic code construction using string formatting",
                suggestion="Never use string formatting to construct executable code."
            ))
        
        return issues
    
    def _scan_information_disclosure(self, tree: ast.AST) -> List[SecurityIssue]:
        """Scan for potential information disclosure."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                
                # Check for stack trace exposure
                if func_name in ['traceback.print_exc', 'traceback.format_exc']:
                    issues.append(SecurityIssue(
                        level=SecurityLevel.LOW,
                        category="Information Disclosure",
                        description="Stack trace exposure",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Avoid exposing detailed error information in production."
                    ))
                
                # Check for debug information
                elif func_name in ['pdb.set_trace', 'pprint.pprint']:
                    issues.append(SecurityIssue(
                        level=SecurityLevel.LOW,
                        category="Information Disclosure",
                        description="Debug code detected",
                        line_number=getattr(node, 'lineno', None),
                        suggestion="Remove debug code before production deployment."
                    ))
        
        return issues
    
    def _get_function_name(self, func_node: ast.AST) -> str:
        """Extract function name from AST node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        elif isinstance(func_node, ast.Call):
            return self._get_function_name(func_node.func)
        return ""
    
    def _get_full_function_name(self, func_node: ast.AST) -> str:
        """Extract full function name including module from AST node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            if isinstance(func_node.value, ast.Name):
                return f"{func_node.value.id}.{func_node.attr}"
            else:
                return func_node.attr
        return ""
    
    def _is_system_function(self, func_node: ast.AST) -> bool:
        """Check if function is a system-related function."""
        full_name = self._get_full_function_name(func_node)
        return full_name in self.system_functions
    
    def _calculate_risk_level(self, issues: List[SecurityIssue]) -> Tuple[str, bool, Dict[str, int]]:
        """Calculate overall risk level and safety status."""
        summary = {
            "total_issues": len(issues),
            "critical": sum(1 for issue in issues if issue.level == SecurityLevel.CRITICAL),
            "high": sum(1 for issue in issues if issue.level == SecurityLevel.HIGH),
            "medium": sum(1 for issue in issues if issue.level == SecurityLevel.MEDIUM),
            "low": sum(1 for issue in issues if issue.level == SecurityLevel.LOW)
        }
        
        # Determine overall risk level
        if summary["critical"] > 0:
            risk_level = "critical"
            is_safe = False
        elif summary["high"] > 0:
            risk_level = "high"
            is_safe = False
        elif summary["medium"] > 2:
            risk_level = "high"
            is_safe = False
        elif summary["medium"] > 0:
            risk_level = "medium"
            is_safe = False
        elif summary["low"] > 5:
            risk_level = "medium"
            is_safe = False
        elif summary["low"] > 0:
            risk_level = "low"
            is_safe = True  # Low-risk issues don't make code unsafe
        else:
            risk_level = "safe"
            is_safe = True
        
        return risk_level, is_safe, summary
    
    def _format_issue(self, issue: SecurityIssue) -> str:
        """Format security issue for string representation."""
        line_info = f" (line {issue.line_number})" if issue.line_number else ""
        return f"[{issue.level.value.upper()}] {issue.category}: {issue.description}{line_info}"


def scan_for_security_issues(code: str) -> Dict[str, Any]:
    """
    Main function to scan Python code for security issues.
    
    Args:
        code (str): The Python code to scan
        
    Returns:
        Dict[str, Any]: Security scan results containing:
            - is_safe: Boolean indicating if code is considered safe
            - issues: List of security issues found as strings
            - risk_level: Overall risk assessment
            - summary: Summary of findings by severity
    """
    scanner = SecurityScanner()
    return scanner.scan_for_security_issues(code)


# Example usage and testing
if __name__ == "__main__":
    # Test with various security issues
    test_codes = [
        # Safe code
        '''
def safe_function(x, y):
    """A safe function that adds two numbers."""
    return x + y

class SafeClass:
    """A safe class."""
    def __init__(self, value):
        self.value = value
''',
        
        # Code with security issues
        '''
import os
import pickle

def unsafe_function(user_input):
    # Critical issues
    result = eval(user_input)  # Code injection
    exec("print('hello')")     # Code execution
    
    # High-risk issues
    password = "hardcoded_secret_123"  # Hardcoded secret
    file_content = open(user_input, 'r').read()  # Unvalidated file access
    
    # Medium-risk issues
    os.system(f"ls {user_input}")  # Command injection
    
    return result
''',
    ]
    
    for i, test_code in enumerate(test_codes, 1):
        print(f"\n=== Test Case {i} ===")
        result = scan_for_security_issues(test_code)
        print(f"Is Safe: {result['is_safe']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Total Issues: {result['summary']['total_issues']}")
        
        if result['issues']:
            print("Issues found:")
            for issue in result['issues']:
                print(f"  - {issue}")
