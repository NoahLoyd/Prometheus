"""
Code Quality Assessor for Prometheus AGI System

This module provides comprehensive code quality assessment functionality
for generated Python code, ensuring safety, modularity, and production-readiness.
"""

import ast
import io
import sys
import subprocess
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple
import logging
from contextlib import redirect_stderr, redirect_stdout


class CodeQualityAssessor:
    """
    Comprehensive code quality assessment tool for Python code.
    
    Evaluates code using multiple metrics including:
    - Syntax validation via AST parsing
    - Style compliance via flake8
    - Complexity analysis
    - Security considerations
    - Best practices adherence
    """
    
    def __init__(self):
        """Initialize the code quality assessor."""
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.max_complexity = 10
        self.max_line_length = 88
        self.max_function_lines = 50
        self.max_class_methods = 20
        
    def assess_code_quality(self, code: str) -> Dict[str, Any]:
        """
        Assess the quality of Python code and return a comprehensive report.
        
        Args:
            code (str): The Python code to assess
            
        Returns:
            Dict[str, Any]: Assessment results containing:
                - score: Quality rating from 0-100
                - issues: List of strings describing problems
                - metrics: Detailed quality metrics
                - suggestions: Improvement recommendations
        """
        if not isinstance(code, str):
            return {
                "score": 0,
                "issues": ["Invalid input: code must be a string"],
                "metrics": {},
                "suggestions": ["Provide valid Python code as string input"]
            }
        
        if not code.strip():
            return {
                "score": 0,
                "issues": ["Empty code provided"],
                "metrics": {},
                "suggestions": ["Provide non-empty Python code"]
            }
        
        issues = []
        metrics = {}
        suggestions = []
        
        # 1. Syntax validation
        syntax_valid, syntax_issues = self._check_syntax(code)
        if not syntax_valid:
            return {
                "score": 0,
                "issues": syntax_issues,
                "metrics": {"syntax_valid": False},
                "suggestions": ["Fix syntax errors before proceeding"]
            }
        
        metrics["syntax_valid"] = True
        
        # 2. Parse AST for analysis
        try:
            tree = ast.parse(code)
        except Exception as e:
            return {
                "score": 0,
                "issues": [f"Failed to parse code: {str(e)}"],
                "metrics": {"syntax_valid": False},
                "suggestions": ["Fix code structure and syntax"]
            }
        
        # 3. Style and linting checks
        style_score, style_issues = self._check_style(code)
        issues.extend(style_issues)
        metrics["style_score"] = style_score
        
        # 4. Complexity analysis
        complexity_score, complexity_issues, complexity_metrics = self._analyze_complexity(tree)
        issues.extend(complexity_issues)
        metrics.update(complexity_metrics)
        
        # 5. Security analysis
        security_score, security_issues = self._check_security(tree, code)
        issues.extend(security_issues)
        metrics["security_score"] = security_score
        
        # 6. Best practices check
        practices_score, practices_issues = self._check_best_practices(tree, code)
        issues.extend(practices_issues)
        metrics["best_practices_score"] = practices_score
        
        # 7. Documentation analysis
        doc_score, doc_issues = self._check_documentation(tree, code)
        issues.extend(doc_issues)
        metrics["documentation_score"] = doc_score
        
        # Calculate overall score
        score = self._calculate_overall_score(
            style_score, complexity_score, security_score, 
            practices_score, doc_score, len(issues)
        )
        
        # Generate suggestions
        suggestions = self._generate_suggestions(metrics, issues)
        
        return {
            "score": score,
            "issues": issues,
            "metrics": metrics,
            "suggestions": suggestions
        }
    
    def _check_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Check code syntax validity."""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]
        except Exception as e:
            return False, [f"Parse error: {str(e)}"]
    
    def _check_style(self, code: str) -> Tuple[int, List[str]]:
        """Check code style using flake8."""
        issues = []
        
        try:
            # Create temporary file for flake8 analysis
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_file.flush()
                
                # Run flake8
                result = subprocess.run(
                    ['flake8', '--max-line-length=88', '--ignore=E203,W503', tmp_file.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode != 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            # Parse flake8 output and clean it up
                            parts = line.split(':', 3)
                            if len(parts) >= 4:
                                issue = f"Line {parts[1]}: {parts[3].strip()}"
                                issues.append(issue)
                
                # Clean up
                os.unlink(tmp_file.name)
                
        except subprocess.TimeoutExpired:
            issues.append("Style check timed out")
        except FileNotFoundError:
            self.logger.warning("flake8 not found, skipping style checks")
        except Exception as e:
            self.logger.warning(f"Style check failed: {str(e)}")
        
        # Calculate style score (100 - (number of issues * 5), minimum 0)
        style_score = max(0, 100 - len(issues) * 5)
        
        return style_score, issues
    
    def _analyze_complexity(self, tree: ast.AST) -> Tuple[int, List[str], Dict[str, Any]]:
        """Analyze code complexity."""
        issues = []
        metrics = {}
        
        # Count various code elements
        functions = []
        classes = []
        lines_of_code = 0
        max_complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_complexity = self._calculate_cyclomatic_complexity(node)
                func_lines = self._count_function_lines(node)
                
                functions.append({
                    'name': node.name,
                    'complexity': func_complexity,
                    'lines': func_lines
                })
                
                max_complexity = max(max_complexity, func_complexity)
                
                if func_complexity > self.max_complexity:
                    issues.append(f"Function '{node.name}' has high complexity: {func_complexity}")
                
                if func_lines > self.max_function_lines:
                    issues.append(f"Function '{node.name}' is too long: {func_lines} lines")
            
            elif isinstance(node, ast.ClassDef):
                class_methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                classes.append({
                    'name': node.name,
                    'methods': len(class_methods)
                })
                
                if len(class_methods) > self.max_class_methods:
                    issues.append(f"Class '{node.name}' has too many methods: {len(class_methods)}")
        
        metrics.update({
            'functions_count': len(functions),
            'classes_count': len(classes),
            'max_complexity': max_complexity,
            'avg_complexity': sum(f['complexity'] for f in functions) / len(functions) if functions else 0
        })
        
        # Calculate complexity score
        complexity_penalty = min(50, len(issues) * 10 + max_complexity * 2)
        complexity_score = max(0, 100 - complexity_penalty)
        
        return complexity_score, issues, metrics
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _count_function_lines(self, node: ast.FunctionDef) -> int:
        """Count lines in a function."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        return 1
    
    def _check_security(self, tree: ast.AST, code: str) -> Tuple[int, List[str]]:
        """Check for potential security issues."""
        issues = []
        
        # Dangerous function calls
        dangerous_functions = {
            'eval', 'exec', 'compile', '__import__',
            'open', 'input', 'raw_input'
        }
        
        # Dangerous modules
        dangerous_modules = {
            'os', 'subprocess', 'sys', 'pickle',
            'shelve', 'marshal', 'dill'
        }
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
                    issues.append(f"Potentially unsafe function call: {node.func.id}")
            
            # Check for dangerous imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_modules:
                            issues.append(f"Import of potentially unsafe module: {alias.name}")
                elif node.module in dangerous_modules:
                    issues.append(f"Import from potentially unsafe module: {node.module}")
        
        # Check for hardcoded secrets (basic patterns)
        secret_patterns = ['password', 'secret', 'key', 'token', 'api_key']
        lines = code.lower().split('\n')
        for i, line in enumerate(lines, 1):
            for pattern in secret_patterns:
                if pattern in line and '=' in line:
                    issues.append(f"Possible hardcoded secret at line {i}")
                    break
        
        security_score = max(0, 100 - len(issues) * 15)
        return security_score, issues
    
    def _check_best_practices(self, tree: ast.AST, code: str) -> Tuple[int, List[str]]:
        """Check adherence to Python best practices."""
        issues = []
        
        # Check for global variables
        globals_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                globals_count += 1
        
        if globals_count > 2:
            issues.append(f"Too many global variables: {globals_count}")
        
        # Check for proper exception handling
        bare_excepts = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_excepts += 1
        
        if bare_excepts > 0:
            issues.append(f"Avoid bare except clauses: {bare_excepts} found")
        
        # Check for magic numbers (simple heuristic)
        magic_numbers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Num) and not isinstance(node.n, bool):
                if node.n not in [0, 1, -1] and node.n not in magic_numbers:
                    magic_numbers.append(node.n)
        
        if len(magic_numbers) > 3:
            issues.append("Consider using named constants instead of magic numbers")
        
        practices_score = max(0, 100 - len(issues) * 10)
        return practices_score, issues
    
    def _check_documentation(self, tree: ast.AST, code: str) -> Tuple[int, List[str]]:
        """Check documentation quality."""
        issues = []
        
        functions_with_docstrings = 0
        total_functions = 0
        classes_with_docstrings = 0
        total_classes = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Str)):
                    functions_with_docstrings += 1
            
            elif isinstance(node, ast.ClassDef):
                total_classes += 1
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Str)):
                    classes_with_docstrings += 1
        
        # Check documentation coverage
        if total_functions > 0:
            func_doc_ratio = functions_with_docstrings / total_functions
            if func_doc_ratio < 0.8:
                issues.append(f"Low function documentation coverage: {func_doc_ratio:.1%}")
        
        if total_classes > 0:
            class_doc_ratio = classes_with_docstrings / total_classes
            if class_doc_ratio < 0.9:
                issues.append(f"Low class documentation coverage: {class_doc_ratio:.1%}")
        
        doc_score = max(0, 100 - len(issues) * 20)
        return doc_score, issues
    
    def _calculate_overall_score(self, style_score: int, complexity_score: int,
                               security_score: int, practices_score: int,
                               doc_score: int, issue_count: int) -> int:
        """Calculate the overall quality score."""
        # Weighted average of different aspects
        weights = {
            'style': 0.2,
            'complexity': 0.25,
            'security': 0.3,
            'practices': 0.15,
            'documentation': 0.1
        }
        
        weighted_score = (
            style_score * weights['style'] +
            complexity_score * weights['complexity'] +
            security_score * weights['security'] +
            practices_score * weights['practices'] +
            doc_score * weights['documentation']
        )
        
        # Apply penalty for high number of issues
        issue_penalty = min(30, issue_count * 2)
        final_score = max(0, int(weighted_score - issue_penalty))
        
        return final_score
    
    def _generate_suggestions(self, metrics: Dict[str, Any], issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on assessment results."""
        suggestions = []
        
        if metrics.get('style_score', 100) < 80:
            suggestions.append("Improve code formatting and style compliance")
        
        if metrics.get('max_complexity', 0) > self.max_complexity:
            suggestions.append("Reduce function complexity by breaking down large functions")
        
        if metrics.get('security_score', 100) < 90:
            suggestions.append("Review and address security concerns")
        
        if metrics.get('documentation_score', 100) < 70:
            suggestions.append("Add comprehensive docstrings to functions and classes")
        
        if metrics.get('functions_count', 0) == 0 and metrics.get('classes_count', 0) == 0:
            suggestions.append("Consider organizing code into functions or classes")
        
        # Add specific suggestions based on common issues
        if any('magic number' in issue.lower() for issue in issues):
            suggestions.append("Define constants for numeric values")
        
        if any('global' in issue.lower() for issue in issues):
            suggestions.append("Minimize use of global variables")
        
        return suggestions


def assess_code_quality(code: str) -> Dict[str, Any]:
    """
    Main function to assess Python code quality.
    
    Args:
        code (str): The Python code to assess
        
    Returns:
        Dict[str, Any]: Assessment results containing:
            - score: Quality rating from 0-100
            - issues: List of strings describing problems
            - metrics: Detailed quality metrics
            - suggestions: Improvement recommendations
    """
    assessor = CodeQualityAssessor()
    return assessor.assess_code_quality(code)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample code
    sample_code = '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b
'''
    
    result = assess_code_quality(sample_code)
    print(f"Code Quality Score: {result['score']}/100")
    print(f"Issues found: {len(result['issues'])}")
    for issue in result['issues']:
        print(f"  - {issue}")
