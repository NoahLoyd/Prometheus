"""
Behavioral Simulator for Prometheus AGI System

This module provides behavioral simulation and analysis for generated Python code,
using AST analysis to infer runtime behavior and detect potentially harmful patterns
before execution.
"""

import ast
import inspect
import re
from typing import Dict, Any, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque


class BehaviorType(Enum):
    """Types of behaviors that can be detected."""
    FILE_OPERATION = "file_operation"
    NETWORK_ACCESS = "network_access"
    SYSTEM_CALL = "system_call"
    PROCESS_SPAWN = "process_spawn"
    MEMORY_ACCESS = "memory_access"
    EXCEPTION_HANDLING = "exception_handling"
    LOOP_BEHAVIOR = "loop_behavior"
    RECURSION = "recursion"
    DYNAMIC_CODE = "dynamic_code"
    DATA_MUTATION = "data_mutation"
    EXTERNAL_DEPENDENCY = "external_dependency"


class RiskLevel(Enum):
    """Risk levels for behavioral patterns."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BehaviorPattern:
    """Represents a detected behavioral pattern."""
    behavior_type: BehaviorType
    risk_level: RiskLevel
    description: str
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    context: Optional[str] = None
    mitigation: Optional[str] = None


@dataclass
class FunctionBehavior:
    """Represents the behavior profile of a function."""
    name: str
    complexity: int = 0
    max_depth: int = 0
    loops: List[str] = field(default_factory=list)
    calls_made: Set[str] = field(default_factory=set)
    variables_modified: Set[str] = field(default_factory=set)
    exceptions_handled: Set[str] = field(default_factory=set)
    side_effects: List[str] = field(default_factory=list)
    is_recursive: bool = False
    has_infinite_loop_risk: bool = False


class BehavioralSimulator:
    """
    Advanced behavioral simulator for Python code analysis.
    
    Uses AST analysis to predict runtime behavior, detect harmful patterns,
    and assess the safety of code execution before it runs.
    """
    
    def __init__(self):
        """Initialize the behavioral simulator."""
        self.logger = logging.getLogger(__name__)
        
        # System operation patterns
        self.file_operations = {
            'open', 'file', 'read', 'write', 'close', 'seek', 'tell',
            'readline', 'readlines', 'writelines', 'flush', 'truncate'
        }
        
        self.system_calls = {
            'os.system', 'os.popen', 'os.spawn', 'os.exec',
            'subprocess.call', 'subprocess.run', 'subprocess.Popen',
            'subprocess.check_call', 'subprocess.check_output'
        }
        
        self.network_operations = {
            'socket.socket', 'urllib.request', 'urllib.urlopen',
            'requests.get', 'requests.post', 'requests.put', 'requests.delete',
            'http.client', 'ftplib', 'smtplib', 'poplib', 'imaplib'
        }
        
        self.process_operations = {
            'multiprocessing.Process', 'threading.Thread',
            'concurrent.futures', 'asyncio.create_task'
        }
        
        self.dangerous_patterns = {
            'eval', 'exec', 'compile', '__import__',
            'globals', 'locals', 'vars', 'dir',
            'getattr', 'setattr', 'delattr', 'hasattr'
        }
        
        # Control flow keywords that affect behavior
        self.control_flow = {
            'if', 'elif', 'else', 'for', 'while', 'try', 'except',
            'finally', 'with', 'break', 'continue', 'return', 'yield'
        }
        
    def simulate_behavior(self, code: str) -> Dict[str, Any]:
        """
        Simulate the behavioral patterns of Python code.
        
        Args:
            code (str): The Python code to analyze
            
        Returns:
            Dict[str, Any]: Behavioral simulation results containing:
                - behavior_safe: Boolean indicating if behavior is safe
                - alerts: List of behavioral alerts and warnings
                - function_behaviors: Detailed behavior analysis per function
                - risk_assessment: Overall risk assessment
                - execution_flow: Predicted execution flow analysis
        """
        if not isinstance(code, str):
            return {
                "behavior_safe": False,
                "alerts": ["Invalid input: code must be a string"],
                "function_behaviors": {},
                "risk_assessment": {"level": "critical", "score": 0},
                "execution_flow": {}
            }
        
        if not code.strip():
            return {
                "behavior_safe": True,
                "alerts": [],
                "function_behaviors": {},
                "risk_assessment": {"level": "safe", "score": 100},
                "execution_flow": {}
            }
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "behavior_safe": False,
                "alerts": [f"Syntax error prevents behavior analysis: {str(e)}"],
                "function_behaviors": {},
                "risk_assessment": {"level": "high", "score": 10},
                "execution_flow": {}
            }
        except Exception as e:
            return {
                "behavior_safe": False,
                "alerts": [f"Failed to parse code for behavior analysis: {str(e)}"],
                "function_behaviors": {},
                "risk_assessment": {"level": "high", "score": 10},
                "execution_flow": {}
            }
        
        # Analyze behavioral patterns
        patterns = []
        function_behaviors = {}
        
        # 1. Analyze function behaviors
        function_behaviors = self._analyze_function_behaviors(tree)
        
        # 2. Detect system operation patterns
        system_patterns = self._detect_system_operations(tree)
        patterns.extend(system_patterns)
        
        # 3. Analyze control flow and complexity
        flow_patterns = self._analyze_control_flow(tree)
        patterns.extend(flow_patterns)
        
        # 4. Detect dangerous patterns
        danger_patterns = self._detect_dangerous_patterns(tree, code)
        patterns.extend(danger_patterns)
        
        # 5. Analyze exception handling
        exception_patterns = self._analyze_exception_handling(tree)
        patterns.extend(exception_patterns)
        
        # 6. Detect infinite loop risks
        loop_patterns = self._detect_loop_risks(tree)
        patterns.extend(loop_patterns)
        
        # 7. Analyze memory and resource usage
        resource_patterns = self._analyze_resource_usage(tree)
        patterns.extend(resource_patterns)
        
        # 8. Detect recursion patterns
        recursion_patterns = self._detect_recursion_patterns(tree)
        patterns.extend(recursion_patterns)
        
        # 9. Analyze execution flow
        execution_flow = self._analyze_execution_flow(tree)
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(patterns, function_behaviors)
        
        # Determine if behavior is safe
        behavior_safe = risk_assessment["level"] in ["safe", "low"]
        
        # Generate alerts
        alerts = [self._format_pattern_alert(pattern) for pattern in patterns]
        
        return {
            "behavior_safe": behavior_safe,
            "alerts": alerts,
            "function_behaviors": {name: self._serialize_function_behavior(fb) 
                                 for name, fb in function_behaviors.items()},
            "risk_assessment": risk_assessment,
            "execution_flow": execution_flow,
            "patterns_detected": len(patterns),
            "detailed_patterns": patterns  # For advanced analysis
        }
    
    def _analyze_function_behaviors(self, tree: ast.AST) -> Dict[str, FunctionBehavior]:
        """Analyze behavior patterns of individual functions."""
        behaviors = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                behavior = self._analyze_single_function(node)
                behaviors[node.name] = behavior
        
        return behaviors
    
    def _analyze_single_function(self, func_node: ast.FunctionDef) -> FunctionBehavior:
        """Analyze behavior of a single function."""
        behavior = FunctionBehavior(name=func_node.name)
        
        # Calculate complexity (simplified cyclomatic complexity)
        behavior.complexity = self._calculate_complexity(func_node)
        
        # Calculate maximum nesting depth
        behavior.max_depth = self._calculate_max_depth(func_node)
        
        # Analyze function calls
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    behavior.calls_made.add(call_name)
                    
                    # Check for recursive calls
                    if call_name == func_node.name:
                        behavior.is_recursive = True
            
            # Analyze variable assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        behavior.variables_modified.add(target.id)
            
            # Analyze loops
            elif isinstance(node, (ast.For, ast.While)):
                loop_type = "for" if isinstance(node, ast.For) else "while"
                behavior.loops.append(f"{loop_type}_loop")
                
                # Check for potential infinite loops
                if isinstance(node, ast.While):
                    if self._has_infinite_loop_risk(node):
                        behavior.has_infinite_loop_risk = True
            
            # Analyze exception handling
            elif isinstance(node, ast.ExceptHandler):
                if node.type and isinstance(node.type, ast.Name):
                    behavior.exceptions_handled.add(node.type.id)
        
        # Detect side effects
        behavior.side_effects = self._detect_function_side_effects(func_node)
        
        return behavior
    
    def _detect_system_operations(self, tree: ast.AST) -> List[BehaviorPattern]:
        """Detect system operation patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                line_num = getattr(node, 'lineno', None)
                
                # File operations
                if any(op in call_name for op in self.file_operations):
                    patterns.append(BehaviorPattern(
                        behavior_type=BehaviorType.FILE_OPERATION,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"File operation detected: {call_name}",
                        line_number=line_num,
                        mitigation="Ensure proper file path validation and error handling"
                    ))
                
                # System calls
                elif any(call in call_name for call in self.system_calls):
                    patterns.append(BehaviorPattern(
                        behavior_type=BehaviorType.SYSTEM_CALL,
                        risk_level=RiskLevel.HIGH,
                        description=f"System call detected: {call_name}",
                        line_number=line_num,
                        mitigation="Validate all inputs to system calls"
                    ))
                
                # Network operations
                elif any(net in call_name for net in self.network_operations):
                    patterns.append(BehaviorPattern(
                        behavior_type=BehaviorType.NETWORK_ACCESS,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"Network access detected: {call_name}",
                        line_number=line_num,
                        mitigation="Implement proper network security and error handling"
                    ))
                
                # Process operations
                elif any(proc in call_name for proc in self.process_operations):
                    patterns.append(BehaviorPattern(
                        behavior_type=BehaviorType.PROCESS_SPAWN,
                        risk_level=RiskLevel.HIGH,
                        description=f"Process spawning detected: {call_name}",
                        line_number=line_num,
                        mitigation="Monitor process creation and resource usage"
                    ))
        
        return patterns
    
    def _analyze_control_flow(self, tree: ast.AST) -> List[BehaviorPattern]:
        """Analyze control flow complexity and patterns."""
        patterns = []
        
        # Count nested control structures
        max_nesting = 0
        current_nesting = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
        
        if max_nesting > 5:
            patterns.append(BehaviorPattern(
                behavior_type=BehaviorType.LOOP_BEHAVIOR,
                risk_level=RiskLevel.MEDIUM,
                description=f"High nesting complexity detected: {max_nesting} levels",
                mitigation="Consider refactoring to reduce complexity"
            ))
        
        return patterns
    
    def _detect_dangerous_patterns(self, tree: ast.AST, code: str) -> List[BehaviorPattern]:
        """Detect dangerous behavioral patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                line_num = getattr(node, 'lineno', None)
                
                if call_name in self.dangerous_patterns:
                    risk_level = RiskLevel.CRITICAL if call_name in ['eval', 'exec'] else RiskLevel.HIGH
                    patterns.append(BehaviorPattern(
                        behavior_type=BehaviorType.DYNAMIC_CODE,
                        risk_level=risk_level,
                        description=f"Dangerous pattern detected: {call_name}",
                        line_number=line_num,
                        mitigation=f"Avoid using {call_name} with untrusted input"
                    ))
        
        # Check for dynamic code construction
        if re.search(r'exec\s*\(\s*.*\+.*\)', code) or re.search(r'eval\s*\(\s*.*\+.*\)', code):
            patterns.append(BehaviorPattern(
                behavior_type=BehaviorType.DYNAMIC_CODE,
                risk_level=RiskLevel.CRITICAL,
                description="Dynamic code construction detected",
                mitigation="Never construct executable code from user input"
            ))
        
        return patterns
    
    def _analyze_exception_handling(self, tree: ast.AST) -> List[BehaviorPattern]:
        """Analyze exception handling patterns."""
        patterns = []
        
        bare_except_count = 0
        total_try_blocks = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                total_try_blocks += 1
                
                for handler in node.handlers:
                    if handler.type is None:  # Bare except
                        bare_except_count += 1
        
        if bare_except_count > 0:
            patterns.append(BehaviorPattern(
                behavior_type=BehaviorType.EXCEPTION_HANDLING,
                risk_level=RiskLevel.MEDIUM,
                description=f"Bare except clauses detected: {bare_except_count}",
                mitigation="Use specific exception types instead of bare except"
            ))
        
        return patterns
    
    def _detect_loop_risks(self, tree: ast.AST) -> List[BehaviorPattern]:
        """Detect potential infinite loop risks."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if self._has_infinite_loop_risk(node):
                    patterns.append(BehaviorPattern(
                        behavior_type=BehaviorType.LOOP_BEHAVIOR,
                        risk_level=RiskLevel.HIGH,
                        description="Potential infinite loop detected",
                        line_number=getattr(node, 'lineno', None),
                        mitigation="Ensure loop has proper termination conditions"
                    ))
            
            elif isinstance(node, ast.For):
                # Check for nested loops with high complexity
                nested_loops = sum(1 for child in ast.walk(node) 
                                 if isinstance(child, (ast.For, ast.While)) and child != node)
                if nested_loops > 2:
                    patterns.append(BehaviorPattern(
                        behavior_type=BehaviorType.LOOP_BEHAVIOR,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"Deeply nested loops detected: {nested_loops + 1} levels",
                        line_number=getattr(node, 'lineno', None),
                        mitigation="Consider optimizing nested loop structure"
                    ))
        
        return patterns
    
    def _analyze_resource_usage(self, tree: ast.AST) -> List[BehaviorPattern]:
        """Analyze memory and resource usage patterns."""
        patterns = []
        
        # Check for large data structure creation
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp) or isinstance(node, ast.GeneratorExp):
                # Complex comprehensions might use significant memory
                if self._is_complex_comprehension(node):
                    patterns.append(BehaviorPattern(
                        behavior_type=BehaviorType.MEMORY_ACCESS,
                        risk_level=RiskLevel.LOW,
                        description="Complex comprehension detected",
                        line_number=getattr(node, 'lineno', None),
                        mitigation="Monitor memory usage for large datasets"
                    ))
        
        return patterns
    
    def _detect_recursion_patterns(self, tree: ast.AST) -> List[BehaviorPattern]:
        """Detect recursion patterns and risks."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_recursive_function(node):
                    # Check for tail recursion optimization potential
                    if not self._has_tail_recursion(node):
                        patterns.append(BehaviorPattern(
                            behavior_type=BehaviorType.RECURSION,
                            risk_level=RiskLevel.MEDIUM,
                            description=f"Non-tail recursive function: {node.name}",
                            function_name=node.name,
                            line_number=getattr(node, 'lineno', None),
                            mitigation="Consider iterative implementation or tail recursion"
                        ))
        
        return patterns
    
    def _analyze_execution_flow(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze predicted execution flow."""
        flow = {
            "entry_points": [],
            "function_call_graph": {},
            "potential_bottlenecks": [],
            "execution_paths": 1
        }
        
        # Find entry points (module-level code)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                flow["entry_points"].append(type(node).__name__)
        
        # Build function call graph
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                calls = []
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        call_name = self._get_call_name(child)
                        if call_name:
                            calls.append(call_name)
                flow["function_call_graph"][node.name] = calls
        
        # Estimate execution paths (simplified)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                flow["execution_paths"] *= 2  # Each if doubles paths
            elif isinstance(node, (ast.For, ast.While)):
                flow["execution_paths"] *= 10  # Loops multiply paths significantly
        
        return flow
    
    def _calculate_risk_assessment(self, patterns: List[BehaviorPattern], 
                                 behaviors: Dict[str, FunctionBehavior]) -> Dict[str, Any]:
        """Calculate overall risk assessment."""
        risk_scores = {
            RiskLevel.SAFE: 0,
            RiskLevel.LOW: 10,
            RiskLevel.MEDIUM: 25,
            RiskLevel.HIGH: 50,
            RiskLevel.CRITICAL: 100
        }
        
        total_risk = sum(risk_scores[pattern.risk_level] for pattern in patterns)
        
        # Add complexity penalties
        for behavior in behaviors.values():
            if behavior.complexity > 10:
                total_risk += 20
            if behavior.has_infinite_loop_risk:
                total_risk += 30
            if behavior.is_recursive and behavior.complexity > 5:
                total_risk += 15
        
        # Determine overall risk level
        if total_risk == 0:
            risk_level = "safe"
        elif total_risk <= 25:
            risk_level = "low"
        elif total_risk <= 75:
            risk_level = "medium"
        elif total_risk <= 150:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        # Calculate safety score (inverse of risk)
        safety_score = max(0, 100 - min(100, total_risk))
        
        return {
            "level": risk_level,
            "score": safety_score,
            "total_risk_points": total_risk,
            "pattern_count": len(patterns),
            "high_risk_patterns": sum(1 for p in patterns if p.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        }
    
    # Helper methods
    def _get_call_name(self, node: ast.Call) -> str:
        """Extract function call name from AST node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return ""
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _calculate_max_depth(self, node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth in a function."""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return get_depth(node)
    
    def _has_infinite_loop_risk(self, node: ast.While) -> bool:
        """Check if a while loop has infinite loop risk."""
        # Simple heuristic: check if loop variable is modified in body
        if isinstance(node.test, ast.Name):
            loop_var = node.test.id
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == loop_var:
                            return False
            return True
        elif isinstance(node.test, ast.Constant) and node.test.value is True:
            return True
        return False
    
    def _detect_function_side_effects(self, node: ast.FunctionDef) -> List[str]:
        """Detect potential side effects in a function."""
        side_effects = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if any(op in call_name for op in self.file_operations):
                    side_effects.append("file_modification")
                elif any(op in call_name for op in self.system_calls):
                    side_effects.append("system_modification")
                elif any(op in call_name for op in self.network_operations):
                    side_effects.append("network_activity")
        
        return side_effects
    
    def _is_complex_comprehension(self, node: Union[ast.ListComp, ast.GeneratorExp]) -> bool:
        """Check if comprehension is complex."""
        # Count generators and conditionals
        generator_count = len(node.generators)
        condition_count = sum(len(gen.ifs) for gen in node.generators)
        return generator_count > 1 or condition_count > 2
    
    def _is_recursive_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is recursive."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if call_name == node.name:
                    return True
        return False
    
    def _has_tail_recursion(self, node: ast.FunctionDef) -> bool:
        """Check if recursive function uses tail recursion."""
        # Simplified check: last statement is return with recursive call
        if node.body:
            last_stmt = node.body[-1]
            if isinstance(last_stmt, ast.Return) and isinstance(last_stmt.value, ast.Call):
                call_name = self._get_call_name(last_stmt.value)
                return call_name == node.name
        return False
    
    def _format_pattern_alert(self, pattern: BehaviorPattern) -> str:
        """Format behavioral pattern as alert string."""
        line_info = f" (line {pattern.line_number})" if pattern.line_number else ""
        func_info = f" in {pattern.function_name}" if pattern.function_name else ""
        return f"[{pattern.risk_level.value.upper()}] {pattern.description}{line_info}{func_info}"
    
    def _serialize_function_behavior(self, behavior: FunctionBehavior) -> Dict[str, Any]:
        """Serialize FunctionBehavior for JSON output."""
        return {
            "name": behavior.name,
            "complexity": behavior.complexity,
            "max_depth": behavior.max_depth,
            "loops": behavior.loops,
            "calls_made": list(behavior.calls_made),
            "variables_modified": list(behavior.variables_modified),
            "exceptions_handled": list(behavior.exceptions_handled),
            "side_effects": behavior.side_effects,
            "is_recursive": behavior.is_recursive,
            "has_infinite_loop_risk": behavior.has_infinite_loop_risk
        }


def simulate_behavior(code: str) -> Dict[str, Any]:
    """
    Main function to simulate behavioral patterns of Python code.
    
    Args:
        code (str): The Python code to analyze
        
    Returns:
        Dict[str, Any]: Behavioral simulation results containing:
            - behavior_safe: Boolean indicating if behavior is safe
            - alerts: List of behavioral alerts and warnings
    """
    simulator = BehavioralSimulator()
    result = simulator.simulate_behavior(code)
    
    # Return simplified format for backward compatibility
    return {
        "behavior_safe": result["behavior_safe"],
        "alerts": result["alerts"]
    }


# Example usage and testing
if __name__ == "__main__":
    # Test with various behavioral patterns
    test_codes = [
        # Safe code
        '''
def safe_calculator(a, b):
    """Safe arithmetic operations."""
    try:
        result = a + b
        return result
    except TypeError:
        return None
''',
        
        # Code with behavioral risks
        '''
import os
import sys

def risky_function(user_input):
    # System calls
    os.system(f"ls {user_input}")
    
    # File operations
    with open(user_input, 'w') as f:
        f.write("data")
    
    # Infinite loop risk
    while True:
        if user_input == "stop":
            break
    
    # Recursion without base case check
    return risky_function(user_input + "1")

def complex_nested_function():
    for i in range(100):
        for j in range(100):
            for k in range(100):
                if i % 2 == 0:
                    if j % 3 == 0:
                        if k % 5 == 0:
                            print(f"{i}, {j}, {k}")
''',
    ]
    
    for i, test_code in enumerate(test_codes, 1):
        print(f"\n=== Test Case {i} ===")
        result = simulate_behavior(test_code)
        print(f"Behavior Safe: {result['behavior_safe']}")
        print(f"Alerts: {len(result['alerts'])}")
        
        if result['alerts']:
            print("Behavioral alerts:")
            for alert in result['alerts']:
                print(f"  - {alert}")
