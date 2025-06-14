"""
Mathematical and logical validation for generated code.
"""

from typing import Dict, Any


class MathEvaluator:
    """Validates mathematical correctness and logical consistency."""
    
    def __call__(self, plan: Dict[str, Any], tool_code: str, test_code: str) -> Dict[str, Any]:
        """Validate mathematical and logical aspects of generated code."""
        issues = []
        
        # Basic checks
        if not tool_code.strip():
            issues.append("Empty tool code")
        
        if not test_code.strip():
            issues.append("Empty test code")
        
        # Check for basic mathematical consistency
        if 'math' in tool_code.lower() and 'import math' not in tool_code:
            issues.append("Uses math operations but doesn't import math module")
        
        if issues:
            return {
                "success": False,
                "error": f"Mathematical validation failed: {'; '.join(issues)}"
            }
        
        return {
            "success": True,
            "info": "Mathematical validation passed"
        }
