# validators/security_validator.py


def validate_security(code: str) -> bool:
    """
    Very basic placeholder: disallow dangerous imports.
    """
    blocked = ["import os", "import subprocess", "import shutil", "eval", "exec"]
    return not any(b in code for b in blocked)
