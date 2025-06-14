"""
Validator Importer Utility for Promethyn AGI System

This module provides safe, dynamic importing of validator functions and classes using dotted module paths.
Supports flexible module loading with proper error handling and logging for the self-coding AGI system.

Author: Promethyn AGI System
Version: 2.0.0
"""

import importlib
import importlib.util
import os
import sys
import logging
from typing import Optional, Any, Union, Callable
import traceback
from pathlib import Path

# Configure logger for Promethyn
logger = logging.getLogger("promethyn.validator_importer")


def import_validator(validator_path: str) -> Any:
    """
    Dynamically import a validator function or class from a dotted module path.
    
    This function safely loads Python functions or classes from validator modules,
    supporting dotted notation like 'validators.security_validator.validate_security'.
    
    Args:
        validator_path: Dotted module path to the validator function or class
                       Examples:
                       - "validators.security_validator.validate_security"
                       - "core.validators.code_validator.CodeValidator"
                       - "validators.extended_validators.custom_validator"
    
    Returns:
        The imported function, class, or other object
        
    Raises:
        ImportError: If the module or attribute cannot be imported with detailed error message
        
    Examples:
        >>> validate_func = import_validator("validators.security_validator.validate_security")
        >>> result = validate_func(some_data)
        
        >>> ValidatorClass = import_validator("validators.code_validator.CodeValidator")
        >>> validator = ValidatorClass()
    """
    if not validator_path or not isinstance(validator_path, str):
        error_msg = f"Invalid validator_path: {validator_path!r}. Must be a non-empty string."
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    validator_path = validator_path.strip()
    if not validator_path:
        error_msg = "Validator path cannot be empty or whitespace only"
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    logger.debug(f"Attempting to import validator: {validator_path}")
    
    # Split the path into module and attribute parts
    path_parts = validator_path.split('.')
    if len(path_parts) < 2:
        error_msg = (
            f"Invalid validator path '{validator_path}'. "
            f"Expected format: 'module.submodule.function_or_class' "
            f"(minimum 2 parts separated by dots)"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)
    
    # The last part is the function/class name, everything else is the module path
    module_path = '.'.join(path_parts[:-1])
    attribute_name = path_parts[-1]
    
    logger.debug(f"Module path: {module_path}, Attribute: {attribute_name}")
    
    try:
        # First, try to import the module directly
        try:
            module = importlib.import_module(module_path)
            logger.debug(f"Successfully imported module: {module_path}")
        except ModuleNotFoundError as e:
            # If direct import fails, try alternative paths for Promethyn structure
            alternative_paths = _get_alternative_module_paths(module_path)
            module = None
            
            for alt_path in alternative_paths:
                try:
                    logger.debug(f"Trying alternative module path: {alt_path}")
                    module = importlib.import_module(alt_path)
                    logger.debug(f"Successfully imported module via alternative path: {alt_path}")
                    break
                except ModuleNotFoundError:
                    continue
            
            if module is None:
                error_msg = (
                    f"Cannot import module '{module_path}' for validator '{validator_path}'. "
                    f"Tried paths: {[module_path] + alternative_paths}. "
                    f"Original error: {str(e)}"
                )
                logger.error(error_msg)
                raise ImportError(error_msg) from e
        
        # Get the specific attribute (function/class) from the module
        if not hasattr(module, attribute_name):
            available_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
            error_msg = (
                f"Module '{module_path}' has no attribute '{attribute_name}' for validator '{validator_path}'. "
                f"Available attributes: {available_attrs}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        validator_object = getattr(module, attribute_name)
        
        # Validate that the imported object is callable or a class
        if not (callable(validator_object) or isinstance(validator_object, type)):
            logger.warning(
                f"Imported validator '{validator_path}' is not callable or a class. "
                f"Type: {type(validator_object).__name__}"
            )
        
        logger.info(f"Successfully imported validator: {validator_path}")
        return validator_object
        
    except ImportError:
        # Re-raise ImportError as-is to preserve the detailed error message
        raise
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = (
            f"Unexpected error importing validator '{validator_path}': "
            f"{type(e).__name__}: {str(e)}\n"
            f"Traceback:\n{tb_str}"
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e


def _get_alternative_module_paths(module_path: str) -> list[str]:
    """
    Generate alternative module paths to try for Promethyn's directory structure.
    
    Args:
        module_path: Original module path
        
    Returns:
        List of alternative module paths to try
    """
    alternatives = []
    
    # Handle common Promethyn directory structures
    if module_path.startswith('validators.'):
        # Try core.validators prefix
        alternatives.append(module_path.replace('validators.', 'core.validators.', 1))
        
    elif module_path.startswith('core.validators.'):
        # Try without core prefix
        alternatives.append(module_path.replace('core.validators.', 'validators.', 1))
        
    elif not module_path.startswith(('validators.', 'core.')):
        # Try adding validators prefix
        alternatives.append(f'validators.{module_path}')
        alternatives.append(f'core.validators.{module_path}')
    
    # Try with extended_validators subdirectory
    if 'extended_validators' not in module_path:
        if module_path.startswith('validators.'):
            base = module_path.replace('validators.', '', 1)
            alternatives.append(f'validators.extended_validators.{base}')
        else:
            alternatives.append(f'validators.extended_validators.{module_path}')
    
    return alternatives


def validate_validator_object(validator_object: Any, validator_path: str) -> bool:
    """
    Validate that an imported object appears to be a proper validator.
    
    Args:
        validator_object: The imported validator function or class
        validator_path: Path used to import the validator (for logging)
        
    Returns:
        True if the object appears to be a valid validator
    """
    logger.debug(f"Validating validator object for path: {validator_path}")
    
    if validator_object is None:
        logger.warning(f"Validator object is None for path: {validator_path}")
        return False
    
    # Check if it's a callable (function or callable class)
    if callable(validator_object):
        logger.debug(f"Validator '{validator_path}' is callable")
        return True
    
    # Check if it's a class with common validator methods
    if isinstance(validator_object, type):
        validator_methods = ['validate', 'run', '__call__', 'check', 'assess', 'scan', 'analyze']
        found_methods = [method for method in validator_methods 
                        if hasattr(validator_object, method)]
        
        if found_methods:
            logger.debug(f"Validator class '{validator_path}' has methods: {found_methods}")
            return True
        else:
            logger.warning(f"Validator class '{validator_path}' has no standard validator methods")
            return False
    
    logger.warning(f"Validator '{validator_path}' is neither callable nor a class with validator methods")
    return False


def list_available_validators(base_path: str = "validators") -> dict[str, list[str]]:
    """
    List all available validator modules and their exportable functions/classes.
    
    Args:
        base_path: Base directory to search for validators
        
    Returns:
        Dictionary mapping module paths to lists of available attributes
    """
    logger.debug(f"Listing available validators in base path: {base_path}")
    
    validators = {}
    search_dirs = [
        "validators",
        "core/validators", 
        "validators/extended_validators",
        "core/validators/extended_validators"
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        try:
            for root, dirs, files in os.walk(search_dir):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != '__pycache__']
                
                for file in files:
                    if not file.endswith('.py') or file.startswith('__'):
                        continue
                    
                    # Convert file path to module path
                    file_path = os.path.join(root, file)
                    module_name = file[:-3]  # Remove .py extension
                    
                    # Convert filesystem path to dotted module path
                    rel_path = os.path.relpath(file_path, '.').replace(os.sep, '.')
                    module_path = rel_path[:-3]  # Remove .py extension
                    
                    try:
                        module = importlib.import_module(module_path)
                        
                        # Get exportable attributes (functions and classes)
                        exportable = []
                        for attr_name in dir(module):
                            if attr_name.startswith('_'):
                                continue
                            
                            attr = getattr(module, attr_name)
                            if callable(attr) or isinstance(attr, type):
                                exportable.append(attr_name)
                        
                        if exportable:
                            validators[module_path] = exportable
                            
                    except Exception as e:
                        logger.debug(f"Could not analyze module {module_path}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error scanning directory {search_dir}: {e}")
    
    logger.info(f"Found {len(validators)} validator modules with exportable attributes")
    return validators


def get_validator_info(validator_path: str) -> dict[str, Any]:
    """
    Get detailed information about a specific validator.
    
    Args:
        validator_path: Dotted path to the validator
        
    Returns:
        Dictionary containing validator information
    """
    info = {
        "path": validator_path,
        "status": "unknown",
        "type": None,
        "callable": False,
        "module": None,
        "attribute": None,
        "error": None
    }
    
    try:
        validator_object = import_validator(validator_path)
        
        info.update({
            "status": "success",
            "type": type(validator_object).__name__,
            "callable": callable(validator_object),
            "is_valid": validate_validator_object(validator_object, validator_path)
        })
        
        # Split path to get module and attribute info
        path_parts = validator_path.split('.')
        if len(path_parts) >= 2:
            info["module"] = '.'.join(path_parts[:-1])
            info["attribute"] = path_parts[-1]
        
        # Get docstring if available
        if hasattr(validator_object, '__doc__') and validator_object.__doc__:
            info["docstring"] = validator_object.__doc__.strip()
        
        # For classes, get method information
        if isinstance(validator_object, type):
            methods = [method for method in dir(validator_object) 
                      if not method.startswith('_') and callable(getattr(validator_object, method))]
            info["methods"] = methods
            
    except ImportError as e:
        info.update({
            "status": "import_error", 
            "error": str(e)
        })
    except Exception as e:
        info.update({
            "status": "unexpected_error",
            "error": f"{type(e).__name__}: {str(e)}"
        })
    
    return info


def reload_validator(validator_path: str) -> Any:
    """
    Reload a validator module and return the specified validator object.
    
    Useful for development when validator code has changed.
    
    Args:
        validator_path: Dotted path to the validator
        
    Returns:
        The reloaded validator function or class
        
    Raises:
        ImportError: If reloading fails
    """
    logger.debug(f"Reloading validator: {validator_path}")
    
    path_parts = validator_path.split('.')
    if len(path_parts) < 2:
        raise ImportError(f"Invalid validator path for reload: {validator_path}")
    
    module_path = '.'.join(path_parts[:-1])
    
    # Force reload the module if it exists in sys.modules
    if module_path in sys.modules:
        try:
            importlib.reload(sys.modules[module_path])
            logger.debug(f"Reloaded module: {module_path}")
        except Exception as e:
            logger.warning(f"Failed to reload module {module_path}: {e}")
    
    # Import the validator again
    return import_validator(validator_path)


# Backward compatibility function (maintains existing interface)
def import_validator_by_name(name: str) -> Optional[Any]:
    """
    Import a validator by simple name (backward compatibility).
    
    Args:
        name: Simple validator name (e.g., "security_validator")
        
    Returns:
        Imported module or None if not found
    """
    logger.debug(f"Importing validator by name (legacy): {name}")
    
    # Try common validator module patterns
    common_patterns = [
        f"validators.{name}",
        f"core.validators.{name}",
        f"validators.extended_validators.{name}"
    ]
    
    for pattern in common_patterns:
        try:
            module = importlib.import_module(pattern)
            logger.debug(f"Successfully imported validator module: {pattern}")
            return module
        except ModuleNotFoundError:
            continue
    
    logger.warning(f"Could not find validator module for name: {name}")
    return None


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    print("Promethyn Validator Importer - Test Mode")
    print("=" * 50)
    
    # Test listing available validators
    print("\nAvailable validators:")
    available = list_available_validators()
    for module_path, attributes in available.items():
        print(f"  {module_path}: {attributes}")
    
    # Test importing a validator (example)
    test_paths = [
        "validators.example_validator.validate",
        "core.validators.security_validator.SecurityValidator",
        "validators.extended_validators.custom_validator.run"
    ]
    
    print("\nTesting validator imports:")
    for test_path in test_paths:
        try:
            result = import_validator(test_path)
            print(f"  ✓ {test_path}: {type(result).__name__}")
        except ImportError as e:
            print(f"  ✗ {test_path}: {e}")
    
    print("\nValidator importer ready for production use.")
