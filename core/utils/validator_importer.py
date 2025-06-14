"""
Validator Importer Utility for Promethyn AGI System

This module provides safe, dynamic importing of validator modules from the validators/ directory.
Supports flexible module loading with proper error handling and logging for the self-coding AGI system.

Author: Promethyn AGI System
Version: 1.0.0
"""

import importlib
import importlib.util
import os
import sys
import logging
from typing import Optional, Any
import traceback


def import_validator(name: str) -> Optional[Any]:
    """
    Dynamically import a validator module from the validators/ directory.
    
    This function safely loads Python modules containing validators, handling
    various naming conventions and providing detailed error reporting for
    debugging the AGI's self-coding capabilities.
    
    :param name: Name of the validator module (without .py extension)
                 Supports snake_case names like 'security_validator'
    :return: Imported module object or None if import fails
    :raises: Clean error messages for missing files or invalid modules
    """
    logger = logging.getLogger("Promethyn.ValidatorImporter")
    
    # Normalize the validator name (handle various input formats)
    validator_name = name.strip().lower()
    if validator_name.endswith('.py'):
        validator_name = validator_name[:-3]
    
    # Construct possible file paths
    base_paths = [
        f"validators/{validator_name}.py",
        f"validators/extended_validators/{validator_name}.py",
        f"core/validators/{validator_name}.py",
        f"validators/core/{validator_name}.py"
    ]
    
    validator_file_path = None
    
    # Find the actual file path
    for path in base_paths:
        if os.path.exists(path):
            validator_file_path = path
            break
    
    if not validator_file_path:
        error_msg = (
            f"Validator module '{name}' not found. "
            f"Searched paths: {', '.join(base_paths)}. "
            f"Please ensure the validator file exists in the validators/ directory structure."
        )
        logger.warning(error_msg)
        return None
    
    try:
        # Convert file path to module path
        module_name = validator_file_path.replace('/', '.').replace('\\', '.').replace('.py', '')
        
        # Check if module is already loaded and reload if necessary
        if module_name in sys.modules:
            logger.debug(f"Reloading existing validator module: {module_name}")
            importlib.reload(sys.modules[module_name])
            return sys.modules[module_name]
        
        # Use importlib.util for safe dynamic import
        spec = importlib.util.spec_from_file_location(module_name, validator_file_path)
        
        if spec is None:
            error_msg = f"Failed to create module spec for validator '{name}' at path '{validator_file_path}'"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        if spec.loader is None:
            error_msg = f"Module spec has no loader for validator '{name}' at path '{validator_file_path}'"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # Create and execute the module
        module = importlib.util.module_from_spec(spec)
        
        # Add to sys.modules before execution to handle circular imports
        sys.modules[module_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as exec_error:
            # Remove from sys.modules if execution fails
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise exec_error
        
        logger.info(f"Successfully imported validator module: {name} from {validator_file_path}")
        return module
        
    except FileNotFoundError as e:
        error_msg = f"Validator file not found: {validator_file_path}. Details: {str(e)}"
        logger.error(error_msg)
        return None
        
    except ImportError as e:
        error_msg = f"Failed to import validator '{name}': {str(e)}"
        logger.error(error_msg)
        return None
        
    except SyntaxError as e:
        error_msg = (
            f"Syntax error in validator '{name}' at {validator_file_path}. "
            f"Line {e.lineno}: {str(e)}"
        )
        logger.error(error_msg)
        return None
        
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = (
            f"Unexpected error importing validator '{name}' from {validator_file_path}: "
            f"{str(e)}\n{tb_str}"
        )
        logger.error(error_msg)
        return None


def validate_validator_module(module: Any, validator_name: str) -> bool:
    """
    Validate that an imported module contains proper validator functionality.
    
    :param module: Imported validator module
    :param validator_name: Name of the validator for logging
    :return: True if module appears to be a valid validator
    """
    logger = logging.getLogger("Promethyn.ValidatorImporter")
    
    if module is None:
        return False
    
    # Check for common validator patterns
    validator_indicators = [
        'validate',  # validate function
        'run',       # run method
        '__call__',  # callable class
        'check',     # check function
        'assess',    # assess function
        'scan',      # scan function
        'analyze'    # analyze function
    ]
    
    found_indicators = []
    for indicator in validator_indicators:
        if hasattr(module, indicator) and callable(getattr(module, indicator)):
            found_indicators.append(indicator)
    
    # Check for validator classes (look for classes that might be validators)
    validator_classes = []
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                (any(keyword in attr_name.lower() for keyword in ['validator', 'assessor', 'scanner', 'checker']))):
                validator_classes.append(attr_name)
    
    is_valid = len(found_indicators) > 0 or len(validator_classes) > 0
    
    if is_valid:
        logger.debug(f"Validator '{validator_name}' validation passed. "
                    f"Found indicators: {found_indicators}, "
                    f"Found classes: {validator_classes}")
    else:
        logger.warning(f"Validator '{validator_name}' may not be properly structured. "
                      f"No standard validator patterns found.")
    
    return is_valid


def list_available_validators() -> list:
    """
    List all available validator modules in the validators/ directory structure.
    
    :return: List of available validator module names
    """
    logger = logging.getLogger("Promethyn.ValidatorImporter")
    validators = []
    
    search_dirs = [
        "validators",
        "validators/extended_validators", 
        "core/validators",
        "validators/core"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            try:
                for file in os.listdir(search_dir):
                    if (file.endswith('.py') and 
                        not file.startswith('__') and 
                        file != 'extended_validators.py'):
                        validator_name = file[:-3]
                        if validator_name not in validators:
                            validators.append(validator_name)
            except PermissionError as e:
                logger.warning(f"Permission denied accessing validator directory '{search_dir}': {e}")
            except Exception as e:
                logger.warning(f"Error listing validators in '{search_dir}': {e}")
    
    logger.info(f"Found {len(validators)} available validators: {validators}")
    return validators


def get_validator_info(name: str) -> dict:
    """
    Get detailed information about a specific validator module.
    
    :param name: Name of the validator module
    :return: Dictionary containing validator information
    """
    logger = logging.getLogger("Promethyn.ValidatorImporter")
    
    module = import_validator(name)
    if module is None:
        return {
            "name": name,
            "status": "not_found",
            "error": f"Validator '{name}' could not be imported"
        }
    
    info = {
        "name": name,
        "status": "loaded",
        "module_path": getattr(module, '__file__', 'unknown'),
        "is_valid": validate_validator_module(module, name),
        "attributes": [],
        "classes": [],
        "functions": []
    }
    
    # Analyze module contents
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            attr = getattr(module, attr_name)
            info["attributes"].append(attr_name)
            
            if isinstance(attr, type):
                info["classes"].append(attr_name)
            elif callable(attr):
                info["functions"].append(attr_name)
    
    # Check for dependencies
    dependencies = getattr(module, 'REQUIRES', [])
    optional_deps = getattr(module, 'OPTIONAL', [])
    
    if dependencies or optional_deps:
        info["dependencies"] = {
            "required": dependencies,
            "optional": optional_deps
        }
    
    return info


# Convenience function for backward compatibility
def import_validator_fallback(name: str) -> Optional[Any]:
    """
    Fallback validator import function with legacy compatibility.
    
    :param name: Name of the validator module
    :return: Imported module or None
    """
    return import_validator(name)
