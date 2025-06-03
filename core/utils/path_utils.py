import os
import pathlib
import importlib
import logging
from typing import Union, Optional
from types import ModuleType

# Set up logger for import operations
logger = logging.getLogger("promethyn.imports")

class SecurityError(Exception):
    """Custom exception for security-related violations in path operations."""
    pass

def safe_path_join(base_dir: Union[str, pathlib.Path], *paths: str, 
                   must_exist: bool = False) -> pathlib.Path:
    """
    Safely join paths while preventing directory traversal attacks and ensuring paths remain within base_dir.
    
    Args:
        base_dir: The base directory that all paths must remain within
        *paths: Path components to join
        must_exist: If True, raises SecurityError if the final path doesn't exist
        
    Returns:
        pathlib.Path: The normalized, absolute path
        
    Raises:
        SecurityError: If path traversal is attempted or path exists outside base_dir
        ValueError: If base_dir or paths are empty/None
    """
    if not base_dir or not paths:
        raise ValueError("Base directory and at least one path component are required")
    
    # Convert base_dir to absolute Path object
    try:
        base_dir = pathlib.Path(base_dir).resolve()
    except Exception as e:
        raise SecurityError(f"Invalid base directory: {str(e)}")
    
    # Join paths and resolve to absolute path
    try:
        final_path = pathlib.Path(base_dir).joinpath(*paths).resolve()
    except Exception as e:
        raise SecurityError(f"Error joining paths: {str(e)}")
    
    # Verify the path is within base_dir
    try:
        final_path.relative_to(base_dir)
    except ValueError:
        raise SecurityError(f"Path {final_path} is outside of base directory {base_dir}")
    
    # Check if path exists if required
    if must_exist and not final_path.exists():
        raise SecurityError(f"Path does not exist: {final_path}")
        
    return final_path

def import_validator(name: str) -> Optional[ModuleType]:
    """
    Dynamically import validator modules from multiple fallback directories.
    
    Attempts to import a validator module by trying different package paths in order:
    1. "validators.{name}"
    2. "core.validators.{name}"
    
    If all import attempts fail, returns None without raising exceptions.
    
    Args:
        name: The name of the validator module to import (without package prefix)
        
    Returns:
        ModuleType: The imported module if successful
        None: If all import attempts fail
        
    Example:
        >>> validator = import_validator("email")
        >>> if validator:
        ...     result = validator.validate("test@example.com")
    """
    if not name or not isinstance(name, str):
        logger.warning(f"Invalid validator name provided: {repr(name)}")
        return None
    
    # List of package paths to try in order
    fallback_paths = [
        f"validators.{name}",
        f"core.validators.{name}"
    ]
    
    for module_path in fallback_paths:
        try:
            logger.debug(f"Attempting to import validator module: {module_path}")
            module = importlib.import_module(module_path)
            logger.info(f"Successfully imported validator module: {module_path}")
            return module
            
        except ImportError as e:
            logger.debug(f"Failed to import {module_path}: {str(e)}")
            continue
            
        except Exception as e:
            logger.warning(f"Unexpected error importing {module_path}: {str(e)}")
            continue
    
    logger.info(f"No validator module found for '{name}' in any fallback directory")
    return None
