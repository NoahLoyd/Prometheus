import os
import pathlib
from typing import Union, Optional

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
