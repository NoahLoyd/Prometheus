"""
Validation module for Prometheus LLM routing system.

This module provides Pydantic models and validation functions for ensuring
configuration integrity at runtime. It validates model configurations,
routing strategies, and complete system configurations with informative
error messages for debugging and system reliability.
"""

from typing import Dict, List, Any, Optional, Literal, Union
from pathlib import Path
import json
import warnings

from pydantic import BaseModel, Field, field_validator, ValidationError as PydanticValidationError, ConfigDict

from .exceptions import ValidationError, ConfigurationError


class ModelConfig(BaseModel):
    """
    Pydantic model for validating individual model configurations.
    
    Validates essential model parameters including name, type, endpoint,
    and token limits with appropriate constraints and defaults.
    """
    
    model_config = ConfigDict(
        extra='forbid',  # Disallow extra fields
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "name": "gpt-4",
                "type": "remote",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "max_tokens": 4096
            }
        }
    )
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique identifier for the model"
    )
    
    type: Literal["local", "remote"] = Field(
        ...,
        description="Model deployment type - local or remote"
    )
    
    endpoint: Optional[str] = Field(
        None,
        description="API endpoint URL for remote models"
    )
    
    max_tokens: int = Field(
        ...,
        ge=1,
        le=100000,
        description="Maximum number of tokens the model can process"
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate model name format and characters."""
        if not v.replace('-', '').replace('_', '').replace('.', '').isalnum():
            raise ValueError(
                "Model name must contain only alphanumeric characters, "
                "hyphens, underscores, and periods"
            )
        return v.strip()
    
    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v, info):
        """Validate endpoint requirements based on model type."""
        model_type = info.data.get('type') if info.data else None
        
        if model_type == 'remote' and not v:
            raise ValueError("Remote models must have an endpoint specified")
        
        if model_type == 'local' and v:
            raise ValueError("Local models should not have an endpoint specified")
        
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("Endpoint must be a valid HTTP or HTTPS URL")
        
        return v
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        """Validate token limits are reasonable."""
        if v <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        # Warn about unusually high token counts
        if v > 32768:
            warnings.warn(
                f"max_tokens ({v}) is unusually high and may cause "
                "performance issues or API errors"
            )
        
        return v


class RouterConfig(BaseModel):
    """
    Pydantic model for validating router configurations.
    
    Validates routing strategy and associated model configurations
    with comprehensive validation of strategy types and model lists.
    """
    
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "strategy": "priority",
                "models": [
                    {
                        "name": "local-llama",
                        "type": "local",
                        "max_tokens": 2048
                    },
                    {
                        "name": "gpt-4",
                        "type": "remote",
                        "endpoint": "https://api.openai.com/v1/chat/completions",
                        "max_tokens": 4096
                    }
                ]
            }
        }
    )
    
    strategy: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Routing strategy identifier"
    )
    
    models: List[ModelConfig] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of model configurations for routing"
    )
    
    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        """Validate strategy name format."""
        allowed_strategies = {
            'round_robin', 'priority', 'load_balance', 'fallback',
            'voting', 'ensemble', 'adaptive', 'performance_based'
        }
        
        strategy_name = v.lower().strip()
        if strategy_name not in allowed_strategies:
            raise ValueError(
                f"Unknown strategy '{v}'. Allowed strategies: "
                f"{', '.join(sorted(allowed_strategies))}"
            )
        
        return strategy_name
    
    @field_validator('models')
    @classmethod
    def validate_models(cls, v):
        """Validate model list constraints."""
        if not v:
            raise ValueError("At least one model must be configured")
        
        # Check for duplicate model names
        model_names = [model.name for model in v]
        if len(model_names) != len(set(model_names)):
            duplicates = [name for name in model_names if model_names.count(name) > 1]
            raise ValueError(
                f"Duplicate model names found: {', '.join(set(duplicates))}"
            )
        
        # Validate type distribution
        local_models = [m for m in v if m.type == 'local']
        remote_models = [m for m in v if m.type == 'remote']
        
        if len(remote_models) > 10:
            warnings.warn(
                f"Large number of remote models ({len(remote_models)}) "
                "may impact performance"
            )
        
        return v


def validate_model_config(model_config: Dict[str, Any]) -> ModelConfig:
    """
    Validate a single model configuration dictionary.
    
    Args:
        model_config: Dictionary containing model configuration data
        
    Returns:
        Validated ModelConfig instance
        
    Raises:
        ValueError: If validation fails with detailed error information
        
    Example:
        >>> config = {
        ...     "name": "my-model",
        ...     "type": "local",
        ...     "max_tokens": 2048
        ... }
        >>> validated = validate_model_config(config)
        >>> print(validated.name)
        my-model
    """
    try:
        return ModelConfig(**model_config)
    
    except PydanticValidationError as e:
        # Extract and format validation errors
        error_details = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_msg = error['msg']
            error_details.append(f"{field_path}: {error_msg}")
        
        raise ValueError(
            f"Model configuration validation failed:\n" +
            "\n".join(f"  • {detail}" for detail in error_details)
        ) from e
    
    except Exception as e:
        raise ValueError(f"Unexpected error during model validation: {str(e)}") from e


def validate_routing_config(routing_config: Dict[str, Any]) -> RouterConfig:
    """
    Validate a complete routing configuration dictionary.
    
    Args:
        routing_config: Dictionary containing routing configuration data
        
    Returns:
        Validated RouterConfig instance
        
    Raises:
        ValueError: If validation fails with detailed error information
        
    Example:
        >>> config = {
        ...     "strategy": "round_robin",
        ...     "models": [
        ...         {"name": "model1", "type": "local", "max_tokens": 1024},
        ...         {"name": "model2", "type": "remote", 
        ...          "endpoint": "https://api.example.com", "max_tokens": 2048}
        ...     ]
        ... }
        >>> validated = validate_routing_config(config)
        >>> print(validated.strategy)
        round_robin
    """
    try:
        return RouterConfig(**routing_config)
    
    except PydanticValidationError as e:
        # Extract and format validation errors with context
        error_details = []
        for error in e.errors():
            field_path = " -> ".join(str(loc) for loc in error['loc'])
            error_msg = error['msg']
            
            # Add context for common errors
            if 'models' in field_path and 'endpoint' in error_msg:
                error_msg += " (Remote models require valid HTTP/HTTPS endpoints)"
            elif 'strategy' in field_path:
                error_msg += " (Use: round_robin, priority, load_balance, etc.)"
            
            error_details.append(f"{field_path}: {error_msg}")
        
        raise ValueError(
            f"Routing configuration validation failed:\n" +
            "\n".join(f"  • {detail}" for detail in error_details)
        ) from e
    
    except Exception as e:
        raise ValueError(f"Unexpected error during routing validation: {str(e)}") from e


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an entire configuration dictionary with comprehensive checks.
    
    This function validates the complete system configuration including
    models, registry settings, hardware preferences, and routing configurations.
    
    Args:
        config: Complete system configuration dictionary
        
    Returns:
        Validated and normalized configuration dictionary
        
    Raises:
        ValueError: If any part of the configuration is invalid
        ConfigurationError: If configuration structure is fundamentally invalid
        
    Example:
        >>> config = {
        ...     "models": {
        ...         "local-model": {
        ...             "type": "local",
        ...             "model_path": "/path/to/model",
        ...             "max_sequence_length": 2048
        ...         }
        ...     },
        ...     "registry": {"max_cached_models": 3},
        ...     "hardware": {"allow_cpu_fallback": True}
        ... }
        >>> validated = validate_config(config)
    """
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a dictionary")
    
    validated_config = config.copy()
    validation_errors = []
    
    try:
        # Validate top-level structure
        _validate_config_structure(config, validation_errors)
        
        # Validate individual model configurations if present
        if 'models' in config:
            validated_config['models'] = _validate_models_section(
                config['models'], validation_errors
            )
        
        # Validate registry configuration
        if 'registry' in config:
            validated_config['registry'] = _validate_registry_section(
                config['registry'], validation_errors
            )
        
        # Validate hardware configuration
        if 'hardware' in config:
            validated_config['hardware'] = _validate_hardware_section(
                config['hardware'], validation_errors
            )
        
        # Validate routing configurations if present
        if 'routing' in config:
            validated_config['routing'] = _validate_routing_section(
                config['routing'], validation_errors
            )
        
        # Check for validation errors
        if validation_errors:
            raise ValueError(
                f"Configuration validation failed:\n" +
                "\n".join(f"  • {error}" for error in validation_errors)
            )
        
        return validated_config
    
    except ValueError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Configuration validation error: {str(e)}") from e


def _validate_config_structure(config: Dict[str, Any], errors: List[str]) -> None:
    """Validate basic configuration structure."""
    required_sections = {'models'}
    optional_sections = {'registry', 'hardware', 'routing', 'logging'}
    
    # Check for unknown sections
    unknown_sections = set(config.keys()) - required_sections - optional_sections
    if unknown_sections:
        errors.append(f"Unknown configuration sections: {', '.join(unknown_sections)}")
    
    # Check for missing required sections
    missing_sections = required_sections - set(config.keys())
    if missing_sections:
        errors.append(f"Missing required sections: {', '.join(missing_sections)}")


def _validate_models_section(models: Any, errors: List[str]) -> Dict[str, Any]:
    """Validate the models section of configuration."""
    if not isinstance(models, dict):
        errors.append("'models' section must be a dictionary")
        return {}
    
    if not models:
        errors.append("'models' section cannot be empty")
        return {}
    
    validated_models = {}
    
    for model_name, model_config in models.items():
        try:
            # Ensure model_config is a dictionary
            if not isinstance(model_config, dict):
                errors.append(f"Model '{model_name}' configuration must be a dictionary")
                continue
            
            # Add model name to config for validation
            config_with_name = {**model_config, 'name': model_name}
            
            # Validate basic required fields
            if 'type' not in model_config:
                errors.append(f"Model '{model_name}' missing required 'type' field")
                continue
            
            # Set max_tokens from max_sequence_length if not present
            if 'max_tokens' not in config_with_name and 'max_sequence_length' in model_config:
                config_with_name['max_tokens'] = model_config['max_sequence_length']
            elif 'max_tokens' not in config_with_name:
                config_with_name['max_tokens'] = 2048  # Default value
            
            validated_models[model_name] = config_with_name
            
        except Exception as e:
            errors.append(f"Error validating model '{model_name}': {str(e)}")
    
    return validated_models


def _validate_registry_section(registry: Any, errors: List[str]) -> Dict[str, Any]:
    """Validate the registry section of configuration."""
    if not isinstance(registry, dict):
        errors.append("'registry' section must be a dictionary")
        return {}
    
    validated_registry = registry.copy()
    
    # Validate specific registry fields
    if 'max_cached_models' in registry:
        max_cached = registry['max_cached_models']
        if not isinstance(max_cached, int) or max_cached < 1:
            errors.append("'max_cached_models' must be a positive integer")
        elif max_cached > 100:
            errors.append("'max_cached_models' should not exceed 100 for performance reasons")
    
    if 'cache_ttl_seconds' in registry:
        ttl = registry['cache_ttl_seconds']
        if not isinstance(ttl, (int, float)) or ttl < 0:
            errors.append("'cache_ttl_seconds' must be a non-negative number")
    
    return validated_registry


def _validate_hardware_section(hardware: Any, errors: List[str]) -> Dict[str, Any]:
    """Validate the hardware section of configuration."""
    if not isinstance(hardware, dict):
        errors.append("'hardware' section must be a dictionary")
        return {}
    
    validated_hardware = hardware.copy()
    
    # Validate boolean fields
    boolean_fields = ['prefer_larger_models', 'allow_cpu_fallback']
    for field in boolean_fields:
        if field in hardware and not isinstance(hardware[field], bool):
            errors.append(f"'{field}' must be a boolean value")
    
    # Validate percentage fields
    if 'min_free_vram_percent' in hardware:
        percent = hardware['min_free_vram_percent']
        if not isinstance(percent, (int, float)) or not 0 <= percent <= 100:
            errors.append("'min_free_vram_percent' must be a number between 0 and 100")
    
    # Validate CUDA device preferences
    if 'cuda_device_preference' in hardware:
        devices = hardware['cuda_device_preference']
        if not isinstance(devices, list):
            errors.append("'cuda_device_preference' must be a list")
        elif not all(isinstance(d, str) for d in devices):
            errors.append("All CUDA device preferences must be strings")
    
    return validated_hardware


def _validate_routing_section(routing: Any, errors: List[str]) -> Dict[str, Any]:
    """Validate the routing section of configuration."""
    if not isinstance(routing, dict):
        errors.append("'routing' section must be a dictionary")
        return {}
    
    validated_routing = {}
    
    for route_name, route_config in routing.items():
        try:
            validated_route = validate_routing_config(route_config)
            validated_routing[route_name] = validated_route.model_dump()
        except ValueError as e:
            errors.append(f"Routing '{route_name}' validation failed: {str(e)}")
    
    return validated_routing


# Additional utility functions for specific validation tasks

def validate_path(path: Union[str, Path]) -> Path:
    """
    Validate and normalize a file system path.
    
    Args:
        path: File system path as string or Path object
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If path is invalid or inaccessible
    """
    try:
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        return path_obj.resolve()
    
    except Exception as e:
        raise ValueError(f"Invalid path '{path}': {str(e)}") from e


def validate_vram_requirements(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate VRAM requirement specifications.
    
    Args:
        requirements: Dictionary containing VRAM requirements
        
    Returns:
        Validated requirements dictionary
        
    Raises:
        ValueError: If VRAM requirements are invalid
    """
    if not isinstance(requirements, dict):
        raise ValueError("VRAM requirements must be a dictionary")
    
    validated = {}
    
    for field in ['min_vram_bytes', 'optimal_vram_bytes']:
        if field in requirements:
            value = requirements[field]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"'{field}' must be a non-negative integer")
            if value > 1024 * 1024 * 1024 * 1024:  # 1TB limit
                raise ValueError(f"'{field}' exceeds reasonable limits")
            validated[field] = value
    
    # Ensure min <= optimal if both are specified
    if 'min_vram_bytes' in validated and 'optimal_vram_bytes' in validated:
        if validated['min_vram_bytes'] > validated['optimal_vram_bytes']:
            raise ValueError(
                "min_vram_bytes cannot be greater than optimal_vram_bytes"
            )
    
    return {**requirements, **validated}
