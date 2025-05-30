"""
Custom exceptions for the LLMRouter system.
"""

class LLMRouterError(Exception):
    """Base exception for LLMRouter errors"""
    pass

class ModelLoadError(LLMRouterError):
    """Raised when a model fails to load"""
    pass

class VRAMError(LLMRouterError):
    """Raised for VRAM-related issues"""
    pass

class ConfigurationError(LLMRouterError):
    """Raised for configuration-related issues"""
    pass

class ExecutionError(LLMRouterError):
    """Raised when model execution fails"""
    pass

class ValidationError(LLMRouterError):
    """Raised for input validation failures"""
    pass
