from typing import Dict, Optional, Type, Any
from pathlib import Path
import json
import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
import weakref
import torch

from .vram_checker import VRAMChecker
from .base_llm import BaseLLM

@dataclass
class ModelMetadata:
    name: str
    model_type: str
    config: Dict[str, Any]
    last_used: datetime
    is_loaded: bool
    device: Optional[str]
    
class LocalModelRegistry:
    def __init__(
        self,
        config_path: Optional[Path] = None,
        vram_checker: Optional[VRAMChecker] = None,
        cache_ttl: int = 3600,  # Time to live for cached models in seconds
        max_cached: int = 3     # Maximum number of models to keep in memory
    ):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, ModelMetadata] = {}
        self.loaded_models: Dict[str, BaseLLM] = {}
        self.model_locks: Dict[str, threading.Lock] = {}
        self.registry_lock = threading.Lock()
        self.vram_checker = vram_checker or VRAMChecker()
        self.cache_ttl = cache_ttl
        self.max_cached = max_cached
        
        if config_path:
            self._load_config(config_path)
            
    def _load_config(self, config_path: Path):
        """Load model configurations from JSON file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
                
            with self.registry_lock:
                for model_name, model_config in config.get("models", {}).items():
                    self.models[model_name] = ModelMetadata(
                        name=model_name,
                        model_type=model_config.get("type", "unknown"),
                        config=model_config,
                        last_used=datetime.min,
                        is_loaded=False,
                        device=None
                    )
        except Exception as e:
            self.logger.error(f"Failed to load model config: {e}")
            raise

    def register_model(
        self,
        name: str,
        model_type: str,
        config: Dict[str, Any]
    ) -> bool:
        """Dynamically register a new model"""
        with self.registry_lock:
            if name in self.models:
                self.logger.warning(f"Model {name} already registered")
                return False
                
            self.models[name] = ModelMetadata(
                name=name,
                model_type=model_type,
                config=config,
                last_used=datetime.min,
                is_loaded=False,
                device=None
            )
            self.model_locks[name] = threading.Lock()
            
        return True

    def get_model(self, name: str) -> Optional[BaseLLM]:
        """
        Get a model instance, loading it if necessary.
        Thread-safe and VRAM-aware.
        """
        if name not in self.models:
            self.logger.error(f"Model {name} not registered")
            return None
            
        # Get or create model lock
        if name not in self.model_locks:
            with self.registry_lock:
                if name not in self.model_locks:
                    self.model_locks[name] = threading.Lock()
                    
        with self.model_locks[name]:
            # Return cached model if available
            if name in self.loaded_models:
                metadata = self.models[name]
                metadata.last_used = datetime.now()
                return self.loaded_models[name]
                
            # Check VRAM availability
            device = self.vram_checker.get_suitable_device(name)
            if not device:
                self.logger.warning(f"Insufficient VRAM to load model {name}")
                return None
                
            # Clean up old models if needed
            self._cleanup_old_models()
            
            # Load the model
            try:
                model_config = self.models[name].config
                model_class = self._get_model_class(model_config["type"])
                if not model_class:
                    raise ValueError(f"Unknown model type: {model_config['type']}")
                    
                model = model_class(
                    model_config=model_config,
                    device=device
                )
                
                self.loaded_models[name] = model
                self.models[name].is_loaded = True
                self.models[name].device = device
                self.models[name].last_used = datetime.now()
                
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model {name}: {e}")
                return None

    def _cleanup_old_models(self):
        """Unload least recently used models if cache is full"""
        with self.registry_lock:
            loaded_count = len(self.loaded_models)
            if loaded_count < self.max_cached:
                return
                
            # Sort models by last used time
            sorted_models = sorted(
                self.models.items(),
                key=lambda x: x[1].last_used
            )
            
            # Unload oldest models
            for model_name, metadata in sorted_models:
                if metadata.is_loaded:
                    if (datetime.now() - metadata.last_used).total_seconds() > self.cache_ttl:
                        self._unload_model(model_name)
                        loaded_count -= 1
                        if loaded_count < self.max_cached:
                            break

    def _unload_model(self, name: str):
        """Unload a model and free its resources"""
        if name in self.loaded_models:
            try:
                model = self.loaded_models[name]
                if hasattr(model, 'to'):
                    model.to('cpu')
                del self.loaded_models[name]
                self.models[name].is_loaded = False
                self.models[name].device = None
                
                # Force CUDA cache cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"Error unloading model {name}: {e}")

    def _get_model_class(self, model_type: str) -> Optional[Type[BaseLLM]]:
        """Get the appropriate model class based on type"""
        # This would be expanded based on supported model types
        from .local_llm import LocalLLM
        
        model_classes = {
            "local": LocalLLM,
            # Add more model types here
        }
        
        return model_classes.get(model_type)

    def get_loaded_models(self) -> Dict[str, ModelMetadata]:
        """Get information about currently loaded models"""
        return {
            name: metadata 
            for name, metadata in self.models.items() 
            if metadata.is_loaded
        }

    def get_model_status(self, name: str) -> Optional[ModelMetadata]:
        """Get current status of a specific model"""
        return self.models.get(name)
