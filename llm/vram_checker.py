from typing import Dict, Optional, List
import torch
import psutil
import logging
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class VRAMStats:
    total: int  # Total VRAM in bytes
    used: int   # Used VRAM in bytes
    free: int   # Free VRAM in bytes
    device: str # CUDA device identifier

@dataclass
class ModelVRAMRequirement:
    min_vram: int     # Minimum VRAM required in bytes
    optimal_vram: int # Optimal VRAM for best performance
    can_offload: bool # Whether model supports CPU offloading
    priority: int     # Loading priority (1-100, higher = more important)

class VRAMChecker:
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.device_stats: Dict[str, VRAMStats] = {}
        self.model_requirements: Dict[str, ModelVRAMRequirement] = {}
        
        if config_path:
            self._load_config(config_path)
        
        self._init_devices()
    
    def _load_config(self, config_path: Path):
        """Load model VRAM requirements from config file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            for model_name, specs in config.get("models", {}).items():
                self.model_requirements[model_name] = ModelVRAMRequirement(
                    min_vram=specs.get("min_vram_bytes", 0),
                    optimal_vram=specs.get("optimal_vram_bytes", 0),
                    can_offload=specs.get("can_offload", False),
                    priority=specs.get("priority", 50)
                )
        except Exception as e:
            self.logger.error(f"Failed to load VRAM config: {e}")
            raise

    def _init_devices(self):
        """Initialize and catalog available CUDA devices"""
        if not torch.cuda.is_available():
            self.logger.warning("No CUDA devices available")
            return
        
        for device_idx in range(torch.cuda.device_count()):
            device = f"cuda:{device_idx}"
            try:
                with torch.cuda.device(device):
                    total_vram = torch.cuda.get_device_properties(device_idx).total_memory
                    used_vram = torch.cuda.memory_allocated(device_idx)
                    free_vram = total_vram - used_vram
                    
                    self.device_stats[device] = VRAMStats(
                        total=total_vram,
                        used=used_vram,
                        free=free_vram,
                        device=device
                    )
            except Exception as e:
                self.logger.error(f"Failed to initialize device {device}: {e}")

    def get_suitable_device(self, model_name: str) -> Optional[str]:
        """
        Determine the most suitable device for a given model based on VRAM requirements
        """
        if model_name not in self.model_requirements:
            self.logger.warning(f"No VRAM requirements defined for model {model_name}")
            return None
            
        requirements = self.model_requirements[model_name]
        
        # Sort devices by available VRAM
        available_devices = sorted(
            self.device_stats.items(),
            key=lambda x: x[1].free,
            reverse=True
        )
        
        for device_name, stats in available_devices:
            if stats.free >= requirements.min_vram:
                return device_name
                
        if requirements.can_offload:
            return "cpu"
            
        return None

    def can_load_models(self, model_names: List[str]) -> Dict[str, bool]:
        """
        Check if a list of models can be loaded given current VRAM status
        """
        results = {}
        remaining_vram = {
            device: stats.free 
            for device, stats in self.device_stats.items()
        }
        
        # Sort models by priority
        sorted_models = sorted(
            model_names,
            key=lambda x: self.model_requirements.get(x, ModelVRAMRequirement(0,0,False,0)).priority,
            reverse=True
        )
        
        for model in sorted_models:
            if model not in self.model_requirements:
                results[model] = False
                continue
                
            requirements = self.model_requirements[model]
            loaded = False
            
            # Try to fit model on any device
            for device in remaining_vram:
                if remaining_vram[device] >= requirements.min_vram:
                    remaining_vram[device] -= requirements.min_vram
                    loaded = True
                    break
                    
            # Fall back to CPU if possible
            if not loaded and requirements.can_offload:
                loaded = True
                
            results[model] = loaded
            
        return results

    def get_vram_status(self) -> Dict[str, VRAMStats]:
        """Get current VRAM status for all devices"""
        self._refresh_stats()
        return self.device_stats

    def _refresh_stats(self):
        """Refresh VRAM statistics"""
        for device_idx in range(torch.cuda.device_count()):
            device = f"cuda:{device_idx}"
            try:
                with torch.cuda.device(device):
                    used_vram = torch.cuda.memory_allocated(device_idx)
                    total_vram = self.device_stats[device].total
                    self.device_stats[device] = VRAMStats(
                        total=total_vram,
                        used=used_vram,
                        free=total_vram - used_vram,
                        device=device
                    )
            except Exception as e:
                self.logger.error(f"Failed to refresh stats for device {device}: {e}")
