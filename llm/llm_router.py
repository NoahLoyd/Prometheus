from typing import List, Tuple, Dict, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError
import hashlib
import logging
import threading
from pathlib import Path
import torch
import json
from datetime import datetime
import traceback
from contextlib import contextmanager
import warnings

from .exceptions import (
    LLMRouterError,
    ModelLoadError,
    VRAMError,
    ConfigurationError,
    ExecutionError,
    ValidationError
)
from .validation import (
    validate_config,
    validate_model_config,
    validate_path,
    validate_vram_requirements
)
from .vram_checker import VRAMChecker, VRAMStats, ModelVRAMRequirement
from .model_registry import LocalModelRegistry, ModelMetadata
from .base_llm import BaseLLM
from core.logging import Logging

# [Previous imports and ModelPriority enum remain the same...]

class LLMRouter:
    """
    Enhanced LLMRouter with comprehensive error handling and validation
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[Union[str, Path]] = None,
        evaluation_strategy: Optional['EvaluationStrategy'] = None,
        fallback_strategy: Optional['FallbackStrategy'] = None,
        voting_strategy: Optional['VotingStrategy'] = None,
        profiler: Optional['TaskProfiler'] = None,
        feedback_memory: Optional['FeedbackMemory'] = None,
        confidence_scorer: Optional['ConfidenceScorer'] = None,
        raise_on_error: bool = False
    ):
        """
        Initialize enhanced LLMRouter with validation and error handling
        
        Args:
            config: Configuration dictionary
            config_path: Path to configuration file
            evaluation_strategy: Strategy for model performance evaluation
            fallback_strategy: Strategy for fallback in case of failures
            voting_strategy: Strategy for merging or voting on model outputs
            profiler: Task Profiler for task classification
            feedback_memory: Feedback Memory system for storing task results
            confidence_scorer: Confidence Scorer for evaluating outputs
            raise_on_error: If True, raise exceptions instead of falling back
            
        Raises:
            ConfigurationError: If configuration is invalid
            ValidationError: If input validation fails
            FileNotFoundError: If config_path is invalid
        """
        try:
            # Initialize logging first for error tracking
            self.logger = logging.getLogger(__name__)
            self.raise_on_error = raise_on_error
            
            # Load and validate configuration
            self.config = self._load_and_validate_config(config, config_path)
            
            # Initialize core components with error handling
            self.vram_checker = self._init_vram_checker(config_path)
            self.model_registry = self._init_model_registry(config_path)
            
            # Initialize selection strategy
            self.model_selector = self._init_model_selector(feedback_memory)
            
            # Initialize remaining components
            self._init_components(
                evaluation_strategy,
                fallback_strategy,
                voting_strategy,
                profiler,
                feedback_memory,
                confidence_scorer
            )
            
            # Verify CUDA availability if required
            self._verify_cuda_availability()
            
        except Exception as e:
            error_msg = f"LLMRouter initialization failed: {str(e)}"
            self.logger.error(error_msg)
            if self.raise_on_error:
                raise ConfigurationError(error_msg) from e
            else:
                warnings.warn(error_msg)
                self._init_fallback_mode()

    def _load_and_validate_config(
        self,
        config: Optional[Dict],
        config_path: Optional[Union[str, Path]]
    ) -> Dict[str, Any]:
        """Load and validate configuration with error handling"""
        try:
            if config_path:
                path = validate_path(config_path)
                with open(path) as f:
                    config = json.load(f)
            elif not config:
                config = {
                    "models": ["simulated"],
                    "use_simulation": True,
                    "logging": {"level": "INFO"},
                    "hardware": {"allow_cpu_fallback": True}
                }
            
            validate_config(config)
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Configuration error: {e}") from e

    def _init_vram_checker(self, config_path: Optional[Path]) -> VRAMChecker:
        """Initialize VRAMChecker with error handling"""
        try:
            return VRAMChecker(config_path)
        except Exception as e:
            error_msg = f"Failed to initialize VRAMChecker: {e}"
            self.logger.error(error_msg)
            if self.raise_on_error:
                raise VRAMError(error_msg) from e
            return self._create_fallback_vram_checker()

    def _init_model_registry(
        self,
        config_path: Optional[Path]
    ) -> LocalModelRegistry:
        """Initialize ModelRegistry with error handling"""
        try:
            return LocalModelRegistry(
                config_path=config_path,
                vram_checker=self.vram_checker
            )
        except Exception as e:
            error_msg = f"Failed to initialize ModelRegistry: {e}"
            self.logger.error(error_msg)
            if self.raise_on_error:
                raise ConfigurationError(error_msg) from e
            return self._create_fallback_model_registry()

    def _verify_cuda_availability(self):
        """Verify CUDA availability if required"""
        if self.config.get("require_cuda", False) and not torch.cuda.is_available():
            error_msg = "CUDA is required but not available"
            self.logger.error(error_msg)
            if self.raise_on_error:
                raise VRAMError(error_msg)
            else:
                warnings.warn(error_msg)

    def generate_plan(
        self,
        goal: str,
        context: Optional[str] = None,
        task_type: Optional[str] = None,
        timeout: float = 30.0
    ) -> List[Tuple[str, str]]:
        """
        Generate execution plan with comprehensive error handling
        
        Args:
            goal: Primary objective to accomplish
            context: Additional context for the task
            task_type: Optional explicit task type
            timeout: Timeout in seconds for execution
            
        Returns:
            List of execution steps as (step_type, step_content) tuples
            
        Raises:
            ValidationError: If input validation fails
            ExecutionError: If execution fails and raise_on_error is True
        """
        try:
            # Validate inputs
            if not goal:
                raise ValidationError("Goal cannot be empty")
            if not isinstance(goal, str):
                raise ValidationError("Goal must be a string")
                
            self.logger.info(f"Generating plan for goal: {goal}")
            
            # Determine task type with error handling
            task_type = self._determine_task_type(goal, task_type)
            
            # Check cache with error handling
            cached_result = self._check_cache(goal, context, task_type)
            if cached_result is not None:
                return cached_result
                
            # Select and execute models
            selected_models = self._select_models_with_fallback(task_type)
            results = self._execute_models_safely(
                selected_models,
                goal,
                context,
                task_type,
                timeout
            )
            
            # Process results
            final_result = self._process_results_safely(results, goal, task_type)
            
            # Update cache if successful
            self._update_cache(goal, context, task_type, final_result)
            
            return final_result
            
        except Exception as e:
            error_msg = f"Plan generation failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            if self.raise_on_error:
                raise ExecutionError(error_msg) from e
                
            # Return fallback response
            return self._generate_fallback_response(goal, context, task_type, str(e))

    def _determine_task_type(
        self,
        goal: str,
        explicit_type: Optional[str]
    ) -> Optional[str]:
        """Determine task type with error handling"""
        try:
            if explicit_type:
                return explicit_type
            if self.profiler:
                return self.profiler.classify_task(goal)
            return None
        except Exception as e:
            self.logger.warning(f"Task type determination failed: {e}")
            return None

    def _check_cache(
        self,
        goal: str,
        context: Optional[str],
        task_type: Optional[str]
    ) -> Optional[List[Tuple[str, str]]]:
        """Check cache with error handling"""
        try:
            query_hash = self._hash_query(goal, context, task_type)
            with self.cache_lock:
                return self.cache.get(query_hash)
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
            return None

    def _select_models_with_fallback(
        self,
        task_type: Optional[str]
    ) -> List[ModelExecutionContext]:
        """Select models with fallback options"""
        try:
            selected_models = self.model_selector.select_models(
                task_type=task_type,
                min_models=self.config.get("min_models", 1),
                max_models=self.config.get("max_models", 3)
            )
            
            if not selected_models:
                raise ModelLoadError("No models available for execution")
                
            return selected_models
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            if self.raise_on_error:
                raise
            return self._get_fallback_models()

    def _execute_models_safely(
        self,
        models: List[ModelExecutionContext],
        goal: str,
        context: Optional[str],
        task_type: Optional[str],
        timeout: float
    ) -> List[Dict[str, Any]]:
        """Execute models with comprehensive error handling"""
        results = []
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures: Dict[Future, ModelExecutionContext] = {}
            
            # Submit tasks
            for model in models:
                try:
                    future = executor.submit(
                        self._execute_single_model_safely,
                        model,
                        goal,
                        context
                    )
                    futures[future] = model
                except Exception as e:
                    self.logger.error(f"Failed to submit model {model.model_name}: {e}")
                    continue
                    
            # Collect results
            for future in futures:
                model_context = futures[future]
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except TimeoutError:
                    self.logger.error(
                        f"Model {model_context.model_name} execution timed out"
                    )
                    results.append(self._create_timeout_result(model_context))
                except Exception as e:
                    self.logger.error(
                        f"Model {model_context.model_name} execution failed: {e}"
                    )
                    results.append(self._create_error_result(model_context, e))
                    
        return results

    def _execute_single_model_safely(
        self,
        model_context: ModelExecutionContext,
        goal: str,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Execute single model with error handling and resource monitoring"""
        start_time = datetime.now()
        
        try:
            # Handle simulation mode
            if model_context.is_simulation:
                return self._execute_simulation(
                    model_context,
                    goal,
                    context,
                    start_time
                )
                
            # Get and validate model instance
            model = self._get_validated_model(model_context)
            
            # Execute with resource monitoring
            with self._monitor_resources_safely(model_context):
                result = model.generate(
                    prompt=goal,
                    context=context,
                    **self._get_model_params(model_context)
                )
                
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics and return result
            return self._create_success_result(
                model_context,
                result,
                execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            model_context.update_metrics(execution_time, False)
            raise ExecutionError(
                f"Model {model_context.model_name} execution failed: {e}"
            ) from e

    @contextmanager
    def _monitor_resources_safely(self, model_context: ModelExecutionContext):
        """Monitor resources with error handling"""
        initial_vram = None
        try:
            initial_vram = self.vram_checker.get_vram_status()
            yield
        finally:
            try:
                if initial_vram:
                    final_vram = self.vram_checker.get_vram_status()
                    vram_used = {
                        device: final_vram[device].used - initial_vram[device].used
                        for device in final_vram
                    }
                    self.logger.debug(
                        f"VRAM usage for {model_context.model_name}: {vram_used}"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to monitor resources: {e}")

    def _create_success_result(
        self,
        model_context: ModelExecutionContext,
        result: Any,
        execution_time: float
    ) -> Dict[str, Any]:
        """Create success result with validation"""
        try:
            confidence = self._calculate_confidence(result)
            model_context.update_metrics(execution_time, True)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "confidence": confidence,
                "model_context": model_context
            }
        except Exception as e:
            self.logger.error(f"Failed to create success result: {e}")
            return self._create_error_result(model_context, e)

    def _create_error_result(
        self,
        model_context: ModelExecutionContext,
        error: Exception
    ) -> Dict[str, Any]:
        """Create error result with proper formatting"""
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "model_context": model_context,
            "traceback": traceback.format_exc()
        }

    def _create_timeout_result(
        self,
        model_context: ModelExecutionContext
    ) -> Dict[str, Any]:
        """Create timeout result"""
        return {
            "success": False,
            "error": "Execution timed out",
            "error_type": "TimeoutError",
            "model_context": model_context
        }

    def _generate_fallback_response(
        self,
        goal: str,
        context: Optional[str],
        task_type: Optional[str],
        error: str
    ) -> List[Tuple[str, str]]:
        """Generate fallback response when execution fails"""
        try:
            if self.fallback_strategy:
                return self.fallback_strategy.refine_plan(goal, context, task_type)
            return [
                ("error", f"Execution failed: {error}"),
                ("fallback", "Using simulation mode"),
                ("result", str(self._simulate_llm("simulated", goal, context)))
            ]
        except Exception as e:
            self.logger.error(f"Fallback response generation failed: {e}")
            return [("error", "System unavailable")]

    def _init_fallback_mode(self):
        """Initialize system in fallback mode"""
        self.config = {
            "models": ["simulated"],
            "use_simulation": True,
            "logging": {"level": "WARNING"}
        }
        self.logger.warning("System initialized in fallback mode")

    def _create_fallback_vram_checker(self) -> VRAMChecker:
        """Create minimal VRAMChecker for fallback mode"""
        class FallbackVRAMChecker:
            def get_vram_status(self):
                return {"cpu": VRAMStats(0, 0, 0, "cpu")}
            def can_load_models(self, _):
                return {"simulated": True}
        return FallbackVRAMChecker()

    def _create_fallback_model_registry(self) -> LocalModelRegistry:
        """Create minimal ModelRegistry for fallback mode"""
        class FallbackModelRegistry:
            def get_model(self, _):
                return None
            def get_all_models(self):
                return {"simulated": ModelMetadata("simulated", {}, datetime.now())}
        return FallbackModelRegistry()

    def _get_validated_model(
        self,
        model_context: ModelExecutionContext
    ) -> BaseLLM:
        """Get and validate model instance"""
        model = self.model_registry.get_model(model_context.model_name)
        if not model:
            raise ModelLoadError(
                f"Failed to load model {model_context.model_name}"
            )
        return model

    def _execute_simulation(
        self,
        model_context: ModelExecutionContext,
        goal: str,
        context: Optional[str],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Execute in simulation mode"""
        result = self._simulate_llm(
            model_context.model_name,
            goal,
            context
        )
        execution_time = (datetime.now() - start_time).total_seconds()
        return {
            "success": True,
            "result": result,
            "execution_time": execution_time,
            "simulation": True,
            "model_context": model_context
        }

    # [Previous utility methods remain unchanged...]
