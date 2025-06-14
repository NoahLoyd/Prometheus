from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ModelExecutionContext(BaseModel):
    """
    Production-grade model execution context for Promethyn AGI system.
    
    This class tracks model metadata, execution metrics, and provides
    a robust interface for model selection and performance monitoring.
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        frozen=False
    )
    
    name: str = Field(
        ...,
        description="The name of the model (e.g., 'mistral-7b', 'gpt-4', 'claude-3')",
        min_length=1
    )
    
    source: str = Field(
        ...,
        description="Source platform of the model (e.g., 'huggingface', 'openai', 'anthropic', 'local')",
        min_length=1
    )
    
    task_type: Optional[str] = Field(
        default=None,
        description="Specific task type the model is optimized for (e.g., 'code_generation', 'reasoning', 'planning')"
    )
    
    priority: int = Field(
        default=5,
        description="Priority score for model selection (1-10, higher is better)",
        ge=1,
        le=10
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata including performance metrics, latency, success rates, etc."
    )
    
    # Execution tracking fields
    total_executions: int = Field(default=0, description="Total number of executions")
    successful_executions: int = Field(default=0, description="Number of successful executions")
    total_execution_time: float = Field(default=0.0, description="Cumulative execution time in seconds")
    last_execution_time: Optional[datetime] = Field(default=None, description="Timestamp of last execution")
    is_simulation: bool = Field(default=False, description="Whether this is a simulation context")
    
    def __init__(self, **data):
        """Initialize with validation and default metadata setup."""
        super().__init__(**data)
        
        # Initialize default metadata if not provided
        if not self.metadata:
            self.metadata = {
                "avg_execution_time": 0.0,
                "success_rate": 0.0,
                "last_error": None,
                "created_at": datetime.now().isoformat(),
                "vram_usage": {},
                "model_params": {}
            }
    
    @property
    def model_name(self) -> str:
        """Alias for name to maintain compatibility with existing code."""
        return self.name
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100.0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        if self.total_executions == 0:
            return 0.0
        return self.total_execution_time / self.total_executions
    
    def update_metrics(self, execution_time: float, success: bool, error: Optional[str] = None) -> None:
        """
        Update execution metrics after a model run.
        
        Args:
            execution_time: Time taken for execution in seconds
            success: Whether the execution was successful
            error: Optional error message if execution failed
        """
        self.total_executions += 1
        self.total_execution_time += execution_time
        self.last_execution_time = datetime.now()
        
        if success:
            self.successful_executions += 1
        else:
            if self.metadata:
                self.metadata["last_error"] = error or "Unknown error"
        
        # Update metadata metrics
        if self.metadata:
            self.metadata["avg_execution_time"] = self.average_execution_time
            self.metadata["success_rate"] = self.success_rate
            self.metadata["last_updated"] = datetime.now().isoformat()
    
    def reset_metrics(self) -> None:
        """Reset all execution metrics."""
        self.total_executions = 0
        self.successful_executions = 0
        self.total_execution_time = 0.0
        self.last_execution_time = None
        
        if self.metadata:
            self.metadata.update({
                "avg_execution_time": 0.0,
                "success_rate": 0.0,
                "last_error": None,
                "last_updated": datetime.now().isoformat()
            })
    
    def get_performance_score(self) -> float:
        """
        Calculate a performance score based on success rate, execution time, and priority.
        
        Returns:
            Performance score between 0.0 and 10.0
        """
        if self.total_executions == 0:
            return float(self.priority)
        
        # Base score from priority (0-10)
        score = float(self.priority)
        
        # Success rate bonus (0-3 points)
        success_bonus = (self.success_rate / 100.0) * 3.0
        
        # Execution time penalty (faster is better, max penalty -2 points)
        avg_time = self.average_execution_time
        if avg_time > 0:
            time_penalty = min(2.0, avg_time / 10.0)  # Penalty increases with time
            score -= time_penalty
        
        # Apply success bonus
        score += success_bonus
        
        # Ensure score is within bounds
        return max(0.0, min(10.0, score))
    
    def is_healthy(self) -> bool:
        """
        Check if the model context is in a healthy state for execution.
        
        Returns:
            True if the model is healthy (good success rate, recent activity)
        """
        # Consider unhealthy if no successful executions and has tried multiple times
        if self.total_executions >= 3 and self.successful_executions == 0:
            return False
        
        # Consider unhealthy if success rate is very low with enough data
        if self.total_executions >= 5 and self.success_rate < 20.0:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "source": self.source,
            "task_type": self.task_type,
            "priority": self.priority,
            "metadata": self.metadata,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "total_execution_time": self.total_execution_time,
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "is_simulation": self.is_simulation,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time,
            "performance_score": self.get_performance_score(),
            "is_healthy": self.is_healthy()
        }
    
    def __str__(self) -> str:
        """Return a clean summary of the model context."""
        status = "healthy" if self.is_healthy() else "degraded"
        
        if self.total_executions > 0:
            return (
                f"ModelExecutionContext(name='{self.name}', source='{self.source}', "
                f"task_type='{self.task_type}', priority={self.priority}, "
                f"executions={self.total_executions}, success_rate={self.success_rate:.1f}%, "
                f"avg_time={self.average_execution_time:.2f}s, status={status})"
            )
        else:
            return (
                f"ModelExecutionContext(name='{self.name}', source='{self.source}', "
                f"task_type='{self.task_type}', priority={self.priority}, status=new)"
            )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return self.__str__()


class BaseLLM(ABC):
    """Abstract base class for LLM implementations in Promethyn."""
    
    @abstractmethod
    def generate_plan(self, goal: str, context: Optional[str] = None) -> List[Tuple[str, str]]:
        """Generate a plan for achieving the given goal."""
        pass

    @abstractmethod
    def _format_prompt(self, goal: str, context: Optional[str]) -> str:
        """Format the prompt for the specific LLM implementation."""
        pass

    @abstractmethod
    def _parse_plan(self, response: str) -> List[Tuple[str, str]]:
        """Parse the LLM response into a structured plan."""
        pass
    
    def generate(self, prompt: str, context: Optional[str] = None, **kwargs) -> Any:
        """
        Default generate method that can be overridden by implementations.
        
        Args:
            prompt: The input prompt
            context: Optional context
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        # Default implementation calls generate_plan for backward compatibility
        return self.generate_plan(prompt, context)
