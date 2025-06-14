"""
retry_memory.py - Thread-safe persistent tracking of task attempts for Prometheus AGI

This module provides a RetryMemory class for tracking task execution history,
including success/failure tracking, statistics, and disk persistence.
Designed for production use in Colab and other environments.
"""

import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

class RetryMemory:
    """
    Thread-safe persistent memory for tracking task attempt history.
    
    Features:
    - Thread-safe operations with locks
    - Persistent JSON storage
    - Configurable history limits per task
    - Cache invalidation support
    - Comprehensive statistics
    - Production-grade error handling
    """
    
    def __init__(self, 
                 memory_file: str = "retry_memory.json",
                 max_history_per_task: int = 100,
                 auto_save: bool = True,
                 cache_timeout: float = 300.0):
        """
        Initialize RetryMemory instance.
        
        Args:
            memory_file: Path to JSON file for persistent storage
            max_history_per_task: Maximum number of attempts to store per task
            auto_save: Whether to automatically save after each operation
            cache_timeout: Cache invalidation timeout in seconds
        """
        self.memory_file = memory_file
        self.max_history_per_task = max_history_per_task
        self.auto_save = auto_save
        self.cache_timeout = cache_timeout
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory storage
        self._data: Dict[str, Dict] = {
            "tasks": {},  # task_id -> task data
            "global_stats": {
                "total_attempts": 0,
                "total_successes": 0,
                "total_failures": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_updated": datetime.now(timezone.utc).isoformat()
            },
            "metadata": {
                "version": "1.0.0",
                "max_history_per_task": max_history_per_task
            }
        }
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, float] = {}
        
        # Load existing data
        self._load_from_disk()
        
        logger.info(f"RetryMemory initialized with file: {memory_file}")
    
    def log_result(self, 
                   task_id: str, 
                   success: bool, 
                   details: Optional[Dict[str, Any]] = None,
                   execution_time: Optional[float] = None) -> None:
        """
        Log the result of a task attempt.
        
        Args:
            task_id: Unique identifier for the task
            success: Whether the task succeeded
            details: Optional additional details about the attempt
            execution_time: Optional execution time in seconds
        """
        with self._lock:
            try:
                timestamp = datetime.now(timezone.utc).isoformat()
                
                # Initialize task if not exists
                if task_id not in self._data["tasks"]:
                    self._data["tasks"][task_id] = {
                        "history": deque(maxlen=self.max_history_per_task),
                        "total_attempts": 0,
                        "total_successes": 0,
                        "total_failures": 0,
                        "first_attempt": timestamp,
                        "last_attempt": timestamp,
                        "last_success": None,
                        "last_failure": None
                    }
                
                task_data = self._data["tasks"][task_id]
                
                # Create attempt record
                attempt = {
                    "timestamp": timestamp,
                    "success": success,
                    "details": details or {},
                    "execution_time": execution_time
                }
                
                # Add to history (deque automatically handles max size)
                task_data["history"].append(attempt)
                
                # Update counters
                task_data["total_attempts"] += 1
                task_data["last_attempt"] = timestamp
                
                if success:
                    task_data["total_successes"] += 1
                    task_data["last_success"] = timestamp
                    self._data["global_stats"]["total_successes"] += 1
                else:
                    task_data["total_failures"] += 1
                    task_data["last_failure"] = timestamp
                    self._data["global_stats"]["total_failures"] += 1
                
                self._data["global_stats"]["total_attempts"] += 1
                self._data["global_stats"]["last_updated"] = timestamp
                
                # Invalidate cache for this task
                self._invalidate_cache(task_id)
                
                # Auto-save if enabled
                if self.auto_save:
                    self._save_to_disk()
                
                logger.debug(f"Logged {'success' if success else 'failure'} for task: {task_id}")
                
            except Exception as e:
                logger.error(f"Error logging result for task {task_id}: {e}")
                raise
    
    def get_history(self, 
                    task_id: str, 
                    limit: Optional[int] = None,
                    success_only: bool = False,
                    failure_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get attempt history for a task.
        
        Args:
            task_id: Task identifier
            limit: Maximum number of attempts to return (most recent first)
            success_only: Return only successful attempts
            failure_only: Return only failed attempts
            
        Returns:
            List of attempt records
        """
        with self._lock:
            try:
                if task_id not in self._data["tasks"]:
                    return []
                
                # Check cache first
                cache_key = f"history_{task_id}_{limit}_{success_only}_{failure_only}"
                if self._is_cache_valid(cache_key):
                    return self._cache[cache_key].copy()
                
                history = list(self._data["tasks"][task_id]["history"])
                
                # Filter by success/failure if requested
                if success_only:
                    history = [h for h in history if h["success"]]
                elif failure_only:
                    history = [h for h in history if not h["success"]]
                
                # Sort by timestamp (most recent first) and apply limit
                history.sort(key=lambda x: x["timestamp"], reverse=True)
                if limit:
                    history = history[:limit]
                
                # Cache the result
                self._cache[cache_key] = history.copy()
                self._cache_timestamps[cache_key] = time.time()
                
                return history
                
            except Exception as e:
                logger.error(f"Error getting history for task {task_id}: {e}")
                return []
    
    def has_failed_before(self, task_id: str, within_hours: Optional[float] = None) -> bool:
        """
        Check if a task has failed before.
        
        Args:
            task_id: Task identifier
            within_hours: Only consider failures within this many hours
            
        Returns:
            True if the task has failed before
        """
        with self._lock:
            try:
                if task_id not in self._data["tasks"]:
                    return False
                
                task_data = self._data["tasks"][task_id]
                
                if within_hours is None:
                    return task_data["total_failures"] > 0
                
                # Check failures within time window
                cutoff_time = datetime.now(timezone.utc).timestamp() - (within_hours * 3600)
                
                for attempt in task_data["history"]:
                    if not attempt["success"]:
                        attempt_time = datetime.fromisoformat(attempt["timestamp"]).timestamp()
                        if attempt_time >= cutoff_time:
                            return True
                
                return False
                
            except Exception as e:
                logger.error(f"Error checking failure history for task {task_id}: {e}")
                return False
    
    def get_success_count(self, task_id: str) -> int:
        """Get total number of successful attempts for a task."""
        with self._lock:
            if task_id not in self._data["tasks"]:
                return 0
            return self._data["tasks"][task_id]["total_successes"]
    
    def get_failure_count(self, task_id: str) -> int:
        """Get total number of failed attempts for a task."""
        with self._lock:
            if task_id not in self._data["tasks"]:
                return 0
            return self._data["tasks"][task_id]["total_failures"]
    
    def clear(self, task_id: Optional[str] = None) -> None:
        """
        Clear retry memory data.
        
        Args:
            task_id: If provided, clear only this task. Otherwise clear all data.
        """
        with self._lock:
            try:
                if task_id:
                    if task_id in self._data["tasks"]:
                        del self._data["tasks"][task_id]
                        self._invalidate_cache(task_id)
                        logger.info(f"Cleared data for task: {task_id}")
                else:
                    # Clear all tasks but preserve global stats structure
                    self._data["tasks"].clear()
                    self._data["global_stats"] = {
                        "total_attempts": 0,
                        "total_successes": 0,
                        "total_failures": 0,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }
                    self._cache.clear()
                    self._cache_timestamps.clear()
                    logger.info("Cleared all retry memory data")
                
                if self.auto_save:
                    self._save_to_disk()
                    
            except Exception as e:
                logger.error(f"Error clearing data: {e}")
                raise
    
    def get_statistics(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Args:
            task_id: If provided, get stats for specific task. Otherwise get global stats.
            
        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            try:
                if task_id:
                    if task_id not in self._data["tasks"]:
                        return {
                            "task_id": task_id,
                            "exists": False,
                            "total_attempts": 0,
                            "total_successes": 0,
                            "total_failures": 0,
                            "success_rate": 0.0,
                            "failure_rate": 0.0
                        }
                    
                    task_data = self._data["tasks"][task_id]
                    total = task_data["total_attempts"]
                    
                    return {
                        "task_id": task_id,
                        "exists": True,
                        "total_attempts": total,
                        "total_successes": task_data["total_successes"],
                        "total_failures": task_data["total_failures"],
                        "success_rate": (task_data["total_successes"] / total) if total > 0 else 0.0,
                        "failure_rate": (task_data["total_failures"] / total) if total > 0 else 0.0,
                        "first_attempt": task_data["first_attempt"],
                        "last_attempt": task_data["last_attempt"],
                        "last_success": task_data["last_success"],
                        "last_failure": task_data["last_failure"],
                        "recent_history_size": len(task_data["history"])
                    }
                else:
                    # Global statistics
                    global_stats = self._data["global_stats"]
                    total = global_stats["total_attempts"]
                    
                    return {
                        "global": True,
                        "total_tasks": len(self._data["tasks"]),
                        "total_attempts": total,
                        "total_successes": global_stats["total_successes"],
                        "total_failures": global_stats["total_failures"],
                        "success_rate": (global_stats["total_successes"] / total) if total > 0 else 0.0,
                        "failure_rate": (global_stats["total_failures"] / total) if total > 0 else 0.0,
                        "created_at": global_stats["created_at"],
                        "last_updated": global_stats["last_updated"],
                        "memory_file": self.memory_file,
                        "max_history_per_task": self.max_history_per_task
                    }
                    
            except Exception as e:
                logger.error(f"Error getting statistics: {e}")
                return {}
    
    def get_all_task_ids(self) -> List[str]:
        """Get list of all tracked task IDs."""
        with self._lock:
            return list(self._data["tasks"].keys())
    
    def invalidate_cache(self, task_id: Optional[str] = None) -> None:
        """
        Manually invalidate cache entries.
        
        Args:
            task_id: If provided, invalidate cache for specific task. Otherwise clear all.
        """
        with self._lock:
            if task_id:
                self._invalidate_cache(task_id)
            else:
                self._cache.clear()
                self._cache_timestamps.clear()
                logger.debug("Invalidated all cache entries")
    
    def save(self) -> None:
        """Manually save data to disk."""
        with self._lock:
            self._save_to_disk()
    
    def reload(self) -> None:
        """Reload data from disk, discarding in-memory changes."""
        with self._lock:
            self._load_from_disk()
            self._cache.clear()
            self._cache_timestamps.clear()
            logger.info("Reloaded data from disk")
    
    # Private methods
    
    def _load_from_disk(self) -> None:
        """Load data from JSON file."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                # Convert history lists back to deques with proper maxlen
                if "tasks" in loaded_data:
                    for task_id, task_data in loaded_data["tasks"].items():
                        if "history" in task_data:
                            # Convert list to deque with maxlen
                            history_list = task_data["history"]
                            task_data["history"] = deque(
                                history_list[-self.max_history_per_task:], 
                                maxlen=self.max_history_per_task
                            )
                
                # Merge with existing structure
                self._data.update(loaded_data)
                
                logger.info(f"Loaded retry memory from {self.memory_file}")
            else:
                logger.info(f"No existing file found at {self.memory_file}, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading from disk: {e}")
            # Continue with empty data structure
    
    def _save_to_disk(self) -> None:
        """Save data to JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.memory_file) or '.', exist_ok=True)
            
            # Convert deques to lists for JSON serialization
            save_data = json.loads(json.dumps(self._data, default=self._json_serializer))
            
            # Convert deques to lists
            if "tasks" in save_data:
                for task_id, task_data in save_data["tasks"].items():
                    if "history" in task_data:
                        task_data["history"] = list(task_data["history"])
            
            # Write to file atomically
            temp_file = f"{self.memory_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            os.replace(temp_file, self.memory_file)
            
            logger.debug(f"Saved retry memory to {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Error saving to disk: {e}")
            # Clean up temp file if it exists
            temp_file = f"{self.memory_file}.tmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, deque):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _invalidate_cache(self, task_id: str) -> None:
        """Invalidate cache entries for a specific task."""
        keys_to_remove = [k for k in self._cache.keys() if task_id in k]
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is valid."""
        if cache_key not in self._cache:
            return False
        
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self.cache_timeout


# Global instance management
_global_retry_memories: Dict[str, RetryMemory] = {}
_global_lock = threading.Lock()


def get_retry_memory(memory_file: str = "retry_memory.json") -> RetryMemory:
    """
    Get or create a global RetryMemory instance.
    
    This function implements a singleton pattern per memory file to ensure
    thread safety and prevent multiple instances operating on the same file.
    
    Args:
        memory_file: Path to the JSON file for persistent storage
        
    Returns:
        RetryMemory instance
    """
    with _global_lock:
        if memory_file not in _global_retry_memories:
            # Resolve to absolute path for consistent key
            abs_path = os.path.abspath(memory_file)
            
            if abs_path not in _global_retry_memories:
                _global_retry_memories[abs_path] = RetryMemory(memory_file=abs_path)
                logger.info(f"Created new global RetryMemory instance for {abs_path}")
            
            # Also store under original key for convenience
            _global_retry_memories[memory_file] = _global_retry_memories[abs_path]
        
        return _global_retry_memories[memory_file]


# Convenience functions for common operations
def log_task_result(task_id: str, 
                   success: bool, 
                   details: Optional[Dict[str, Any]] = None,
                   execution_time: Optional[float] = None,
                   memory_file: str = "retry_memory.json") -> None:
    """
    Convenience function to log a task result.
    
    Args:
        task_id: Unique identifier for the task
        success: Whether the task succeeded
        details: Optional additional details about the attempt
        execution_time: Optional execution time in seconds
        memory_file: Memory file to use
    """
    retry_memory = get_retry_memory(memory_file)
    retry_memory.log_result(task_id, success, details, execution_time)


def get_task_statistics(task_id: str, memory_file: str = "retry_memory.json") -> Dict[str, Any]:
    """
    Convenience function to get task statistics.
    
    Args:
        task_id: Task identifier
        memory_file: Memory file to use
        
    Returns:
        Dictionary containing task statistics
    """
    retry_memory = get_retry_memory(memory_file)
    return retry_memory.get_statistics(task_id)


def has_task_failed_recently(task_id: str, 
                           within_hours: float = 24.0,
                           memory_file: str = "retry_memory.json") -> bool:
    """
    Convenience function to check if a task has failed recently.
    
    Args:
        task_id: Task identifier
        within_hours: Time window in hours
        memory_file: Memory file to use
        
    Returns:
        True if the task has failed within the specified time window
    """
    retry_memory = get_retry_memory(memory_file)
    return retry_memory.has_failed_before(task_id, within_hours)


if __name__ == "__main__":
    # Example usage and testing
    import doctest
    doctest.testmod()
    
    # Simple demonstration
    retry_mem = get_retry_memory("test_retry_memory.json")
    
    # Log some test results
    retry_mem.log_result("test_task_1", True, {"method": "approach_a"}, 1.5)
    retry_mem.log_result("test_task_1", False, {"method": "approach_b", "error": "timeout"}, 5.0)
    retry_mem.log_result("test_task_2", True, {"method": "approach_c"}, 0.8)
    
    # Get statistics
    print("Task 1 stats:", retry_mem.get_statistics("test_task_1"))
    print("Global stats:", retry_mem.get_statistics())
    print("Task 1 history:", retry_mem.get_history("test_task_1"))
    
    # Clean up test file
    if os.path.exists("test_retry_memory.json"):
        os.remove("test_retry_memory.json")
    
    print("RetryMemory demonstration completed successfully!")
