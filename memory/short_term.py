from collections import OrderedDict
import logging
import json
from typing import Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManagedMemory:
    """A managed memory class that uses OrderedDict to maintain a size-limited memory store"""
    
    def __init__(self, max_size: int = 1000):
        self._memory = OrderedDict()
        self.max_size = max_size
        logger.info(f"Initialized ManagedMemory with max size of {max_size} items")

    def add(self, key: str, value: Any) -> None:
        """
        Add an item to memory, automatically removing oldest items if max size is reached
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        try:
            # Remove oldest item if we're at max capacity
            while len(self._memory) >= self.max_size:
                self._memory.popitem(last=False)
                logger.debug("Removed oldest memory item due to size limit")
            
            # Add new item
            self._memory[key] = value
            logger.debug(f"Added new memory item with key: {key}")
            
        except Exception as e:
            logger.error(f"Error adding item to memory: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve an item from memory
        
        Args:
            key: The key to look up
            
        Returns:
            The stored value or None if not found
        """
        try:
            value = self._memory.get(key)
            if value is not None:
                # Move accessed item to end to mark as recently used
                self._memory.move_to_end(key)
                logger.debug(f"Retrieved memory item with key: {key}")
            else:
                logger.debug(f"Memory item not found for key: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Error retrieving item from memory: {e}")
            return None

    def remove(self, key: str) -> None:
        """
        Remove an item from memory
        
        Args:
            key: The key to remove
        """
        try:
            if key in self._memory:
                del self._memory[key]
                logger.debug(f"Removed memory item with key: {key}")
            else:
                logger.debug(f"Cannot remove - key not found: {key}")
                
        except Exception as e:
            logger.error(f"Error removing item from memory: {e}")
            raise

    def clear(self) -> None:
        """Clear all items from memory"""
        try:
            self._memory.clear()
            logger.info("Cleared all items from memory")
            
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise

    def size(self) -> int:
        """Get current number of items in memory"""
        return len(self._memory)

    def to_dict(self) -> dict:
        """Convert memory contents to a regular dictionary"""
        return dict(self._memory)

    def save_to_file(self, filepath: str) -> None:
        """
        Save memory contents to a JSON file
        
        Args:
            filepath: Path to save the file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f)
            logger.info(f"Saved memory contents to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving memory to file: {e}")
            raise

    def load_from_file(self, filepath: str) -> None:
        """
        Load memory contents from a JSON file
        
        Args:
            filepath: Path to load the file from
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self._memory.clear()
                for key, value in data.items():
                    self.add(key, value)
            logger.info(f"Loaded memory contents from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading memory from file: {e}")
            raise

# Create global memory instance
short_term_memory = ManagedMemory()

def add_to_memory(key: str, value: Any) -> None:
    """
    Add an item to short-term memory
    
    Args:
        key: The key to store the value under
        value: The value to store
    """
    short_term_memory.add(key, value)

def get_from_memory(key: str) -> Optional[Any]:
    """
    Retrieve an item from short-term memory
    
    Args:
        key: The key to look up
        
    Returns:
        The stored value or None if not found
    """
    return short_term_memory.get(key)

def remove_from_memory(key: str) -> None:
    """
    Remove an item from short-term memory
    
    Args:
        key: The key to remove
    """
    short_term_memory.remove(key)

def clear_memory() -> None:
    """Clear all items from short-term memory"""
    short_term_memory.clear()

def get_memory_size() -> int:
    """Get current number of items in short-term memory"""
    return short_term_memory.size()

def save_memory(filepath: str) -> None:
    """
    Save memory contents to file
    
    Args:
        filepath: Path to save the file
    """
    short_term_memory.save_to_file(filepath)

def load_memory(filepath: str) -> None:
    """
    Load memory contents from file
    
    Args:
        filepath: Path to load the file from
    """
    short_term_memory.load_from_file(filepath)
