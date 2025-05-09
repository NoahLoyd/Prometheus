# memory/short_term.py

class ShortTermMemory:
    """
    A class to manage short-term memory for Promethyn AI.
    Stores key-value pairs for quick access during runtime.
    """

    def __init__(self):
        """
        Initializes an empty memory store.
        """
        self.memory_store = {}

    def save(self, key: str, value: any):
        """
        Stores a value under the specified key.

        Args:
            key (str): The key to store the value under.
            value (any): The value to store.
        """
        self.memory_store[key] = value

    def load(self, key: str):
        """
        Retrieves the value stored under the specified key.

        Args:
            key (str): The key whose value should be retrieved.

        Returns:
            any: The value associated with the key, or None if not found.
        """
        return self.memory_store.get(key, None)

    def delete(self, key: str):
        """
        Deletes the value stored under the specified key.

        Args:
            key (str): The key to delete from the memory store.
        """
        if key in self.memory_store:
            del self.memory_store[key]

    def clear(self):
        """
        Clears all key-value pairs from the memory store.
        """
        self.memory_store.clear()
