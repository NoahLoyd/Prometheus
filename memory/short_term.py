import json
import os

class ShortTermMemory:
    def __init__(self, limit=10, filepath="memory.json"):
        self.limit = limit
        self.filepath = filepath
        self.buffer = self._load_memory()

    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.limit:
            self.buffer.pop(0)
        self._save_memory()

    def get_all(self):
        return self.buffer

    def clear(self):
        self.buffer = []
        self._save_memory()

    def _save_memory(self):
        with open(self.filepath, "w") as f:
            json.dump(self.buffer, f)

    def _load_memory(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                return json.load(f)
        return []
