# memory/short_term.py

class ShortTermMemory:
    def __init__(self, limit=10):
        self.limit = limit
        self.buffer = []

    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.limit:
            self.buffer.pop(0)

    def get_all(self):
        return self.buffer

    def clear(self):
        self.buffer = []
