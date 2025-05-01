from memory.short_term import ShortTermMemory

class PrometheusAgent:
    def __init__(self, name="Prometheus", memory_limit=10):
        self.name = name
        self.memory = ShortTermMemory(limit=memory_limit)
        self.tools = []

    def think(self, input_text):
        self.memory.add(input_text)
        return f"{self.name} is thinking about: {input_text}"

    def recall(self):
        return self.memory.get_all()

    def act(self, instruction):
        return f"{self.name} is performing: {instruction}"

