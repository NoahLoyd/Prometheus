# core/agent.py

class PrometheusAgent:
    def __init__(self, name="Prometheus"):
        self.name = name
        self.memory = []
        self.tools = []

    def think(self, input_text):
        self.memory.append(input_text)
        return f"{self.name} is thinking about: {input_text}"

    def act(self, instruction):
        return f"{self.name} is performing: {instruction}"

