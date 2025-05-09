# logging.py
class Logging:
    def __init__(self, memory):
        self.memory = memory

    def log_step(self, step, result):
        self.memory.store_short_term({'step': step, 'result': result})

    def archive_goal(self, goal, results):
        self.memory.store_long_term({'goal': goal, 'results': results})