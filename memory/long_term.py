# memory/long_term.py

import json
import os
from datetime import datetime

class LongTermMemory:
    def __init__(self, filepath="memory.json"):
        self.filepath = filepath
        self.data = {
            "goal_history": [],
            "tool_usage": {},
            "insights": {},
            "logs": []
        }
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Failed to load long-term memory: {e}")
        else:
            self.save()

    def save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save long-term memory: {e}")

    def add_goal_history(self, entry):
        entry["timestamp"] = datetime.now().isoformat()
        self.data["goal_history"].append(entry)
        self.save()

    def log_tool_usage(self, tool_name):
        self.data["tool_usage"][tool_name] = self.data["tool_usage"].get(tool_name, 0) + 1
        self.save()

    def add_insight(self, label, content):
        self.data["insights"][label] = content
        self.save()

    def log_event(self, tag, detail):
        self.data["logs"].append({
            "time": datetime.now().isoformat(),
            "tag": tag,
            "detail": detail
        })
        self.save()

    def get_memory(self):
        return self.data