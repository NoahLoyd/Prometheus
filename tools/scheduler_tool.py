# tools/scheduler_tool.py

from tools.base_tool import BaseTool
from datetime import datetime, timedelta
from typing import List, Dict

class SchedulerTool(BaseTool):
    name = "scheduler"
    description = "Schedules and lists reminders for future background tasks or events."

    def __init__(self):
        self.events: List[Dict] = []

    def run(self, query: str) -> str:
        try:
            q = query.strip().lower()
            if q.startswith("add:"):
                parts = query[4:].split(":", 1)
                if len(parts) != 2:
                    return "Add format: add: <YYYY-MM-DD HH:MM>: <message>"
                time_str, message = parts[0].strip(), parts[1].strip()
                try:
                    event_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                    self.events.append({"time": event_time, "msg": message})
                    return f"Event scheduled for {event_time}: {message}"
                except Exception as e:
                    return f"Invalid datetime format: {e}"
            elif q == "list":
                if not self.events:
                    return "No events scheduled."
                events_sorted = sorted(self.events, key=lambda e: e["time"])
                return "\n".join(f"{idx}: {ev['time']} - {ev['msg']}" for idx, ev in enumerate(events_sorted))
            elif q.startswith("remove:"):
                idx = int(q[7:].strip())
                if 0 <= idx < len(self.events):
                    removed = self.events.pop(idx)
                    return f"Removed event: {removed['time']} - {removed['msg']}"
                return "Invalid event index."
            else:
                return "Unknown scheduler command. Use 'add: <YYYY-MM-DD HH:MM>: <msg>', 'list', or 'remove: <idx>'."
        except Exception as e:
            return f"SchedulerTool error: {e}"
