# tools/note_tool.py

from tools.base_tool import BaseTool
from typing import List

class NoteTool(BaseTool):
    name = "note"
    description = "A tool to save, retrieve, and manage notes for AGI memory or reminders."

    def __init__(self):
        self.notes: List[str] = []

    def run(self, query: str) -> str:
        try:
            q = query.strip()
            if q.lower().startswith("save:"):
                note = q[5:].strip()
                if note:
                    self.notes.append(note)
                    return f"Note saved: {note}"
                return "Cannot save empty note."
            elif q.lower() == "get":
                if self.notes:
                    return "Notes:\n" + "\n".join(f"- {n}" for n in self.notes)
                return "No notes saved."
            elif q.lower().startswith("delete:"):
                idx = int(q[7:].strip())
                if 0 <= idx < len(self.notes):
                    removed = self.notes.pop(idx)
                    return f"Deleted note: {removed}"
                return "Invalid note index."
            elif q.lower() == "count":
                return f"{len(self.notes)} notes saved."
            else:
                return "Unknown note command. Use 'save: <msg>', 'get', 'delete: <idx>', or 'count'."
        except Exception as e:
            return f"NoteTool error: {e}"
