import os
from tools.base_tool import BaseTool

class NoteTool(BaseTool):
    def __init__(self, notes_file="notes.txt"):
        self.notes_file = notes_file

    def run(self, query: str) -> str:
        parts = query.strip().split(":", 1)
        if len(parts) != 2:
            return "Invalid format. Use 'save: message' or 'list'."

        action = parts[0].strip().lower()
        data = parts[1].strip() if len(parts) > 1 else ""

        if action == "save":
            return self._save_note(data)
        elif action == "list":
            return self._list_notes()
        else:
            return "Unknown action. Use 'save' or 'list'."

    def _save_note(self, note: str) -> str:
        if not note:
            return "Cannot save an empty note."
        try:
            with open(self.notes_file, "a") as f:
                f.write(note + "\n")
            return "Note saved."
        except Exception as e:
            return f"Error saving note: {str(e)}"

    def _list_notes(self) -> str:
        if not os.path.exists(self.notes_file):
            return "No notes saved yet."
        try:
            with open(self.notes_file, "r") as f:
                notes = f.read().strip()
            return notes if notes else "No notes saved yet."
        except Exception as e:
            return f"Error reading notes: {str(e)}"
