# tools/note_tool.py
import os
from tools.base_tool import BaseTool

class NoteTool(BaseTool):
    def __init__(self, notes_file="notes.txt"):
        super().__init__(name="note", description="Save and list notes.")
        self.notes_file = notes_file

    def run(self, query: str) -> str:
        if query.startswith("save:"):
            note = query.split("save:", 1)[1].strip()
            return self._save_note(note)
        elif query.strip() == "list":
            return self._list_notes()
        else:
            return "Invalid command. Use 'save: message' to save a note or 'list' to list notes."

    def _save_note(self, note: str) -> str:
        try:
            with open(self.notes_file, "a") as file:
                file.write(note + "\n")
            return "Note saved successfully."
        except Exception as e:
            return f"Error saving note: {e}"

    def _list_notes(self) -> str:
        try:
            if not os.path.exists(self.notes_file):
                return "No notes saved yet."
            with open(self.notes_file, "r") as file:
                notes = file.read().strip()
            return notes if notes else "No notes saved yet."
        except Exception as e:
            return f"Error reading notes: {e}"
