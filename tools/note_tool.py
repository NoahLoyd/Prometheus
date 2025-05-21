# tools/note.py

from tools.base_tool import BaseTool

class NoteTool(BaseTool):
    name = "note"
    description = "A tool to save and retrieve notes."

    def __init__(self):
        self.notes = []

    def run(self, query: str) -> str:
        # Syntax: "save: message" to save, "get" to get all notes
        q = query.strip().lower()
        if q.startswith("save:"):
            note = query[5:].strip()
            if note:
                self.notes.append(note)
                return f"Note saved: {note}"
            return "Cannot save empty note."
        elif q == "get":
            if self.notes:
                return "Notes:\n" + "\n".join(f"- {n}" for n in self.notes)
            return "No notes saved."
        else:
            return "Unknown note command. Use 'save: <message>' or 'get'."
