# tools/note_tool.py
from tools.base_tool import BaseTool

class NoteTool(BaseTool):
    def __init__(self):
        self.notes = []

    def run(self, command: str) -> str:
        if command.startswith("save:"):
            note = command.split("save:", 1)[1].strip()
            self.notes.append(note)
            return f"Saved note: {note}"
        elif command == "list":
            return "\n".join(self.notes) if self.notes else "No notes yet."
        else:
            return "Unknown action. Use 'save:' or 'list'."
