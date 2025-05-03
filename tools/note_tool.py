# tools/note_tool.py

from tools.base_tool import BaseTool

class NoteTool(BaseTool):
    def __init__(self):
        super().__init__(name="note_tool", description="Saves and lists notes.")
        self.notes = []

    def save_note(self, text):
        self.notes.append(text)
        return f"Note saved: {text}"

    def list_notes(self):
        return "\n".join(f"{i+1}. {note}" for i, note in enumerate(self.notes)) or "No notes saved."

    def run(self, action="save", text=""):
        if action == "save":
            return self.save_note(text)
        elif action == "list":
            return self.list_notes()
        else:
            return f"Unknown action '{action}'. Use 'save' or 'list'."
