# tools/note_tool.py
class NoteTool:
    def __init__(self):
        self.notes = []

    def run(self, query: str) -> str:
        if query.strip().lower() == "list":
            return "\n".join(self.notes) if self.notes else "No notes saved."
        elif query.strip().lower().startswith("save:"):
            note = query.partition("save:")[2].strip()
            self.notes.append(note)
            return f"Note saved: {note}"
        else:
            return "Unknown action. Use 'save:' or 'list'."
