
# tools/note_tool.py

class NoteTool:
    def __init__(self):
        self.notes = []

    def execute(self):
        note = input("What should I write down? ")
        self.notes.append(note)
        return f"Note saved: {note}"

    def list_notes(self):
        return self.notes
