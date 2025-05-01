class NoteTool:
    def __init__(self):
        self.notes = []

    def save_note(self, text):
        self.notes.append(text)
        return f"Note saved: {text}"

    def list_notes(self):
        return self.notes

    def __call__(self, *args, **kwargs):
        note = kwargs.get("note", "This is a default note.")
        return self.save_note(note)
