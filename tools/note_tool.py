import os
from tools.base_tool import BaseTool


class NoteTool(BaseTool):
    """
    A tool for saving and listing user notes. Notes are stored in a specified file.
    """

    def __init__(self, notes_file="notes.txt"):
        """
        Initialize the NoteTool with a default or custom notes file.
        :param notes_file: The file to store notes. Defaults to 'notes.txt'.
        """
        self.notes_file = notes_file

    def run(self, query: str) -> str:
        """
        Process a user query to save or list notes.
        :param query: The input query in the format 'note: action: message'.
        :return: A message indicating the result of the action.
        """
        # Split the query into action and data
        parts = query.strip().split(":", 1)
        if len(parts) != 2:
            return "Invalid format. Use 'note: save: message' or 'note: list'."

        action = parts[0].strip().lower()
        data = parts[1].strip()

        if action == "save":
            return self._save_note(data)
        elif action == "list":
            return self._list_notes()
        else:
            return (
                "Unknown action. Valid actions are 'save' (e.g., 'note: save: message') "
                "or 'list' (e.g., 'note: list')."
            )

    def _save_note(self, note: str) -> str:
        """
        Save a note to the notes file.
        :param note: The note to save.
        :return: A success message.
        """
        if not note:
            return "Cannot save an empty note. Please provide a valid message."

        try:
            with open(self.notes_file, "a") as f:
                f.write(note + "\n")
            return "Note saved successfully."
        except Exception as e:
            return f"Error saving note: {str(e)}"

    def _list_notes(self) -> str:
        """
        List all saved notes from the notes file.
        :return: A string containing all saved notes or a message indicating no notes exist.
        """
        if not os.path.exists(self.notes_file):
            return "No notes saved yet."

        try:
            with open(self.notes_file, "r") as f:
                notes = f.read().strip()
            return notes if notes else "No notes saved yet."
        except Exception as e:
            return f"Error reading notes: {str(e)}"
