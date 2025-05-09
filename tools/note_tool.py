# tools/note_tool.py

import os
from tools.base_tool import BaseTool

class NoteTool(BaseTool):
    """
    A tool for saving and listing notes.
    """

    def __init__(self, notes_file="notes.txt"):
        """
        Initialize the NoteTool with a default notes file.

        Args:
            notes_file (str): The name of the file where notes will be stored.
        """
        super().__init__(name="note", description="Save and list notes.")
        self.notes_file = notes_file

    def run(self, query: str) -> str:
        """
        Process a query command for the note tool.

        Args:
            query (str): The command to execute, e.g., "save: message" or "list".

        Returns:
            str: The result of the command execution.
        """
        if query.startswith("save:"):
            # Extract the message to save
            note = query.split("save:", 1)[1].strip()
            return self._save_note(note)
        elif query.strip() == "list":
            # List all saved notes
            return self._list_notes()
        else:
            # Invalid command format
            return "Invalid command. Use 'save: message' to save a note or 'list' to list notes."

    def _save_note(self, note: str) -> str:
        """
        Save a note to the notes file.

        Args:
            note (str): The note to save.

        Returns:
            str: Success or error message.
        """
        if not note:
            return "Cannot save an empty note."
        try:
            with open(self.notes_file, "a") as file:
                file.write(note + "\n")
            return "Note saved successfully."
        except Exception as e:
            return f"Error saving note: {e}"

    def _list_notes(self) -> str:
        """
        List all saved notes from the notes file.

        Returns:
            str: The list of notes or an appropriate message if no notes are saved.
        """
        try:
            # Check if the notes file exists
            if not os.path.exists(self.notes_file):
                return "No notes saved yet."
            # Read the notes from the file
            with open(self.notes_file, "r") as file:
                notes = file.read().strip()
            return notes if notes else "No notes saved yet."
        except Exception as e:
            return f"Error reading notes: {e}"
