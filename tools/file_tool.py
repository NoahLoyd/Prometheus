# tools/file_tool.py

import os
from tools.base_tool import BaseTool

class FileTool(BaseTool):
    """
    A tool for interacting with local files. Supports reading, writing, and listing files.
    """

    def __init__(self):
        super().__init__(name="file", description="Tool for managing local files (read, write, list).")

    def run(self, query: str) -> str:
        parts = query.strip().split(":", 2)

        if len(parts) < 2:
            return "Invalid command format. Use 'file: action: filename[: content]'."

        action = parts[0].strip().lower()

        if action == "list":
            try:
                files = [f for f in os.listdir() if os.path.isfile(f)]
                return "\n".join(files) or "No files found."
            except Exception as e:
                return f"Error listing files: {str(e)}"

        elif action == "read" and len(parts) == 2:
            filename = parts[1].strip()
            try:
                with open(filename, "r") as f:
                    return f.read()
            except FileNotFoundError:
                return f"Error: File '{filename}' not found."
            except Exception as e:
                return f"Error reading file '{filename}': {str(e)}"

        elif action == "write" and len(parts) == 3:
            filename = parts[1].strip()
            content = parts[2].strip()
            try:
                with open(filename, "w") as f:
                    f.write(content)
                return f"Successfully written to '{filename}'."
            except Exception as e:
                return f"Error writing to file '{filename}': {str(e)}"

        else:
            return "Invalid command. Use 'file: list', 'file: read: filename', or 'file: write: filename: content'."