# tools/file_tool.py

import os
from tools.base_tool import BaseTool

class FileTool(BaseTool):
    def __init__(self):
        super().__init__(name="file", description="Tool for managing local files (read, write, list).")

    def run(self, query: str) -> str:
        parts = query.strip().split(":", 2)

        if len(parts) < 2:
            return "Invalid command format. Use 'action: filename[: content]'."

        action = parts[0].strip().lower()

        if action == "list":
            return self._list_files()
        elif action == "read":
            filename = parts[1].strip()
            return self._read_file(filename)
        elif action == "write" and len(parts) == 3:
            filename = parts[1].strip()
            content = parts[2].strip()
            return self._write_file(filename, content)
        else:
            return "Invalid command. Use 'list', 'read', or 'write'."

    def _list_files(self) -> str:
        try:
            files = [f for f in os.listdir() if os.path.isfile(f)]
            return "\n".join(files) or "No files found."
        except Exception as e:
            return f"Error listing files: {str(e)}"

    def _read_file(self, filename: str) -> str:
        try:
            with open(filename, "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File '{filename}' not found."
        except Exception as e:
            return f"Error reading file '{filename}': {str(e)}"

    def _write_file(self, filename: str, content: str) -> str:
        try:
            with open(filename, "w") as f:
                f.write(content)
            return f"Successfully written to '{filename}'."
        except Exception as e:
            return f"Error writing to file '{filename}': {str(e)}"