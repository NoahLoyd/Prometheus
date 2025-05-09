# tools/file_tool.py
import os
from tools.base_tool import BaseTool

class FileTool(BaseTool):
    def __init__(self):
        super().__init__(name="file", description="Read and write files.")

    def run(self, query: str) -> str:
        command, _, data = query.partition(":")
        command = command.strip().lower()
        if command == "read":
            return self._read_file(data.strip())
        elif command == "write":
            filename, _, content = data.partition(":")
            return self._write_file(filename.strip(), content.strip())
        else:
            return "Invalid command. Use 'read: filename' or 'write: filename: content'."

    def _read_file(self, filename: str) -> str:
        try:
            with open(filename, "r") as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {e}"

    def _write_file(self, filename: str, content: str) -> str:
        try:
            with open(filename, "w") as file:
                file.write(content)
            return f"File '{filename}' written successfully."
        except Exception as e:
            return f"Error writing to file: {e}"
            