# tools/file_tool.py
import os
from tools.base_tool import BaseTool

class FileTool(BaseTool):
    def run(self, command: str) -> str:
        if command == "list":
            return ", ".join(os.listdir())
        elif command.startswith("read:"):
            filename = command.split("read:", 1)[1].strip()
            try:
                with open(filename, "r") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"
        elif command.startswith("write:"):
            parts = command.split("write:", 1)[1].split("::")
            if len(parts) != 2:
                return "Use format: write: filename.txt::content"
            filename, content = parts
            with open(filename.strip(), "w") as f:
                f.write(content.strip())
            return f"Wrote to {filename.strip()}"
        else:
            return "Unknown file command."
