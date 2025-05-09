import os
from tools.base_tool import BaseTool

class FileTool(BaseTool):
    def run(self, query: str) -> str:
        parts = query.strip().split(":", 2)
        if len(parts) < 2:
            return "Invalid format. Use: 'file: action: filename[: content]'"

        action = parts[0].strip().lower()

        if action == "list":
            files = [f for f in os.listdir() if os.path.isfile(f)]
            return "\n".join(files) or "No files found."

        elif action == "read":
            filename = parts[1].strip()
            try:
                with open(filename, "r") as f:
                    return f.read()
            except FileNotFoundError:
                return f"File '{filename}' not found."

        elif action == "write" and len(parts) == 3:
            filename = parts[1].strip()
            content = parts[2].strip()
            with open(filename, "w") as f:
                f.write(content)
            return f"Written to '{filename}'."

        else:
            return "Invalid action. Use: 'file: list', 'file: read: filename', or 'file: write: filename: content'"