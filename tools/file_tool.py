# tools/file_tool.py

from tools.base_tool import BaseTool
import os

class FileTool(BaseTool):
    name = "file"
    description = "A tool to read and write files from disk. Use with caution."

    def run(self, query: str) -> str:
        try:
            q = query.strip().lower()
            if q.startswith("read:"):
                path = query[5:].strip()
                if not os.path.isfile(path):
                    return f"File not found: {path}"
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            elif q.startswith("write:"):
                parts = query[6:].split(":", 1)
                if len(parts) != 2:
                    return "Write format: write: <path>: <content>"
                path, content = parts[0].strip(), parts[1].strip()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Written to {path}."
            elif q.startswith("append:"):
                parts = query[7:].split(":", 1)
                if len(parts) != 2:
                    return "Append format: append: <path>: <content>"
                path, content = parts[0].strip(), parts[1].strip()
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
                return f"Appended to {path}."
            elif q.startswith("delete:"):
                path = query[7:].strip()
                if not os.path.isfile(path):
                    return f"File not found: {path}"
                os.remove(path)
                return f"Deleted {path}."
            else:
                return "Unknown file command. Use 'read: <path>', 'write: <path>: <content>', 'append: <path>: <content>', or 'delete: <path>'."
        except Exception as e:
            return f"FileTool error: {e}"
