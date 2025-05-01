# tools/file_tool.py
import os

class FileTool:
    def __call__(self, action="read", filename="default.txt", content=""):
        if action == "write":
            with open(filename, "w") as f:
                f.write(content)
            return f"Content written to '{filename}'."
        
        elif action == "read":
            try:
                with open(filename, "r") as f:
                    return f.read()
            except FileNotFoundError:
                return f"File '{filename}' not found."
        
        elif action == "list":
            files = [f for f in os.listdir() if os.path.isfile(f)]
            return f"Files: {', '.join(files)}"
        
        else:
            return f"Unknown action: {action}"
