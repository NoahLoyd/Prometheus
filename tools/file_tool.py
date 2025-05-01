       # tools/file_tool.py

import os

class FileTool:
    def read(self, filename):
        if not os.path.exists(filename):
            return f"File '{filename}' not found."
        with open(filename, "r") as f:
            return f.read()

    def write(self, filename, content):
        with open(filename, "w") as f:
            f.write(content)
        return f"Content written to '{filename}'."

    def list_files(self, directory="."):
        try:
            return os.listdir(directory)
        except Exception as e:
            return str(e)

    def __call__(self, action="list", filename="", content=""):
        if action == "read":
            return self.read(filename)
        elif action == "write":
            return self.write(filename, content)
        elif action == "list":
            return self.list_files(filename or ".")
        else:
            return "Unknown file action."
