import os
from tools.base_tool import BaseTool

class FileTool(BaseTool):
    def __init__(self):
        super().__init__(name="file", description="Handles file read, write, and list operations.")

    def run(self, query: str) -> str:
        return self.handle(query)

    def handle(self, query: str) -> str:
        try:
            parts = query.strip().split(" ", 2)
            command = parts[0].lower()

            if command == "write" and len(parts) == 3:
                filename = parts[1]
                content = parts[2]
                with open(filename, "w") as f:
                    f.write(content)
                return f"Content written to '{filename}'."

            elif command == "read" and len(parts) >= 2:
                filename = parts[1]
                with open(filename, "r") as f:
                    return f.read()

            elif command == "list":
                files = [f for f in os.listdir() if os.path.isfile(f)]
                return f"Files: {', '.join(files)}"

            else:
                return "Usage:\n- write <filename> <content>\n- read <filename>\n- list"

        except Exception as e:
            return f"Error: {e}"
