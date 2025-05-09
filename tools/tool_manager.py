# tools/tool_manager.py

def run_tool(self, command):
    """Parse and run tool using 'tool_name: input' format."""
    if ":" not in command:
        return "Invalid command format. Use 'tool_name: input'"

    parts = command.split(":", 1)
    if len(parts) != 2:
        return "Command must follow the format 'tool_name: input'."

    tool_name, query = parts
    tool_name = tool_name.strip().lower()
    query = query.strip()

    return self.call_tool(tool_name, query)
