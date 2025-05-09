class ToolManager:
    """
    A class to manage and run tools using the 'tool_name: input' format.
    """

    def __init__(self, tools=None):
        """
        Initialize the ToolManager with a list of tools.
        Each tool should have a .run(query) method and a unique name.
        """
        self.tools = tools or {}

    def register_tool(self, tool_name, tool_instance):
        """
        Register a tool with the manager.
        :param tool_name: The name of the tool (str).
        :param tool_instance: The tool instance (must have a .run() method).
        """
        self.tools[tool_name.lower()] = tool_instance

    def call_tool(self, tool_name, query):
        """
        Call the specified tool with the given query.
        :param tool_name: The name of the tool (str).
        :param query: The query string to pass to the tool.
        :return: The result of the tool's execution.
        """
        tool = self.tools.get(tool_name.lower())
        if not tool:
            return f"Error: Tool '{tool_name}' not found."
        if not hasattr(tool, "run"):
            return f"Error: Tool '{tool_name}' does not have a 'run' method."
        return tool.run(query)

    def run_tool(self, command):
        """
        Parse and run a tool using the 'tool_name: input' format.
        :param command: The command string (e.g., "tool_name: input").
        :return: The result of the tool's execution.
        """
        if ":" not in command:
            return "Invalid command format. Use 'tool_name: input'."

        parts = command.split(":", 1)
        if len(parts) != 2:
            return "Command must follow the format 'tool_name: input'."

        tool_name, query = parts
        tool_name = tool_name.strip().lower()
        query = query.strip()

        return self.call_tool(tool_name, query)
