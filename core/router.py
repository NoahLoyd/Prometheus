# core/router.py

from core.reasoning import ReasoningEngine

class CommandRouter:
    def __init__(self, agent):
        self.agent = agent
        self.reasoning = ReasoningEngine(agent)

    def interpret(self, command):
        command = command.lower().strip()

        if "calculate" in command:
            expression = command.replace("calculate", "").strip()
            return self.agent.act("calculator", expression)

        elif "summarize" in command and "http" in command:
            url = self._extract_url(command)
            return self.agent.act("internet", url)

        elif "file" in command or "read" in command or "write" in command or "list" in command:
            parsed = self._parse_file_command(command)
            return self.agent.act("file", **parsed)

        elif "note" in command or "write down" in command:
            return self.agent.act("notepad")

        elif command.startswith("remember"):
            info = command.replace("remember", "").strip()
            self.agent.memory.add(info)
            return f"Remembered: {info}"

        elif "what do you remember" in command or "recall memory" in command:
            return self.agent.recall()

        elif "clear memory" in command or "forget everything" in command:
            self.agent.memory.clear()
            return "Memory cleared."

        elif "what can you do" in command or "list tools" in command or "available tools" in command:
            tool_names = list(self.agent.tool_manager.tools.keys())
            return f"Available tools: {', '.join(tool_names)}"

        elif "analyze" in command or "what should i do" in command:
            goal = command.replace("analyze", "").replace("what should i do", "").strip()
            return self.reasoning.analyze_goal(goal)

        else:
            return "I didn't understand that command. Try 'calculate', 'summarize', 'note', or 'remember'."

    def _extract_url(self, text):
        words = text.split()
        for word in words:
            if word.startswith("http"):
                return word
        return ""

    def _parse_file_command(self, command):
        # Default values
        action = "read"
        filename = "default.txt"
        content = ""

        if "write" in command:
            action = "write"
            try:
                # Extract filename
                if "to" in command:
                    to_index = command.split().index("to")
                    filename = command.split()[to_index + 1]

                # Extract content
                if "saying" in command:
                    saying_index = command.lower().index("saying") + len("saying ")
                    content = command[saying_index:].split(" to")[0].strip()
            except Exception:
                content = "No content provided."

        elif "read" in command:
            action = "read"
            try:
                if "file" in command:
                    file_index = command.split().index("file")
                    filename = command.split()[file_index + 1]
            except Exception:
                filename = "default.txt"

        elif "list" in command:
            action = "list"

        return {
            "action": action,
            "filename": filename,
            "content": content
        }
