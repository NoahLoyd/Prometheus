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
            return f"Available tools: {', '.join(self.agent.tools.list_tools())}"

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
           
