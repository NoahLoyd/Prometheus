# core/router.py

class CommandRouter:
    def __init__(self, agent):
        self.agent = agent

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

        else:
            return "I didn't understand that command."

    def _extract_url(self, text):
        words = text.split()
        for word in words:
            if word.startswith("http"):
                return word
        return ""
