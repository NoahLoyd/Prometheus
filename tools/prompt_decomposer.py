from typing import Dict, Any

class PromptDecomposer:
    """
    Decomposes natural language prompts into actionable coding plans.
    """

    def decompose(self, prompt: str) -> Dict[str, Any]:
        """
        Given a prompt, returns a structured plan dictionary.
        This is a basic rule-based implementation; can be replaced with LLM logic.
        """
        # Very basic logic for demo; extend with NLP/LLM for advanced AGI use.
        plan = {}

        prompt = prompt.strip().lower()
        if "track time" in prompt or "time spent" in prompt:
            plan = {
                "file": "tools/time_tracker.py",
                "class": "TimeTrackerTool",
                "code": (
                    "from tools.base_tool import BaseTool\n"
                    "import time\n\n"
                    "class TimeTrackerTool(BaseTool):\n"
                    "    name = 'time_tracker'\n"
                    "    description = 'Tracks time spent on tasks.'\n\n"
                    "    def __init__(self):\n"
                    "        self.tasks = {}\n"
                    "        self.active_task = None\n"
                    "        self.start_time = None\n\n"
                    "    def run(self, query: str) -> str:\n"
                    "        cmd = query.strip().lower()\n"
                    "        if cmd.startswith('start:'):\n"
                    "            task = cmd[len('start:'):].strip()\n"
                    "            if self.active_task:\n"
                    "                return f'Already tracking {self.active_task}'\n"
                    "            self.active_task = task\n"
                    "            self.start_time = time.time()\n"
                    "            return f'Started tracking {task}'\n"
                    "        elif cmd == 'stop':\n"
                    "            if not self.active_task:\n"
                    "                return 'No active task.'\n"
                    "            elapsed = time.time() - self.start_time\n"
                    "            self.tasks[self.active_task] = self.tasks.get(self.active_task, 0) + elapsed\n"
                    "            msg = f'Stopped {self.active_task}. Time: {elapsed:.2f} seconds.'\n"
                    "            self.active_task = None\n"
                    "            self.start_time = None\n"
                    "            return msg\n"
                    "        elif cmd == 'report':\n"
                    "            if not self.tasks:\n"
                    "                return 'No tasks tracked.'\n"
                    "            return '\\n'.join(f'{t}: {s:.2f} sec' for t, s in self.tasks.items())\n"
                    "        else:\n"
                    "            return 'Commands: start:<task>, stop, report.'\n"
                ),
                "test": (
                    "from tools.time_tracker import TimeTrackerTool\n"
                    "tt = TimeTrackerTool()\n"
                    "print(tt.run('start: coding'))\n"
                    "import time; time.sleep(1)\n"
                    "print(tt.run('stop'))\n"
                    "print(tt.run('report'))\n"
                )
            }
        else:
            plan = {
                "file": "tools/undefined_module.py",
                "class": "UndefinedModule",
                "code": (
                    "# Placeholder for undefined module\n"
                    "class UndefinedModule:\n"
                    "    pass\n"
                ),
                "test": "print('No test defined.')"
            }
        return plan
