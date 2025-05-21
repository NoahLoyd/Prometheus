import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

class AddOnNotebook:
    """
    Persistent logging utility for Promethyn's strategic intelligence and prompts.
    Logs user prompts that couldn't be built and self-generated strategic suggestions.
    Each entry has timestamp, source (entry_type), and details.
    """

    def __init__(self, log_path: Optional[str] = "addons/notebook_log.jsonl"):
        self.log_path = log_path
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log(self, entry_type: str, content: Dict[str, Any]):
        """
        Log an entry with timestamp and type.
        entry_type: 'user_prompt', 'self_suggestion', etc.
        content: dict with details.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": entry_type,
            "content": content
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_logs(self, entry_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns logs filtered by entry_type (or all if None).
        """
        if not os.path.isfile(self.log_path):
            return []
        logs = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if not entry_type or entry.get("source") == entry_type:
                        logs.append(entry)
                except Exception:
                    continue
        return logs
