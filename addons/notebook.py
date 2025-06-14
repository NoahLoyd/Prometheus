import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

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

    def log(self, tag: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log an entry with timestamp, tag, and message.
        
        Args:
            tag: A tag to categorize the log entry (e.g., 'tool_manager', 'user_prompt', 'ERROR')
            message: The main log message as a string
            metadata: Optional dictionary of additional metadata to include in the log entry
        """
        # Build the base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tag": tag,
            "message": message
        }
        
        # Merge metadata if provided, avoiding overwrite of protected keys
        if metadata:
            protected_keys = {"timestamp", "tag", "message"}
            for key, value in metadata.items():
                if key not in protected_keys:
                    log_entry[key] = value
        
        # Write the log entry
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _log_legacy(self, entry_type: str, content: Dict[str, Any]):
        """
        Legacy logging method for backward compatibility.
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
        Searches both 'source' (legacy) and 'tag' (new) fields.
        """
        if not os.path.isfile(self.log_path):
            return []
        logs = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if not entry_type or entry.get("source") == entry_type or entry.get("tag") == entry_type:
                        logs.append(entry)
                except Exception:
                    continue
        return logs
