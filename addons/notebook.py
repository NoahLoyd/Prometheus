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

    def log(self, component: str, log_type: str, message: Union[str, Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None):
        """
        Log an entry with timestamp, component, and log type.
        
        Args:
            component: The component generating the log (e.g., 'tool_manager', 'user_prompt')
            log_type: The type of log event (e.g., 'INITIALIZATION', 'ERROR', 'SUCCESS')
            message: The main log message (string) or content (dict)
            metadata: Optional dictionary of additional metadata to merge into the log entry
        
        For backward compatibility, also supports the old 2-arg format:
        - log(entry_type, content_dict)  # Original format
        """
        # Handle backward compatibility for old 2-arg calls
        if metadata is None and isinstance(message, dict) and log_type is None:
            # This is likely the old format: log(entry_type, content_dict)
            return self._log_legacy(component, message)
        
        # Build the base log entry with new format
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": component,
            "log_type": log_type
        }
        
        # Handle message content
        if isinstance(message, str):
            log_entry["message"] = message
        elif isinstance(message, dict):
            log_entry["content"] = message
        else:
            # Convert other types to string for safety
            log_entry["message"] = str(message)
        
        # Merge metadata if provided, avoiding overwrite of protected keys
        if metadata:
            protected_keys = {"timestamp", "component", "log_type", "message", "content"}
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
        Searches both 'source' (legacy) and 'component' (new) fields.
        """
        if not os.path.isfile(self.log_path):
            return []
        logs = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if not entry_type or entry.get("source") == entry_type or entry.get("component") == entry_type:
                        logs.append(entry)
                except Exception:
                    continue
        return logs
