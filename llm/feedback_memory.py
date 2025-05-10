# llm/feedback_memory.py

import sqlite3

class FeedbackMemory:
    """
    Records feedback and task results into a persistent memory system.
    """

    def __init__(self, db_path: str = "feedback_memory.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Create the feedback table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT,
                model_name TEXT,
                success INTEGER,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def record_feedback(self, task_type: str, model_name: str, success: bool, confidence: float):
        """
        Record feedback into the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (task_type, model_name, success, confidence)
            VALUES (?, ?, ?, ?)
        """, (task_type, model_name, int(success), confidence))
        conn.commit()
        conn.close()