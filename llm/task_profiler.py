import logging
from typing import Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import os

class TaskProfiler:
    """
    Profiles tasks and predicts task types using a trained classifier.
    Falls back to a safe default if the classifier model is absent.

    The interface is stable for future integration of a trained classifier.
    """

    def __init__(self, model_path: str = "task_classifier.pkl", fallback_type: str = "always_tool"):
        """
        Initialize the Task Profiler with a pre-trained classifier if available.
        If not, operate in fallback mode.

        :param model_path: Path to the serialized classifier model.
        :param fallback_type: Task type to use in fallback mode.
        """
        self.model_path = model_path
        self.fallback_type = fallback_type
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[SVC] = None
        self._ready = False
        self._load_model()

    def _load_model(self):
        """
        Attempt to load the task classification model and vectorizer.
        If the model file is missing or corrupted, fall back gracefully.
        """
        if not os.path.isfile(self.model_path):
            logging.warning(
                f"[TaskProfiler] Model file '{self.model_path}' not found. "
                f"Falling back to '{self.fallback_type}' mode. "
                "Promethyn will continue self-coding, but task classification is disabled until a model is provided."
            )
            self._ready = False
            return

        try:
            with open(self.model_path, "rb") as file:
                self.vectorizer, self.classifier = pickle.load(file)
            self._ready = True
        except Exception as e:
            logging.warning(
                f"[TaskProfiler] Failed to load model '{self.model_path}': {e}. "
                f"Reverting to '{self.fallback_type}' mode."
            )
            self._ready = False

    def is_ready(self) -> bool:
        """
        Check if the classifier model is loaded and operational.

        :return: True if classifier is loaded, else False (fallback mode).
        """
        return self._ready

    def classify_task(self, goal: str) -> str:
        """
        Classify the task type based on the provided goal.
        Falls back to a default value if model is unavailable.

        :param goal: The user-provided task goal.
        :return: Predicted task type (e.g., 'reasoning', 'coding'), or fallback type.
        """
        if not self.is_ready():
            # Fallback mode: Always return the default/fallback type.
            return self.fallback_type

        # Model is available, perform actual prediction.
        features = self.vectorizer.transform([goal])
        return self.classifier.predict(features)[0]
