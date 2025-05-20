from typing import Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle

class TaskProfiler:
    """
    Profiles tasks and predicts task types using a trained classifier.
    """

    def __init__(self, model_path: str = "task_classifier.pkl"):
        """
        Initialize the Task Profiler with a pre-trained classifier.
        :param model_path: Path to the serialized classifier model.
        """
        self.model_path = model_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[SVC] = None
        self._load_model()

    def _load_model(self):
        """Load the task classification model and vectorizer."""
        with open(self.model_path, "rb") as file:
            self.vectorizer, self.classifier = pickle.load(file)

    def classify_task(self, goal: str) -> str:
        """
        Classify the task type based on the goal.
        :param goal: The user-provided task goal.
        :return: Predicted task type (e.g., 'reasoning', 'coding').
        """
        features = self.vectorizer.transform([goal])
        return self.classifier.predict(features)[0]
