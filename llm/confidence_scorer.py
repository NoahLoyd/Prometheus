from typing import List, Any

class ConfidenceScorer:
    """
    Scores LLM outputs for confidence, ranking, or filtering.
    """
    def __init__(self, logger: Any = None):
        self.logger = logger

    def score(self, outputs: List[str], context: Any = None) -> List[float]:
        """
        Assigns a confidence score (0.0 - 1.0) to each output.

        :param outputs: List of model outputs.
        :param context: Optional context for scoring.
        :return: List of confidence scores.
        """
        if not outputs:
            if self.logger:
                self.logger.warning("No outputs to score for confidence.")
            return []
        scores = [1.0 for _ in outputs]
        if self.logger:
            self.logger.info(f"ConfidenceScorer assigned uniform scores: {scores}.")
        return scores
