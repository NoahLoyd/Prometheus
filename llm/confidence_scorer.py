from typing import Dict

class ConfidenceScorer:
    """
    Evaluates the confidence of model outputs based on coherence, relevance, and success metrics.
    """

    def compute_confidence(self, result: Dict) -> float:
        """
        Compute confidence score for a model output.
        :param result: Model output dictionary containing 'coherence', 'relevance', and 'success'.
        :return: Confidence score (0 to 1).
        """
        coherence = result.get("coherence", 0)
        relevance = result.get("relevance", 0)
        success = 1 if result.get("success", False) else 0
        return (0.5 * coherence + 0.4 * relevance + 0.1 * success)
