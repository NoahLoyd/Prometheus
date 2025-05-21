from typing import Any, List, Optional

class EvaluationStrategy:
    """
    Abstract base class for evaluating LLM outputs.
    """
    def evaluate(self, outputs: List[str], context: Optional[str] = None) -> int:
        """
        Evaluate a list of outputs and select the best one.

        :param outputs: List of outputs from different models.
        :param context: Optional context for evaluation.
        :return: Index of the best output.
        """
        raise NotImplementedError("EvaluationStrategy.evaluate() must be implemented by subclasses.")

class DefaultEvaluationStrategy(EvaluationStrategy):
    """
    Default implementation: selects the first output unless overridden.
    Accepts a logger for production use.
    """
    def __init__(self, logger: Any = None):
        self.logger = logger

    def evaluate(self, outputs: List[str], context: Optional[str] = None) -> int:
        if not outputs:
            if self.logger:
                self.logger.warning("No outputs to evaluate.")
            return -1
        if self.logger:
            self.logger.info(f"Evaluating {len(outputs)} outputs using DefaultEvaluationStrategy.")
        return 0
