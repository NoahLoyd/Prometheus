from typing import List, Any

class VotingStrategy:
    """
    Abstract base class for voting-based aggregation of LLM outputs.
    """

    def vote(self, outputs: List[str], context: Any = None) -> int:
        """
        Vote on a list of outputs to select the best one.

        :param outputs: List of outputs from different models.
        :param context: Optional context for voting.
        :return: Index of the selected output.
        """
        raise NotImplementedError("VotingStrategy.vote() must be implemented by subclasses.")

class PlanVotingStrategy(VotingStrategy):
    """
    Votes on outputs by majority/plurality (for plans).
    """

    def __init__(self, logger: Any = None):
        self.logger = logger

    def vote(self, outputs: List[str], context: Any = None) -> int:
        if not outputs:
            if self.logger:
                self.logger.warning("No outputs to vote on.")
            return -1
        # Placeholder: implement a real voting algorithm in production
        if self.logger:
            self.logger.info("PlanVotingStrategy selected the first output (stub).")
        return 0

class FragmentVotingStrategy(VotingStrategy):
    """
    Votes on outputs by fragments or sections (for fragmented outputs).
    """

    def __init__(self, logger: Any = None):
        self.logger = logger

    def vote(self, outputs: List[str], context: Any = None) -> int:
        if not outputs:
            if self.logger:
                self.logger.warning("No outputs to vote on.")
            return -1
        # Placeholder: implement a fragment-based voting algorithm in production
        if self.logger:
            self.logger.info("FragmentVotingStrategy selected the first output (stub).")
        return 0
