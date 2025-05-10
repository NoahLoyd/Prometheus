from typing import List, Tuple, Dict
from core.logging import Logging


class PlanVotingStrategy:
    def merge_or_vote(self, results: List[Dict]) -> List[Tuple[str, str]]:
        ...


class FragmentVotingStrategy(PlanVotingStrategy):
    def __init__(self, logger: Logging):
        self.logger = logger

    def merge_or_vote(self, results: List[Dict]) -> List[Tuple[str, str]]:
        successful_plans = [result["plan"] for result in results if result["success"]]

        fragment_scores = {}
        for plan in successful_plans:
            for step in plan:
                fragment_scores[step] = fragment_scores.get(step, 0) + 1

        for fragment, score in fragment_scores.items():
            historical_score = self.logger.get_fragment_success_rate(fragment)
            fragment_scores[fragment] += historical_score

        merged_plan = sorted(fragment_scores.keys(), key=lambda step: fragment_scores[step], reverse=True)
        return merged_plan
