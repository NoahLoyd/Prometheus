from typing import List, Dict


class VotingStrategy:
    """
    Combines multiple successful outputs via majority voting or scoring.
    """

    def merge_or_vote(self, results: List[Dict]) -> List[Dict]:
        """
        Merge results or perform majority voting to select the best output.

        :param results: List of successful results.
        :return: A merged or voted output plan.
        """
        successful_results = [result for result in results if result["success"]]
        if not successful_results:
            raise ValueError("No successful results to merge or vote on.")

        # Simple majority voting example
        plans = [result["plan"] for result in successful_results]
        return max(set(plans), key=plans.count)  # Return the most common plan
