# evaluation.py
class Evaluation:
    def __init__(self, memory):
        self.memory = memory

    def rank_steps(self, results):
        # Weighted scoring based on context and performance
        return sorted(results, key=lambda x: self._compute_score(x), reverse=True)

    def _compute_score(self, result):
        score = 2 if result["success"] else -1
        if "substituted_tool" in result:
            score -= 1  # Penalize substitutions
        return score

    def generate_summary(self, results):
        # Generate insights and summaries
        insights = [res.get("insight", "") for res in results if res["success"]]
        self.memory.log_insights(insights)
        return "\n".join(insights)