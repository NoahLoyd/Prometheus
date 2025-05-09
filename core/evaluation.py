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

    def reflect_on_performance(self, results):
        # Summarize success/failure ratios, error patterns, and generate recommendations
        total_steps = len(results)
        successful_steps = [res for res in results if res["success"]]
        failed_steps = [res for res in results if not res["success"]]
        substituted_tools = [
            res["substituted_tool"] for res in results if "substituted_tool" in res
        ]

        reflection = {
            "total_steps": total_steps,
            "success_ratio": len(successful_steps) / total_steps if total_steps > 0 else 0,
            "failure_ratio": len(failed_steps) / total_steps if total_steps > 0 else 0,
            "error_patterns": self._analyze_errors(failed_steps),
            "substituted_tools": substituted_tools,
            "improvement_recommendations": self._generate_recommendations(failed_steps)
        }
        return reflection

    def _analyze_errors(self, failed_steps):
        error_patterns = {}
        for step in failed_steps:
            error_message = step.get("error", "Unknown error")
            error_patterns[error_message] = error_patterns.get(error_message, 0) + 1
        return error_patterns

    def _generate_recommendations(self, failed_steps):
        recommendations = []
        for step in failed_steps:
            tool = step["tool_name"]
            recommendations.append(f"Consider alternative strategies for tool: {tool}")
        return recommendations

    def compare_with_past_goals(self, goal, results):
        # Retrieve similar past goals and evaluate relative performance
        past_goals = self.memory.retrieve_related_goals(goal)
        comparison = []
        for past_goal in past_goals:
            success_rate = sum(1 for res in past_goal["results"] if res["success"]) / len(past_goal["results"])
            comparison.append({
                "past_goal": past_goal["goal"],
                "past_success_rate": success_rate,
                "current_success_rate": sum(1 for res in results if res["success"]) / len(results),
            })
        return comparison