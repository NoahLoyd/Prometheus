import csv
import time
from datetime import datetime
import re
from core.planner_llm import LLMPlanner  # Import LLMPlanner

class StrategicBrain:
    def __init__(self, tool_manager, short_term_memory, long_term_memory):
        self.tool_manager = tool_manager
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.goal = None
        self.plan = []
        self.tags = []
        self.llm_planner = LLMPlanner()  # Instantiate LLMPlanner

    def log(self, key, value):
        timestamp = datetime.now().isoformat()
        self.short_term_memory.save(f"{timestamp}::{key}", value)
        self.long_term_memory.log_event(key, value)

    def extract_tags(self, goal):
        words = re.findall(r'\w+', goal.lower())
        tags = set()
        if "money" in words or "$" in words:
            tags.add("money")
        if "learn" in words:
            tags.add("learning")
        if "audience" in words or "followers" in words:
            tags.add("audience")
        if "fitness" in words or "health" in words:
            tags.add("fitness")
        return list(tags) if tags else ["general"]

    def memory_aware_plan(self, goal):
        tags = self.extract_tags(goal)
        previous_goals = []
        for tag in tags:
            previous_goals.extend(self.long_term_memory.summarize_goals_by_tag(tag))
        self.tags = tags
        return previous_goals

    def set_goal(self, goal):
        self.goal = goal
        self.memory_aware_plan(goal)
        self.plan = self.plan_goal(goal)
        self.short_term_memory.save("current_goal", goal)
        self.short_term_memory.save("current_plan", self.plan)
        self.log("goal_set", {"goal": goal, "tags": self.tags})

    def plan_goal(self, goal, use_llm=True):
        if use_llm:
            try:
                steps = self.llm_planner.plan(goal)
                self.plan = steps
                self.log("llm_planner_success", steps)
                return steps
            except Exception as e:
                self.log("llm_planner_failure", str(e))
                # Fall back to static planning logic

        goal = goal.lower()
        steps = []

        if "money" in goal or "$" in goal:
            steps.append(("calculator", "1000 / 30"))
            steps.append(("internet", "Search: best ways to make money online 2025"))
            steps.append(("summarizer", "Summarize top 3 money-making strategies"))
            steps.append(("note", f"save: Strategy plan for goal: {goal}"))

        if "audience" in goal or "followers" in goal:
            steps.append(("internet", "Search: how to grow an audience in 2025"))
            steps.append(("summarizer", "Summarize audience growth strategies"))
            steps.append(("note", f"save: Growth plan for goal: {goal}"))

        if "learn" in goal:
            steps.append(("internet", f"Search: best resources to learn about {goal}"))
            steps.append(("summarizer", "Summarize top learning paths"))
            steps.append(("note", f"save: Learning plan for {goal}"))

        if "fitness" in goal or "health" in goal:
            steps.append(("internet", f"Search: best fitness plans for {goal}"))
            steps.append(("summarizer", "Summarize effective fitness strategies"))
            steps.append(("note", f"save: Fitness plan for goal: {goal}"))

        if not steps:
            steps.append(("note", f"save: Custom plan requested: {goal}"))

        self.plan = steps
        return steps

    def execute_step(self, tool_name, query):
        try:
            result = self.tool_manager.call_tool(tool_name, query)
            self.log(f"{tool_name}_success", query)
            self.short_term_memory.save(f"{tool_name}_result", result)
            self.long_term_memory.log_tool_usage(tool_name)
            return {"success": True, "result": result}
        except Exception as e:
            error_message = str(e)
            self.log(f"{tool_name}_error", error_message)
            return {"success": False, "error": error_message}

    def execute_plan(self):
        if not self.plan:
            raise ValueError("No plan set. Use set_goal() first.")
        results = []
        for i, (tool_name, query) in enumerate(self.plan):
            self.log("step_started", f"{tool_name}: {query}")
            result = self.execute_step(tool_name, query)
            results.append({
                "tool_name": tool_name,
                "query": query,
                **result
            })
            self.short_term_memory.save(f"step_{i}_result", result)
            time.sleep(1)
        return results

    def retry_or_replan(self, failed_steps):
        new_steps = []
        tool_failures = {}
        for step in failed_steps:
            tool = step["tool_name"]
            query = step["query"]
            tool_failures[tool] = tool_failures.get(tool, 0) + 1
            if tool_failures[tool] > 2:  # Switch tools after 2 failures
                tool = "note"  # Example of switching to a different tool
                new_query = f"Alternative strategy for: {query}"
            else:
                new_query = f"Retry: {query}"
            self.log("replan", {"tool": tool, "query": new_query})
            new_steps.append((tool, new_query))
        return new_steps

    def rank_steps(self, results):
        ranked = []
        for step in results:
            score = 2 if step["success"] else -1
            ranked.append({
                "tool": step["tool_name"],
                "query": step["query"],
                "score": score
            })
        self.short_term_memory.save("ranked_steps", ranked)
        return ranked

    def evaluate_results(self, step_results):
        structured = {
            "successful_steps": [
                {"tool": s["tool_name"], "query": s["query"], "result": s["result"]}
                for s in step_results if s["success"]
            ],
            "failed_steps": [
                {"tool": s["tool_name"], "query": s["query"], "error": s["error"]}
                for s in step_results if not s["success"]
            ]
        }
        self.short_term_memory.save("evaluation_summary", structured)
        self.long_term_memory.add_insight("evaluation_summary", structured)
        self.log("evaluation", structured)

        scores = {}
        for s in structured["successful_steps"]:
            scores[s["tool"]] = scores.get(s["tool"], 0) + 1
        best_tool = max(scores, key=scores.get, default=None)
        if best_tool:
            self.long_term_memory.add_insight("most_effective_tool", best_tool)

        error_reasons = {}
        for s in structured["failed_steps"]:
            reason = "syntax/tool mismatch" if "syntax" in s["error"].lower() else "other"
            error_reasons[reason] = error_reasons.get(reason, 0) + 1
        self.long_term_memory.add_insight("failure_reasons", error_reasons)

        return structured

    def generate_goal_summary(self, goal_result, format="dict", filepath=None):
        summary = {
            "Goal": goal_result["goal"],
            "Tags": ", ".join(self.tags),
            "Success": goal_result["success"],
            "Details": goal_result["results"]
        }
        if format == "csv":
            headers = ["Goal", "Tags", "Success", "Details"]
            rows = [[summary["Goal"], summary["Tags"], summary["Success"], str(summary["Details"])]]
            if filepath:
                with open(filepath, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    writer.writerows(rows)
        elif format == "markdown":
            markdown_output = f"# Goal Summary\n\n" \
                              f"**Goal:** {summary['Goal']}\n\n" \
                              f"**Tags:** {summary['Tags']}\n\n" \
                              f"**Success:** {summary['Success']}\n\n" \
                              f"**Details:**\n\n{summary['Details']}\n"
            if filepath:
                with open(filepath, mode="w") as file:
                    file.write(markdown_output)
        return summary

    def summarize_top_tools(self):
        tool_usage = self.long_term_memory.get_memory().get("tool_usage", {})
        sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        return [{"tool": tool, "usage_count": count} for tool, count in sorted_tools]

    def archive_goal_history(self, final):
        entry = {
            "goal": final["goal"],
            "success": final["success"],
            "timestamp": datetime.now().isoformat(),
            "summary": self.evaluate_results(final["results"]),
            "ranked_steps": final["step_scores"],
            "tags": self.tags
        }
        self.short_term_memory.save(f"history::{entry['timestamp']}", entry)
        self.long_term_memory.add_goal_history(entry)

    def achieve_goal(self):
        if not self.goal or not self.plan:
            raise ValueError("Goal and plan must be set before execution.")
        step_results = self.execute_plan()
        failed = [s for s in step_results if not s["success"]]
        if failed:
            retries = self.retry_or_replan(failed)
            self.plan = retries
            retry_results = self.execute_plan()
            step_results.extend(retry_results)
        ranked = self.rank_steps(step_results)
        final = {
            "goal": self.goal,
            "results": step_results,
            "failures": failed,
            "success": all(r["success"] for r in step_results),
            "step_scores": ranked
        }
        self.short_term_memory.save("goal_result", final)
        self.archive_goal_history(final)
        self.log("goal_complete", final["success"])
        return final
       