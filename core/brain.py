# core/brain.py

import time
from datetime import datetime


class StrategicBrain:
    """
    Strategic reasoning engine for Promethyn.
    Accepts high-level goals, creates plans, executes them, retries if needed,
    and logs all activity to memory.
    """

    def __init__(self, tool_manager, memory):
        self.tool_manager = tool_manager
        self.memory = memory
        self.goal = None
        self.plan = []

    def log(self, key, value):
        timestamp = datetime.now().isoformat()
        self.memory.save(f"{timestamp}::{key}", value)

    def set_goal(self, goal):
        self.goal = goal
        self.plan = self.plan_goal(goal)
        self.memory.save("current_goal", goal)
        self.memory.save("current_plan", self.plan)
        self.log("goal_set", goal)

    def plan_goal(self, goal):
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
            steps.append(("note", f"save: Fitness plan for {goal}"))

        if not steps:
            steps.append(("note", f"save: Custom plan requested: {goal}"))

        return steps

    def execute_step(self, tool_name, query):
        try:
            result = self.tool_manager.call_tool(tool_name, query)
            self.log(f"{tool_name}_success", query)
            self.memory.save(f"{tool_name}_result", result)
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
            self.memory.save(f"step_{i}_result", result)
            time.sleep(1)

        return results

    def retry_or_replan(self, failed_steps):
        new_steps = []
        for step in failed_steps:
            tool = step["tool_name"]
            query = step["query"]

            if "retry" not in query.lower():
                new_query = f"Retry: {query}"
                tool = step["tool_name"]
            else:
                new_query = f"Log failure for {query}"
                tool = "note"

            self.log("replan", {"tool": tool, "query": new_query})
            new_steps.append((tool, new_query))

        return new_steps

    def rank_steps(self, results):
        ranked = []
        for step in results:
            score = 2 if step["success"] else -1  # Factor in positive and negative impacts
            ranked.append({
                "tool": step["tool_name"],
                "query": step["query"],
                "score": score
            })
        self.memory.save("ranked_steps", ranked)
        return ranked

    def evaluate_results(self, step_results):
        structured_summary = {
            "successful_steps": [
                {"tool": step["tool_name"], "query": step["query"], "result": step["result"]}
                for step in step_results if step["success"]
            ],
            "failed_steps": [
                {"tool": step["tool_name"], "query": step["query"], "error": step["error"]}
                for step in step_results if not step["success"]
            ]
        }
        self.memory.save("evaluation_summary", structured_summary)
        self.log("evaluation", structured_summary)
        return structured_summary

    def archive_goal_history(self, final):
        entry = {
            "goal": final["goal"],
            "success": final["success"],
            "timestamp": datetime.now().isoformat(),
            "summary": self.evaluate_results(final["results"]),
            "ranked_steps": final["step_scores"]
        }
        self.memory.save(f"history::{entry['timestamp']}", entry)

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
        self.memory.save("goal_result", final)
        self.archive_goal_history(final)
        self.log("goal_complete", final["success"])
        return final
       