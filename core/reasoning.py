# core/reasoning.py

class ReasoningEngine:
    def __init__(self, agent):
        self.agent = agent

    def analyze_goal(self, goal):
        if not goal:
            return "Please describe a goal you'd like help with."

        plan = self.generate_plan(goal)
        return f"To achieve the goal: '{goal}', Prometheus recommends:\n\n{plan}"

    def generate_plan(self, goal):
        goal = goal.lower()

        if "website" in goal:
            return (
                "- Research the website’s purpose and target audience.\n"
                "- Generate a name, tagline, and domain idea.\n"
                "- Write basic HTML/CSS/JS structure using the code tool.\n"
                "- Save files to local or remote storage.\n"
                "- Deploy to a static host or GitHub Pages.\n"
                "- Monitor for feedback and improve UX."
            )

        elif "money" in goal or "income" in goal:
            return (
                "- Identify fastest possible monetization methods (e.g. affiliate site, product, services).\n"
                "- Use Prometheus to generate landing pages and offers.\n"
                "- Build and deploy content using website + notepad tools.\n"
                "- Track metrics and adjust strategy every few days.\n"
                "- Automate as much as possible using saved tasks or cron jobs."
            )

        elif "learn" in goal or "study" in goal:
            return (
                "- Define what you want to learn and why.\n"
                "- Create a spaced repetition plan.\n"
                "- Use Prometheus to summarize resources and quiz you.\n"
                "- Store flashcards or progress in memory.json.\n"
                "- Reflect weekly on retention and understanding."
            )

        else:
            return (
                "- Break the goal into specific steps or subgoals.\n"
                "- Identify tools or models that can help.\n"
                "- Execute one task at a time, store progress.\n"
                "- Adjust plan based on outcome.\n"
                "- Reflect, document, and improve."
            )
