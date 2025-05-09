import time
import os
import sys
from datetime import datetime
from termcolor import colored
from core.brain import StrategicBrain
from core.memory import Memory  # Assuming memory is implemented
from core.tool_manager import ToolManager  # Assuming tool manager is implemented
from openai_llm import OpenAILLM  # Assuming LLM is implemented

class PromethynShell:
    def __init__(self):
        # Initialize StrategicBrain with dependencies
        memory = Memory()
        tool_manager = ToolManager()
        llm = OpenAILLM()
        self.brain = StrategicBrain(llm, tool_manager, memory)
        self.todo_queue = []

    def start(self):
        self.print_header()
        while True:
            user_input = self.get_user_input()
            if user_input.lower() in {"exit", "quit"}:
                self.print_goodbye()
                break
            elif user_input.lower() == "daily":
                self.daily_mode()
            else:
                self.process_goal(user_input)

    def print_header(self):
        print(colored("Welcome to Promethyn Shell!", "cyan", attrs=["bold"]))
        print(colored("Type your goals, or type 'daily' to enter daily mode.", "yellow"))

    def print_goodbye(self):
        print(colored("\nGoodbye! Promethyn will remember this session.", "green"))

    def get_user_input(self):
        try:
            return input(colored("\nWhat is your goal? > ", "blue"))
        except KeyboardInterrupt:
            self.print_goodbye()
            sys.exit(0)

    def process_goal(self, goal):
        # Check memory for similar past goals
        similar_goals = self.brain.logging.memory.retrieve_related_goals(goal)
        if similar_goals:
            print(colored("\nPromethyn remembers similar goals from the past:", "magenta"))
            for past_goal in similar_goals:
                print(f" - {past_goal['goal']} (Success: {past_goal['success']})")
                if "meta_lessons" in past_goal:
                    print(f"   Lessons: {past_goal['meta_lessons']}")
        
        # Set and achieve the goal
        print(colored("\nSetting the goal...", "cyan"))
        self.brain.set_goal(goal)
        print(colored("Goal set! Starting execution...\n", "green", attrs=["bold"]))
        
        # Stream step-by-step execution
        results = self.brain.achieve_goal(batch_mode=False)
        self.stream_execution_results(results["results"])
        
        # Log reflection and insights
        self.log_reflection_and_insights(results)

    def stream_execution_results(self, results):
        for step in results:
            tool = step["tool_name"]
            query = step["query"]
            success = step["success"]
            output = step.get("result", step.get("error", "No output"))
            color = "green" if success else "red"
            
            print(colored(f"\n[TOOL: {tool}] Executing: {query}", "yellow"))
            time.sleep(1)  # Simulate time delay for execution
            print(colored(f"Output: {output}", color))

    def log_reflection_and_insights(self, results):
        print(colored("\nGenerating reflection and insights...", "cyan"))
        reflection = results["reflection"]
        insights = results["insights"]
        
        # Display reflection
        print(colored("\nReflection Summary:", "magenta", attrs=["bold"]))
        print(f"- Success ratio: {reflection['success_ratio']:.2%}")
        print(f"- Failure ratio: {reflection['failure_ratio']:.2%}")
        print(f"- Error patterns: {reflection['error_patterns']}")
        print(f"- Substituted tools: {reflection['substituted_tools']}")
        print("\nImprovement Recommendations:")
        for rec in reflection["improvement_recommendations"]:
            print(f" - {rec}")
        
        # Log insights
        print(colored("\nInsights Generated:", "magenta", attrs=["bold"]))
        for insight in insights:
            print(f" - {insight}")
        
        # Auto-schedule follow-ups if needed
        self.auto_schedule_followups(reflection)

    def auto_schedule_followups(self, reflection):
        recommendations = reflection["improvement_recommendations"]
        if recommendations:
            print(colored("\nAuto-scheduling follow-ups...", "cyan"))
            for rec in recommendations:
                self.todo_queue.append(rec)
            print(colored(f"{len(recommendations)} follow-ups added to the queue.", "green"))
            self.run_todo_queue()

    def run_todo_queue(self):
        print(colored("\nRunning scheduled tasks from the queue...", "cyan"))
        while self.todo_queue:
            task = self.todo_queue.pop(0)
            print(colored(f"\nExecuting follow-up: {task}", "yellow"))
            # Treat follow-up task as a new goal
            self.process_goal(task)

    def daily_mode(self):
        print(colored("\nEntering Daily Mode!", "cyan", attrs=["bold"]))
        daily_goals = []
        while True:
            goal = input(colored("\nAdd a goal for today (or type 'start' to execute): > ", "blue"))
            if goal.lower() == "start":
                break
            elif goal:
                daily_goals.append(goal)
        
        print(colored("\nStarting daily execution...", "green"))
        for goal in daily_goals:
            self.process_goal(goal)
        
        # Generate daily strategy report
        print(colored("\nGenerating daily strategy report...", "cyan"))
        daily_report = self.brain.generate_daily_review()
        self.display_daily_report(daily_report)

    def display_daily_report(self, report):
        print(colored("\n--- DAILY STRATEGY REPORT ---", "magenta", attrs=["bold"]))
        print(f"Date: {report['date']}")
        print(f"Goals Achieved: {report['goals_achieved']}")
        print(f"Tools Used: {report['tools_used']}")
        print(f"Failures: {report['failures']}")
        print("\nMeta-Lessons Learned:")
        for lesson in report["meta_lessons"]:
            print(f" - {lesson}")
        print(colored("\n--- END OF REPORT ---", "magenta", attrs=["bold"]))


# Entry point
if __name__ == "__main__":
    shell = PromethynShell()
    shell.start()