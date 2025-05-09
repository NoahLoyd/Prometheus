from core.logging import Logging
from core.brain import StrategicBrain


def main():
    """
    Main entry point for the Promethyn Shell.
    """
    # Define LLM configuration
    llm_config = {
        "mistral-7b": {
            "type": "local",
            "path": "/models/mistral-7b",
            "device": "cpu",
            "tags": ["strategy", "planning"]
        }
    }

    # Initialize Memory and StrategicBrain
    memory = Logging()
    brain = StrategicBrain(memory, llm_config)

    # Example goal
    goal = "Optimize logistics for supply chain management."
    brain.think(goal)


if __name__ == "__main__":
    main()
     