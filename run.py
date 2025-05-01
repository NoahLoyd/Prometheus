from core.agent import PrometheusAgent

if __name__ == "__main__":
    agent = PrometheusAgent()

    print(agent.think("What is 10 * 3?"))
    print(agent.act("calculator", "10 * 3"))

    print("\nMemory contents:")
    print(agent.recall())
