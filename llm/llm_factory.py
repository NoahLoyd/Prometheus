from typing import Dict
from llm.llm_router import LLMRouter
from llm.evaluation_strategy import DefaultEvaluationStrategy
from llm.fallback_strategy import ChainOfThoughtFallbackStrategy
from llm.voting_strategy import FragmentVotingStrategy
from core.logging import Logging

def build_llm_router(llm_config: Dict[str, Dict]) -> LLMRouter:
    """
    Factory function to build an LLMRouter with default strategies.

    :param llm_config: Configuration dictionary for LLM models
    :return: An instance of LLMRouter
    """
    logger = Logging()

    evaluation_strategy = DefaultEvaluationStrategy(logger)
    fallback_strategy = ChainOfThoughtFallbackStrategy(logger)
    voting_strategy = FragmentVotingStrategy(logger)

    return LLMRouter(
        config=llm_config,
        evaluation_strategy=evaluation_strategy,
        fallback_strategy=fallback_strategy,
        voting_strategy=voting_strategy
    )
