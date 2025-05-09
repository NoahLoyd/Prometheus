from typing import List, Tuple, Dict
from .local_llm import LocalLLM
from .base_llm import BaseLLM

class LLMRouter:
    def __init__(self, config: Dict[str, Dict]):
        """
        Initialize the router with a configuration of models.

        :param config: Dictionary with model names as keys and their parameters as values.
                       Example: {"mistral": {"path": "mistral-7b", "type": "local"}}
        """
        self.config = config
        self.models = {}

    def _get_model(self, name: str) -> BaseLLM:
        if name not in self.models:
            # Lazy load model based on type
            model_config = self.config[name]
            if model_config["type"] == "local":
                self.models[name] = LocalLLM(model_path=model_config["path"], device=model_config.get("device", "cpu"))
            # Add other model types here
        return self.models[name]

    def generate_plan(self, goal: str, context: Optional[str] = None) -> List[Tuple[str, str]]:
        # Logic to pick the best model based on scoring
        best_model_name = self._select_best_model()
        model = self._get_model(best_model_name)
        return model.generate_plan(goal, context)

    def _select_best_model(self) -> str:
        # Placeholder scoring logic
        return list(self.config.keys())[0]