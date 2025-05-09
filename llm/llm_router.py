    def _evaluate_results(self, results: List[Dict], goal: str, task_type: Optional[str]) -> Dict:
        """
        Evaluate and compare results from multiple models to select the best one.

        :param results: List of result dictionaries from models.
        :param goal: The goal or input for evaluation.
        :param task_type: Optional task type for evaluation relevance.
        :return: The best result dictionary.
        """
        def score_result(result: Dict) -> float:
            """
            Score a result based on its tool diversity, tag alignment, and past performance.
            """
            if not result["success"]:
                return 0.0

            plan = result["plan"]
            tool_diversity = len(set(step[0] for step in plan))  # Unique tools used
            tag_alignment = sum(1 for tag in task_type if tag in plan)  # Alignment with tags
            past_success = self.logger.get_model_success_rate(result["model_name"])

            return 0.5 * tool_diversity + 0.3 * tag_alignment + 0.2 * past_success

        # Score all results and select the best
        scored_results = [(score_result(result), result) for result in results]
        best_result = max(scored_results, key=lambda x: x[0])[1]

        return best_result

    def self_refine_plan(self, goal: str, context: Optional[str], task_type: Optional[str]) -> List[Tuple[str, str]]:
        """
        Attempt to refine the plan using a fallback meta-model if all models fail.

        :param goal: The goal or input to the models.
        :param context: Optional context for the models.
        :param task_type: Optional task type for relevance scoring.
        :return: A refined list of (tool_name, query) tuples representing the plan.
        """
        try:
            meta_model = self._get_model("fallback_meta_model")
            memory_context = self.logger.retrieve_long_term_memory(goal, task_type)
            refined_plan = meta_model.generate_plan(goal, f"{context}\n{memory_context}")
            return refined_plan
        except Exception as e:
            raise RuntimeError(f"Meta-model failed to refine the plan: {str(e)}")

    def _update_feedback_weights(self, model_name: str, task_type: Optional[str], success: bool):
        """
        Update feedback-based weights for a specific model and task type.

        :param model_name: The name of the model.
        :param task_type: The type of task the model was used for.
        :param success: Whether the model succeeded.
        """
        key = (model_name, task_type)
        current_weight = self.feedback_weights.get(key, 1.0)
        adjustment = 0.1 if success else -0.1
        self.feedback_weights[key] = max(0.1, current_weight + adjustment)  # Ensure weight stays positive

    def _merge_or_vote_plans(self, results: List[Dict]) -> List[Tuple[str, str]]:
        """
        Merge or vote on plans from multiple successful models.

        :param results: List of result dictionaries from models.
        :return: The selected comprehensive or consistent plan.
        """
        successful_plans = [result["plan"] for result in results if result["success"]]

        # Voting logic: Select the most common steps across plans
        step_votes = {}
        for plan in successful_plans:
            for step in plan:
                step_votes[step] = step_votes.get(step, 0) + 1

        # Sort steps by votes and return the most common plan
        merged_plan = sorted(step_votes.keys(), key=lambda step: step_votes[step], reverse=True)
        return merged_plan

    def generate_plan(self, goal: str, context: Optional[str] = None, task_type: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Generate a plan using the best available models in parallel and compare their outputs.

        :param goal: The goal or input to the models.
        :param context: Optional context for the models.
        :param task_type: Optional task type for relevance scoring.
        :return: A list of (tool_name, query) tuples representing the plan.
        """
        top_models = self._select_top_models(task_type, num_models=3)
        results = self._execute_in_parallel(top_models, goal, context)

        # If all models fail, attempt to refine the plan
        if not any(result["success"] for result in results):
            return self.self_refine_plan(goal, context, task_type)

        # If multiple models succeed, merge or vote on their plans
        if sum(1 for result in results if result["success"]) > 1:
            return self._merge_or_vote_plans(results)

        # Otherwise, evaluate and return the best single result
        best_result = self._evaluate_results(results, goal, task_type)
        self._update_model_profiles(best_result["model_name"], task_type, success=True)

        return best_result["plan"]