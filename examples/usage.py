from pathlib import Path
from llm.llm_router import LLMRouter

# Initialize router with config
router = LLMRouter(config_path=Path("llm/model_config.json"))

# Generate a plan
result = router.generate_plan(
    goal="Analyze the performance implications of using async/await in this codebase",
    context="We're working on a high-throughput service with strict latency requirements"
)

# The router will:
# 1. Check available VRAM
# 2. Select appropriate models
# 3. Execute in parallel when possible
# 4. Handle failures gracefully
# 5. Clean up resources automatically
