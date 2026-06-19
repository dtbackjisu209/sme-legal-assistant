from __future__ import annotations

from dataclasses import dataclass

from ai_legal_assistant.application.use_cases.rewrite_legal_query import (
    BuildLegalQueryPlanUseCase,
)
from ai_legal_assistant.domain.services.query_expansion_policy import QueryExpansionPolicy
from ai_legal_assistant.infrastructure.llm.huggingface_causal_llm import (
    HuggingFaceCausalLLM,
    HuggingFaceCausalLLMConfig,
)
from ai_legal_assistant.infrastructure.nlp.vietnamese_legal_query_normalizer import (
    VietnameseLegalQueryNormalizer,
)
from ai_legal_assistant.infrastructure.query_expansion.llm_legal_query_planner import (
    LLMLegalQueryPlanner,
)


@dataclass(frozen=True)
class QueryPlannerConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    device: str | None = None
    max_input_tokens: int = 4096
    max_new_tokens: int = 500
    trust_remote_code: bool = False


def build_query_planner(
    config: QueryPlannerConfig = QueryPlannerConfig(),
) -> BuildLegalQueryPlanUseCase:
    llm = HuggingFaceCausalLLM(
        HuggingFaceCausalLLMConfig(
            model_name_or_path=config.model_name_or_path,
            device=config.device,
            max_input_tokens=config.max_input_tokens,
            max_new_tokens=config.max_new_tokens,
            trust_remote_code=config.trust_remote_code,
        )
    )
    return BuildLegalQueryPlanUseCase(
        normalizer=VietnameseLegalQueryNormalizer(),
        planner=LLMLegalQueryPlanner(llm=llm),
        policy=QueryExpansionPolicy(),
    )
