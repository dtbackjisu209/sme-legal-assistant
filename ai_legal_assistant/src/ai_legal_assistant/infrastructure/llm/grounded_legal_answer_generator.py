from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai_legal_assistant.application.dto.answer_generation_dto import (
    GenerateGroundedAnswerRequest,
)
from ai_legal_assistant.application.ports.llm_port import TextGenerationPort
from ai_legal_assistant.domain.entities.competition_submission import (
    CompetitionQuestion,
    ResolvedLegalContext,
)


SYSTEM_PROMPT = """Bạn là trợ lý pháp lý Việt Nam. Trả lời trực tiếp, rõ ràng và thận trọng.
Chỉ sử dụng các căn cứ được cung cấp trong phần NGỮ CẢNH PHÁP LÝ. Không dùng kiến thức bên ngoài,
không bịa số Điều, khoản, thời hạn, mức tiền hoặc tên văn bản. Khi ngữ cảnh chưa đủ để kết luận,
hãy nói rõ phạm vi thông tin mà các căn cứ đã cung cấp thay vì suy đoán.

Viết câu trả lời bằng tiếng Việt tự nhiên, có thể chia đoạn ngắn. Nêu Điều và tên văn bản khi căn
cứ đó thực sự hỗ trợ kết luận. Không nhắc đến "ngữ cảnh", "tài liệu truy xuất", prompt hay mô hình.
Không xuất JSON hoặc Markdown code fence."""


@dataclass(frozen=True)
class GroundedLegalAnswerGeneratorConfig:
    max_context_characters: int = 16_000
    max_characters_per_context: int = 3_000


class GroundedLegalAnswerGenerator:
    """Turns retrieved legal text into an answer while keeping citation selection deterministic."""

    def __init__(
        self,
        llm: TextGenerationPort,
        config: GroundedLegalAnswerGeneratorConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or GroundedLegalAnswerGeneratorConfig()
        if self.config.max_context_characters <= 0:
            raise ValueError("max_context_characters must be positive.")
        if self.config.max_characters_per_context <= 0:
            raise ValueError("max_characters_per_context must be positive.")

    def generate(
        self,
        *,
        question: CompetitionQuestion,
        contexts: Sequence[ResolvedLegalContext],
    ) -> str:
        if not contexts:
            raise ValueError(f"No legal context was retrieved for question id={question.question_id}.")
        user_prompt = (
            f"CÂU HỎI:\n{question.question}\n\n"
            f"NGỮ CẢNH PHÁP LÝ:\n{self._render_contexts(contexts)}\n\n"
            "Hãy trả lời câu hỏi chỉ dựa trên các căn cứ trên."
        )
        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        ).strip()
        if not answer:
            raise ValueError(f"Answer model returned an empty answer for id={question.question_id}.")
        return answer

    def generate_batch(
        self,
        *,
        requests: Sequence[GenerateGroundedAnswerRequest],
    ) -> tuple[str, ...]:
        if not requests:
            return ()
        prompts = tuple(self._build_user_prompt(request) for request in requests)
        answers = self.llm.generate_batch(
            system_prompt=SYSTEM_PROMPT,
            user_prompts=prompts,
        )
        if len(answers) != len(requests):
            raise ValueError("LLM returned an unexpected batch size.")
        cleaned_answers = tuple(answer.strip() for answer in answers)
        empty_request = next(
            (
                request
                for request, answer in zip(requests, cleaned_answers, strict=True)
                if not answer
            ),
            None,
        )
        if empty_request is not None:
            raise ValueError(
                "Answer model returned an empty answer for "
                f"id={empty_request.question.question_id}."
            )
        return cleaned_answers

    def _build_user_prompt(self, request: GenerateGroundedAnswerRequest) -> str:
        if not request.contexts:
            raise ValueError(
                "No legal context was retrieved for "
                f"question id={request.question.question_id}."
            )
        return (
            f"CÂU HỎI:\n{request.question.question}\n\n"
            f"NGỮ CẢNH PHÁP LÝ:\n{self._render_contexts(request.contexts)}\n\n"
            "Hãy trả lời câu hỏi chỉ dựa trên các căn cứ trên."
        )

    def _render_contexts(self, contexts: Sequence[ResolvedLegalContext]) -> str:
        rendered: list[str] = []
        consumed = 0
        for index, context in enumerate(contexts, start=1):
            available = self.config.max_context_characters - consumed
            if available <= 0:
                break
            text = context.text.strip()[: self.config.max_characters_per_context]
            if len(text) > available:
                text = text[:available]
            if not text:
                continue
            if context.citation is None:
                heading = f"Căn cứ {index}"
            else:
                heading = (
                    f"{context.citation.article}, {context.citation.document_title}"
                )
            item = f"[{heading}]\n{text}"
            rendered.append(item)
            consumed += len(item)
        if not rendered:
            raise ValueError("Retrieved legal context did not contain any usable text.")
        return "\n\n".join(rendered)
