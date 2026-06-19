from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HuggingFaceCausalLLMConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    device: str | None = None
    max_input_tokens: int = 4096
    max_new_tokens: int = 500
    trust_remote_code: bool = False
    enable_thinking: bool = False


class HuggingFaceCausalLLM:
    def __init__(self, config: HuggingFaceCausalLLMConfig) -> None:
        if not config.model_name_or_path.strip():
            raise ValueError("model_name_or_path cannot be empty.")
        if config.max_input_tokens <= 0 or config.max_new_tokens <= 0:
            raise ValueError("Token limits must be positive.")

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                StoppingCriteria,
                StoppingCriteriaList,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Local Qwen query planning requires torch and transformers."
            ) from exc

        self.config = config
        self._torch = torch
        self._stopping_criteria_base = StoppingCriteria
        self._stopping_criteria_list = StoppingCriteriaList
        self._device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        dtype = "auto" if self._device.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        forbidden_phrases: tuple[str, ...] = (),
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.enable_thinking,
        )
        inputs = self._tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        stopping_criteria = self._json_stopping_criteria(inputs["input_ids"].shape[1])
        eos_token_ids = [self._tokenizer.eos_token_id]
        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_token_ids:
            eos_token_ids.append(im_end_id)
        bad_words_ids = [
            token_ids
            for phrase in forbidden_phrases
            if (token_ids := self._tokenizer.encode(phrase, add_special_tokens=False))
        ]
        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=eos_token_ids,
                stopping_criteria=stopping_criteria,
                bad_words_ids=bad_words_ids or None,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        output = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if "</think>" in output:
            output = output.split("</think>", maxsplit=1)[1].strip()
        return output

    def _json_stopping_criteria(self, prompt_length: int):
        tokenizer = self._tokenizer
        base = self._stopping_criteria_base

        class CompleteJsonObject(base):
            def __call__(self, input_ids, scores, **kwargs):
                generated_text = tokenizer.decode(
                    input_ids[0, prompt_length:],
                    skip_special_tokens=True,
                )
                return HuggingFaceCausalLLM._has_complete_json_object(generated_text)

        return self._stopping_criteria_list([CompleteJsonObject()])

    @staticmethod
    def _has_complete_json_object(text: str) -> bool:
        started = False
        in_string = False
        escaped = False
        depth = 0
        for character in text:
            if not started:
                if character == "{":
                    started = True
                    depth = 1
                continue
            if escaped:
                escaped = False
                continue
            if character == "\\" and in_string:
                escaped = True
                continue
            if character == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False
