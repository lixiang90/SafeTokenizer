from typing import List, Dict, Union
from transformers import AutoTokenizer


class SafeTokenizer:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self._build_special_tokens_set()

    def _build_special_tokens_set(self):
        self.special_tokens = set()
        self.special_token_to_id = {}
        self.special_id_to_token = {}

        if hasattr(self.tokenizer, "added_tokens_encoder"):
            for token, token_id in self.tokenizer.added_tokens_encoder.items():
                self.special_tokens.add(token)
                self.special_token_to_id[token] = token_id
                self.special_id_to_token[token_id] = token

        special_tokens_map = getattr(self.tokenizer, "special_tokens_map", {}) or {}
        for key, token in special_tokens_map.items():
            if isinstance(token, str):
                self.special_tokens.add(token)
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                self.special_token_to_id[token] = token_id
                self.special_id_to_token[token_id] = token
            elif isinstance(token, list):
                for t in token:
                    self.special_tokens.add(t)
                    token_id = self.tokenizer.convert_tokens_to_ids(t)
                    self.special_token_to_id[t] = token_id
                    self.special_id_to_token[token_id] = t

    def encode(self, data: List[Dict[str, Union[str, int]]]) -> List[int]:
        all_token_ids = []

        for item in data:
            item_type = item.get("type")
            content = item.get("content")

            if item_type == "text":
                if not isinstance(content, str):
                    raise ValueError(
                        f"For 'text' type, content must be a string, got {type(content)}"
                    )
                token_ids = self.tokenizer.encode(content, add_special_tokens=False)
                all_token_ids.extend(token_ids)

            elif item_type == "special":
                if isinstance(content, str):
                    token_id = self.tokenizer.convert_tokens_to_ids(content)
                    if token_id is None:
                        raise ValueError(f"Unknown special token: {content}")
                    all_token_ids.append(token_id)
                elif isinstance(content, int):
                    all_token_ids.append(content)
                else:
                    raise ValueError(
                        f"For 'special' type, content must be a string or int, got {type(content)}"
                    )

            else:
                raise ValueError(
                    f"Unknown type: {item_type}. Must be 'text' or 'special'"
                )

        return all_token_ids

    def decode(self, token_ids: List[int]) -> List[Dict[str, Union[str, int]]]:
        if not token_ids:
            return []

        result = []
        current_text_tokens = []

        for token_id in token_ids:
            if token_id in self.special_id_to_token:
                if current_text_tokens:
                    text = self.tokenizer.decode(current_text_tokens)
                    result.append({"type": "text", "content": text})
                    current_text_tokens = []

                special_token = self.special_id_to_token[token_id]
                result.append({"type": "special", "content": special_token})
            else:
                current_text_tokens.append(token_id)

        if current_text_tokens:
            text = self.tokenizer.decode(current_text_tokens)
            result.append({"type": "text", "content": text})

        return result

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        return cls(tokenizer)
