from typing import List, Dict, Union
from transformers import AutoTokenizer


class SafeTokenizer:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self._build_special_tokens_set()
        self._check_split_special_tokens_support()

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

    def _check_split_special_tokens_support(self):
        """检测底层 tokenizer 是否支持 split_special_tokens 参数。"""
        try:
            self.tokenizer.encode("", add_special_tokens=False, split_special_tokens=True)
            self._split_special_tokens = True
        except TypeError:
            self._split_special_tokens = False

    def _find_special_token_ranges(self, content: str) -> List[tuple]:
        """找出文本中所有多字符特殊 token 的出现区间，返回已排序且合并后的区间列表。"""
        intervals = []
        for token in self.special_tokens:
            if len(token) <= 1:
                continue
            start = 0
            while True:
                idx = content.find(token, start)
                if idx == -1:
                    break
                intervals.append((idx, idx + len(token)))
                start = idx + 1
        if not intervals:
            return []
        intervals.sort()
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _encode_text_safely(self, content: str) -> List[int]:
        """安全地编码普通文本，确保文本中的特殊 token 字符串不会被解析为特殊 token ID。

        如果底层 tokenizer 支持 split_special_tokens，则优先使用该参数来拆分 added tokens。
        否则，仅对文本中实际包含的特殊 token 区间逐字符编码，其余部分仍使用正常编码。
        """
        if self._split_special_tokens:
            return self.tokenizer.encode(
                content, add_special_tokens=False, split_special_tokens=True
            )

        # Fallback：定位文本中的特殊 token 区间，仅对这些区间逐字符编码
        ranges = self._find_special_token_ranges(content)
        if not ranges:
            return self.tokenizer.encode(content, add_special_tokens=False)

        token_ids = []
        prev_end = 0
        for start, end in ranges:
            if start > prev_end:
                token_ids.extend(
                    self.tokenizer.encode(content[prev_end:start], add_special_tokens=False)
                )
            for char in content[start:end]:
                token_ids.extend(self.tokenizer.encode(char, add_special_tokens=False))
            prev_end = end
        if prev_end < len(content):
            token_ids.extend(
                self.tokenizer.encode(content[prev_end:], add_special_tokens=False)
            )
        return token_ids

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
                token_ids = self._encode_text_safely(content)
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
