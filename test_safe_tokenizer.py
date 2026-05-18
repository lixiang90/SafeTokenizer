from safe_tokenizer import SafeTokenizer
from transformers import AutoTokenizer


def test_safe_tokenizer():
    print("Loading tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )
    safe_tokenizer = SafeTokenizer(base_tokenizer)

    print("\n=== Test 1: Basic encode/decode ===")
    data = [
        {"type": "text", "content": "Hello, world!"},
        {"type": "special", "content": "<|endoftext|>"},
        {"type": "text", "content": "How are you?"},
    ]

    print(f"Input data: {data}")
    token_ids = safe_tokenizer.encode(data)
    print(f"Encoded token ids: {token_ids}")
    decoded = safe_tokenizer.decode(token_ids)
    print(f"Decoded data: {decoded}")

    print("\n=== Test 2: Special token by ID ===")
    data2 = [
        {"type": "text", "content": "Test"},
        {"type": "special", "content": 151643},
    ]
    print(f"Input data: {data2}")
    token_ids2 = safe_tokenizer.encode(data2)
    print(f"Encoded token ids: {token_ids2}")
    decoded2 = safe_tokenizer.decode(token_ids2)
    print(f"Decoded data: {decoded2}")

    print("\n=== Test 3: from_pretrained ===")
    safe_tokenizer2 = SafeTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )
    token_ids3 = safe_tokenizer2.encode(data)
    print(f"Encoded with from_pretrained: {token_ids3}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_safe_tokenizer()
