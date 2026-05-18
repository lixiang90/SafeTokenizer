# SafeTokenizer

一个安全的 Hugging Face Tokenizer 包装器，强制**严格分离普通文本与特殊控制 Token**，从根本上避免用户输入与模型指令之间的歧义和注入风险。

---

## 问题背景：为什么需要 SafeTokenizer？

### DeepSeek `<think>` 标签事件的启示

2026 年，DeepSeek 模型因其在回复中使用 `<think>...</think>` 标签展示思维链（Chain-of-Thought）而受到广泛关注。然而，这也暴露了一个严重的安全隐患：当用户输入中包含与特殊控制标记相似的字符串时，如果系统**未能严格区分"普通文本"和"控制指令"**，就会引发不可预测的行为。

具体表现包括：
- **用户输入注入**：攻击者故意在输入中插入 `<think>` 或 `<think`（未闭合）等字符串，模型可能将其误认为是系统内部标记，导致输出混乱、泄露其他会话信息，或被诱导输出本不应展示的内部推理过程。
- **输出解析混乱**：下游应用在解析模型输出时，如果用户生成的内容中恰好包含与特殊标记相同的字符串，应用可能错误地将这部分用户文本当作系统标记处理，导致界面渲染异常或逻辑错误。
- **信息泄露风险**：暴露的 `<think>` 思维链被证实可显著增加提示注入攻击的成功率（Trend Micro 研究）。攻击者可以通过观察思维链来发现模型的防护规则漏洞，进而绕过安全限制。

**根本原因在于**：传统 Tokenizer 的 `encode(text)` 接口将整个输入视为一个普通字符串，无论其中是否包含特殊 Token。这导致用户输入中的字符串与系统特殊标记之间没有明确的边界。

---

## 核心设计：显式类型声明

`SafeTokenizer` 抛弃了纯字符串输入，要求调用者通过**结构化数据**显式声明每一段内容的类型：

| 类型 | 含义 | 示例 |
|------|------|------|
| `text` | 普通用户文本，将被当作纯文本编码 | `{"type": "text", "content": "Hello, world!"}` |
| `special` | 特殊控制 Token，直接插入对应的 Token ID | `{"type": "special", "content": "<\|endoftext\|>"}` |

这种设计带来了以下关键优势：

### 1. 彻底消除注入歧义
用户输入的内容**必须**被标记为 `type: "text"`。在编码时，`SafeTokenizer` 会主动屏蔽特殊 Token 的识别：

- **优先策略**：如果底层 Tokenizer 支持 `split_special_tokens` 参数，编码时会强制启用该参数，使文本中的特殊 Token 字符串被拆分为普通子词（Subword），而不是被识别为独立的特殊 Token ID。
- **兼容回退**：对于不支持该参数的旧版 Tokenizer，会**智能定位文本中出现的特殊 Token 字符串**，仅对这些局部区间逐字符编码，其余文本仍使用正常的子词编码。这既保证了安全性，又最大限度保留了 Tokenization 效率。

这意味着即使文本中包含 `<think>`、`<|im_start|>`、`<|endoftext|>` 等字符串，也只会被当作普通字符序列处理，**绝不会被解析为具有控制意义的特殊 Token**。

### 2. 精确控制特殊标记的插入
系统提示、消息分隔符、结束标记等特殊 Token **必须**被显式声明为 `type: "special"`。这确保了只有开发者明确意图插入的控制标记才会进入 Token 序列，避免了因字符串拼接失误导致的安全漏洞。

### 3. 解码时可追溯、可过滤
`decode()` 接口同样返回结构化数据，精确区分输出中的普通文本和特殊 Token。这使得下游应用可以：
- **安全地过滤内部思维链**：轻松识别并移除 `<think>` 等特殊标记及其内容，防止内部推理泄露给用户。
- **准确渲染输出**：不会因为用户文本中恰好包含与特殊标记相同的字符串而错误解析。

---

## 安装

```bash
pip install transformers
```

本项目仅依赖 `transformers` 库。

---

## 使用方法

### 基本编码与解码

```python
from safe_tokenizer import SafeTokenizer
from transformers import AutoTokenizer

# 包装现有的 Tokenizer
base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
safe_tokenizer = SafeTokenizer(base_tokenizer)

# 构建结构化输入：严格区分文本和特殊 Token
data = [
    {"type": "text", "content": "Hello, world!"},
    {"type": "special", "content": "<|endoftext|>"},
    {"type": "text", "content": "How are you?"},
]

# 编码
token_ids = safe_tokenizer.encode(data)
print(token_ids)

# 解码（返回结构化数据）
decoded = safe_tokenizer.decode(token_ids)
print(decoded)
# 输出:
# [
#   {"type": "text", "content": "Hello, world!"},
#   {"type": "special", "content": "<|endoftext|>"},
#   {"type": "text", "content": "How are you?"}
# ]
```

### 防止用户输入注入特殊标记

假设一个聊天应用需要构建如下对话结构：

```python
user_input = "用户输入了 <think"  # 恶意或误输入的未闭合标记

messages = [
    {"type": "text", "content": "User: "},
    {"type": "text", "content": user_input},  # 始终被视为纯文本！
    {"type": "special", "content": "<|im_end|>"},
    {"type": "text", "content": "\nAssistant: "},
]

token_ids = safe_tokenizer.encode(messages)
```

在这个例子中，即使 `user_input` 包含 `<think`，它也会被当作普通字符进行文本编码，**不会**被模型误解为思维链的开始标记。特殊标记 `<|im_end|>` 则是通过 `special` 类型精确插入的。

### 通过 Token ID 插入特殊标记

```python
data = [
    {"type": "text", "content": "Test"},
    {"type": "special", "content": 151643},  # 使用已知的 Token ID
]
token_ids = safe_tokenizer.encode(data)
```

### 使用 from_pretrained 快捷创建

```python
safe_tokenizer = SafeTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
```

### 访问底层 Tokenizer 属性

`SafeTokenizer` 会自动代理对底层 `AutoTokenizer` 的属性访问：

```python
vocab_size = safe_tokenizer.vocab_size
model_max_length = safe_tokenizer.model_max_length
```

---

## API 参考

### `SafeTokenizer(tokenizer: AutoTokenizer)`

包装一个现有的 Hugging Face Tokenizer。

#### `encode(data: List[Dict[str, Union[str, int]]]) -> List[int]`

将结构化数据编码为 Token ID 列表。

- **`text` 类型**：`content` 必须为字符串。编码时会强制屏蔽特殊 Token 识别（优先使用 `split_special_tokens=True`，不支持则逐字符编码），确保内容中的特殊 Token-like 字符串不会被解析为特殊 Token ID。
- **`special` 类型**：`content` 可为字符串（特殊 Token 名称）或整数（Token ID）。直接插入对应的 Token ID。

#### `decode(token_ids: List[int]) -> List[Dict[str, Union[str, int]]]`

将 Token ID 列表解码为结构化数据。

- 连续的非特殊 Token 会被合并为一个 `text` 条目。
- 特殊 Token 会被识别为独立的 `special` 条目。

#### `from_pretrained(pretrained_model_name_or_path, *args, **kwargs) -> SafeTokenizer`

类方法，直接从模型名称或路径创建 `SafeTokenizer`。

---

## 安全最佳实践

1. **永远将用户输入标记为 `text`**：无论用户输入什么内容，包含何种特殊字符或标记-like 字符串，都应使用 `{"type": "text", "content": user_input}`。
2. **永远将系统标记标记为 `special`**：对话模板中的分隔符、结束标记、系统提示边界等，都应显式声明为 `special`。
3. **解码后过滤敏感标记**：在将模型输出生成展示给用户之前，使用 `decode()` 的输出结构来识别并安全移除 `<think>`、`<|im_start|>` 等内部标记。
4. **不信任纯字符串拼接**：避免使用 f-string 或字符串拼接来组装包含特殊标记的提示。使用结构化输入替代。

---

## 与 DeepSeek 事件的关联

| 风险场景 | 传统字符串拼接 | SafeTokenizer 方案 |
|---------|-------------|------------------|
| 用户输入 `<think` | 可能因 Tokenizer 解析规则被误处理，导致模型进入不可控的思维链模式 | 强制声明为 `text` 类型，始终作为普通字符编码，无歧义 |
| 系统标记被用户文本模仿 | 字符串匹配可能误将用户文本识别为系统标记 | `special` 类型直接操作 Token ID，与文本编码路径完全隔离 |
| 输出中包含 `<think>` 泄露 | 纯字符串解析难以区分"用户碰巧输入了 `<think>`"和"模型的思维链" | `decode()` 返回结构化数据，可精确识别并过滤所有特殊标记 |

---

## License

MIT
