# Skill-Enhanced Text-to-Image Pipeline

這是一套整合 **Skill LLM、LoRA Retrieval 與 Qwen-Image 2.1** 的文字生成圖片流程。

主要流程：

```text
User Prompt
→ Skill LLM
→ rewritten_prompt + style_keywords
→ LoRA Retrieval
→ Prompt Builder
→ Qwen-Image 2.1
→ Generated Image
```

---

## 檔案說明

- `skill_t2i.py`：主程式
- `lora_library.json`：LoRA 資料與檢索資訊
- `prompts.txt`：範例 Prompt，可直接修改
- `requirements.txt`：測試成功環境的完整套件版本

---

## 執行環境

測試成功環境：

```text
Windows 11 + WSL2
Ubuntu 24.04
Python 3.12
NVIDIA RTX 5070 Ti 16GB
PyTorch 2.11.0+cu128
```

不需要使用完全相同的硬體環境，只要 NVIDIA Driver、PyTorch 與 CUDA 相容即可。

PyTorch 建議依照自己的 GPU 與系統環境選擇適合的 CUDA 版本。

---

## 必要套件

建議安裝：

```bash
pip install torch torchvision
pip install transformers diffusers accelerate
pip install sentence-transformers
pip install peft safetensors pillow
```

若需要重現測試環境，可參考：

```text
requirements.txt
```

---

## 模型準備

以下模型為目前主程式的**預設設定**，可依需求更換。

### Skill LLM

預設：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

設定位置：

```python
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
```

第一次執行時會由 Hugging Face 自動下載並快取。

用途：

- 改寫原始 Prompt
- 保留原始語意
- 產生 `rewritten_prompt`
- 擷取 `style_keywords`

若要更換其他 LLM，可修改 `LLM_MODEL`，並依新模型需求調整 `SkillLLM`。

---

### Embedding Model

預設：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

設定位置：

```python
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

用途：

- 將 `style_keywords` 與 LoRA 描述轉成 embedding
- 使用 cosine similarity 做 LoRA Retrieval

可更換其他 Embedding Model；若模型介面不同，需同步調整 `LoRARetriever`。

---

### Image Model

預設：

```text
Qwen/Qwen-Image-2.1
```

建議先從 Hugging Face 下載至本機，再修改：

```python
IMAGE_MODEL = "/your/path/Qwen-Image-2.1"
```

目前程式使用：

```python
local_files_only=True
```

因此預設情況下，模型需先存在本機。

若更換其他影像生成模型，主要需調整 `ImageGenerator` 的模型載入、推論參數及 LoRA 載入方式。

---

## 使用方式

### 直接輸入 Prompt

執行：

```bash
python skill_t2i.py
```

程式會讓使用者輸入一筆或多筆 Prompt，輸入空白行後開始執行。

---

### 使用 Prompt File

提供的：

```text
prompts.txt
```

為示範 Prompt，可直接修改。

每一行代表一筆 Prompt。

執行：

```bash
python skill_t2i.py --prompt-file prompts.txt
```

---

## 預設參數

| Parameter | Default |
|---|---:|
| Resolution | 1024 × 1024 |
| Steps | 40 |
| Seed | 42 |
| LoRA Weight | 0.3 |
| Similarity Threshold | 0.45 |
| Top-K LoRA | 2 |
| true_cfg_scale | 1.0 |

---

## 模組分工

### `SkillLLM`

負責：

```text
User Prompt
→ rewritten_prompt
→ style_keywords
```

若要更換 LLM、修改 System Prompt 或 Prompt 改寫規則，主要修改此模組。

---

### `LoRARetriever`

負責：

```text
style_keywords
→ Embedding
→ Cosine Similarity
→ Threshold
→ Top-K
→ Selected LoRA
```

目前預設：

```text
Threshold = 0.45
Top-K = 2
```

若沒有 LoRA 通過 Threshold，則使用 Prompt-only mode。

若要更換 Embedding Model、Similarity 計算方式、Threshold 或 Top-K，主要修改此模組。

---

### `build_final_prompt()`

負責整合：

```text
rewritten_prompt
+ style_keywords
+ LoRA trigger
→ final_prompt
```

若要修改 Prompt 組合方式，修改此函式即可。

---

### `ImageGenerator`

負責：

- 載入 Qwen-Image 2.1
- CPU Offload
- 載入 LoRA
- 設定 LoRA Weight
- 圖片生成
- 圖片儲存

若要更換影像生成模型，主要修改此模組。

注意：不同 Base Model 的 LoRA 通常不能直接共用。

---

## LoRA Library

LoRA 設定存放於：

```text
lora_library.json
```

主要欄位格式：

```json
{
  "name": "Example Style",
  "repo": "example/repository",
  "weight_name": "example.safetensors",
  "trigger": "example_style",
  "description": "Example visual style",
  "keywords": [
    "anime illustration",
    "vibrant colors"
  ],
  "base_model": "Qwen-Image-2.1",
  "weight": 0.3
}
```

若要新增 LoRA，只需新增對應項目至 `lora_library.json`。

---

## 輸出結果

預設輸出：

```text
results/
├── result_01_original.png
├── result_01_enhanced.png
├── result.json
└── result_timing.txt
```

其中：

- `Original Image`：使用原始 Prompt
- `Enhanced Image`：使用改寫後 Prompt、Style Keywords 與選中的 LoRA
- 兩張圖片使用相同 Seed，方便比較

---

## 修改位置快速對照

| 想修改的功能 | 主要修改位置 |
|---|---|
| 更換 Skill LLM | `SkillLLM` |
| 修改 Prompt 改寫規則 | `SkillLLM.analyze()` |
| 更換 Embedding Model | `EMBED_MODEL` / `LoRARetriever` |
| 修改 Threshold / Top-K | `LoRARetriever` |
| 新增或修改 LoRA | `lora_library.json` |
| 修改 Prompt 組合方式 | `build_final_prompt()` |
| 更換 Image Model | `ImageGenerator` |
| 修改輸入方式 | `input_prompts()` / `load_prompts_from_file()` |

整體流程採模組化設計，因此若需要替換 LLM、Embedding Model、LoRA Library 或 Image Generation Model，通常只需修改對應模組即可。
