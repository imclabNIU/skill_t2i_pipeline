# Skill-Enhanced Text-to-Image Pipeline

## 專案介紹

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

提供網頁 UI、API 與 FIFO queue，由 coordinator 接收請求，GPU worker 依序執行；worker 忙碌時，新請求會保留排隊。

worker 啟動後載入模型並持續重用；預設不在生成完成後卸載 GPU 模型。對話式介面保留描述、送出時間與圖片歷史，重新整理後會從輸出儲存載入。

## Docker 用法

需要 Linux、NVIDIA GPU、相容驅動、Docker Compose 與 NVIDIA Container Toolkit。將下載好的模型放入本目錄的 `models/`，也可先啟動服務再放入模型：

```text
models/
├── base-models/Qwen-Image-2.1/
├── llm/Qwen2.5-1.5B-Instruct/
├── embedding/paraphrase-multilingual-MiniLM-L12-v2/
└── loras/
    ├── mikkoani/qi21-mikkoani.safetensors
    ├── genshin-2d/genshintcg_style_st3500.safetensors
    ├── zenless-zone-zero/zzz_style_v1.6_3500.safetensors
    ├── mikkoph/qi21-mikkoph.safetensors
    └── filmstills/filmstills_qwen21.safetensors
```

在本目錄執行；已有 `.env` 時直接編輯：

```bash
cp .env.example .env
docker compose -f compose.yaml -f compose.standalone.yaml config --quiet
docker compose -f compose.yaml -f compose.standalone.yaml up -d --build
```

開啟 <http://localhost:7860/>。模型目錄唯讀掛載至 `/data/models`，輸出與快取保存於 named volumes；預設離線載入，不重新下載模型。

缺少模型時，介面會列出模型名稱與缺少的檔案，描述可先送出排隊；模型檔案完整後會自動開始載入。更換已載入的基礎模型或量化設定須重新啟動 worker；外部 LoRA 清單仍可原子更新。

設定說明見 [.env.example](.env.example)。可調整模型路徑、LoRA 清單、量化、顯存預算與圖片參數；宿主模型目錄不同時設定 `SKILL_MODELS_DIR`。預設使用 `4bit`、`OFFLOAD_MODE=none`、`GPU_MEMORY_GIB=auto` 與 VAE tiling；顯存預算限制的是 PyTorch allocator，實際需求依模型與解析度而異。

查看日誌及停止服務：

```bash
docker compose -f compose.yaml -f compose.standalone.yaml logs -f worker
docker compose -f compose.yaml -f compose.standalone.yaml down
```

停止前先等待工作完成；保留 volumes 可在下次啟動時沿用輸出及佇列。此啟動方式沒有登入機制，請在受控網路使用。

## 本機直接執行

需要 Linux、Python 3.12 與支援 CUDA 12.8 的 NVIDIA 驅動／GPU。沿用上述 `models/` 目錄，在本目錄建立虛擬環境並安裝依賴：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

內附 LoRA 清單使用容器路徑。先產生使用本機絕對路徑的清單：

```bash
python - <<'PY'
import json
from pathlib import Path

models = Path('models').resolve()
library = json.loads(Path('lora_library.json').read_text())
for item in library:
    item['repo'] = str(models / 'loras' / Path(item['repo']).name)
Path('lora_library.local.json').write_text(json.dumps(library, ensure_ascii=False, indent=2))
PY
```

在 `prompts.txt` 每行填入一筆提示詞，執行：

```bash
HF_HUB_OFFLINE=1 python skill_t2i.py \
  --prompt-file prompts.txt \
  --output-dir results \
  --llm-model ./models/llm/Qwen2.5-1.5B-Instruct \
  --embed-model ./models/embedding/paraphrase-multilingual-MiniLM-L12-v2 \
  --image-model ./models/base-models/Qwen-Image-2.1 \
  --lora-library ./lora_library.local.json \
  --quantization 4bit \
  --offload-mode none \
  --gpu-memory-gib auto \
  --vae-tiling
```

圖片與 `result.json` 會寫入 `results/`。可用 `--width`、`--height`、`--steps` 與 `--seed` 調整生圖；其他參數執行 `python skill_t2i.py --help` 查看。換模型時調整模型路徑與相容的 LoRA 清單，影像模型須為含 `model_index.json` 的完整 Diffusers 目錄。
