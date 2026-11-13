import os
import re
import json
import gc
import time
import argparse
import math
import inspect
from pathlib import Path

import torch
import transformers
import diffusers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig as TransformersBitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util
from diffusers import AutoPipelineForText2Image, BitsAndBytesConfig as DiffusersBitsAndBytesConfig


# ============================================================
# Configuration
# ============================================================

LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "/data/models/base-models/Qwen-Image-2.1")

EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LORA_LIBRARY = os.environ.get("LORA_LIBRARY", "./lora_library.json")

MAX_LORAS = int(os.environ.get("MAX_LORAS", "2"))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.45"))
DEFAULT_LORA_WEIGHT = float(os.environ.get("DEFAULT_LORA_WEIGHT", "0.3"))
LOCAL_FILES_ONLY = os.environ.get("HF_HUB_OFFLINE", "0").upper() in ("1", "ON", "YES", "TRUE")


def quantization_config(config_type, mode):
    if mode == "none":
        return None
    if mode == "8bit":
        return config_type(load_in_8bit=True)
    if mode == "4bit":
        return config_type(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    raise ValueError("Quantization must be none, 8bit or 4bit")


def configure_gpu_memory(requested):
    if not torch.cuda.is_available():
        raise RuntimeError("This pipeline requires a CUDA GPU")
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    limit = int(total * 0.9) if requested == "auto" else int(requested * 1024**3)
    if limit > total:
        raise ValueError(f"GPU memory budget exceeds visible device memory ({total / 1024**3:.2f} GiB); use auto or a smaller budget")
    torch.cuda.set_per_process_memory_fraction(limit / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    print(f"[Memory] PyTorch allocator limit: {limit / 1024**3:.2f} GiB; CUDA context/external allocations are additional")
    return limit / 1024**3


# ============================================================
# Utility
# ============================================================

def now():
    return time.perf_counter()


def extract_json(text):
    text = text.strip()

    # 1. 先嘗試標準 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. 擷取 ```json ... ``` 或第一個 { 到最後一個 }
    match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.S
    )

    if match:
        candidate = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end <= start:
            raise ValueError(
                f"LLM did not return valid JSON:\n{text}"
            )

        candidate = text[start:end + 1]

    # 3. 先正常解析
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 4. Python JSON strict=False 可容忍字串中的部分控制字元
    try:
        return json.loads(candidate, strict=False)
    except Exception:
        pass

    # 5. 清除 LLM 偶爾產生的裸換行、Tab 與其他控制字元
    cleaned = candidate
    cleaned = cleaned.replace("\r", " ")
    cleaned = cleaned.replace("\n", " ")
    cleaned = cleaned.replace("\t", " ")
    cleaned = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        "",
        cleaned
    )

    try:
        return json.loads(cleaned)
    except Exception as e:
        print("\n[ERROR] Raw LLM output:")
        print(text)
        print("\n[ERROR] Cleaned candidate:")
        print(cleaned)
        raise ValueError(
            f"LLM did not return valid JSON after cleanup: {e}"
        )

def load_lora_library(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Skill LLM
# ============================================================

class SkillLLM:

    def __init__(self, model_name, quantization="none"):
        print(f"[Skill] Loading LLM: {model_name}")

        config = quantization_config(TransformersBitsAndBytesConfig, quantization)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=LOCAL_FILES_ONLY)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=LOCAL_FILES_ONLY,
            dtype=torch.bfloat16,
            quantization_config=config,
            device_map={"": torch.cuda.current_device()} if config is not None else "auto"
        )

#First, translate the user's input into faithful English without adding,
#removing, or changing any meaning. Store this translation as original_prompt.

#Then use original_prompt as the basis for rewriting and enhancement.

#Return ONLY valid JSON in exactly this format:
#{
#  "original_prompt": "...",
#  "rewritten_prompt": "...",
#  "style_keywords": [
#    "...",
#    "..."
# ]
#}

    def analyze(self, user_prompt):
        system_prompt = '''
You are a text-to-image prompt enhancement assistant.

Your goal is to preserve the user's original creative intent while expanding it into a richer, more complete,
and visually descriptive prompt that helps a text-to-image model generate a more accurate and visually coherent image.

Follow these rules carefully:

1. Preserve the user's original meaning and explicit attributes exactly.
   Keep all important subjects, objects, actions, environments, relationships, mood, and requested details.
   Do not change explicitly specified attributes such as colors, numbers, object types, species, locations,
   time of day, or spatial relationships.

2. Do not merely translate or lightly paraphrase the user's prompt.
   The rewritten_prompt must provide more useful visual information than the original prompt.
   It should expand the user's description into a fuller image-generation instruction while preserving the same meaning.

3. Expand the prompt into a fuller image description that directly improves image generation.
   When appropriate, enrich the subject appearance, scene details, spatial relationships,
   foreground and background structure, visual focus, lighting, atmosphere, composition,
   camera perspective, color mood, texture, and level of detail.

4. Added details must strengthen the visual representation of the user's original idea.
   Prefer adding details that help the image model understand:
   - what the main visual focus should be
   - how subjects and objects are positioned
   - how foreground and background are organized
   - how depth and spatial separation should appear
   - how the scene is lit
   - what visual atmosphere should be conveyed
   - what textures or surface qualities are visible
   - how the overall composition should be structured

5. Do not introduce new subjects, objects, actions, locations, events, or story elements that change the meaning of the original prompt. Visual details used only to clarify composition, lighting, depth, texture, spatial relationships, or scene presentation are allowed as long as they do not alter the original content.

6. Do not add details only to make the prompt longer.
   Every added detail should help the text-to-image model better understand or render the requested scene.

7. rewritten_prompt should describe the visual content and scene clearly and in sufficient detail for image generation.
   It should read like a complete image description rather than a translation of the original sentence.

8. Extract visual style concepts separately into style_keywords.
   style_keywords should use normalized English visual terms that are useful for semantic LoRA retrieval.

9. Do not include:
   - LoRA names
   - LoRA trigger words
   - model names
   - repository names
   - file names
   - model IDs

10. Do not invent or force a visual style if the user did not request or clearly imply one.

11. Prefer specific visual style terms over vague descriptive words.

Avoid vague terms such as:
- beautiful
- nice
- cool
- amazing
- artistic

Prefer concrete terms such as:
- hand-drawn illustration
- expressive line art
- visible brush strokes
- watercolor painting
- oil painting
- cel shading
- anime illustration
- realistic photography
- cinematic lighting
- soft pastel colors
- vibrant colorful palette
- dark moody atmosphere

12. Merge synonymous or highly overlapping style descriptions into one clear style keyword.

Example:

User:
"一隻貓坐在東京街頭，我想要手繪感，而且顏色鮮豔"

Good rewritten_prompt:
"A cat sitting on a Tokyo street as the main visual focus, with recognizable urban street elements arranged around it and layered background details creating clear depth. 
The subject is clearly separated from the surrounding environment, with a balanced street-level composition that keeps attention on the cat while preserving the atmosphere of the Tokyo streetscape."

Good style_keywords:
[
  "hand-drawn illustration",
  "vibrant colorful palette"
]

Another example:

User:
"一個女生站在海邊看夕陽，感覺很孤單"

Good rewritten_prompt:
"A young woman standing alone by the seaside and facing the setting sun over the horizon, with her figure clearly separated from the wide coastal background. 
The open space around her and the distant horizon emphasize her visual isolation and reinforce the lonely atmosphere of the scene"

Good style_keywords:
[
  "melancholic atmosphere",
  "warm sunset lighting"
]

Return ONLY valid JSON in exactly this format:

{
  "rewritten_prompt": "...",
  "style_keywords": [
    "...",
    "..."
  ]
}
'''

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        result = extract_json(text)
        result.setdefault("style_keywords", [])

        return result

    def unload(self):
        print("[Skill] Unloading LLM...")

        del self.model
        del self.tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# LoRA Retriever
# ============================================================

class LoRARetriever:

    def __init__(self, library, embedding_model):
        print(f"[Retriever] Loading embedding model: {embedding_model}")

        self.library = None
        self.encoder = SentenceTransformer(embedding_model, local_files_only=LOCAL_FILES_ONLY)
        self.set_library(library)

    def set_library(self, library):
        if self.library == library:
            return

        texts = []

        for item in library:
            text = " ; ".join([
                item.get("name", ""),
                item.get("description", ""),
                " ".join(item.get("keywords", []))
            ])
            texts.append(text)

        embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        self.library = library
        self.library_embeddings = embeddings

    def retrieve(
        self,
        style_keywords,
        base_model,
        top_k=2,
        threshold=0.45
    ):
        if not style_keywords or not self.library:
            return []

        query = " ; ".join(style_keywords)

        query_embedding = self.encoder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        scores = util.cos_sim(
            query_embedding,
            self.library_embeddings
        )[0]

        ranked = sorted(
            enumerate(scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        selected = []

        print("\n[Retriever] Ranking:")

        for index, score in ranked:
            item = self.library[index]

            compatible = (
                item.get("base_model", "").lower()
                == base_model.lower()
            )

            print(
                f"{item['name']:<30}"
                f" score={score:.3f}"
                f" compatible={compatible}"
            )

            if not compatible:
                continue

            if score < threshold:
                continue

            selected_item = dict(item)
            selected_item["similarity"] = round(score, 4)

            selected.append(selected_item)

            if len(selected) >= top_k:
                break

        return selected


# ============================================================
# Prompt Builder
# ============================================================

def build_final_prompt(
    rewritten_prompt,
    style_keywords,
    selected_loras
):
    parts = [rewritten_prompt.strip()]

    for keyword in style_keywords:
        if keyword.strip():
            parts.append(keyword.strip())

    for item in selected_loras:
        trigger = item.get("trigger", "").strip()

        if trigger:
            parts.append(trigger)

    final_parts = []
    seen = set()

    for part in parts:
        key = part.lower()

        if key not in seen:
            seen.add(key)
            final_parts.append(part)

    return ", ".join(final_parts)


# ============================================================
# Image Generator
# ============================================================

class ImageGenerator:

    def __init__(self, model_name, quantization="none", offload_mode="model", vae_tiling=False):
        print(f"\n[Image] Loading base model: {model_name}")
        if offload_mode not in ("none", "model", "sequential"):
            raise ValueError("Offload mode must be none, model or sequential")
        if quantization != "none" and offload_mode == "sequential":
            raise ValueError("Sequential offload requires unquantized models")
        components = {}
        if quantization != "none":
            config = AutoPipelineForText2Image.load_config(model_name, local_files_only=True)
            for name, specification in config.items():
                if name not in ("transformer", "unet") and not name.startswith("text_encoder"):
                    continue
                library, class_name = specification
                if class_name is None:
                    continue
                if library not in ("diffusers", "transformers"):
                    raise ValueError(f"Unsupported model component library: {library}")
                module = diffusers if library == "diffusers" else transformers
                config_type = DiffusersBitsAndBytesConfig if library == "diffusers" else TransformersBitsAndBytesConfig
                components[name] = getattr(module, class_name).from_pretrained(
                    model_name, subfolder=name, dtype=torch.bfloat16,
                    quantization_config=quantization_config(config_type, quantization),
                    device_map={"": torch.cuda.current_device()}, local_files_only=True)
                if offload_mode == "model":
                    components[name].to("cpu")
                    torch.cuda.empty_cache()
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            local_files_only=True,
            **components
        )
        if vae_tiling:
            if not hasattr(getattr(self.pipe, "vae", None), "enable_tiling"):
                raise ValueError("Selected pipeline does not support VAE tiling; set VAE_TILING=0")
            self.pipe.vae.enable_tiling()
        if offload_mode == "model":
            self.pipe.enable_model_cpu_offload()
        elif offload_mode == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.loras_loaded = False
        print(f"[Image] Quantization: {quantization}; offload: {offload_mode}")

    def load_loras(self, loras):
        if not loras:
            print("[LoRA] No LoRA selected -> prompt-only mode")
            return

        adapter_names = []
        adapter_weights = []
        self.loras_loaded = True

        print("\n[LoRA] Loading adapters:")

        for i, item in enumerate(loras):
            adapter_name = f"style_{i}"

            kwargs = {
                "adapter_name": adapter_name,
                "local_files_only": LOCAL_FILES_ONLY
            }

            if item.get("weight_name"):
                kwargs["weight_name"] = item["weight_name"]

            weight = float(
                item.get(
                    "weight",
                    DEFAULT_LORA_WEIGHT
                )
            )

            print(
                f"{adapter_name}"
                f" -> {item['repo']}"
                f" weight={weight}"
            )

            self.pipe.load_lora_weights(
                item["repo"],
                **kwargs
            )

            adapter_names.append(adapter_name)
            adapter_weights.append(weight)

        self.pipe.set_adapters(
            adapter_names,
            adapter_weights=adapter_weights
        )

        print(f"[LoRA] Active adapters: {adapter_names}")

    def generate(
        self,
        prompt,
        output,
        width,
        height,
        steps,
        seed
    ):
        print("\n[Image] Final Prompt:")
        print(prompt)

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(seed)

        guidance = {}
        if "true_cfg_scale" in inspect.signature(self.pipe.__call__).parameters:
            guidance["true_cfg_scale"] = float(os.environ.get("TRUE_CFG_SCALE", "1.0"))

        # inference_mode 比 no_grad 更適合純推論，可避免建立不必要的
        # autograd 狀態，減少額外記憶體使用。
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
                **guidance
            )

        image = result.images[0]
        image.save(output)

        print(f"\n[Done] Image saved: {output}")

    def clear_loras(self):
        if self.loras_loaded:
            self.pipe.unload_lora_weights()
            self.loras_loaded = False


# ============================================================
# Main Pipeline
# ============================================================

class ResidentPipeline:
    def __init__(self, args):
        self.args = args
        self.gpu_limit = configure_gpu_memory(args.gpu_memory_gib)
        self.skill = SkillLLM(args.llm_model, quantization=args.quantization)
        self.retriever = LoRARetriever(load_lora_library(args.lora_library), args.embed_model)
        self.image_generator = ImageGenerator(args.image_model, quantization=args.quantization,
            offload_mode=args.offload_mode, vae_tiling=args.vae_tiling)

    def generate(self, prompt, output_dir):
        args = argparse.Namespace(**(vars(self.args) | {'output_dir': str(output_dir)}))
        run(args, [prompt], self)


def run(args, prompts, pipeline=None):
    total_start = now()
    timing = {}
    pipeline = pipeline or ResidentPipeline(args)
    gpu_limit = pipeline.gpu_limit
    skill = pipeline.skill
    retriever = pipeline.retriever
    image_generator = pipeline.image_generator
    torch.cuda.reset_peak_memory_stats()

    print("=" * 60)
    print("SKILL TEXT-TO-IMAGE PIPELINE")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Stage 1 - Skill LLM
    # --------------------------------------------------------

    skill_results = []

    start = now()

    for i, prompt in enumerate(prompts):
        print()
        print(f"[Skill {i + 1}/{len(prompts)}]")
        print(f"Original Prompt: {prompt}")

        result = skill.analyze(prompt)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

        skill_results.append({
            "original_prompt": prompt,
            "skill_result": result
        })

    timing["skill_batch"] = now() - start

    start = now()

    # --------------------------------------------------------
    # LoRA Retriever
    # --------------------------------------------------------

    library = load_lora_library(args.lora_library)
    lora_base_model = args.lora_base_model
    if lora_base_model == "auto":
        lora_base_model = Path(args.image_model).name

    start = now()

    retriever.set_library(library)

    timing["retriever_refresh"] = now() - start

    all_results = []

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for i, item in enumerate(skill_results):
        print()
        print("=" * 60)
        print(f"[Generate {i + 1}/{len(skill_results)}]")
        print("=" * 60)

        original_prompt = item["original_prompt"]
        skill_result = item["skill_result"]

        #translated_prompt = skill_result["original_prompt"] #初始中文翻譯英文時用
        rewritten_prompt = skill_result["rewritten_prompt"]

        style_keywords = skill_result.get(
            "style_keywords",
            []
        )

        current_seed = args.seed + i

        # ====================================================
        # A. Original Prompt Baseline
        # ====================================================

        baseline_output = str(
            output_dir /
            f"result_{i + 1:02d}_original.png"
        )

        print()
        print("[Baseline Image]")
        #print(f"Prompt: {translated_prompt}") #翻譯時用
        print(f"Prompt: {original_prompt}")
        print(f"Seed: {current_seed}")

        image_generator.clear_loras()

        original_start = now()

        image_generator.generate(
            #translated_prompt, #翻譯英文用
            original_prompt,
            baseline_output,
            args.width,
            args.height,
            args.steps,
            current_seed
        )
        original_generation_time = now() - original_start

        # ====================================================
        # B. Retrieve LoRA
        # ====================================================

        selected_loras = retriever.retrieve(
            style_keywords,
            lora_base_model,
            args.top_k,
            args.threshold
        )

        print()
        print("[Selected LoRAs]")

        if selected_loras:
            for lora in selected_loras:
                print(
                    f"{lora['name']} "
                    f"| score={lora['similarity']} "
                    f"| weight={lora.get('weight', DEFAULT_LORA_WEIGHT)}"
                )
        else:
            print(
                "None -> enhanced image "
                "will use prompt-only mode"
            )

        # ====================================================
        # C. Final Prompt
        # ====================================================

        final_prompt = build_final_prompt(
            rewritten_prompt,
            style_keywords,
            selected_loras
        )

        enhanced_output = str(
            output_dir /
            f"result_{i + 1:02d}_enhanced.png"
        )

        print()
        print("[Enhanced Image]")
        print(f"Rewritten Prompt: {rewritten_prompt}")
        print(f"Style Keywords: {style_keywords}")
        print(f"Final Prompt: {final_prompt}")
        print(f"Seed: {current_seed}")

        image_generator.clear_loras()

        image_generator.load_loras(
            selected_loras
        )

        enhanced_start = now()

        image_generator.generate(
            final_prompt,
            enhanced_output,
            args.width,
            args.height,
            args.steps,
            current_seed
        )
        enhanced_generation_time = now() - enhanced_start

        all_results.append({
            "index": i + 1,
            #"translated_prompt": translated_prompt, #翻譯用
            "original_prompt": original_prompt,
            "rewritten_prompt": rewritten_prompt,
            "style_keywords": style_keywords,
            "selected_loras": selected_loras,
            "final_prompt": final_prompt,
            "original_output": baseline_output,
            "enhanced_output": enhanced_output,
            "seed": current_seed,
            "original_generation_time": original_generation_time,
            "enhanced_generation_time": enhanced_generation_time
        })

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    timing["total"] = now() - total_start

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)

    for name, seconds in timing.items():
        print(
            f"{name:<25}: "
            f"{seconds:.3f} sec"
        )

    # --------------------------------------------------------
    # Save Metadata
    # --------------------------------------------------------

    metadata = {
        "results": all_results,
        "runtime": {
            "quantization": args.quantization,
            "models_resident": True,
            "offload_mode": args.offload_mode,
            "vae_tiling": args.vae_tiling,
            "requested_gpu_memory_gib": args.gpu_memory_gib,
            "allocator_limit_gib": gpu_limit,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        },
        "generation": {
            "base_model": args.image_model,
            "pipeline_class": type(image_generator.pipe).__name__,
            "lora_base_model": lora_base_model,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "base_seed": args.seed,
            "top_k": args.top_k,
            "threshold": args.threshold
        },
        "timing": timing
    }

    metadata_path = str(
        output_dir / "result.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    timing_path = str(
        output_dir / "result_timing.txt"
    )

    with open(
        timing_path,
        "w",
        encoding="utf-8"
    ) as f:
        for name, seconds in timing.items():
            f.write(
                f"{name}: "
                f"{seconds:.3f} sec\n"
            )
    print()
    print(f"[Done] Metadata: {metadata_path}")
    print(f"[Done] Timing: {timing_path}")


# ============================================================
# Prompt Input
# ============================================================

def input_prompts():
    prompts = []

    print("請輸入 Prompt，每行一筆")
    print("直接輸入空白行後開始執行")

    while True:
        prompt = input(
            f"{len(prompts) + 1}> "
        ).strip()

        if not prompt:
            break

        prompts.append(prompt)

    return prompts


def load_prompts_from_file(path):
    prompts = []

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:
        for line in f:
            prompt = line.strip()

            if prompt:
                prompts.append(prompt)

    print(
        f"[Prompt File] Loaded "
        f"{len(prompts)} prompts from: {path}"
    )

    return prompts


# ============================================================
# Arguments
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    def gpu_memory(value):
        if value == "auto":
            return value
        try:
            amount = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError("GPU memory must be auto or a finite positive GiB value") from None
        if not math.isfinite(amount) or amount <= 0:
            raise argparse.ArgumentTypeError("GPU memory must be auto or a finite positive GiB value")
        return amount

    parser.add_argument("--quantization", choices=("none", "8bit", "4bit"),
                        default=os.environ.get("QUANTIZATION", "4bit"))
    parser.add_argument("--offload-mode", choices=("none", "model", "sequential"),
                        default=os.environ.get("OFFLOAD_MODE", "none"))
    parser.add_argument("--gpu-memory-gib", type=gpu_memory,
                        default=os.environ.get("GPU_MEMORY_GIB", "auto"),
                        help="PyTorch allocator budget in GiB; auto uses 90%% of visible GPU memory")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction,
                        default=os.environ.get("VAE_TILING", "1"),
                        help="Decode VAE in tiles to reduce large-image memory peaks")
    parser.add_argument("--lora-base-model", default=os.environ.get("LORA_BASE_MODEL", "auto"),
                        help="LoRA compatibility key; auto uses the image model directory name")

    parser.add_argument(
        "--output-dir",
        default="results"
    )

    parser.add_argument(
        "--prompt-file",
        default=None
    )

    parser.add_argument(
        "--width",
        type=int,
        default=int(os.environ.get("WIDTH", "1024"))
    )

    parser.add_argument(
        "--height",
        type=int,
        default=int(os.environ.get("HEIGHT", "1024"))
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("STEPS", "40"))
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", "42"))
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=MAX_LORAS
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD
    )

    parser.add_argument(
        "--llm-model",
        default=LLM_MODEL
    )

    parser.add_argument(
        "--image-model",
        default=IMAGE_MODEL
    )

    parser.add_argument(
        "--embed-model",
        default=EMBED_MODEL
    )

    parser.add_argument(
        "--lora-library",
        default=LORA_LIBRARY
    )

    args = parser.parse_args(argv)
    if isinstance(args.vae_tiling, str):
        if args.vae_tiling not in ("0", "1"):
            parser.error("VAE_TILING must be 0 or 1")
        args.vae_tiling = args.vae_tiling == "1"
    if args.quantization not in ("none", "8bit", "4bit"):
        parser.error("QUANTIZATION must be none, 8bit or 4bit")
    if args.offload_mode not in ("none", "model", "sequential"):
        parser.error("OFFLOAD_MODE must be none, model or sequential")
    if args.quantization != "none" and args.offload_mode == "sequential":
        parser.error("Sequential offload requires QUANTIZATION=none; quantized models support model or none")
    return args


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    if args.prompt_file:
        prompts = load_prompts_from_file(
            args.prompt_file
        )
    else:
        prompts = input_prompts()

    if not prompts:
        print("沒有輸入任何 Prompt")
        exit()

    run(
        args,
        prompts
    )
