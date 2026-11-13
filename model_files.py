import json
import os
import struct
from pathlib import Path


def inspect_models():
    missing, revision = [], []

    def check(path, model, kind=None):
        path = Path(path)
        try:
            stat = path.stat()
            revision.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino))
            if stat.st_size == 0:
                raise ValueError('檔案尚未完整')
            if kind == 'json':
                data = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(data, (dict, list)):
                    raise ValueError('模型設定格式無效')
                return data
            if path.suffix == '.safetensors':
                with path.open('rb') as source:
                    length = struct.unpack('<Q', source.read(8))[0]
                    if length > 100_000_000:
                        raise ValueError('權重標頭無效')
                    header = json.loads(source.read(length))
                end = max(value['data_offsets'][1] for key, value in header.items() if key != '__metadata__')
                if 8 + length + end != stat.st_size:
                    raise ValueError('權重尚未上傳完整')
        except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
            reason = '找不到檔案' if isinstance(error, FileNotFoundError) else '檔案未完成或格式無效'
            missing.append({'model': model, 'path': str(path), 'reason': reason})
        return None

    def weights(folder, model):
        indexes = sorted(folder.glob('*.index.json'))
        files = set()
        for index in indexes:
            data = check(index, model, 'json')
            if isinstance(data, dict):
                files.update(folder / name for name in data.get('weight_map', {}).values())
        if not files:
            files.update(folder.glob('*.safetensors'))
            files.update(folder.glob('*.bin'))
        if not files:
            missing.append({'model': model, 'path': str(folder), 'reason': '找不到模型權重'})
        for path in sorted(files):
            check(path, model)

    image = Path(os.environ.get('IMAGE_MODEL', '/data/models/base-models/Qwen-Image-2.1'))
    llm = os.environ.get('LLM_MODEL', '/data/models/llm/Qwen2.5-1.5B-Instruct')
    embed = os.environ.get('EMBED_MODEL', '/data/models/embedding/paraphrase-multilingual-MiniLM-L12-v2')
    offline = os.environ.get('HF_HUB_OFFLINE', '1').upper() in ('1', 'ON', 'YES', 'TRUE')

    def remote(name):
        return not offline and not name.startswith(('/', '.', '~')) and not Path(name).is_dir()

    for name, model in ((llm, '提示詞改寫模型'), (embed, 'LoRA 比對模型')):
        if remote(name):
            revision.append((name, 'online'))
            continue
        folder = Path(name)
        check(folder / 'config.json', model, 'json')
        weights(folder, model)
    if not remote(llm):
        check(Path(llm) / 'tokenizer_config.json', '提示詞改寫模型', 'json')
    if not remote(embed):
        check(Path(embed) / 'modules.json', 'LoRA 比對模型', 'json')
    config = check(image / 'model_index.json', '影像生成模型', 'json')
    if isinstance(config, dict):
        for name, specification in config.items():
            if name.startswith('_') or not isinstance(specification, list) or len(specification) != 2 or specification[1] is None:
                continue
            folder = image / name
            if name in ('transformer', 'unet', 'vae') or name.startswith('text_encoder'):
                check(folder / 'config.json', '影像生成模型', 'json')
                weights(folder, '影像生成模型')
            elif name == 'scheduler':
                check(folder / 'scheduler_config.json', '影像生成模型', 'json')
            elif name.startswith('tokenizer'):
                check(folder / 'tokenizer_config.json', '影像生成模型', 'json')
            elif name == 'processor':
                check(folder / 'preprocessor_config.json', '影像生成模型', 'json')
    catalog = check(Path(os.environ.get('LORA_LIBRARY', '/app/lora_library.json')), 'LoRA 清單', 'json')
    family = os.environ.get('LORA_BASE_MODEL', 'auto')
    family = image.name if family == 'auto' else family
    if isinstance(catalog, list):
        for item in catalog:
            if item.get('base_model', '').lower() == family.lower() and item.get('weight_name') and not remote(item['repo']):
                check(Path(item['repo']) / item['weight_name'], 'LoRA：' + item['name'])
    return missing, tuple(revision)
