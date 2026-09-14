from qwen_tts.qwen_tts import Qwen3TTSTokenizer, Qwen3TTSModel
from huggingface_hub import snapshot_download

from src.config import get_config


def load_tokenizer():
    config = get_config().training
    tokenizer = None

    if config.tokenizer_path:
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            config.tokenizer_path,
            device_map="auto",
        )
    else:
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            config.tokenizer,
            device_map="auto",
        )
    print("Tokenizer Loaded")
    return tokenizer


def load_model():
    config = get_config().training
    model = None

    try:
        model = Qwen3TTSModel.from_pretrained(
            config.model_path,
            device_map="auto",
            attn_implementation=config.attn_implementation
        )
        print(f"Loaded model at {config.model_path}")

        return model

    except Exception:
        print(f"Downloading {config.model} into {config.model_path}")

        snapshot_download(config.model, local_dir=config.model_path)

    model = Qwen3TTSModel.from_pretrained(
        config.model_path, attn_implementation=config.attn_implementation
    )
    print(f"Loaded model at {config.model_path}")

    return model
