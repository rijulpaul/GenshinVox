from qwen_tts import Qwen3TTSTokenizer, Qwen3TTSModel
from huggingface_hub import snapshot_download

from src.config import get_config


def load_tokenizer():
    config = get_config().training
    tokenizer = None

    try:
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            config.tokenizer_path,
        )
        print(f"Successfully loaded tokenizer at {config.tokenizer_path}")

        return tokenizer

    except Exception:
        print(f"Tokenizer not found at {config.tokenizer_path}")
        print(f"Downloading {config.tokenizer} into {config.tokenizer_path}")

        snapshot_download(config.tokenizer, local_dir=config.tokenizer_path)

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        config.tokenizer_path,
    )
    print(f"Successfully loaded tokenizer at {config.tokenizer_path}")

    return tokenizer


def load_model():
    config = get_config().training
    model = None

    try:
        model = Qwen3TTSModel.from_pretrained(
            config.model_path, attn_implementation=config.attn_implementation
        )
        print(f"Successfully loaded model at {config.model_path}")

        return model

    except Exception:
        print(f"Model not found at {config.model_path}")
        print(f"Downloading {config.model} into {config.model_path}")

        snapshot_download(config.model, local_dir=config.model_path)

    model = Qwen3TTSModel.from_pretrained(
        config.model_path, attn_implementation=config.attn_implementation
    )
    print(f"Successfully loaded model at {config.model_path}")

    return model
