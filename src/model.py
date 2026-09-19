from qwen_tts.qwen_tts import Qwen3TTSTokenizer, Qwen3TTSModel
from huggingface_hub import snapshot_download

from src.config import get_config
from src.log import info, warning


def load_tokenizer():
    config = get_config().training
    tokenizer = None

    if config.tokenizer_path:
        info(f"Loading tokenizer from local path: {config.tokenizer_path}")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            config.tokenizer_path,
            device_map="auto",
        )
    else:
        info(f"Loading tokenizer from hub: {config.tokenizer}")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            config.tokenizer,
            device_map="auto",
        )
    info("Tokenizer Loaded")
    return tokenizer


def load_model():
    config = get_config().training
    model = None

    try:
        info(f"Loading model from local path: {config.model_path}")
        model = Qwen3TTSModel.from_pretrained(
            config.model_path,
            attn_implementation=config.attn_implementation
        )
        info(f"Loaded model at {config.model_path}")

        return model

    except Exception:
        warning(f"Model not found at {config.model_path}; downloading {config.model} into it")

        snapshot_download(config.model, local_dir=config.model_path)
        info(f"Model downloaded to {config.model_path}")

    info(f"Loading model from local path: {config.model_path}")
    model = Qwen3TTSModel.from_pretrained(
        config.model_path, attn_implementation=config.attn_implementation
    )
    info(f"Loaded model at {config.model_path}")

    return model
