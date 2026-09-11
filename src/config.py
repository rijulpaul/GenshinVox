from pydantic import BaseModel
from peft import LoraConfig
from typing import Literal

SelectRefStrategy = Literal["random", "good enough", "best"]
WandbMode = Literal["online", "offline", "disabled"]


class TrainingConfig(BaseModel):
    lr: float = 2e-6
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False
    epochs: int
    batch_size: int
    model: str
    model_path: str = "./model"
    tokenizer: str
    tokenizer_path: str = "./tokenizer"
    attn_implementation: str = "sdpa"
    speaker_name: str
    gradient_accumulation_steps: int = 1
    enable_experiment_tracking: bool = False
    model_checkpoint: CheckpointConfig
    training_checkpoint: CheckpointConfig | None = None
    resume_training_path: str | None = None


class DatasetConfig(BaseModel):
    dataset: str
    local_path: str | None = None
    audio_column: str = "audio"
    transcript_column: str = "transcript"
    speaker_column: str = "speaker"
    speaker_name: str | None
    is_processed: bool = False


class WandbConfig(BaseModel):
    """wandb / experiment tracking settings (integrated via accelerate)."""

    project: str = "genshinvox"
    entity: str | None = None
    run_name: str | None = None
    mode: WandbMode = "offline"  # "online" | "offline" | "disabled"
    log_every_n_steps: int = 10


class CheckpointConfig(BaseModel):
    save_every_n_epochs: int = 1
    output_path: str = "./output"
    upload_to_hub: bool = False
    repo_id: str | None = None
    private: bool = False
    revision: str = "main"  # branch name
    path_in_repo: str | None = None  # subfolder inside the repo; supports {epoch}
    commit_message: str = (
        "Upload checkpoint-epoch-{epoch}"  # supports {epoch}, {global_step}
    )
    upload_every_n_epochs: int = 1


class ProcessConfig(BaseModel):
    top_db_to_trim: int = 30
    target_sample_rate: int = 24000
    select_ref_audio_strategy: SelectRefStrategy = "good enough"
    ref_audio_min_duration: float = 5
    ref_audio_max_duration: float = 15


class BaseConfig(BaseModel):
    lora: LoraConfig | None = None
    training: TrainingConfig
    dataset: DatasetConfig
    process: ProcessConfig
    wandb: WandbConfig


__base_config = None


def load_config(config_file: str):
    """Load a config from a .json, .yaml, or .yml file."""
    global __base_config
    raw_config = _read_config_file(config_file)
    __base_config = BaseConfig(**raw_config)
    return __base_config


def _read_config_file(config_file: str):
    """Parse a config file, dispatching on its file extension."""
    from pathlib import Path
    path = Path(config_file)

    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    elif path.suffix.lower() in {".json"}:
        import json
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    else:
        raise ValueError(f"Unsupport format for {config_file} use .yaml, .yml or .json extensions only")


def get_config():
    return __base_config
