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
    model_path: str = "./model"     # Required during model checkpoint setup
    tokenizer: str
    tokenizer_path: str = ""
    attn_implementation: str = "flash-attn"
    speaker_name: str       # The speaker_name to set in the model
    gradient_accumulation_steps: int = 1
    enable_experiment_tracking: bool = False    # Use WandB to track experiment
    model_checkpoint: CheckpointConfig          # Model Checkpoint save inference ready model with modified state dict and differ from a Training Checkpoint
    training_checkpoint: CheckpointConfig | None = None     # To enable training resumability
    resume_training_path: str | None = None


class DatasetConfig(BaseModel):
    dataset: str
    subset: str | None = None
    train_split: str = "train"
    test_split: str | None = None   # If left empty and TrainingConfig is not None, test split will be generated from original dataset
    test_size: int = 0              # Works only if test_split is None, amount of audio samples to test with, supports fractions and numbers, test_size = 0 skips testing
    local_path: str | None = None   # Loads all the parquets in the specified directory (!!!wont read sub-directories)
    audio_column: str = "audio"
    transcript_column: str = "transcript"
    speaker_column: str | None  # Specify if you want to filter based on a speaker/column
    speaker_name: str | None    # To keep rows with the specified speaker/value. Leave empty to use the entire dataset
    is_processed: bool = False  # if True, skips preprocessing


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
    path_in_repo: str | None = None  # subfolder inside the repo; supports {epoch}, {global_step}
    commit_message: str = "Upload checkpoint-epoch-{epoch}"  # supports {epoch}, {global_step}
    upload_every_n_epochs: int = 1


class ProcessConfig(BaseModel):
    top_db_to_trim: int = 30
    target_sample_rate: int = 24000
    select_ref_audio_strategy: SelectRefStrategy = "good enough"
        # random: selects random audio as ref
        # good enough: first audio to fit between the range provided below
        # best: selects longest possible audio as reference
    ref_audio_min_duration: float = 5
    ref_audio_max_duration: float = 15

class TestingConfig(BaseModel):     # If None, testing is skipped
    word_error_rate: bool = True
    speaker_similarity: bool = True
    save_samples: bool = True
    output_path: str = "./output/{epoch}" # supports {epoch}, {global_step}, to push_to_hub, output_path should be inside the model output_path


class BaseConfig(BaseModel):
    lora: LoraConfig | None = None          # If None, full finetuning is performed
    training: TrainingConfig
    dataset: DatasetConfig
    process: ProcessConfig
    wandb: WandbConfig
    testing: TestingConfig | None = None    # If not None, will run everytime the model checkpoint is saved, otherwise skips testing


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
