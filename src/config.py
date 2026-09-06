from pydantic import BaseModel
from peft import LoraConfig
from typing import Literal
import json

SelectRefStrategy = Literal["random", "good enough", "best possible"]


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
    output_path: str = "./output"
    gradient_accumulation_steps: int = 1


class DatasetConfig(BaseModel):
    dataset: str
    local_path: str | None = None
    audio_column: str = "audio"
    transcript_column: str = "transcript"
    speaker_column: str = "speaker"
    speaker_name: str | None


class ProcessConfig(BaseModel):
    top_db_to_trim: int = 30
    target_sample_rate: int = 24000
    select_ref_audio_strategy: SelectRefStrategy = "random"
    ref_audio_min_duration: float = 5
    ref_audio_max_duration: float = 15


class BaseConfig(BaseModel):
    lora: LoraConfig
    training: TrainingConfig
    dataset: DatasetConfig
    process: ProcessConfig


__base_config = None


def load_config(config_file: str):
    global __base_config
    with open(config_file, "r") as file:
        config = json.load(file)
    __base_config = BaseConfig(**config)
    return __base_config


def get_config():
    return __base_config
