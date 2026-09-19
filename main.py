import argparse
import dotenv
import torch

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from datasets import load_from_disk
from transformers import AutoConfig

from src.config import load_config
from src.finetune import finetune
from src.peft import get_peft_model
from src.model import load_model, load_tokenizer
from src.dataset import load_dataset, TTSDataset
from src.preprocess import preprocess
from src.extract_feature import extract_feature
from src.log import info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Load config from a JSON or YAML file."
    )

    args = parser.parse_args()
    info(f"Received config file: {args.config}")

    dotenv.load_dotenv()
    info("Loaded .env environment variables")

    config = load_config(args.config)
    info(
        f"Configuration loaded: model={config.training.model}, "
        f"epochs={config.training.epochs}, batch_size={config.training.batch_size}"
    )

    deepspeed_plugin = None
    if config.training.deepspeed_zero_stage:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=config.training.deepspeed_zero_stage,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            gradient_clipping=1.0,
        )
        info(
            f"DeepSpeed enabled (ZeRO stage {config.training.deepspeed_zero_stage}, "
            f"grad_accum={config.training.gradient_accumulation_steps})"
        )
    else:
        info("DeepSpeed disabled (deepspeed_zero_stage is None)")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
        mixed_precision="bf16",
        log_with="wandb",
    )
    info(
        f"Accelerator initialized: device={accelerator.device}, "
        f"num_processes={accelerator.num_processes}, bf16={accelerator.mixed_precision}"
    )

    train_dataset, test_dataset = load_dataset()
    info(f"Loaded dataset: {len(train_dataset)} train samples")
    if test_dataset is not None:
        info(f"Loaded dataset: {len(test_dataset)} test samples")
    else:
        info("No test dataset configured (test_size=0 / no test_split)")

    # Make sure the model/cache exists before every process tries to load it
    # Prevent ChildFailedError caused by multiple parallel model download
    if accelerator.is_main_process:
        info("Main process loading model (ensuring model cache is populated)...")
        model = load_model()

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        info("Worker processes loading model from local cache...")
        model = load_model()

    info("All processes have the model loaded")

    if accelerator.is_main_process:
        if not config.dataset.is_processed:
            info(
                f"Preprocessing train dataset ({len(train_dataset)} samples)..."
            )
            train_dataset = preprocess(train_dataset)
            info(f"Preprocessing complete: {len(train_dataset)} samples remain")

        info("Loading tokenizer...")
        tokenizer = load_tokenizer()
        info("Tokenizer loaded")

        info("Extracting features (audio codes + reference mel)...")
        train_dataset, ref_mel = extract_feature(
            dataset=train_dataset, tokenizer=tokenizer, processor=model.processor
        )
        info(
            f"Feature extraction complete: {len(train_dataset)} samples, "
            f"reference mel shape={tuple(ref_mel.shape)}"
        )

        info("Saving processed dataset to disk (data/train, data/ref_mel)...")
        train_dataset.save_to_disk('data/train')
        torch.save(ref_mel,'data/ref_mel')
        info("Processed dataset saved to disk")

        del tokenizer

    accelerator.wait_for_everyone()

    info("Loading processed dataset from disk...")
    train_dataset = load_from_disk('data/train')
    ref_mel = torch.load('data/ref_mel', map_location="cpu")
    info(
        f"Loaded processed dataset: {len(train_dataset)} samples, "
        f"ref_mel={tuple(ref_mel.shape)}"
    )

    model_config = AutoConfig.from_pretrained(config.training.model)
    train_dataset = TTSDataset(train_dataset, model.processor, ref_mel, model_config)
    info(f"Built TTSDataset with {len(train_dataset)} samples")

    if config.lora:
        if accelerator.is_main_process:
            info("Applying LoRA adapters to the model...")
        model.model = get_peft_model(model.model)
        info("LoRA model ready")
    else:
        info("LoRA disabled: running full fine-tuning")

    info("Starting fine-tuning...")
    finetune(model, train_dataset, test_dataset, accelerator)
    info("Fine-tuning finished")