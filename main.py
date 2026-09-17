import argparse
import dotenv
import torch

from transformers import AutoConfig
from accelerate import Accelerator
from datasets import load_from_disk

from src.config import load_config
from src.finetune import finetune
from src.peft import get_peft_model
from src.model import load_model, load_tokenizer
from src.dataset import load_dataset, TTSDataset
from src.preprocess import preprocess
from src.extract_feature import extract_feature

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Load config from a JSON or YAML file."
    )

    args = parser.parse_args()

    dotenv.load_dotenv()

    config = load_config(args.config)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
    )

    train_dataset, test_dataset = load_dataset()
    model = load_model()

    if accelerator.is_main_process:
        if not config.dataset.is_processed:
            train_dataset = preprocess(train_dataset)

        tokenizer = load_tokenizer()

        train_dataset, ref_mel = extract_feature(
            dataset=train_dataset, tokenizer=tokenizer, processor=model.processor
        )

        train_dataset.save_to_disk('data/train')
        torch.save(ref_mel,'data/ref_mel')

        del tokenizer

    accelerator.wait_for_everyone()

    train_dataset = load_from_disk('data/train')
    ref_mel = torch.load('data/ref_mel', map_location="cpu")

    model_config = AutoConfig.from_pretrained(config.training.model)
    train_dataset = TTSDataset(train_dataset, model.processor, ref_mel, model_config)

    if config.lora:
        print("LoRa Finetuning Enabled")
        model.model = get_peft_model(model.model)

    finetune(model, train_dataset, test_dataset, accelerator)
