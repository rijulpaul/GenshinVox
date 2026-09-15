import argparse
import dotenv

from transformers import AutoConfig

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

    train_dataset, test_dataset = load_dataset()
    if not config.dataset.is_processed:
        train_dataset = preprocess(train_dataset)

    model = load_model()
    tokenizer = load_tokenizer()

    model_config = AutoConfig.from_pretrained(config.training.model)

    train_dataset, ref_mel = extract_feature(
        dataset=train_dataset, tokenizer=tokenizer, processor=model.processor
    )
    del tokenizer

    train_dataset = TTSDataset(train_dataset, model.processor, ref_mel, model_config)

    if config.lora:
        print("LoRa Finetuning Enabled")
        model.model = get_peft_model(model.model)

    finetune(model, train_dataset, test_dataset)
