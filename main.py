import argparse

from transformers import AutoConfig

from src.config import load_config
from src.finetune import finetune
from src.model import load_model, load_tokenizer
from src.dataset import load_dataset, TTSDataset
from src.preprocess import preprocess
from src.extract_feature import extract_feature

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Load config from the specified JSON file."
    )

    args = parser.parse_args()

    config = load_config(args.config)

    dataset = load_dataset()
    dataset = preprocess(dataset)

    tokenizer = load_tokenizer()
    model = load_model()

    config = AutoConfig.from_pretrained(config.training.model)

    dataset, ref_mel = extract_feature(
        dataset=dataset, tokenizer=tokenizer, processor=model.processor
    )
    dataset = TTSDataset(dataset, model.processor, ref_mel, config)

    finetune(model, dataset)
