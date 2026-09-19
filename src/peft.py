from peft import get_peft_model as gpm
from src.config import get_config
from src.log import info


def get_peft_model(model):
    peft_config = get_config().lora
    info("Wrapping model with LoRA adapters...")
    model = gpm(model, peft_config)
    model.print_trainable_parameters()
    info("LoRA adapters attached; model ready for training")

    return model
