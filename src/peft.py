from peft import get_peft_model as gpm
from src.config import get_config


def get_peft_model(model):
    peft_config = get_config().lora

    model = gpm(model, peft_config)
    model.print_trainable_parameters()

    return model
