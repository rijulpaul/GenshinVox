from peft import get_peft_model as gpm
from src.config import get_config
from collections import OrderedDict


def get_peft_model(model):
    peft_config = get_config().lora
    model = gpm(model, peft_config)
    model.print_trainable_parameters()

    return model

def build_state_dict(state_dict):
    config = get_config().lora
    new_state_dict = OrderedDict()
    for k in state_dict:
        lk = k.split('.')
        if lk[-2]=="base_layer":
            # Is affected by LoRa
            base = k
            a = k.replace('base_layer','lora_A.default')
            b = k.replace('base_layer','lora_B.default')
            weights = merge_lora(state_dict[base],state_dict[a],state_dict[b],config.lora_alpha,config.r)
            nk = k.replace('base_model.model.','')    # Remove 'base_model.model' prefix
            nk = nk.replace('base_layer.','')         # Remove 'base_model.model' prefix
            new_state_dict[nk] = weights
        elif lk[-3] in ["lora_A","lora_B"]:
            # Pray to God we identified the base_layer and executed the merge
            continue
        else:
            nk = k.replace('base_model.model.','')    # Remove 'base_model.model' prefix
            new_state_dict[nk] = state_dict[k]

    return new_state_dict


def merge_lora(base_weight, lora_A, lora_B, alpha, rank):
    scaling = alpha / rank

    # LoRA update
    delta_weight = lora_B @ lora_A

    # Merge into base
    merged_weight = base_weight + scaling * delta_weight

    return merged_weight
