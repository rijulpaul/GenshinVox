from src.config import get_config
"""Experiment tracking helpers (wandb integrated via accelerate)."""


def tracker_config(training_config, lora_config=None):
    """Flatten the training config into a dict for wandb hyperparameter logging.

    When LoRA is used (`lora_config` is not None), its config is added under the
    "lora/" prefix so both full-finetune and LoRA runs are fully described.
    """
    tracker = {
        "lr": training_config.lr,
        "beta1": training_config.betas[0],
        "beta2": training_config.betas[1],
        "eps": training_config.eps,
        "weight_decay": training_config.weight_decay,
        "amsgrad": training_config.amsgrad,
        "epochs": training_config.epochs,
        "batch_size": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "speaker_name": training_config.speaker_name,
        "model": training_config.model,
        "model_path": training_config.model_path,
    }

    if lora_config is not None:
        lora = lora_config.to_dict()
        tracker.update(
            {f"lora/{k}": v for k, v in lora.items() if not k.startswith("_")}
        )

    return tracker


def setup_tracker(accelerator,**setup_kwargs):
    """Initialize a wandb tracker through accelerate (no-op if disabled)."""
    config = get_config()
    wandb_config = config.wandb

    init_kwargs = {"wandb": {}}
    if wandb_config.entity:
        init_kwargs["wandb"]["entity"] = wandb_config.entity
    if wandb_config.run_name:
        init_kwargs["wandb"]["name"] = wandb_config.run_name
    if wandb_config.mode:
        init_kwargs["wandb"]["mode"] = wandb_config.mode

    for k,v in setup_kwargs.items():
        init_kwargs["wandb"][k] = v

    accelerator.init_trackers(
        wandb_config.project,
        config=tracker_config(config.training, config.lora),
        init_kwargs=init_kwargs,
    )
