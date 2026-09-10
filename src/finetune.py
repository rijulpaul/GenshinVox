import os, shutil, json, time

from src.config import get_config

from accelerate import Accelerator
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.optim import AdamW


def _tracker_config(config):
    """Flatten the training config into a dict for wandb hyperparameter logging."""
    return {
        "lr": config.lr,
        "beta1": config.betas[0],
        "beta2": config.betas[1],
        "eps": config.eps,
        "weight_decay": config.weight_decay,
        "amsgrad": config.amsgrad,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "speaker_name": config.speaker_name,
        "model": config.model,
        "model_path": config.model_path,
    }


def _setup_tracker(accelerator, training_config, wandb_config):
    """Initialize a wandb tracker through accelerate (no-op if disabled)."""
    init_kwargs = {"wandb": {}}
    if wandb_config.entity:
        init_kwargs["wandb"]["entity"] = wandb_config.entity
    if wandb_config.run_name:
        init_kwargs["wandb"]["name"] = wandb_config.run_name
    if wandb_config.mode:
        init_kwargs["wandb"]["mode"] = wandb_config.mode
    accelerator.init_trackers(
        wandb_config.project,
        config=_tracker_config(training_config),
        init_kwargs=init_kwargs,
    )


def _upload_checkpoint_to_hf(config, output_dir, epoch, global_step):
    """Upload a saved checkpoint directory to the Hugging Face Hub."""
    from huggingface_hub import create_repo, upload_folder

    if not config.upload_to_hub or epoch % config.upload_every_n_epochs != 0:
        return

    if not config.repo_id:
        raise ValueError("hf.repo_id is required to upload checkpoints to the Hub")

    token = os.environ["HF_TOKEN"]

    if not token:
        raise ValueError(
            "Environment variable HF_TOKEN with hugging face api key needed to authenticate"
        )

    create_repo(
        repo_id=config.repo_id,
        repo_type="model",
        private=config.private,
        exist_ok=True,
        token=token,
    )

    commit_message = (
        config.commit_message.format(epoch=epoch, global_step=global_step)
        or f"Upload checkpoint-epoch-{epoch}"
    )

    path_in_repo = None
    if config.path_in_repo:
        path_in_repo = config.path_in_repo.format(epoch=epoch, global_step=global_step)
    upload_folder(
        repo_id=config.repo_id,
        repo_type="model",
        folder_path=output_dir,
        revision=config.revision,
        commit_message=commit_message,
        path_in_repo=path_in_repo,
        token=config.token,
    )


def finetune(model, dataset):

    config = get_config()
    training_config = config.training
    wandb_config = config.wandb
    model_ckpt_config = training_config.model_checkpoint
    train_ckpt_config = training_config.training_checkpoint

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = AdamW(
        model.model.parameters(),
        lr=training_config.lr,
        betas=training_config.betas,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
        amsgrad=training_config.amsgrad,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
    )

    if training_config.enable_experiment_tracking:
        _setup_tracker(accelerator, training_config, wandb_config)

    model, optimizer, dataloader = accelerator.prepare(
        model.model, optimizer, dataloader
    )

    # Resume training from a checkpoint
    if training_config.resume_training_path:
        accelerator.load_state(training_config.resume_training_path)

    model.train()

    target_speaker_embedding = None

    # --- experiment tracking state ---
    global_step = 0
    micro_steps = 0
    running_loss = 0.0
    running_talker_loss = 0.0
    running_sub_talker_loss = 0.0

    for epoch in range(training_config.epochs):
        epoch_start = time.perf_counter()
        epoch_loss = epoch_talker_loss = epoch_sub_talker_loss = 0.0
        epoch_samples = 0

        for step, batch in enumerate(dataloader):
            break
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model.talker.text_projection(
                        model.talker.model.text_embedding(input_text_ids)
                    )
                    * text_embedding_mask
                )

                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids)
                    * codec_embedding_mask
                )

                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = (
                        model.talker.code_predictor.get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1][:, :-1, :]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = (
                    model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    if grad_norm is not None:
                        grad_norm = grad_norm.item()

                optimizer.step()
                optimizer.zero_grad()

            # --- track metrics ---
            loss_value = loss.item()
            talker_loss_value = outputs.loss.item()
            sub_talker_loss_value = sub_talker_loss.item()

            running_loss += loss_value
            running_talker_loss += talker_loss_value
            running_sub_talker_loss += sub_talker_loss_value
            micro_steps += 1
            epoch_loss += loss_value
            epoch_talker_loss += talker_loss_value
            epoch_sub_talker_loss += sub_talker_loss_value
            epoch_samples += batch["input_ids"].shape[0]

            if accelerator.sync_gradients:
                global_step += 1
                if (
                    training_config.enable_experiment_tracking
                    and global_step % wandb_config.log_every_n_steps == 0
                ):
                    accelerator.log(
                        {
                            "train/loss": running_loss / micro_steps,
                            "train/talker_loss": running_talker_loss / micro_steps,
                            "train/sub_talker_loss": (
                                running_sub_talker_loss / micro_steps
                            ),
                            "train/grad_norm": grad_norm,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/micro_steps": micro_steps,
                        },
                        step=global_step,
                    )
                running_loss = 0.0
                running_talker_loss = 0.0
                running_sub_talker_loss = 0.0
                micro_steps = 0

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss_value:.4f}"
                )

        if training_config.enable_experiment_tracking:
            epoch_duration = time.perf_counter() - epoch_start
            accelerator.log(
                {
                    "epoch/epoch": epoch,
                    "epoch/loss": epoch_loss / len(dataloader),
                    "epoch/talker_loss": epoch_talker_loss / len(dataloader),
                    "epoch/sub_talker_loss": (epoch_sub_talker_loss / len(dataloader)),
                    "epoch/duration_sec": epoch_duration,
                    "epoch/samples_per_sec": (
                        epoch_samples / epoch_duration if epoch_duration > 0 else 0.0
                    ),
                },
                step=global_step,
            )

        if accelerator.is_main_process:
            accelerator.wait_for_everyone()

            # Save Training Checkpoint Locally
            if train_ckpt_config and epoch % train_ckpt_config.save_every_n_epochs == 0:
                accelerator.save_state(
                    train_ckpt_config.output_path.format(
                        epoch=epoch, global_step=global_step
                    )
                )

            # Save model checkpoint Locally
            if epoch % model_ckpt_config.save_every_n_epochs == 0:
                output_dir = os.path.join(
                    model_ckpt_config.output_path.format(
                        epoch=epoch, global_step=global_step
                    )
                )

                shutil.copytree(
                    training_config.model_path, output_dir, dirs_exist_ok=True
                )

                input_config_file = os.path.join(
                    training_config.model_path, "config.json"
                )
                output_config_file = os.path.join(output_dir, "config.json")
                with open(input_config_file, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                    config_dict["tts_model_type"] = "custom_voice"
                    talker_config = config_dict.get("talker_config", {})
                    talker_config["spk_id"] = {training_config.speaker_name: 3000}
                    talker_config["spk_is_dialect"] = {
                        training_config.speaker_name: False
                    }
                    config_dict["talker_config"] = talker_config

                with open(output_config_file, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = {
                    k: v.detach().to("cpu")
                    for k, v in unwrapped_model.state_dict().items()
                }

                drop_prefix = "speaker_encoder"
                keys_to_drop = [
                    k for k in state_dict.keys() if k.startswith(drop_prefix)
                ]
                for k in keys_to_drop:
                    del state_dict[k]

                weight = state_dict["talker.model.codec_embedding.weight"]
                state_dict["talker.model.codec_embedding.weight"][3000] = (
                    target_speaker_embedding[0]
                    .detach()
                    .to(weight.device)
                    .to(weight.dtype)
                )
                save_path = os.path.join(output_dir, "model.safetensors")
                save_file(state_dict, save_path)

            # --- upload checkpoints to Hugging Face Hub ---
            for ckpt_config in [model_ckpt_config, train_ckpt_config]:
                try:
                    output_dir = os.path.join(
                        ckpt_config.output_path.format(
                            epoch=epoch, global_step=global_step
                        )
                    )
                    _upload_checkpoint_to_hf(
                        ckpt_config, output_dir, epoch, global_step
                    )
                    accelerator.print(
                        f"Uploaded {output_dir} to Hugging Face Hub: "
                        f"{ckpt_config.repo_id}"
                    )
                except Exception as exc:
                    accelerator.print(
                        "Warning: failed to upload checkpoint to Hugging Face Hub: "
                        f"{exc}"
                    )

    accelerator.end_training()
