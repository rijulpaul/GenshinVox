import os, shutil, json, time, torch
import numpy as np
import soundfile as sf
from peft import PeftModel

from src.config import get_config
import src.eval as eval
from src.experiment_tracking import setup_tracker
from src.log import info, warning

from qwen_tts.qwen_tts import Qwen3TTSModel

from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.optim import AdamW


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

    commit_message = config.commit_message.format(epoch=f"{epoch:03d}", global_step=global_step)

    path_in_repo = None
    if config.path_in_repo:
        path_in_repo = config.path_in_repo.format(epoch=f"{epoch:03d}", global_step=global_step)

    upload_folder(
        repo_id=config.repo_id,
        repo_type="model",
        folder_path=output_dir,
        revision=config.revision,
        commit_message=commit_message,
        path_in_repo=path_in_repo,
        token=token,
        ignore_patterns=["*.md"]
    )

def finetune(model, train_dataset, test_dataset, accelerator):

    config = get_config()
    training_config = config.training
    model_ckpt_config = training_config.model_checkpoint
    train_ckpt_config = training_config.training_checkpoint

    info(
        f"Initializing fine-tuning: model={training_config.model}, "
        f"lr={training_config.lr}, epochs={training_config.epochs}, "
        f"batch_size={training_config.batch_size}, "
        f"grad_accum={training_config.gradient_accumulation_steps}"
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    info(f"Train dataloader ready: {len(dataloader)} batches")

    optimizer = AdamW(
        model.model.parameters(),
        lr=training_config.lr,
        betas=training_config.betas,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
        amsgrad=training_config.amsgrad,
    )

    # --- experiment tracking state ---
    start_epoch = 0
    global_step = 0
    micro_steps = 0
    running_loss = 0.0
    running_talker_loss = 0.0
    running_sub_talker_loss = 0.0

    # wandb setup kwargs used if resuming training
    setup_kwargs = {}

    # Resume training from a checkpoint
    if training_config.resume_training_path:
        path = training_config.resume_training_path
        accelerator.load_state(path)
        with open(os.path.join(path,'train_info.json'),'r') as file:
            resume_info = json.load(file)
            start_epoch = resume_info.epoch+1
            global_step = resume_info.global_step
            setup_kwargs['id'] = resume_info.run_id
            setup_kwargs['resume'] = 'must'
            info(
                f"Resuming training from {path}: "
                f"resuming at epoch {start_epoch}, global step {global_step}"
            )

    if training_config.enable_experiment_tracking:
        setup_tracker(accelerator, **setup_kwargs)

    model, optimizer, dataloader = accelerator.prepare(
        model.model, optimizer, dataloader
    )
    info(f"Model, optimizer, dataloader prepared ({len(dataloader)} accelerator batches)")

    model.train()
    info("Model set to train mode")
    base_model = accelerator.unwrap_model(model)

    target_speaker_embedding = None

    for epoch in range(start_epoch,training_config.epochs):
        info(f"===== Starting epoch {epoch} / {training_config.epochs - 1} =====")
        epoch_start = time.perf_counter()
        epoch_loss = epoch_talker_loss = epoch_sub_talker_loss = 0.0
        epoch_samples = 0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = base_model.speaker_encoder(
                    ref_mels.to(accelerator.device).to(base_model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    base_model.talker.text_projection(
                        base_model.talker.model.text_embedding(input_text_ids)
                    )
                    * text_embedding_mask
                )

                input_codec_embedding = (
                    base_model.talker.model.codec_embedding(input_codec_ids)
                    * codec_embedding_mask
                )

                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = (
                        base_model.talker.code_predictor.get_input_embeddings()[i - 1](
                            codec_ids[:, :, i]
                        )
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = base_model.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1][:, :-1, :]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                _, sub_talker_loss = (
                    base_model.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    if grad_norm is not None:
                        grad_norm = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)

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
            global_step+=1

            if accelerator.sync_gradients:
                if training_config.enable_experiment_tracking:
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
                    f"Epoch {epoch:03d} | Step {step} | Loss: {loss_value:.4f}"
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

        epoch_avg = (
            epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        )
        info(
            f"Epoch {epoch} complete | avg loss={epoch_avg:.4f} | "
            f"duration={time.perf_counter() - epoch_start:.1f}s"
        )

        if train_ckpt_config and epoch % train_ckpt_config.save_every_n_epochs == 0:
            output_dir = train_ckpt_config.output_path.format(
                epoch=f"{epoch:03d}", global_step=global_step
            )
            accelerator.save_state(output_dir)
            info(f"Training checkpoint (resumable) saved to {output_dir}")
            run = accelerator.get_tracker('wandb',unwrap=True)
            run_id = run.id if hasattr(run,"id") else None
            data = {
                "epoch": epoch,
                "global step": global_step,
                "run_id": run_id
            }
            with open(os.path.join(output_dir,'train_info.json'),'w') as file:
                json.dump(data,file,indent=4)

            if accelerator.is_main_process:
                # Save model checkpoint Locally and setup for inference
                if epoch % model_ckpt_config.save_every_n_epochs == 0:

                    output_dir = os.path.join(
                        model_ckpt_config.output_path.format(
                            epoch=f"{epoch:03d}", global_step=global_step
                        )
                    )

                    if config.lora:
                        # save adapters
                        base_model.save_pretrained(
                            os.path.join(output_dir,"adapter")
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
                        talker_config["spk_id"] = {training_config.speaker_name.lower(): 3000}
                        talker_config["spk_is_dialect"] = {
                            training_config.speaker_name.lower(): False
                        }
                        config_dict["talker_config"] = talker_config

                    with open(output_config_file, "w", encoding="utf-8") as f:
                        json.dump(config_dict, f, indent=2, ensure_ascii=False)


                    state_dict = {
                        k: v.detach().to("cpu")
                        for k, v in base_model.state_dict().items()
                    }

                    for k,v in state_dict.items():
                        print(k,v)

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
                    info(f"Model checkpoint saved to {save_path}")

        accelerator.wait_for_everyone()

    accelerator.end_training()
    info("Training complete; accelerator tracker ended")
