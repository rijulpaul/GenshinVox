# GenshinVox

Fine-tuning pipeline for the Qwen3-TTS-12Hz-Base models ([0.6B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) or [1.7B](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)).

[Config-driven](#configuration) and model/data agnostic:
- **Model** — any Qwen3-TTS-12Hz `-Base` checkpoint, by HF id or local path.
- **Data** — any dataset the `datasets` library can load with an `Audio()` column (HF hub, local Parquet, CSV/JSON, audio folder).
- **Full fine-tuning or [LoRA](#using-lora-adapter)**, with optional wandb tracking, distributed training via `accelerate` with `DeepSpeed ZeRO` integrated, WER/speaker-similarity eval, Hub uploads.

---

## Requirements

- **python**: >=3.12
- **ffmpeg**: >=4 and <=9
- **cu128** compatible GPU 

## Setup

```bash
uv sync                                  # Python 3.12+
uv add flash-attn                        # optional but recommended
uv add deepspeed                         # For optimizer/gradients/model sharding (optional)
echo "HF_TOKEN=hf_..." >> .env           # required for hub access/uploads and speaker similarity testing model download (pyannote/wespeaker-voxceleb-resnet34-LM).
echo "WANDB_API_KEY=..." >> .env         # only for online wandb
cp config/example.yaml config/your_run.yaml
```

See `pyproject.toml` for the full dependency list.

### On Linux
```bash
uv add torchcodec   # Installs cu128 version

# Optionally
uv add flash-attn deepspeed # wheels index provided in pyproject.toml
```

### On Windows
> [!NOTE] 
> No cuda version of torchcodec is available on windows (cpu variant should be installed).  
Same goes for deepspeed and flash-attn with pre-built wheels  

> [!TIP]
> Inorder to install flash-attn or deepspeed on windows remove wheels index for these (lines `40-41` in `pyproject.toml`)   
```toml
flash-attn = [{ index = "wheels" }]
deepspeed = [{ index = "wheels" }]
```

---

## Configuration

Start from an example and edit:
- [`config/example.yaml`](config/example.yaml) / [`config/example.json`](config/example.json)
- Every field is a Pydantic model in [`src/config.py`](src/config.py) — the schema is the source of truth.

Highlights:

- **`lora`** — include to enable LoRA, omit for full fine-tuning.
- **`training`** — lr, epochs, batch_size, model/tokenizer, `attn_implementation`, `speaker_name`, `gradient_accumulation_steps`, checkpoint save/upload settings, `resume_training_path`, `deepspeed_zero_stage`.
  - Default `lr` is **`2e-6`** — the official `2e-5` causes severe generation problems.
- **`dataset`** — HF id or local path, `audio_column`/`transcript_column`, `speaker_column`/`speaker_name`, train/test split, `is_processed`.
- **`testing`** — optional WER + speaker-similarity eval and sample saving each epoch.
- **`wandb`** — `project`/`entity`/`run_name`/`mode`.

Checkpoint `output_path`, `path_in_repo`, and `commit_message` support `{epoch}` and `{global_step}`.

---

## Running

```bash
uv run accelerate launch main.py --config config/your_run.yaml
uv run accelerate launch --num_processes <n> main.py --config config/your_run.yaml  # multi-GPU
```

---

## Outputs

- **Model checkpoints** — inference-ready dir (`config.json`, `model.safetensors`, plus `adapter/` for LoRA). Load with `Qwen3TTSModel.from_pretrained(...)` → `generate_custom_voice(speaker="<speaker_name>")`.
- **Training checkpoints** — resumable state via `accelerator.save_state`.
- **Eval samples** — `*.wav` per test sample.

---

## Why LoRA saves full model?
- The model is passed a ref_audio whose embedding is added to the state_dict along with other changes essentialy converting it from `Base` to `CustomVoice`.
- This makes it easier to use adapter without passing a ref_audio and ref_text. Also, the model is used to run evaluations post training.

## Using LoRA adapter
1. Load the model saved with it and attach adapter to it.
```python
# from qwen_tts.qwen_tts import Qwen3TTSModel # if within this repo
from qwen_tts import Qwen3TTSModel  # If using official qwen_tts library

tts = Qwen3TTSModel.from_pretrained(
    path_to_model,
    attn_implementation="flash_attention_2", # sdpa, eager, etc.
    devicie_map="cuda" # auto, cpu
)

from peft import PeftModel

tts.model = PeftModel.from_pretrained(
    tts.model,
    path_to_adapter
)

wav, sr = tts.generate_custom_voice(
    text = text,
    speaker = speaker
)

```
2. Load a Base model, attach the adapter and pass a `ref_audio` representing the same voice actor.
```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

from peft import PeftModel

model.model = PeftModel.from_pretrained(
    model.model,
    path_to_adapter
)

ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
ref_text  = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

# Save the resulting audio
sf.write("output_voice_clone.wav", wavs[0], sr)
```
---

## Changes vs. the official `sft_12hz.py`

**Bug fixes:**
- **Double label-shift** — the official script shifts `inputs_embeds`/`attention_mask`/`labels` (`[:, 1:]`) AND HF's loss shifts again. This pipeline passes unshifted tensors so labels are shifted exactly once.
- **Text-projection** — the official feeds 1024-dim text embeddings (0.6B) straight into a 2048-dim codec sum. This pipeline applies `model.talker.text_projection(...)` to match dims.

**Additions:** `datasets.Audio()` loading instead of JSONL, LoRA + merge handling, JSON/YAML config, distributed training, wandb tracking, WER/speaker-similarity eval, resume, Hub uploads.

---

## License / attribution

Vendored `qwen_tts/` is **Apache-2.0 © Alibaba Qwen Team**. Training logic builds on their `finetuning/sft_12hz.py`. See `qwen_tts/` for license texts.
