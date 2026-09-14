import torch
import random
from typing import List
from tqdm import tqdm

from src.config import get_config

from qwen_tts.qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def extract_feature(dataset, processor, tokenizer):
    """ Pick a reference audio and convert to mel_spectrogram
        Convert rest of audio and transcripts to embeddings
    """
    config = get_config()
    dataset_config = config.dataset
    processing_config = config.process

    audio_column = dataset_config.audio_column
    transcript_column = dataset_config.transcript_column

    search_size = len(dataset)

    select_ref_strategy = processing_config.select_ref_audio_strategy
    ref_min_duration = processing_config.ref_audio_min_duration
    ref_max_duration = processing_config.ref_audio_max_duration

    ref_idx = None
    ref_duration = 0.0

    if select_ref_strategy == "random":
        ref_idx = random.randint(0, search_size - 1)
        audio = dataset[ref_idx][audio_column]
        ref_duration = len(audio["array"]) / audio["sampling_rate"]
    else:
        for i in tqdm(
            range(search_size),
            desc="Finding reference audio",
        ):
            audio = dataset[i][audio_column]

            duration = len(audio["array"]) / audio["sampling_rate"]

            if (
                duration <= ref_max_duration
                and duration > ref_min_duration
                and duration > ref_duration
            ):
                ref_idx = i
                ref_duration = duration
                if select_ref_strategy == "good enough":
                    break

        if ref_idx is None:
            raise RuntimeError("Could not find suitable reference audio.")

    ref_audio = dataset[ref_idx][audio_column]

    print(
        f"Using index {ref_idx} as reference\n Duration: {ref_duration:.2f}s\n Selection Strategy: {select_ref_strategy}"
    )

    dataset = dataset.filter(
        lambda _, idx: idx != ref_idx,
        with_indices=True,
        desc="Removing reference sample",
    )

    def _process_batch(example):

        text_id = _tokenize_text(example[transcript_column], processor)

        audios = example[audio_column]["array"]

        codes = tokenizer.encode([audios], sr=processing_config.target_sample_rate)

        audio_code = codes.audio_codes[0]

        return {"audio_codes": audio_code, "text_ids": text_id}

    encoded = dataset.map(
        _process_batch,
        remove_columns=[audio_column, transcript_column],
        desc="Extracting audio codes",
    )

    ref_mel = extract_mels(audio=ref_audio["array"], sr=ref_audio["sampling_rate"])

    return encoded, ref_mel


def _tokenize_text(text, processor) -> List[torch.Tensor]:
    text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    input = processor(
        text=text,
        return_tensors="pt",
        padding=True,
    )

    input_id = input["input_ids"]
    input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

    return input_id


@torch.inference_mode()
def extract_mels(audio, sr):
    assert sr == 24000, "Only support 24kHz audio sampling rate"
    mels = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    return mels
