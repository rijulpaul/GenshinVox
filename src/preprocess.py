import re
import librosa
import numpy as np

from src.config import get_config
from src.log import info


def preprocess(dataset):
    config = get_config()
    dataset_config = config.dataset
    process_config = config.process

    def filter(example):
        # Filter out speaker if a speaker name is given, otherwise use the whole dataset
        if dataset_config.speaker_name and example[dataset_config.speaker_column] != dataset_config.speaker_name:
            return False
        # Remove Transcripts with Player specific replacements
        return example[dataset_config.transcript_column][0] != "#"

    if dataset_config.speaker_name:
        info(f"Filtering dataset by speaker '{dataset_config.speaker_name}'...")
    else:
        info("No speaker filter configured; keeping the whole dataset")
    dataset = dataset.filter(filter)
    info(f"After speaker/#-transcript filter: {len(dataset)} rows")

    info("Normalizing transcripts (stripping tags, collapsing whitespace)...")
    process_text = lambda text: __process_text(
        text,
        dataset_config.transcript_column
    )
    dataset = dataset.map(process_text)
    info(f"Transcripts normalized: {len(dataset)} rows")

    # Remove rows with symbol only transcript
    info("Removing rows with symbol-only transcripts...")
    dataset = dataset.filter(
        lambda x: bool(re.search(r"[a-zA-Z]", x[dataset_config.transcript_column]))
    )
    info(f"After symbol-only transcript filter: {len(dataset)} rows")

    info(
        f"Processing audio: mono, resample {process_config.target_sample_rate}Hz, "
        f"trim top_db={process_config.top_db_to_trim}"
    )
    process_audio = lambda audio: __process_audio(
        audio,
        process_config,
        dataset_config.audio_column
    )
    dataset = dataset.map(process_audio)
    info(f"Audio processing complete: {len(dataset)} rows")

    return dataset


def __to_mono(array):
    if array.ndim > 1:
        array = array.mean(axis=-1)
    return array


def __resample(array, orig_sr, target_sr):

    if orig_sr != target_sr:
        array = librosa.resample(
            array,
            orig_sr=orig_sr,
            target_sr=target_sr,
        )

    return array


def __trim(array, top_db):
    trimmed, _ = librosa.effects.trim(
        array,
        top_db=top_db,
    )

    return trimmed


def __process_audio(example, process_config, audio_column):
    audio = example[audio_column]

    array = np.asarray(audio["array"], dtype=np.float32)

    orig_sr = audio["sampling_rate"]
    target_sr = process_config.target_sample_rate

    top_db = process_config.top_db_to_trim

    array = __to_mono(array)
    array = __resample(array, orig_sr, target_sr)
    array = __trim(array, top_db)

    return {
        "audio": {
            "array": array.astype(np.float32),
            "sampling_rate": target_sr,
        }
    }


def __process_text(example, transcript_column):
    text = example[transcript_column]

    # Remove HTML/XML-style tags
    text = re.sub(r"<[^>]*>", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return {transcript_column: text}
