import re
import librosa
import numpy as np

from functools import Placeholder, partial

from src.config import get_config


def preprocess(dataset):
    config = get_config()
    dataset_config = config.dataset
    process_config = config.process

    # Filter out speaker if a speaker name is given, otherwise use the whole dataset
    if dataset_config.speaker_name:
        dataset = dataset.filter(
            lambda x: x[dataset_config.speaker_column] == dataset_config.speaker_name
        )

    process_text = partial(
        __process_text, Placeholder, dataset_config.transcript_column
    )
    dataset = dataset.map(process_text)

    # Remove rows with symbol only transcript
    dataset = dataset.filter(
        lambda x: bool(re.search(r"[a-zA-Z]", x[dataset_config.transcript_column]))
    )

    process_audio = partial(
        __process_audio, Placeholder, process_config, dataset_config.audio_column
    )
    dataset = dataset.map(process_audio)

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

    # Remove content enclosed in (), [], or {}
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}", "", text)

    # Remove HTML/XML-style tags
    text = re.sub(r"<[^>]*>", "", text)

    # Genshin specific replacements
    text = re.sub(r"{NICKNAME}", "Traveler", text).strip()
    text = re.sub(r"{F#he}{M#she}", "He", text).strip()
    text = re.sub(r"{F#his}{M#her}", "his", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    return {transcript_column: text}
