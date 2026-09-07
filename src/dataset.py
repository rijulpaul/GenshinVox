from typing import Any, List, Tuple, Union

import numpy as np
import torch
from qwen_tts.qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from torch.utils.data import Dataset

AudioLike = Union[
    str,  # wav path, URL, base64
    np.ndarray,  # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):
    def __init__(self, dataset, processor, ref_mel, config: Qwen3TTSConfig, lag_num=-1):
        self.dataset = dataset
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.ref_mel = ref_mel

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        text_ids = torch.Tensor(item["text_ids"])
        audio_codes = torch.Tensor(item["audio_codes"])

        return {
            "text_ids": text_ids[:, :-5],  # 1 , t
            "audio_codes": audio_codes,  # t, 16
            "ref_mel": self.ref_mel,
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [
            b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch
        ]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)

        for i, data in enumerate(batch):
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = (
                self.config.tts_pad_token_id
            )
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            # codec channel
            # input_ids[i,   :3, 1] = 0
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,  # for speaker embedding
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[i, 8 : 8 + text_ids_len - 3, 1] = (
                self.config.talker_config.codec_pad_id
            )
            input_ids[i, 8 + text_ids_len - 3, 1] = (
                self.config.talker_config.codec_pad_id
            )
            input_ids[i, 8 + text_ids_len - 2, 1] = (
                self.config.talker_config.codec_bos_id
            )
            input_ids[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1
            ] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = (
                self.config.talker_config.codec_eos_token_id
            )

            codec_0_labels[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len
            ] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = (
                self.config.talker_config.codec_eos_token_id
            )

            codec_ids[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :
            ] = audio_codecs

            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # for speaker embedding

            codec_mask[
                i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len
            ] = True
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        ref_mels = [self.ref_mel] * len(batch)
        ref_mels = torch.cat(ref_mels, dim=0)

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }


from datasets import load_dataset as ld, Audio
from src.config import get_config


def load_dataset():
    config = get_config().dataset
    dataset = None
    if config.local_path:
        try:
            dataset = ld(
                "parquet",
                data_files={"train": config.local_path + "/*.parquet"},
                split="train",
            )
            print(f"Successfully loaded dataset at {config.local_path}")
        except Exception:
            print(f"Dataset not found at {config.local_path}")
            print(f"Downloading dataset into {config.local_path}")
            dataset = ld(config.dataset, split="train")
    else:
        dataset = ld(config.dataset, split="train")

    dataset = dataset.cast_column(config.audio_column, Audio(decode=True))
    return dataset
