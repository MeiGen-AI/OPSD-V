# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb


def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        array_shape = txn.get(f"{array_name}_shape".encode()).decode()
        array_shape = tuple(map(int, array_shape.split()))
    return array_shape


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, row_index, shape=None):
    data_key = f"{array_name}_{row_index}_data".encode()
    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)
    if row_bytes is None:
        raise KeyError(f"Missing LMDB key: {array_name}_{row_index}_data")

    if dtype == str:
        array = row_bytes.decode()
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)

    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array


def _lmdb_has_key(env, key: str) -> bool:
    with env.begin() as txn:
        return txn.get(key.encode()) is not None



class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class InferencePromptEmbedsVideoLMDBDataset(Dataset):
    """Flexible LMDB dataset for inference/training with precomputed prompt embeddings."""

    def __init__(self, data_path: str, max_pair: int = int(1e8), require_gt_latents: bool = False):
        self.env = lmdb.open(
            data_path, readonly=True, lock=False, readahead=False, meminit=False
        )
        self.max_pair = max_pair

        if _lmdb_has_key(self.env, "prompts_shape"):
            self.prompt_key = "prompts"
        elif _lmdb_has_key(self.env, "text_shape"):
            self.prompt_key = "text"
        else:
            raise KeyError("Neither `prompts_shape` nor `text_shape` found in LMDB.")
        self.prompts_shape = get_array_shape_from_lmdb(self.env, self.prompt_key)

        if not _lmdb_has_key(self.env, "prompt_embeds_shape"):
            raise KeyError("`prompt_embeds_shape` not found in LMDB.")
        self.prompt_embeds_shape = get_array_shape_from_lmdb(self.env, "prompt_embeds")

        self.has_gt_latents = False
        self.latent_key = None
        self.latents_shape = None
        if _lmdb_has_key(self.env, "latents_shape"):
            self.latent_key = "latents"
            self.latents_shape = get_array_shape_from_lmdb(self.env, self.latent_key)
            self.has_gt_latents = True
        elif _lmdb_has_key(self.env, "video_shape"):
            self.latent_key = "video"
            self.latents_shape = get_array_shape_from_lmdb(self.env, self.latent_key)
            self.has_gt_latents = True
        if require_gt_latents and not self.has_gt_latents:
            raise KeyError(
                "Neither `latents_shape` nor `video_shape` found in LMDB, but gt latents are required."
            )

    def __len__(self):
        return min(self.prompts_shape[0], self.max_pair)

    def __getitem__(self, idx):
        prompts = retrieve_row_from_lmdb(self.env, self.prompt_key, str, idx)
        prompt_embeds = retrieve_row_from_lmdb(
            self.env,
            "prompt_embeds",
            np.float16,
            idx,
            shape=self.prompt_embeds_shape[1:],
        )
        batch = {
            "prompts": prompts,
            "prompt_embeds": torch.tensor(prompt_embeds, dtype=torch.float32),
            "idx": idx,
        }

        if self.has_gt_latents:
            gt_latents = retrieve_row_from_lmdb(
                self.env,
                self.latent_key,
                np.float16,
                idx,
                shape=self.latents_shape[1:],
            )
            batch["gt_latents"] = torch.tensor(gt_latents, dtype=torch.float32)

        return batch


def cycle(dl):
    while True:
        for data in dl:
            yield data
