"""
utils.py  –  Vocabulary builder, dataset loader, and image transforms
             for the Image Captioning System.
"""

import os
import json
import pickle
import re
from collections import Counter

import nltk
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Download NLTK tokenizer data silently
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────
class Vocabulary:
    """
    Builds a word↔index mapping from a list of captions.

    Special tokens
    --------------
    <pad> = 0   <start> = 1   <end> = 2   <unk> = 3
    """

    PAD   = "<pad>"
    START = "<start>"
    END   = "<end>"
    UNK   = "<unk>"

    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._counter: Counter        = Counter()

        for tok in (self.PAD, self.START, self.END, self.UNK):
            self._add(tok)

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _add(self, word: str) -> None:
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return nltk.tokenize.word_tokenize(text.lower())

    # ── Public API ────────────────────────────────────────────────────────────
    def build_from_captions(self, captions: list[str]) -> None:
        for cap in captions:
            self._counter.update(self.tokenize(cap))
        for word, freq in self._counter.items():
            if freq >= self.freq_threshold:
                self._add(word)

    def numericalize(self, text: str) -> list[int]:
        unk = self.word2idx[self.UNK]
        return [self.word2idx.get(w, unk) for w in self.tokenize(text)]

    def __len__(self) -> int:
        return len(self.word2idx)

    # ── Serialization ─────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            return pickle.load(f)

    def to_dict(self) -> dict:
        return {"word2idx": self.word2idx, "idx2word": self.idx2word}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        vocab = cls()
        vocab.word2idx = d["word2idx"]
        vocab.idx2word = {int(k): v for k, v in d["idx2word"].items()}
        return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Image transforms
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    return get_val_transform(image_size)


# ─────────────────────────────────────────────────────────────────────────────
# Flickr8k Dataset (simple caption file format)
# ─────────────────────────────────────────────────────────────────────────────
class Flickr8kDataset(Dataset):
    """
    Loads the Flickr8k dataset from:
        data/images/        – JPEG images
        data/captions.txt   – one caption per line: "<image_filename> <caption>"

    Expected captions.txt format
    ----------------------------
    1000268201_693b08cb0e.jpg  A child in a pink dress is climbing up a set of stairs .
    ...
    """

    def __init__(
        self,
        img_dir: str,
        captions_file: str,
        vocab: Vocabulary,
        transform=None,
    ):
        self.img_dir  = img_dir
        self.vocab    = vocab
        self.transform = transform or get_val_transform()

        self.image_files: list[str]       = []
        self.captions:    list[list[int]] = []

        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts     = line.split(None, 1)
                if len(parts) < 2:
                    continue
                img_name, caption = parts[0], parts[1]
                img_path = os.path.join(img_dir, img_name)
                if not os.path.isfile(img_path):
                    continue

                tokens = (
                    [vocab.word2idx[Vocabulary.START]]
                    + vocab.numericalize(caption)
                    + [vocab.word2idx[Vocabulary.END]]
                )
                self.image_files.append(img_path)
                self.captions.append(tokens)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img     = Image.open(self.image_files[idx]).convert("RGB")
        img     = self.transform(img)
        caption = torch.tensor(self.captions[idx], dtype=torch.long)
        return img, caption


# ─────────────────────────────────────────────────────────────────────────────
# Collate function (pad sequences to the same length in a batch)
# ─────────────────────────────────────────────────────────────────────────────
class CaptionCollate:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images   = [item[0].unsqueeze(0) for item in batch]
        captions = [item[1] for item in batch]

        images   = torch.cat(images, dim=0)
        lengths  = [len(c) for c in captions]
        targets  = torch.zeros(len(captions), max(lengths), dtype=torch.long)
        targets.fill_(self.pad_idx)
        for i, cap in enumerate(captions):
            targets[i, : lengths[i]] = cap

        return images, targets, torch.tensor(lengths)
