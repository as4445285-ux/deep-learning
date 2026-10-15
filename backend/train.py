"""
train.py  –  Full training pipeline for the Image Captioning System.

Usage
-----
    python backend/train.py [--options]

Expected data layout
--------------------
    data/
    ├── images/          ← Flickr8k (or any) JPEG images
    └── captions.txt     ← one "<filename> <caption>" per line

Download Flickr8k
-----------------
    Request the dataset at:  https://illinois.edu/fb/sec/1713398
    Or use the Kaggle mirror: https://www.kaggle.com/datasets/adityajn105/flickr8k
"""

import argparse
import os
import time
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import EncoderCNN, DecoderRNN
from utils import (
    Vocabulary,
    Flickr8kDataset,
    CaptionCollate,
    get_train_transform,
    get_val_transform,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✔  Checkpoint saved → {path}")


def load_checkpoint(path: str, encoder, decoder, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_loss   = checkpoint.get("loss", float("inf"))
    print(f"  ✔  Loaded checkpoint (epoch {checkpoint.get('epoch', 0)})")
    return start_epoch, best_loss


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(encoder, decoder, loader, optimizer, criterion, device, pad_idx):
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for step, (images, captions, _) in enumerate(loader):
        images   = images.to(device)
        captions = captions.to(device)

        features = encoder(images)
        outputs  = decoder(features, captions[:, :-1])          # teacher forcing

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            captions[:, 1:].reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"    step {step+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(encoder, decoder, loader, criterion, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0

    for images, captions, _ in loader:
        images   = images.to(device)
        captions = captions.to(device)

        features = encoder(images)
        outputs  = decoder(features, captions[:, :-1])

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            captions[:, 1:].reshape(-1),
        )
        total_loss += loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")
    parser.add_argument("--img_dir",        default="data/images",      help="Path to image folder")
    parser.add_argument("--captions_file",  default="data/captions.txt",help="Path to captions file")
    parser.add_argument("--weights_dir",    default="backend/weights",   help="Where to save checkpoints")
    parser.add_argument("--embed_size",     type=int,   default=256)
    parser.add_argument("--hidden_size",    type=int,   default=512)
    parser.add_argument("--num_layers",     type=int,   default=2)
    parser.add_argument("--freq_threshold", type=int,   default=5,      help="Min word frequency for vocab")
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--resume",         default=None,               help="Path to checkpoint to resume from")
    parser.add_argument("--val_split",      type=float, default=0.1,    help="Fraction of data for validation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {device}\n")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_path = os.path.join(args.weights_dir, "vocab.pkl")
    if os.path.isfile(vocab_path):
        vocab = Vocabulary.load(vocab_path)
        print(f"  ✔  Loaded vocabulary ({len(vocab)} words)")
    else:
        print("  Building vocabulary …")
        vocab = Vocabulary(freq_threshold=args.freq_threshold)
        captions = []
        with open(args.captions_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    captions.append(parts[1])
        vocab.build_from_captions(captions)
        os.makedirs(args.weights_dir, exist_ok=True)
        vocab.save(vocab_path)
        print(f"  ✔  Vocabulary built ({len(vocab)} words) and saved.")

    pad_idx = vocab.word2idx[Vocabulary.PAD]

    # ── Datasets ──────────────────────────────────────────────────────────────
    full_dataset = Flickr8kDataset(
        args.img_dir, args.captions_file, vocab, transform=get_train_transform()
    )
    val_size   = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = get_val_transform()   # type: ignore[union-attr]

    collate_fn = CaptionCollate(pad_idx)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"  Train samples: {train_size}  |  Val samples: {val_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder = EncoderCNN(embed_size=args.embed_size).to(device)
    decoder = DecoderRNN(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.linear.parameters()) + list(decoder.parameters()),
        lr=args.lr,
    )
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )
    criterion  = nn.CrossEntropyLoss(ignore_index=pad_idx)

    start_epoch = 1
    best_loss   = float("inf")

    if args.resume:
        start_epoch, best_loss = load_checkpoint(
            args.resume, encoder, decoder, optimizer
        )

    # ── Train loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch}/{args.epochs}")
        print(f"{'─'*60}")

        train_loss = train_one_epoch(
            encoder, decoder, train_loader, optimizer, criterion, device, pad_idx
        )
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"\n  ▶  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  ({elapsed:.1f}s)"
        )

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                {
                    "epoch":   epoch,
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss":    val_loss,
                    "vocab":   vocab.to_dict(),
                    "embed_size":  args.embed_size,
                    "hidden_size": args.hidden_size,
                    "num_layers":  args.num_layers,
                },
                os.path.join(args.weights_dir, "best_model.pth"),
            )

        # Save latest
        save_checkpoint(
            {
                "epoch":   epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss":    val_loss,
                "vocab":   vocab.to_dict(),
                "embed_size":  args.embed_size,
                "hidden_size": args.hidden_size,
                "num_layers":  args.num_layers,
            },
            os.path.join(args.weights_dir, "latest_model.pth"),
        )

    print("\n✅  Training complete!")


if __name__ == "__main__":
    main()
