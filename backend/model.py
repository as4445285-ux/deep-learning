"""
model.py  –  ResNet50 CNN Encoder + LSTM Decoder for Image Captioning
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# Encoder: ResNet-50 (pretrained, last FC removed)
# ─────────────────────────────────────────────────────────────────────────────
class EncoderCNN(nn.Module):
    """
    Extracts a fixed-size feature vector from an image using a pretrained
    ResNet-50 backbone.  The final classification layer is replaced by a
    linear projection to `embed_size`.
    """

    def __init__(self, embed_size: int = 256, fine_tune: bool = False):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze backbone by default
        for param in resnet.parameters():
            param.requires_grad = fine_tune

        # Replace the classifier head
        modules = list(resnet.children())[:-1]          # drop the FC layer
        self.backbone = nn.Sequential(*modules)
        self.linear   = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn        = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(images)             # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        features = self.bn(self.linear(features))        # (B, embed_size)
        return features


# ─────────────────────────────────────────────────────────────────────────────
# Decoder: LSTM language model
# ─────────────────────────────────────────────────────────────────────────────
class DecoderRNN(nn.Module):
    """
    Autoregressive LSTM that generates a caption word-by-word conditioned on
    the CNN image embedding.
    """

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_size)
        self.lstm    = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear  = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,   # (B, embed_size)
        captions: torch.Tensor,   # (B, seq_len)   – token ids, including <start>
    ) -> torch.Tensor:
        embeddings = self.dropout(self.embed(captions))           # (B, T, E)
        # Prepend image feature as the first time-step
        features   = features.unsqueeze(1)                         # (B, 1, E)
        inputs     = torch.cat((features, embeddings), dim=1)     # (B, T+1, E)
        hiddens, _ = self.lstm(inputs)                             # (B, T+1, H)
        outputs    = self.linear(hiddens)                          # (B, T+1, V)
        return outputs

    @torch.no_grad()
    def sample(
        self,
        features: torch.Tensor,   # (1, embed_size)
        vocab,                     # Vocabulary object
        max_len: int = 25,
        temperature: float = 1.0,
        beam_size: int = 1,
    ) -> list[str]:
        """Greedy or temperature-scaled sampling to generate a caption."""
        self.eval()
        inputs   = features.unsqueeze(1)           # (1, 1, E)
        states   = None
        word_ids = []

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)   # (1, 1, H)
            outputs = self.linear(hiddens.squeeze(1))      # (1, V)

            if temperature != 1.0:
                outputs = outputs / temperature
                probs   = torch.softmax(outputs, dim=-1)
                word_id = torch.multinomial(probs, 1).item()
            else:
                word_id = outputs.argmax(dim=-1).item()

            word_ids.append(word_id)
            if word_id == vocab.word2idx.get("<end>", -1):
                break

            inputs = self.embed(
                torch.tensor([[word_id]], device=features.device)
            )                                              # (1, 1, E)

        words = [
            vocab.idx2word[i]
            for i in word_ids
            if vocab.idx2word.get(i) not in ("<start>", "<end>", "<pad>")
        ]
        return words


# ─────────────────────────────────────────────────────────────────────────────
# Full Model wrapper
# ─────────────────────────────────────────────────────────────────────────────
class ImageCaptioner(nn.Module):
    """Combines encoder + decoder into a single module."""

    def __init__(
        self,
        embed_size: int = 256,
        hidden_size: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 2,
    ):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs  = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocab, max_len=25, temperature=1.0):
        features = self.encoder(image)
        return self.decoder.sample(features, vocab, max_len, temperature)
