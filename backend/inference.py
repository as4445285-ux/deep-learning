"""
inference.py  – Real image captioning inference using Hugging Face BLIP.

Usage (command-line)
--------------------
    python backend/inference.py --image path/to/image.jpg

Usage (import)
--------------
    from inference import CaptionPredictor
    predictor = CaptionPredictor()
    caption = predictor.predict("path/to/image.jpg")
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
except Exception as exc:  # pragma: no cover
    BlipForConditionalGeneration = None
    BlipProcessor = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

# Allow running from either project root or backend/
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


class CaptionPredictor:
    """Loads a pretrained BLIP captioning model from Hugging Face."""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = "auto"):
        self.model_name = model_name
        self._error = None
        self._ready = False
        self._demo = False

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if BlipProcessor is None or BlipForConditionalGeneration is None:
            self._error = f"transformers could not be imported: {TRANSFORMERS_IMPORT_ERROR}"
            raise RuntimeError(self._error)

        self.cache_dir = os.path.join(_HERE, ".hf_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            self.processor = BlipProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
            self.model.to(self.device)
            self.model.eval()
            self._ready = True
            print(f"[inference] BLIP model loaded ✔ ({self.model_name}) on {self.device}")
        except Exception as exc:
            self._error = str(exc)
            self._ready = False
            print(f"[inference] BLIP model failed to load: {exc}")
            raise

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def predict(self, image_path: str, temperature: float = 1.0, max_len: int = 25) -> str:
        pil = Image.open(image_path).convert("RGB")
        return self.predict_from_pil(pil, temperature=temperature, max_len=max_len)

    def predict_from_pil(
        self,
        pil_image: Image.Image,
        temperature: float = 1.0,
        max_len: int = 25,
    ) -> str:
        if not self._ready:
            raise RuntimeError(self._error or "Caption model is not ready.")

        device_inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = device_inputs["pixel_values"].to(self.device)

        try:
            generation_kwargs = {
                "pixel_values": pixel_values,
                "max_new_tokens": max_len,
                "do_sample": temperature != 1.0,
            }
            if temperature != 1.0:
                generation_kwargs["temperature"] = max(0.1, float(temperature))

            generated_ids = self.model.generate(**generation_kwargs)
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return caption if caption else "Unable to generate a caption for this image."
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP Image Caption Generation")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=25)
    args = parser.parse_args()

    predictor = CaptionPredictor(model_name=args.model)
    t0 = time.time()
    caption = predictor.predict(args.image, args.temperature, args.max_len)
    elapsed = (time.time() - t0) * 1000

    print(f"\n📷  Image   : {args.image}")
    print(f"💬  Caption : {caption}")
    print(f"⏱  Time    : {elapsed:.1f} ms\n")
