"""ONNX-based text embedder for all-MiniLM-L6-v2.

Handles model download from HuggingFace, tokenization, and L2-normalized
embedding generation. No index management — that's the database's job.
"""

import logging
from pathlib import Path

import numpy as np

from ember.config import HF_REPO, get_model_dir

logger = logging.getLogger(__name__)


class Embedder:
    """Produces L2-normalized 384-dim float32 embeddings via ONNX Runtime."""

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or get_model_dir()
        self._session = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        """Download model files from HuggingFace if not cached locally."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "model.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"

        if model_path.exists() and tokenizer_path.exists():
            return

        from huggingface_hub import hf_hub_download

        if not model_path.exists():
            downloaded = hf_hub_download(
                repo_id=HF_REPO, filename="onnx/model.onnx",
                local_dir=self.model_dir,
            )
            p = Path(downloaded)
            if p != model_path:
                p.rename(model_path)

        if not tokenizer_path.exists():
            downloaded = hf_hub_download(
                repo_id=HF_REPO, filename="tokenizer.json",
                local_dir=self.model_dir,
            )
            p = Path(downloaded)
            if p != tokenizer_path:
                p.rename(tokenizer_path)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._ensure_model()
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            tok.enable_truncation(max_length=256)
            tok.enable_padding(pad_id=0, pad_token="[PAD]", length=256)
            self._tokenizer = tok
        return self._tokenizer

    @property
    def session(self):
        if self._session is None:
            self._ensure_model()
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(self.model_dir / "model.onnx")
            )
        return self._session

    def _run_onnx(self, texts: list[str]) -> np.ndarray:
        """Run ONNX inference. Returns (N, 384) float32, L2-normalized."""
        encoded = [self.tokenizer.encode(t) for t in texts]
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encoded], dtype=np.int64
        )
        token_type_ids = np.zeros_like(input_ids)

        output = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        last_hidden = output[0]  # (N, seq_len, 384)

        # Mean pooling with attention mask
        mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1), last_hidden.shape
        )
        embeddings = np.sum(last_hidden * mask_expanded, axis=1) / np.clip(
            mask_expanded.sum(axis=1), a_min=1e-9, a_max=None
        )

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return (embeddings / norms).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        """Encode text to L2-normalized float32 vector, shape (384,)."""
        return self._run_onnx([text])[0]

    def batch_embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalized float32 vectors, shape (N, 384)."""
        return self._run_onnx(texts)


def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return vec.astype(np.float32).flatten().tobytes()
