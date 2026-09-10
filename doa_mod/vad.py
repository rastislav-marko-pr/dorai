"""vad — Neural VAD ONNX wrapper for speech activity detection.

Wraps the neural VAD ONNX model (vad.onnx) to provide per-chunk speech
probability. Used by auto_calibrate and live_doa to gate DOA estimation
on speech activity: a bearing from a silent or noise-only frame points at
the room's noise field, not at the talker.

The model expects 16 kHz mono audio in 512-sample (32 ms) chunks.
Higher sample rates are downsampled internally. The LSTM state is carried
across chunks for streaming operation.

IMPORTANT: The VAD model requires a 64-sample CONTEXT prefix prepended to
each 512-sample chunk (so the ONNX model actually receives 576 samples).
The context is the tail of the previous chunk, maintained automatically.
Without it the model outputs near-zero probabilities for all audio.
"""
from __future__ import annotations

import os

import numpy as np

# Neural VAD constants
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 512        # 32 ms at 16 kHz — the NEW samples per call
_CONTEXT_SAMPLES = 64       # rolling context prepended to each chunk
_STATE_DIM = 128            # LSTM hidden size in v5


def _default_model_path() -> str:
    """Path to vad.onnx next to this file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vad.onnx')


class NeuralVAD:
    """Streaming Neural VAD via ONNX Runtime.

    Usage::

        vad = NeuralVAD()                    # loads vad.onnx
        prob = vad.process_chunk(audio_512)   # returns float in [0, 1]
        vad.reset()                           # clear LSTM state

    For longer buffers, use process_buffer() which splits into 512-sample
    chunks and returns per-chunk probabilities plus an overall decision.

    The 64-sample context window is managed automatically: each call to
    process_chunk() saves the last 64 samples for the next call, matching
    the official model's OnnxWrapper calling convention.
    """

    def __init__(self, model_path: str | None = None):
        import onnxruntime as ort
        path = model_path or _default_model_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Neural VAD model not found at {path}. "
                f"Place vad.onnx in the doa/ directory.")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            path, providers=['CPUExecutionProvider'], sess_options=opts)
        self.reset()

    def reset(self):
        """Clear LSTM state and context — call between unrelated audio segments."""
        self._state = np.zeros((2, 1, _STATE_DIM), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def process_chunk(self, audio: np.ndarray) -> float:
        """Run one 512-sample chunk at 16 kHz. Returns speech probability.

        The LSTM state and 64-sample context are updated in-place for streaming.
        The model actually receives 576 samples (64 context + 512 new).
        """
        x = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        # Prepend rolling context (last 64 samples of previous chunk)
        x_with_ctx = np.concatenate([self._context, x], axis=1)
        out, self._state = self.session.run(
            ['output', 'stateN'],
            {
                'input': x_with_ctx,
                'state': self._state,
                'sr': np.array(_SAMPLE_RATE, dtype=np.int64),
            })
        # Save tail as context for next chunk
        self._context = x_with_ctx[:, -_CONTEXT_SAMPLES:]
        return float(out[0, 0])

    def process_buffer(self, audio: np.ndarray, fs: int = _SAMPLE_RATE,
                       threshold: float = 0.5) -> dict:
        """Process a longer buffer, return per-chunk probs and overall decision.

        If fs != 16000, the audio is decimated (simple slicing, not filtered —
        fine for VAD energy detection, not for analysis).

        Returns dict with:
            probs: list of per-chunk speech probabilities
            is_speech: True if any chunk exceeds threshold
            max_prob: highest speech probability in the buffer
            mean_prob: mean speech probability across chunks
        """
        audio = np.asarray(audio, dtype=np.float32).ravel()

        # Downsample to 16 kHz if needed (simple decimation, adequate for VAD)
        if fs != _SAMPLE_RATE:
            ratio = fs / _SAMPLE_RATE
            indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]

        probs = []
        for start in range(0, len(audio) - _CHUNK_SAMPLES + 1, _CHUNK_SAMPLES):
            chunk = audio[start:start + _CHUNK_SAMPLES]
            probs.append(self.process_chunk(chunk))

        if not probs:
            return dict(probs=[], is_speech=False, max_prob=0.0, mean_prob=0.0)

        max_p = float(max(probs))
        mean_p = float(np.mean(probs))

        return dict(
            probs=probs,
            is_speech=max_p >= threshold,
            max_prob=max_p,
            mean_prob=mean_p,
        )

