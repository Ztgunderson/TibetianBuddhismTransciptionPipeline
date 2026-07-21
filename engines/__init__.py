"""Inference-engine adapters behind one interface, so the benchmark runner treats the
choice of engine as just another axis of the sweep (like model or quant level).

Engine A — `ggml.py`  — whisper.cpp built with CUDA. Primary; owns the FP16/Q8_0/Q4_0
                        ladder (Loop 1) and imatrix quantization (Loop 2).
Engine B — `ct2.py`   — faster-whisper / CTranslate2 INT8. Second arm; owns INT8
                        calibration-cache quantization (Loop 2). Built only after Loop 1.

Both conform to `Engine` (see `base.py`). Nothing else in the codebase imports a concrete
engine directly — `get_engine(name, **cfg)` is the only entry point.
"""

from engines.base import Engine, Transcription, get_engine

__all__ = ["Engine", "Transcription", "get_engine"]
