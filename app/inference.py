"""One inference at a time across STT and the local LLM on the same GPU."""

from threading import RLock

GPU_INFERENCE_LOCK = RLock()
