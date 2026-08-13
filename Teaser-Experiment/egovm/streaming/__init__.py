"""Streaming KV-cache backend for the Δt sweep (EXTENDING.md, modules X0–X9).

The full-reprocess sweep (M0–M9) re-encodes each window from scratch. This package runs the
same design against a bounded, streaming KV cache over the same Qwen3-VL-8B backbone, so the
streaming-vs-full-reprocess comparison is within-backbone by construction.

Nothing here is imported by M0–M9; the two backends never share a process.
"""
