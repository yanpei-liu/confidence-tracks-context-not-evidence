"""X0 — the streaming-backend adapter.

Wraps the EgoStream FlexibleCache (Forte et al., arXiv 2605.31557) so it presents the same
*measurement* the full-reprocess scorer does: a softmax over option-letter logits at the first
generated position, with no generation.

The upstream checkout ships no LICENSE and is not a git repository, so its source is imported
from outside this tree rather than vendored (EXTENDING.md §1b). Everything in this file is ours;
`_ingest_batch` mirrors the call sequence upstream requires but adds the timestamp channel the
image path otherwise lacks.

Three things here are not cosmetic:

  * **Shims.** Upstream pins torch 2.6 / transformers 4.57; `egovm` is torch 2.13 /
    transformers 5.15. Three one-line adaptations, applied at import time from our code.
  * **Timestamps.** Qwen3-VL interleaves "<t seconds>" markers only on the *video* path.
    EgoStream ingests on the *image* path, so unmodified the model has no elapsed-time input at
    all — and elapsed time is the sweep's independent variable. We interleave the markers
    ourselves. X1 check (6) verifies the model can actually read them.
  * **The master/secondary clone.** Querying mutates the cache (`set_QA(True)` stops pruning but
    `update()` still concatenates the question tokens, and nothing truncates them). Upstream
    describes the clone in the paper's §B.6 and does not ship it. `copy.deepcopy` is it.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..config import INPUT_RES, OPTION_LETTERS
from ..scoring import PROMPT, option_block
from . import sconf

# --------------------------------------------------------------------------------------
# import-time adaptation of the upstream package
# --------------------------------------------------------------------------------------

_SHIMS: list[str] = []


def _install_cv2_stub() -> None:
    """`src/video_sampler.py` imports cv2 at module load and `egovm` has no opencv.

    We never use their sampler — frames come from our own 0.2 fps bank — but the import has to
    succeed. The stub is deliberately minimal: anything that actually decodes video will fail
    loudly rather than silently return garbage.
    """
    if "cv2" in sys.modules:
        return
    import numpy as np
    from PIL import Image as _Image

    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2RGB = 4
    cv2.INTER_AREA = 3
    cv2.cvtColor = lambda a, code: a[:, :, ::-1].copy()
    cv2.resize = lambda a, size, interpolation=None: np.asarray(
        _Image.fromarray(a).resize(size, _Image.LANCZOS)
    )

    class _VideoCapture:  # never used; present only so the import resolves
        def __init__(self, *a, **k):
            raise RuntimeError("cv2 stub: EgoStream's video sampler is not used by egovm")

    cv2.VideoCapture = _VideoCapture
    cv2.CAP_PROP_FPS = 5
    sys.modules["cv2"] = cv2
    _SHIMS.append("cv2_stub")


def _install_cache_shims() -> None:
    from Caches.FlexibleCacheLayer import FlexibleCacheLayer

    # transformers 5.x made CacheLayerMixin.get_max_length abstract; 4.57 did not, so upstream
    # never defined it and the class cannot be instantiated.
    if "get_max_length" in getattr(FlexibleCacheLayer, "__abstractmethods__", ()):
        FlexibleCacheLayer.get_max_length = lambda self: self.get_max_cache_shape()
        FlexibleCacheLayer.__abstractmethods__ = frozenset(
            m for m in FlexibleCacheLayer.__abstractmethods__ if m != "get_max_length"
        )
        _SHIMS.append("FlexibleCacheLayer.get_max_length")

    # transformers 5.x calls get_mask_sizes(query_length: int); 4.57 passed a cache_position
    # tensor and upstream does `.shape[0]` on it.
    def _get_mask_sizes(self, cache_position):
        q = int(cache_position) if isinstance(cache_position, int) else int(cache_position.shape[0])
        return int(self.get_seq_length()) + q, 0

    FlexibleCacheLayer.get_mask_sizes = _get_mask_sizes
    _SHIMS.append("FlexibleCacheLayer.get_mask_sizes")


def import_egostream():
    """Put the upstream package on the path and apply the shims. Idempotent."""
    root = sconf.EGOSTREAM_ROOT
    if not root.is_dir():
        raise SystemExit(
            f"EgoStream checkout not found at {root}. Set EGOVM_EGOSTREAM to its "
            "Online_Cache_Optimization directory."
        )
    _install_cv2_stub()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _install_cache_shims()
    from src.model_operations import _prime_cache_sw
    from src.tensor_ops import (
        compute_projected_embeddings, compute_text_embeddings, compute_visual_embeddings,
        process_text_conversation, replace_placeholder,
    )

    return types.SimpleNamespace(
        prime_cache=_prime_cache_sw,
        compute_visual_embeddings=compute_visual_embeddings,
        compute_projected_embeddings=compute_projected_embeddings,
        process_text_conversation=process_text_conversation,
        compute_text_embeddings=compute_text_embeddings,
        replace_placeholder=replace_placeholder,
    )


def cache_config(budget_label: str = sconf.DEFAULT_BUDGET, offload_dir: Path | None = None,
                 enable_offload: bool = True, retrieve: bool = True) -> dict:
    """The upstream cache config with our budget and paths substituted in.

    Read from `Configs_files/conf_sing_vid.json` rather than restated, so the pruning pipeline
    stays whatever upstream ships and `cache_policy` on our records describes something real.
    """
    cfg = json.loads(sconf.UPSTREAM_CACHE_CONF.read_text())["cache"]
    cfg["mutable_token_size"] = sconf.BUDGETS[budget_label]
    cfg["hat_num_frames"] = sconf.HAT_NUM_FRAMES
    cfg["debug_cache_layout"] = False
    cfg["debug_tracking"] = False
    cfg["enable_disk_offload"] = bool(enable_offload)
    cfg["offload_dir"] = str(offload_dir or (sconf.OFFLOAD_ROOT / f"pid{os.getpid()}"))
    cfg.setdefault("retrieve", {})["enabled"] = bool(retrieve)
    if not enable_offload:
        # the storage step has nothing to write to; leaving it enabled makes the layer build an
        # offload store it never uses
        for step in cfg.get("pruning_pipeline", []):
            if str(step.get("domain")) == "storage":
                step["enabled"] = False
    return cfg


def policy_string(cfg: dict) -> str:
    parts = [s["name"] for s in cfg.get("pruning_pipeline", []) if s.get("enabled")]
    if cfg.get("retrieve", {}).get("enabled"):
        parts.append("rekv_retrieve")
    return "+".join(parts) or "none"


# --------------------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------------------


@dataclass
class StreamingScorer:
    model: object
    processor: object
    ops: object
    letter_ids: list[int]
    device: str
    input_res: int = INPUT_RES
    letters: tuple = tuple(OPTION_LETTERS)
    tokens_per_frame: int = sconf.TOKENS_PER_FRAME
    _stats: dict = field(default_factory=dict)

    # ---- construction ----
    @classmethod
    def load(cls, model_id: str = sconf.MODEL_ID, dtype=torch.bfloat16, attn: str = "sdpa",
             input_res: int = INPUT_RES, letters=OPTION_LETTERS):
        ops = import_egostream()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map="cuda:0", attn_implementation=attn
        )
        model.eval()

        # transformers 5.x get_rope_index requires mm_token_type_ids (0=text/1=image/2=video);
        # 4.57 had no such argument. Derive it from the <|image_pad|> positions. Upstream
        # discards this call's result anyway — it only wants rope_deltas.
        img_pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        _orig = model.model.get_rope_index

        def _rope(input_ids=None, mm_token_type_ids=None, **kw):
            if mm_token_type_ids is None and input_ids is not None:
                mm_token_type_ids = (input_ids == img_pad).int()
            return _orig(input_ids=input_ids, mm_token_type_ids=mm_token_type_ids, **kw)

        model.model.get_rope_index = _rope
        if "get_rope_index" not in _SHIMS:
            _SHIMS.append("get_rope_index.mm_token_type_ids")

        ids = []
        for L in letters:
            enc = processor.tokenizer.encode(L, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(f"letter {L!r} is not a single token: {enc}")
            ids.append(enc[0])
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"option letters collide: {ids}")

        return cls(model, processor, ops, ids, "cuda:0", input_res, tuple(letters),
                   sconf.visual_tokens_per_frame(input_res))

    @staticmethod
    def shims() -> list[str]:
        return list(_SHIMS)

    # ---- cache lifecycle ----
    def new_cache(self, budget_label: str = sconf.DEFAULT_BUDGET, offload_dir: Path | None = None,
                  enable_offload: bool = True, retrieve: bool = True):
        cfg = cache_config(budget_label, offload_dir, enable_offload, retrieve)
        cache = self.ops.prime_cache(
            model=self.model, processor=self.processor, cache_cfg=cfg,
            prompt_path=str(sconf.PRIMING_PROMPT), verbose=False,
        )
        # Upstream sizes the protected recent-frame region as `hat_num_frames * 448`, valid only
        # at its own 896x512. With frame-aware pruning on, `get_hat_size()` recomputes it from
        # the real frame token history, so this only corrects the static fallback — but leaving
        # it wrong mis-sizes capacity accounting if pruning is ever disabled, with no error.
        if hasattr(cache, "set_hat_size"):
            cache.set_hat_size(sconf.HAT_NUM_FRAMES * self.tokens_per_frame)
        cache.egovm_cfg = cfg
        cache.egovm_budget = budget_label
        cache.egovm_frames = 0
        cache.egovm_ingest_s = 0.0
        return cache

    @staticmethod
    def release(cache) -> None:
        if cache is not None and hasattr(cache, "cleanup_offload"):
            try:
                cache.cleanup_offload()
            except Exception:  # noqa: BLE001 — cleanup must never kill a long job
                pass

    # ---- ingestion ----
    def _prepare(self, frames, times_s, timestamps: bool):
        """Build (input_embeddings, visual_pos_mask, frame_token_counts) for one batch.

        `times_s` are window-relative seconds — the same coordinate the full-reprocess sweep's
        "<t seconds>" markers use, so the two regimes present the same clock.
        """
        r = self.input_res
        pil = [im if im.size == (r, r) else im.resize((r, r)) for im in frames]
        vin = self.processor.image_processor(images=pil, return_tensors="pt").to(self.device)
        pixel_values = vin.pixel_values.to(torch.bfloat16)
        grid = vin.image_grid_thw

        vision_out, n_vis = self.ops.compute_visual_embeddings(pixel_values, self.model, grid_thw=grid)
        projected, deepstack = self.ops.compute_projected_embeddings(vision_out)

        content = []
        for t in times_s:
            if timestamps:
                content.append({"type": "text", "text": f"<{t:.0f} seconds>"})
            content.append({"type": "image"})
        conversation = [{"role": "user", "content": content}]

        ids0 = self.ops.process_text_conversation(conversation, self.processor, self.device)
        # compute_text_embeddings splits n_vis evenly across the <|image_pad|> placeholders, so
        # interleaved text is transparent to it as long as every frame is its own placeholder.
        text_emb, expanded = self.ops.compute_text_embeddings(ids0, n_vis, self.processor, self.model)
        embeds, vis_mask = self.ops.replace_placeholder(expanded, text_emb, projected, self.processor)

        per_frame = int(grid.prod(dim=-1)[0].item()) // 4
        counts = [per_frame] * len(pil)
        if sum(counts) != int(n_vis):
            raise RuntimeError(f"frame token accounting: {sum(counts)} != {n_vis}")
        return embeds, vis_mask, counts, expanded, grid, deepstack, per_frame

    @torch.inference_mode()
    def ingest(self, cache, frames: list, times_s: list[float], timestamps: bool = True) -> dict:
        """Append one batch of frames to `cache`. Returns per-batch stats."""
        import time

        embeds, vis_mask, counts, expanded, grid, deepstack, per_frame = self._prepare(
            frames, times_s, timestamps
        )
        seq_len = embeds.shape[1]

        # rope_deltas are computed for their side effect on the model's internal state, exactly
        # as upstream does; the return value is unused.
        self.model.model.get_rope_index(
            input_ids=expanded, image_grid_thw=grid, video_grid_thw=None,
            attention_mask=torch.ones_like(expanded),
        )

        cache.prepare_frame_batch(
            frame_token_counts=counts, visual_pos_mask=vis_mask, appended_seq_len=seq_len,
        )
        past = cache.get_seq_length()
        step = cache.global_token_counter
        attn = torch.ones((1, past + seq_len), dtype=torch.long, device=self.device)
        pos = torch.arange(step, step + seq_len, dtype=torch.long, device=self.device).unsqueeze(0)

        t0 = time.perf_counter()
        self.model.model.language_model(
            inputs_embeds=embeds, past_key_values=cache, attention_mask=attn, position_ids=pos,
            visual_pos_masks=vis_mask, deepstack_visual_embeds=deepstack, use_cache=True,
        )
        cache.finalize_frame_batch()
        dt = time.perf_counter() - t0

        cache.global_token_counter = step + seq_len
        cache.egovm_frames += len(frames)
        cache.egovm_ingest_s += dt
        return {
            "n_frames": len(frames), "tokens_per_frame": per_frame, "appended_seq_len": seq_len,
            "live_tokens": int(cache.get_seq_length()), "ingest_s": dt,
        }

    # ---- query ----
    @torch.inference_mode()
    def query(self, cache, question: str, choices: dict, retrieve: bool = True) -> dict:
        """Score one question against a *clone* of `cache`; the master is left untouched.

        The clone is mandatory, not an optimisation: `update()` concatenates the question tokens
        onto the live KV and nothing removes them, so querying the master directly would
        contaminate every later checkpoint.
        """
        import time

        t0 = time.perf_counter()
        clone = copy.deepcopy(cache)
        clone_s = time.perf_counter() - t0

        cfg = getattr(cache, "egovm_cfg", {})
        rcfg = cfg.get("retrieve", {}) or {}
        retrieved = None
        if retrieve and rcfg.get("enabled") and hasattr(clone, "retrieve_for_qa"):
            # question-conditioned, so it has to run per query and on the clone
            retrieved = clone.retrieve_for_qa(
                method=rcfg.get("method", "rekv"), model=self.model, processor=self.processor,
                question=question,
                top_k_tokens=rcfg.get("top_k_tokens", cfg.get("mutable_token_size", 8192)),
                block_size_frames=rcfg.get("block_size_frames", 1),
                include_live_mutable=rcfg.get("include_live_mutable", True),
                max_offloaded_chunks=rcfg.get("max_offloaded_chunks", None),
                dtype=torch.bfloat16, device=self.model.device, verbose=False,
            )
        clone.set_QA(True)

        text = PROMPT.format(question=question, options=option_block(choices))
        ti = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": text}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )
        input_ids = ti["input_ids"].to(self.device)
        n_q = int(input_ids.shape[1])
        past = clone.get_seq_length()
        step = getattr(clone, "global_token_counter", past)
        attn = torch.ones((1, past + n_q), dtype=torch.long, device=self.device)
        pos = torch.arange(step, step + n_q, device=self.device).unsqueeze(0)

        out = self.model.forward(input_ids=input_ids, past_key_values=clone,
                                 attention_mask=attn, position_ids=pos, use_cache=True)
        logits = out.logits[0, -1].float()
        probs = torch.softmax(logits[self.letter_ids], dim=-1).tolist()
        logps = torch.log_softmax(logits, dim=-1)[self.letter_ids].tolist()
        i = max(range(len(probs)), key=lambda k: probs[k])

        live = int(past)
        del clone
        return {
            "letter_probs": {L: round(p, 6) for L, p in zip(self.letters, probs)},
            "letter_logprobs": {L: round(v, 4) for L, v in zip(self.letters, logps)},
            "pred": self.letters[i],
            "confidence": round(probs[i], 6),
            "entropy": round(-sum(q * math.log(max(q, 1e-12)) for q in probs), 6),
            "n_input_tokens": live + n_q,
            "n_visual_tokens": int(getattr(cache, "egovm_frames", 0) * self.tokens_per_frame),
            "live_tokens": live,
            "clone_s": round(clone_s, 3),
            "retrieved": bool(retrieved and retrieved.get("retrieved")),
        }
