"""X0 — verify the streaming adapter imports and pin the upstream it depends on.

No GPU. Emits `artifacts/x0_backend_report.json`:

  * sha256 of every upstream file the adapter imports, so a silent edit upstream is caught
    rather than absorbed (there is no commit to pin — the checkout is not a git repository);
  * the shims actually applied;
  * the licence status, which is why the source is imported rather than vendored;
  * the token cost model, probed from the processor on CPU.

Run: python -m egovm.streaming.x0_report
"""
from __future__ import annotations

import hashlib
import json
import sys

from ..config import INPUT_RES, PATCH_PX
from . import sconf


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = sconf.EGOSTREAM_ROOT
    rep: dict = {
        "egostream_root": str(root),
        "is_git_checkout": (root.parent / ".git").exists(),
        "license_file_present": any(
            (root.parent / n).exists() for n in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING")
        ),
        "vendored_into_repo": False,
        "vendoring_note": (
            "Upstream ships no LICENSE and is not a git checkout, so there is no grant "
            "permitting redistribution and no commit to pin. The adapter imports it from "
            "$EGOVM_EGOSTREAM and records the checksums below instead."
        ),
    }

    missing = [d for d in sconf.UPSTREAM_DEPS if not (root / d).exists()]
    rep["missing_upstream_files"] = missing
    rep["upstream_sha256"] = {d: sha256(root / d) for d in sconf.UPSTREAM_DEPS if (root / d).exists()}

    # --- import path ---
    failures = []
    try:
        from .backend import StreamingScorer, cache_config, import_egostream, policy_string

        import_egostream()
        rep["shims"] = StreamingScorer.shims()
        cfg = cache_config()
        rep["cache_policy"] = policy_string(cfg)
        rep["cache_config"] = {
            k: cfg[k] for k in
            ("mutable_token_size", "use_sink_tokens", "sink_sys_prompt", "use_hat_tokens",
             "hat_num_frames", "intra_frame_pruning", "inter_frame_pruning", "enable_disk_offload")
            if k in cfg
        }
        rep["cache_config"]["retrieve"] = cfg.get("retrieve", {})
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        failures.append(f"import: {type(e).__name__}: {e}")

    # --- cost model, probed rather than asserted ---
    try:
        from PIL import Image
        from transformers import AutoProcessor

        ip = AutoProcessor.from_pretrained(sconf.MODEL_ID).image_processor
        probe = {}
        for res in (INPUT_RES, 448, 896):
            w, h = (res, res) if res != 896 else (896, 512)
            grid = ip(images=[Image.new("RGB", (w, h), (90, 90, 90))],
                      return_tensors="pt").image_grid_thw
            probe[f"{w}x{h}"] = int(grid.prod(dim=-1).sum()) // 4
        rep["tokens_per_frame_probe"] = probe
        expect = sconf.visual_tokens_per_frame(INPUT_RES)
        got = probe[f"{INPUT_RES}x{INPUT_RES}"]
        if got != expect:
            failures.append(f"tokens/frame at {INPUT_RES}: expected {expect}, probed {got}")
        rep["tokens_per_frame"] = got
        rep["cost_model"] = f"(res//{PATCH_PX})**2  [image path: no temporal merge]"
        rep["full_reprocess_tokens_per_frame"] = expect // 2
        rep["ratio_vs_full_reprocess"] = 2.0
    except Exception as e:  # noqa: BLE001
        failures.append(f"cost probe: {type(e).__name__}: {e}")

    rep["live_caps"] = {b: sconf.live_cap(b) for b in sconf.BUDGETS}
    rep["failures"] = failures
    rep["ok"] = not failures and not missing

    sconf.X0_REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nX0 {'OK' if rep['ok'] else 'FAILED'} -> {sconf.X0_REPORT}")
    if not rep["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
