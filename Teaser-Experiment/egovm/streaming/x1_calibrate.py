"""X1 — calibrate the streaming backend on a GPU.

The analogue of M0. Seven checks, each of which catches a failure that is otherwise silent:

  1. **tokens/frame constant in batch size** — the resolution-confound regression test. If the
     processor ever starts fitting frames to a total pixel budget, per-frame resolution becomes
     a function of window length, i.e. of Δt, and the sweep measures resolution instead of
     memory.
  2. **cache saturates at sink + mutable + hat**, separately for each budget. This is the
     context-length axis; if the budgets do not actually separate, X6 measures nothing.
  3. **deepcopy leaves the master intact** — without it, multi-checkpoint querying silently
     contaminates every later Δt.
  4. option letters single-token.
  5. **synthetic colour MCQ** — the model demonstrably reads frames through the image path.
  6. **timestamp probe** — markers-on beats markers-off on an elapsed-time question. The image
     path carries no clock of its own, so if injection does not reach the model the experiment
     has no independent variable.
  7. **throughput and disk volume**, with offload+retrieval on and off. Ingestion writes every
     layer's evicted KV; retrieval reads layer 0's back. This decides the cost model and
     whether `offload_dir` must be node-local.

Artifact: artifacts/x1_calibration.json
Run: python -m egovm.streaming.x1_calibrate
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ..config import INPUT_RES
from ..m3_frames import frame_path
from . import sconf
from .backend import StreamingScorer, cache_config, policy_string

COLOURS = {"a": ("red", (205, 25, 25)), "b": ("green", (25, 150, 25)),
           "c": ("blue", (25, 25, 205)), "d": ("yellow", (225, 210, 30))}


def synth(rgb, res=INPUT_RES) -> Image.Image:
    im = Image.new("RGB", (res, res), rgb)
    d = ImageDraw.Draw(im)
    d.rectangle([res // 8, res // 8, res - res // 8, res - res // 8],
                outline=(255, 255, 255), width=max(4, res // 40))
    return im


def bank_frames(start_gid: int, n: int) -> list[Image.Image]:
    out = []
    for g in range(start_gid, start_gid + n):
        with Image.open(frame_path(g)) as im:
            out.append(im.convert("RGB").copy())
    return out


def dir_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> None:
    rep: dict = {"checks": [], "failures": []}

    def ok(name, **kw):
        rep["checks"].append({"check": name, "ok": True, **kw})
        print(f"[OK]   {name} {kw}", flush=True)

    def bad(name, err, **kw):
        rep["failures"].append({"check": name, "error": str(err)[:1500], **kw})
        print(f"[FAIL] {name}: {err}", flush=True)

    rep["env"] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available() else None,
        "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
        "input_res": INPUT_RES,
        "offload_root": str(sconf.OFFLOAD_ROOT),
    }

    t0 = time.time()
    sc = StreamingScorer.load()
    rep["env"]["load_s"] = round(time.time() - t0, 1)
    rep["shims"] = sc.shims()

    # ---- (4) letters ----
    if len(set(sc.letter_ids)) == len(sc.letters):
        ok("single_token_letters", ids=sc.letter_ids)
    else:
        bad("single_token_letters", sc.letter_ids)

    off = sconf.OFFLOAD_ROOT / "x1"
    shutil.rmtree(off, ignore_errors=True)

    # ---- (1) tokens per frame must not depend on batch size ----
    try:
        tpf = {}
        cache = sc.new_cache("default", offload_dir=off / "tpf")
        base = cache.get_seq_length()
        rep["priming_tokens"] = base
        for n in (4, 16, 64):
            frames = [synth((40, 40, 40)) for _ in range(n)]
            times = [i * sconf.GRID_S for i in range(n)]
            st = sc.ingest(cache, frames, times, timestamps=True)
            tpf[n] = st["tokens_per_frame"]
        vals = set(tpf.values())
        expect = sconf.visual_tokens_per_frame(INPUT_RES)
        if len(vals) == 1 and vals.pop() == expect:
            ok("tokens_per_frame_constant", per_batch=tpf, expected=expect)
        else:
            bad("tokens_per_frame_constant", f"{tpf}, expected {expect}")
        StreamingScorer.release(cache)
        del cache
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); bad("tokens_per_frame_constant", e)

    # ---- (2) saturation cap per budget ----
    caps = {}
    try:
        for label in ("small", "default", "large"):
            cache = sc.new_cache(label, offload_dir=off / f"cap_{label}")
            n_needed = (sconf.live_cap(label) // sconf.TOKENS_PER_FRAME) + 80
            seen = []
            for i in range(0, n_needed, 16):
                k = min(16, n_needed - i)
                st = sc.ingest(cache, [synth((60, 90, 120)) for _ in range(k)],
                               [(i + j) * sconf.GRID_S for j in range(k)], timestamps=True)
                seen.append(st["live_tokens"])
            caps[label] = {"observed_max": max(seen), "final": seen[-1],
                           "expected_cap": sconf.live_cap(label),
                           "frames_ingested": cache.egovm_frames}
            StreamingScorer.release(cache)
            del cache
            torch.cuda.empty_cache()
        # the cap is an upper bound the layer enforces after each batch; allow one batch of slack
        slack = 16 * sconf.TOKENS_PER_FRAME
        bad_b = [b for b, v in caps.items() if v["final"] > v["expected_cap"] + slack]
        separated = caps["large"]["final"] > 2 * caps["default"]["final"] > 2 * caps["small"]["final"]
        if not bad_b and separated:
            ok("budget_saturation", caps=caps)
        else:
            bad("budget_saturation", f"over-cap={bad_b} separated={separated}", caps=caps)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); bad("budget_saturation", e, caps=caps)

    # ---- (5) the model reads frames, and (3) the master survives a query ----
    q_colour = "What is the dominant colour of the scene in the video?"
    ch_colour = {k: v[0] for k, v in COLOURS.items()}
    try:
        cache = sc.new_cache("default", offload_dir=off / "colour")
        for gold, (_name, rgb) in COLOURS.items():
            for i in range(0, 32, 16):
                sc.ingest(cache, [synth(rgb) for _ in range(16)],
                          [(i + j) * sconf.GRID_S for j in range(16)], timestamps=True)
            before = cache.get_seq_length()
            r = sc.query(cache, q_colour, ch_colour)
            after = cache.get_seq_length()
            rep.setdefault("colour_probe", []).append(
                {"gold": gold.upper(), "pred": r["pred"], "confidence": r["confidence"],
                 "letter_probs": r["letter_probs"], "master_intact": before == after})
            if before != after:
                bad("master_unchanged_after_query", f"{before} -> {after}")
            # reset between colours: the cache is a memory, so a stale colour would leak
            StreamingScorer.release(cache); del cache
            cache = sc.new_cache("default", offload_dir=off / "colour")
        StreamingScorer.release(cache); del cache

        probe = rep["colour_probe"]
        n_correct = sum(p["pred"] == p["gold"] for p in probe)
        if all(p["master_intact"] for p in probe):
            ok("master_unchanged_after_query", n=len(probe))
        if n_correct >= 3:
            ok("model_reads_frames", correct=f"{n_correct}/{len(probe)}")
        else:
            bad("model_reads_frames", f"only {n_correct}/{len(probe)} synthetic colours correct")
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); bad("model_reads_frames", e)

    # ---- (6) timestamp probe: does the injected clock reach the model? ----
    # An earlier version ingested one 600 s window and asked how long it was. That is not
    # diagnostic: markers-on and markers-off both answered "10 minutes" at p = 0.999, because
    # 10 minutes is also the model's prior for a stream of that many frames. Nothing was
    # learned about whether the markers were read.
    #
    # So: hold the *frames* fixed (120 real frames, always) and vary only the injected clock,
    # labelling the same footage as spanning 1 min, 10 min or 1 h. If the answer tracks the
    # markers, the channel reaches the model — nothing else in the input changed. Markers-off
    # is the control: with no clock in the context there is nothing to track.
    q_time = ("Based on the timestamps in the video stream you have observed, approximately how "
              "much video time has elapsed since the stream began?")
    ch_time = {"a": "About 1 minute", "b": "About 10 minutes",
               "c": "About 1 hour", "d": "About 4 hours"}
    SPANS = [(60.0, "A"), (600.0, "B"), (3600.0, "C")]
    NF = 120
    try:
        tprobe = {"on": {}, "off": {}}
        for mode in (True, False):
            key = "on" if mode else "off"
            for span, gold in SPANS:
                cache = sc.new_cache("default", offload_dir=off / f"ts_{key}_{int(span)}")
                for i in range(0, NF, 16):
                    k = min(16, NF - i)
                    times = [(i + j) * span / (NF - 1) for j in range(k)]
                    sc.ingest(cache, bank_frames(3000 + i, k), times, timestamps=mode)
                r = sc.query(cache, q_time, ch_time)
                tprobe[key][int(span)] = {
                    "gold": gold, "pred": r["pred"], "p_gold": r["letter_probs"][gold],
                    "letter_probs": r["letter_probs"]}
                StreamingScorer.release(cache); del cache
                torch.cuda.empty_cache()
        rep["timestamp_probe"] = tprobe
        on_preds = [v["pred"] for v in tprobe["on"].values()]
        off_preds = [v["pred"] for v in tprobe["off"].values()]
        n_on_correct = sum(v["pred"] == v["gold"] for v in tprobe["on"].values())
        rep["timestamp_probe"]["summary"] = {
            "on_preds": on_preds, "off_preds": off_preds,
            "on_distinct": len(set(on_preds)), "off_distinct": len(set(off_preds)),
            "on_correct": n_on_correct,
        }
        # the gate: with the frames held fixed, the answer must move with the injected clock
        if len(set(on_preds)) >= 2 and n_on_correct >= 2:
            ok("timestamps_readable", on_preds=on_preds, off_preds=off_preds,
               on_correct=f"{n_on_correct}/3")
        else:
            bad("timestamps_readable",
                f"frames held fixed, clock varied over {[s for s, _ in SPANS]}s: "
                f"markers-on predicted {on_preds} ({n_on_correct}/3 correct), "
                f"markers-off predicted {off_preds}. The injected clock does not move the answer.")
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); bad("timestamps_readable", e)

    # ---- (7) throughput and disk volume ----
    q_real = "What was the camera wearer doing most recently?"
    ch_real = {"a": "Sitting at a desk", "b": "Walking outdoors",
               "c": "Cooking in a kitchen", "d": "Driving a car"}
    try:
        thr = {}
        for tag, (offload, retrieve) in {"offload_retrieve": (True, True),
                                         "no_offload": (False, False)}.items():
            d = off / f"thr_{tag}"
            shutil.rmtree(d, ignore_errors=True)
            torch.cuda.reset_peak_memory_stats()
            cache = sc.new_cache("default", offload_dir=d, enable_offload=offload,
                                 retrieve=retrieve)
            n = 256
            t_start = time.time()
            warm_s = None
            for i in range(0, n, 16):
                sc.ingest(cache, bank_frames(5000 + i, 16),
                          [(i + j) * sconf.GRID_S for j in range(16)], timestamps=True)
                if i == 48:  # skip the unsaturated head when reporting steady state
                    warm_s = time.time() - t_start
                    warm_n = i + 16
            total_s = time.time() - t_start
            steady = (total_s - warm_s) / (n - warm_n)
            tq = time.time()
            r = sc.query(cache, q_real, ch_real, retrieve=retrieve)
            query_s = time.time() - tq
            thr[tag] = {
                "frames": n,
                "s_per_frame_all": round(total_s / n, 4),
                "s_per_frame_steady": round(steady, 4),
                "query_s": round(query_s, 3),
                "clone_s": r["clone_s"],
                "retrieved": r["retrieved"],
                "live_tokens": r["live_tokens"],
                # tokens the layers report having evicted to the store, vs bytes actually on
                # disk. The first run measured 0 bytes despite ~16k tokens over the cap, so
                # both are recorded: they answer different questions (is the method doing what
                # it says, and does offload_dir need node-local scratch).
                "offloaded_tokens": int(cache.get_offloaded_token_count())
                if hasattr(cache, "get_offloaded_token_count") else None,
                "presented_tokens": n * sconf.TOKENS_PER_FRAME,
                "offload_bytes": dir_bytes(d),
                "offload_gb_per_733_frames": round(dir_bytes(d) / n * 733 / 2**30, 2),
                "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            StreamingScorer.release(cache); del cache
            shutil.rmtree(d, ignore_errors=True)
            torch.cuda.empty_cache()
        rep["throughput"] = thr
        ok("throughput", **{k: v["s_per_frame_steady"] for k, v in thr.items()})
        # project the programme cost from the measured steady-state rate
        sf = thr["offload_retrieve"]["s_per_frame_steady"]
        rep["cost_projection_gpu_h"] = {
            "x5_main_273622f": round(273622 * sf / 3600, 1),
            "x6_budget_547244f": round(547244 * sf / 3600, 1),
            "x7_prior_109182f": round(109182 * sf / 3600, 1),
            "x7_tokmatch_144404f": round(144404 * sf / 3600, 1),
            "x8_nots_273622f": round(273622 * sf / 3600, 1),
            "s_per_frame_used": sf,
        }
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc(); bad("throughput", e)

    shutil.rmtree(off, ignore_errors=True)
    rep["cache_policy"] = policy_string(cache_config())
    rep["ok"] = not rep["failures"]
    sconf.X1_REPORT.write_text(json.dumps(rep, indent=2))
    print("\n=== X1 " + ("OK" if rep["ok"] else "FAILED") + f" -> {sconf.X1_REPORT}")
    print(json.dumps({k: v for k, v in rep.items() if k != "upstream_sha256"}, indent=2)[:4000])
    if not rep["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
