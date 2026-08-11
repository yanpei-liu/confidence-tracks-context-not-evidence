"""M0 — environment + model smoke test.

Verifies, on a GPU node, that:
  1. Qwen3-VL-8B-Instruct loads;
  2. each option letter is a single token and the four are distinct;
  3. **visual tokens per frame is constant in the number of frames.** Qwen3-VL's video
     processor otherwise smart-resizes to a total pixel budget, which would make per-frame
     resolution a decreasing function of window length — i.e. of Δt. If that were left on,
     the whole sweep would be measuring resolution, not memory. This check is the one that
     catches it;
  4. the option-letter softmax is a valid distribution;
  5. the model actually reads the frames — a synthetic MCQ whose answer is visible only in
     the video is answered correctly, and the answer tracks the frames when they change.

Artifacts: artifacts/m0_env_report.json, artifacts/m0_smoke.jsonl
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time

from PIL import Image, ImageDraw

from .config import ARTIFACTS, INPUT_RES, MODEL_HUB_ID, MODEL_ID, visual_tokens_per_frame

COLOURS = {
    "a": ("red", (205, 25, 25)),
    "b": ("green", (25, 150, 25)),
    "c": ("blue", (25, 25, 205)),
    "d": ("yellow", (225, 210, 30)),
}


def synth_frame(rgb: tuple[int, int, int], res: int) -> Image.Image:
    im = Image.new("RGB", (res, res), rgb)
    d = ImageDraw.Draw(im)
    d.rectangle([res // 8, res // 8, res - res // 8, res - res // 8],
                outline=(255, 255, 255), width=max(4, res // 40))
    return im


def main() -> None:
    import torch

    from .scoring import LetterScorer

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available() else None,
        "model_hub_id": MODEL_HUB_ID,
        "model_path": MODEL_ID,
        "input_res": INPUT_RES,
        "hostname": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
    }
    for mod in ("transformers", "accelerate", "PIL", "numpy"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception as e:  # noqa: BLE001
            env[mod] = f"ERR {e}"

    t0 = time.time()
    sc = LetterScorer.load()
    env["load_s"] = round(time.time() - t0, 1)
    env["letter_token_ids"] = sc.letter_ids

    records, failures = [], []

    # (3) tokens-per-frame must not depend on frame count
    tpf = {}
    for n_frames in (8, 64, 256):
        frames = [synth_frame((40, 40, 40), INPUT_RES) for _ in range(n_frames)]
        t0 = time.time()
        r = sc.score(frames, "How many white rectangles are visible?",
                     {"a": "one", "b": "two", "c": "three", "d": "four"})
        r.update(test="token_accounting", n_frames=n_frames, input_res=INPUT_RES,
                 expected_visual_tokens=int(n_frames * visual_tokens_per_frame()),
                 wall_s=round(time.time() - t0, 2))
        tpf[n_frames] = r["n_visual_tokens"] / n_frames
        records.append(r)
        s = sum(r["letter_probs"].values())
        if abs(s - 1.0) > 1e-4:
            failures.append(f"letter probs sum to {s} at n_frames={n_frames}")
    env["visual_tokens_per_frame"] = tpf
    env["expected_tokens_per_frame"] = visual_tokens_per_frame()
    if len(set(round(v, 3) for v in tpf.values())) != 1:
        failures.append(f"tokens/frame varies with frame count: {tpf} — resolution is not pinned")
    if abs(list(tpf.values())[0] - visual_tokens_per_frame()) > 0.01:
        failures.append(f"tokens/frame {tpf} != expected {visual_tokens_per_frame()}")

    # (3b) the "<t seconds>" markers Qwen3-VL writes into the prompt must reflect real
    # stream time. The default (no metadata) assumes fps=24 and would describe a 60-minute
    # window as ~16 seconds — the model would be told the evidence was moments ago.
    import re

    frames = [synth_frame((40, 40, 40), INPUT_RES) for _ in range(12)]
    inp = sc.build_inputs(frames, "q?", {c: "x" for c in "abcd"})
    dec = sc.processor.tokenizer.decode(inp["input_ids"][0].tolist())
    stamps = [float(x) for x in re.findall(r"<([\d.]+) seconds>", dec)]
    env["timestamp_markers_12_frames"] = stamps
    # 12 frames at 0.2 fps span 55 s; markers are pair-averaged, so 2.5, 12.5, ... 52.5
    if not stamps or max(stamps) < 40.0:
        failures.append(f"frame timestamps look wrong for a 55 s window: {stamps}")

    offs = [0, 1, 2, 60, 61, 62]
    inp2 = sc.build_inputs([synth_frame((40, 40, 40), INPUT_RES) for _ in offs],
                           "q?", {c: "x" for c in "abcd"}, frame_offsets=offs)
    dec2 = sc.processor.tokenizer.decode(inp2["input_ids"][0].tolist())
    stamps2 = [float(x) for x in re.findall(r"<([\d.]+) seconds>", dec2)]
    env["timestamp_markers_nonuniform"] = stamps2
    if not stamps2 or max(stamps2) < 290.0:
        failures.append(f"non-uniform frame offsets not honoured: {stamps2} (expect ~307 s max)")

    # (5) does the answer follow the frames?
    q = "What is the dominant colour of the scene?"
    ch = {k: v[0] for k, v in COLOURS.items()}
    correct = 0
    for gold, (_, rgb) in COLOURS.items():
        frames = [synth_frame(rgb, INPUT_RES) for _ in range(16)]
        r = sc.score(frames, q, ch)
        r.update(test="frames_are_read", gold=gold.upper(), correct=r["pred"] == gold.upper())
        correct += r["correct"]
        records.append(r)
    env["frames_are_read_score"] = f"{correct}/4"
    if correct < 3:
        failures.append(f"model may not be reading the video: {correct}/4 on a trivial probe")

    # throughput estimate for planning the sweep
    env["seconds_per_eval"] = {str(r["n_frames"]): r["wall_s"]
                               for r in records if r["test"] == "token_accounting"}
    env["failures"] = failures

    with (ARTIFACTS / "m0_smoke.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (ARTIFACTS / "m0_env_report.json").write_text(json.dumps(env, indent=1))

    print(json.dumps(env, indent=1))
    for r in records:
        print(f"  {r['test']:18s} n={r.get('n_frames','-'):>4} "
              f"vis_tok={r['n_visual_tokens']} (exp {r.get('expected_visual_tokens','-')}) "
              f"pred={r['pred']} conf={r['confidence']:.3f} "
              f"{'gold=' + r['gold'] if 'gold' in r else ''}")
    if failures:
        print("\nM0 FAILURES:")
        for x in failures:
            print("  -", x)
        sys.exit(1)
    print("\nM0 OK")


if __name__ == "__main__":
    main()
