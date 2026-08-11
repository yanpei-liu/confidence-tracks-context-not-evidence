"""M3 — build the global 0.2 fps frame bank.

Every evaluation window is a slice of one global frame grid, so frames are decoded exactly
once and then re-read from disk by every (question, Δt, arm) that covers them. Without this,
the sweep would re-decode ~570k frame-samples instead of ~37.6k.

Grid: frame ``g`` is at stream second ``g * 5.0``. Chunk ``i`` covers stream ``[30i, 30i+30)``
and therefore contributes exactly frames ``6i .. 6i+5`` at in-chunk offsets 0,5,10,15,20,25 s.

Keyframes in these files are 12.5 s apart, so seeking cannot hit a 5 s grid — each chunk is
fully decoded once by ffmpeg and 6 frames are written.

Artifacts:
  $DATA/frames/<bucket>/<g>.jpg
  artifacts/m3_frame_index.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import (
    ARTIFACTS, CHUNK_NOMINAL_S, FRAME_H, FRAME_W, FRAMES, SAMPLE_FPS, VIDEO_DIR,
)
from .m2_stream_index import StreamIndex

GRID_S = 1.0 / SAMPLE_FPS  # 5.0 s
PER_CHUNK = int(round(CHUNK_NOMINAL_S * SAMPLE_FPS))  # 6
BUCKET = 1000


def frame_path(g: int):
    return FRAMES / f"{g // BUCKET:05d}" / f"{g:07d}.jpg"


def extract_chunk(args) -> tuple[int, list[int], str]:
    """Decode one chunk to PER_CHUNK jpgs. Returns (chunk_idx, missing_gids, error)."""
    i, rel_path = args
    g0 = i * PER_CHUNK
    targets = [g0 + k for k in range(PER_CHUNK)]
    if all(frame_path(g).exists() and frame_path(g).stat().st_size > 0 for g in targets):
        return i, [], ""
    outdir = frame_path(g0).parent
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / f".tmp_{i}_%02d.jpg"
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-threads", "2",
        "-i", str(VIDEO_DIR / rel_path),
        # fps=0.2 emits the frame nearest each of 0,5,10,15,20,25 s into the chunk, so
        # frame g lands at stream second 5g exactly as the index defines it.
        "-vf", f"fps={SAMPLE_FPS},scale={FRAME_W}:{FRAME_H}",
        "-frames:v", str(PER_CHUNK), "-q:v", "3", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except Exception as e:  # noqa: BLE001 - record and continue; a few bad chunks are tolerable
        return i, targets, f"{type(e).__name__}: {e}"[:200]
    missing = []
    for k, g in enumerate(targets):
        src = outdir / f".tmp_{i}_{k+1:02d}.jpg"
        if src.exists() and src.stat().st_size > 0:
            os.replace(src, frame_path(g))
        else:
            missing.append(g)
    for k in range(PER_CHUNK + 2):
        stale = outdir / f".tmp_{i}_{k+1:02d}.jpg"
        if stale.exists():
            stale.unlink()
    return i, missing, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N chunks")
    a = ap.parse_args()

    idx = StreamIndex.load()
    jobs = [(i, p) for i, p in enumerate(idx.paths)]
    if a.limit:
        jobs = jobs[: a.limit]
    n_expected = len(jobs) * PER_CHUNK
    print(f"M3: {len(jobs)} chunks -> {n_expected} frames, {a.workers} workers", flush=True)

    missing, errors, done = [], {}, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(extract_chunk, j) for j in jobs]
        for fut in as_completed(futs):
            i, miss, err = fut.result()
            missing.extend(miss)
            if err:
                errors[idx.paths[i]] = err
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)} chunks, {len(missing)} missing", flush=True)

    doc = {
        "grid_s": GRID_S,
        "sample_fps": SAMPLE_FPS,
        "per_chunk": PER_CHUNK,
        "n_chunks": len(jobs),
        "n_frames_expected": n_expected,
        "n_frames_missing": len(missing),
        "missing_gids": sorted(missing)[:5000],
        "frame_w": FRAME_W,
        "frame_h": FRAME_H,
        "errors": errors,
        "frames_root": str(FRAMES),
    }
    out = ARTIFACTS / "m3_frame_index.json"
    out.write_text(json.dumps(doc, indent=1))

    got = n_expected - len(missing)
    print(f"M3 {'OK' if not missing else 'PARTIAL'}: {got}/{n_expected} frames "
          f"({100*got/n_expected:.2f}%), {len(errors)} chunk errors -> {out}", flush=True)
    if len(missing) > 0.02 * n_expected:
        sys.exit(f"too many missing frames: {len(missing)}")


if __name__ == "__main__":
    main()
