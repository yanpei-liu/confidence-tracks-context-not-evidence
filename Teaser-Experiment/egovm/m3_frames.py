"""M3 — build the global 0.2 fps frame bank.

Every evaluation window is a slice of one global frame grid, so frames are decoded exactly
once and then re-read from disk by every (question, Δt, arm) that covers them. Without this
the sweep would re-decode ~1.5M frame-samples instead of ~37.6k.

Grid: frame ``g`` sits at cumulative **stream** second ``g * 5.0``. Which chunk supplies it,
and at what in-chunk offset, comes from the M2 index rather than from an assumed uniform
layout — 8 chunks are clipped short because the recording restarted and they overlap the
next chunk in wall-clock.

Keyframes in these files are 12.5 s apart, so seeking cannot hit a 5 s grid: each chunk is
fully decoded once and the wanted frames are picked by exact source-frame number (20 fps).

Artifacts:
  $DATA/frames/<bucket>/<g>.jpg
  artifacts/m3_frame_index.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import ARTIFACTS, FRAME_H, FRAME_W, FRAMES, SAMPLE_FPS, SRC_FPS, VIDEO_DIR
from .m2_stream_index import StreamIndex

GRID_S = 1.0 / SAMPLE_FPS  # 5.0 s
BUCKET = 1000


def frame_path(g: int):
    return FRAMES / f"{g // BUCKET:05d}" / f"{g:07d}.jpg"


def chunk_frame_jobs(idx: StreamIndex) -> list[tuple[int, str, list[tuple[int, float]]]]:
    """For each chunk: (chunk_i, rel_path, [(gid, in_chunk_offset_s), ...])."""
    jobs = []
    for i, (c0, dur) in enumerate(zip(idx.cum, idx.durs)):
        g_lo = math.ceil(c0 / GRID_S - 1e-9)
        g_hi = math.ceil((c0 + dur) / GRID_S - 1e-9) - 1
        want = [(g, g * GRID_S - c0) for g in range(g_lo, g_hi + 1)]
        want = [(g, off) for g, off in want if 0.0 <= off < dur]
        if want:
            jobs.append((i, idx.paths[i], want))
    return jobs


def extract_chunk(args) -> tuple[int, list[int], str]:
    """Decode one chunk and write its share of the grid. Returns (chunk_i, missing, error)."""
    i, rel_path, want = args
    gids = [g for g, _ in want]
    if all(frame_path(g).exists() and frame_path(g).stat().st_size > 0 for g in gids):
        return i, [], ""

    outdir = frame_path(gids[0]).parent
    outdir.mkdir(parents=True, exist_ok=True)
    # exact source-frame numbers; select is evaluated on the decoded frame index
    nums = [int(round(off * SRC_FPS)) for _, off in want]
    expr = "+".join(f"eq(n\\,{n})" for n in nums)
    tmp = outdir / f".tmp_{i}_%03d.jpg"
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-threads", "2",
        "-i", str(VIDEO_DIR / rel_path),
        "-vf", f"select='{expr}',scale={FRAME_W}:{FRAME_H}",
        "-vsync", "0", "-frames:v", str(len(nums)), "-q:v", "3", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except Exception as e:  # noqa: BLE001 - record and continue; a few bad chunks are tolerable
        return i, gids, f"{type(e).__name__}: {str(e)[:160]}"

    missing = []
    for k, g in enumerate(gids):
        src = outdir / f".tmp_{i}_{k+1:03d}.jpg"
        if src.exists() and src.stat().st_size > 0:
            frame_path(g).parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, frame_path(g))
        else:
            missing.append(g)
    for k in range(len(gids) + 2):
        stale = outdir / f".tmp_{i}_{k+1:03d}.jpg"
        if stale.exists():
            stale.unlink()
    return i, missing, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N chunks")
    a = ap.parse_args()

    idx = StreamIndex.load()
    jobs = chunk_frame_jobs(idx)
    if a.limit:
        jobs = jobs[: a.limit]
    n_expected = sum(len(w) for _, _, w in jobs)
    n_grid = math.floor(idx.total_s / GRID_S) + 1
    print(f"M3: {len(jobs)} chunks -> {n_expected} frames "
          f"(grid has {n_grid} points over {idx.total_s/3600:.2f} h), "
          f"{a.workers} workers", flush=True)

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
        "n_chunks": len(jobs),
        "n_frames_expected": n_expected,
        "max_gid": max(g for _, _, w in jobs for g, _ in w),
        "n_frames_missing": len(missing),
        "missing_gids": sorted(missing)[:20000],
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
