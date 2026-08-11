"""M2 — stream-time index over the A1 video chunks.

The A1 recording is 6,266 x 30 s mp4 chunks over 7 days, with 31 gaps (6 overnight of
10-14 h, several intra-day of 2-3 h). Wall-clock Δt would silently jump an overnight gap,
so every Δt in this experiment is measured in *cumulative stream seconds*: the amount of
video the model has actually observed between two points.

Artifact: artifacts/m2_stream_index.json
  chunks: [[abs_start_s, rel_path, duration_s], ...] sorted by abs_start_s
  cum:    [cumulative stream seconds at the start of each chunk]
"""
from __future__ import annotations

import bisect
import json
import re
import statistics
from dataclasses import dataclass

from .config import ARTIFACTS, CHUNK_NOMINAL_S, PARTICIPANT, VIDEO_DIR
from .timeparse import abs_seconds

_NAME_RE = re.compile(r"^DAY(\d+)_" + PARTICIPANT + r"_(\d{8})\.mp4$")


def scan_chunks() -> list[tuple[float, str]]:
    out = []
    for day_dir in sorted(VIDEO_DIR.glob("DAY*")):
        for p in day_dir.glob("*.mp4"):
            m = _NAME_RE.match(p.name)
            if not m:
                raise ValueError(f"unexpected filename: {p}")
            t = abs_seconds(int(m.group(1)), m.group(2))
            out.append((t, f"{day_dir.name}/{p.name}"))
    out.sort()
    return out


def probe_duration(path) -> float:
    import av

    with av.open(str(path)) as c:
        return float(c.duration) / 1_000_000.0


@dataclass
class StreamIndex:
    """Monotone map between wall-clock absolute seconds and cumulative stream seconds."""

    starts: list[float]
    paths: list[str]
    durs: list[float]
    cum: list[float]

    @classmethod
    def load(cls, path=None) -> "StreamIndex":
        d = json.loads((path or ARTIFACTS / "m2_stream_index.json").read_text())
        starts = [c[0] for c in d["chunks"]]
        paths = [c[1] for c in d["chunks"]]
        durs = [c[2] for c in d["chunks"]]
        return cls(starts, paths, durs, d["cum"])

    @property
    def total_s(self) -> float:
        return self.cum[-1] + self.durs[-1]

    def to_stream(self, abs_s: float) -> float:
        """Wall-clock -> stream seconds. Times inside a gap snap to the gap's leading edge."""
        i = bisect.bisect_right(self.starts, abs_s) - 1
        if i < 0:
            return 0.0
        return self.cum[i] + min(max(abs_s - self.starts[i], 0.0), self.durs[i])

    def to_abs(self, stream_s: float) -> float:
        """Stream seconds -> wall-clock."""
        i = bisect.bisect_right(self.cum, stream_s) - 1
        i = min(max(i, 0), len(self.cum) - 1)
        return self.starts[i] + min(max(stream_s - self.cum[i], 0.0), self.durs[i])

    def chunk_at_stream(self, stream_s: float) -> int:
        i = bisect.bisect_right(self.cum, stream_s) - 1
        return min(max(i, 0), len(self.cum) - 1)


def main(n_probe: int = 60) -> None:
    chunks = scan_chunks()
    if not chunks:
        raise SystemExit(f"no chunks under {VIDEO_DIR} — is the download finished?")

    # Probe a stratified sample of real durations to justify the 30 s nominal value.
    step = max(1, len(chunks) // n_probe)
    probed = []
    for i in range(0, len(chunks), step):
        probed.append(probe_duration(VIDEO_DIR / chunks[i][1]))
    dev = [abs(d - CHUNK_NOMINAL_S) for d in probed]

    durs = [CHUNK_NOMINAL_S] * len(chunks)
    cum, acc = [], 0.0
    for d in durs:
        cum.append(acc)
        acc += d

    starts = [c[0] for c in chunks]
    gaps = []
    for i in range(len(chunks) - 1):
        g = starts[i + 1] - starts[i] - durs[i]
        if g > 1.0:
            gaps.append({"after": chunks[i][1], "before": chunks[i + 1][1], "gap_s": round(g, 2)})

    doc = {
        "participant": PARTICIPANT,
        "n_chunks": len(chunks),
        "total_stream_s": acc,
        "total_stream_h": acc / 3600.0,
        "nominal_chunk_s": CHUNK_NOMINAL_S,
        "probe": {
            "n": len(probed),
            "median_duration_s": statistics.median(probed),
            "max_abs_deviation_s": max(dev),
        },
        "n_gaps": len(gaps),
        "gaps": gaps,
        "chunks": [[t, p, d] for (t, p), d in zip(chunks, durs)],
        "cum": cum,
    }
    out = ARTIFACTS / "m2_stream_index.json"
    out.write_text(json.dumps(doc))

    # verification: round-trip the map at a few thousand points
    idx = StreamIndex(starts, [p for _, p in chunks], durs, cum)
    import random

    rnd = random.Random(0)
    worst = 0.0
    for _ in range(5000):
        s = rnd.uniform(0, idx.total_s - 0.01)
        worst = max(worst, abs(idx.to_stream(idx.to_abs(s)) - s))
    assert worst < 1e-6, worst
    assert all(cum[i] < cum[i + 1] for i in range(len(cum) - 1))
    print(
        f"M2 OK: {len(chunks)} chunks, {acc/3600:.2f} h stream, {len(gaps)} gaps, "
        f"probe max|dur-30s| = {max(dev):.3f}s, roundtrip err {worst:.2e} -> {out}"
    )


if __name__ == "__main__":
    main()
