"""M5 — no-memory-required control questions.

Risk 2 in CLAUDE.md: Δt is confounded with token count, interference and recency, so a drop
in memory accuracy could just be generic long-context degradation. The control is a question
answerable from the *checkpoint instant alone*: it rides on exactly the same context as the
memory question, so if control accuracy stays flat while memory accuracy falls, generic
degradation is ruled out.

Source: EgoLifeCap DenseCaption — first-person Chinese captions at ~1-3 s granularity.
`A1_JAKE_DAY{n}_HH000000.srt` holds offsets from that nominal hour.

Construction, for each distinct checkpoint:
  gold        = the caption covering the checkpoint instant (must be distinctive: >= MIN_CHARS
                characters and occurring at most MAX_DUP times in the whole day)
  distractors = 3 captions drawn >= FAR_S of stream time away, with low character-bigram
                overlap against the gold *and* against each other, and not occurring anywhere
                within NEAR_S of the checkpoint

The control is posed in Chinese because the captions are Chinese; translating them would
inject an error source. This is fine because the control is only ever compared against
*itself* across Δt (is it flat?), never against the English memory questions.

Artifact: artifacts/m5_control_questions.jsonl
"""
from __future__ import annotations

import collections
import json
import random
import re

from .config import ARTIFACTS, CAPTION_DIR, PARTICIPANT
from .m2_stream_index import StreamIndex
from .timeparse import abs_seconds

MIN_CHARS = 6
GOLD_MIN_CHARS = 8
MAX_DUP = 8

# A caption like "我稍微向左右摆头" ("I turn my head slightly left and right") is a true
# description of the instant but carries nothing a 0.2 fps frame can be scored against, so
# such items are not used as the gold answer. They stay eligible as distractors, where being
# hard to rule in is harmless.
_LOW_CONTENT = re.compile(
    r"^(?:我)?(?:又|再|继续|然后|接着|稍微|慢慢|快速)*"
    r"(?:抬头|低头|摆头|转头|回头|扭头|点头|站着|坐着|站起来|坐下|走着|停下|"
    r"看了一眼|看着前面|向左|向右|向前|向后|环顾|四处看看|张望|发呆|不动)"
    r"[^，,。]{0,4}$"
)
FAR_S = 1800.0   # distractors come from >= 30 min away in stream time
NEAR_S = 300.0   # ...and must not occur within 5 min of the checkpoint
MAX_BIGRAM_J = 0.20

STEM = "我现在正在做什么？"
_SRT_NAME = re.compile(r"^" + PARTICIPANT + r"_DAY(\d+)_(\d{8})\.srt$")
_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")


def parse_srt(path, base_abs_s: float) -> list[tuple[float, float, str]]:
    out, lines = [], path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        m = _TIME.search(lines[i])
        if not m:
            i += 1
            continue
        a = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000
        b = int(m.group(5)) * 3600 + int(m.group(6)) * 60 + int(m.group(7)) + int(m.group(8)) / 1000
        text = []
        i += 1
        while i < len(lines) and lines[i].strip():
            text.append(lines[i].strip())
            i += 1
        t = " ".join(text).strip()
        if t:
            out.append((base_abs_s + a, base_abs_s + b, t))
    return out


def load_captions() -> list[tuple[float, float, str]]:
    caps = []
    for day_dir in sorted(CAPTION_DIR.glob("DAY*")):
        for p in sorted(day_dir.glob("*.srt")):
            m = _SRT_NAME.match(p.name)
            if not m:
                continue
            caps.extend(parse_srt(p, abs_seconds(int(m.group(1)), m.group(2))))
    caps.sort()
    return caps


def bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    rng = random.Random(20260811)
    idx = StreamIndex.load()
    caps = load_captions()
    if not caps:
        raise SystemExit(f"no captions under {CAPTION_DIR}")
    freq = collections.Counter(c[2] for c in caps)

    # index captions in stream time
    scaps = [(idx.to_stream(a), idx.to_stream(b), t) for a, b, t in caps]
    scaps.sort()
    s_starts = [c[0] for c in scaps]

    plan = [json.loads(l) for l in (ARTIFACTS / "m4_eval_plan.jsonl").read_text().splitlines()]
    checkpoints = sorted({(r["qid"], r["dt_s"], r["checkpoint_stream_s"]) for r in plan})

    import bisect

    # pre-filter once: distinctive captions only, with their bigrams cached
    pool_idx = [(a, t) for a, _, t in scaps
                if len(re.sub(r"\s+", "", t)) >= MIN_CHARS and freq[t] <= MAX_DUP]
    bigram_cache = {t: bigrams(t) for _, t in pool_idx}
    print(f"caption pool: {len(pool_idx)} distinctive of {len(scaps)}")

    rows, skipped = [], collections.Counter()
    for qid, dt, ck in checkpoints:
        j = bisect.bisect_right(s_starts, ck) - 1
        # take the caption covering ck, else the nearest one ending before it
        gold = None
        for k in range(j, max(j - 12, -1), -1):
            if k < 0:
                break
            a, b, t = scaps[k]
            # 30 s back is still the last frame or two of the window at 0.2 fps; the actual
            # lag is recorded per item so it can be checked against the result
            if ck - a > 30.0:
                break
            txt = re.sub(r"\s+", "", t)
            if (len(txt) >= GOLD_MIN_CHARS and freq[t] <= MAX_DUP
                    and not _LOW_CONTENT.match(txt)):
                gold = (a, b, t)
                break
        if gold is None:
            skipped["no_distinctive_gold"] += 1
            continue

        gb = bigrams(gold[2])
        # captions occurring anywhere near the checkpoint are excluded outright: a "wrong"
        # option that also describes the current instant would make the control unanswerable
        i_lo = bisect.bisect_left(s_starts, ck - NEAR_S)
        i_hi = bisect.bisect_right(s_starts, ck + NEAR_S)
        near = {t for _, _, t in scaps[i_lo:i_hi]}

        # rejection-sample from the pre-filtered distinctive pool rather than scanning all
        # ~130k captions per checkpoint
        picked: list[str] = []
        picked_bg: list[set[str]] = []
        for _ in range(4000):
            if len(picked) == 3:
                break
            j = rng.randrange(len(pool_idx))
            a_s, t = pool_idx[j]
            if abs(a_s - ck) < FAR_S or t in near or t in picked:
                continue
            tb = bigram_cache[t]
            if jaccard(tb, gb) > MAX_BIGRAM_J:
                continue
            if any(jaccard(tb, pb) > MAX_BIGRAM_J for pb in picked_bg):
                continue
            picked.append(t)
            picked_bg.append(tb)
        if len(picked) < 3:
            skipped["not_enough_distractors"] += 1
            continue

        opts = picked + [gold[2]]
        rng.shuffle(opts)
        gi = opts.index(gold[2])
        rows.append({
            "control_id": f"{qid}|{dt}",
            "qid": qid, "dt_s": dt,
            "checkpoint_stream_s": ck,
            "gold_caption_stream_s": round(gold[0], 2),
            "lag_s": round(ck - gold[0], 2),
            "question": STEM,
            "choices": {k: v for k, v in zip("abcd", opts)},
            "answer": "ABCD"[gi],
            "language": "zh",
        })

    out = ARTIFACTS / "m5_control_questions.jsonl"
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # verification
    assert all(len(set(r["choices"].values())) == 4 for r in rows), "duplicate options"
    assert all(r["choices"]["abcd"[i]] for r in rows for i in range(4))
    ansd = collections.Counter(r["answer"] for r in rows)
    lags = sorted(r["lag_s"] for r in rows)
    print(f"M5 OK: {len(rows)}/{len(checkpoints)} checkpoints got a control -> {out}")
    print(f"  skipped: {dict(skipped)}")
    print(f"  gold-caption lag behind checkpoint: median {lags[len(lags)//2]:.1f}s max {lags[-1]:.1f}s")
    print(f"  answer-letter balance: {dict(sorted(ansd.items()))}")
    print(f"  captions loaded: {len(caps)}")


if __name__ == "__main__":
    main()
