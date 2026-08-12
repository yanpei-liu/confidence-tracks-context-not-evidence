"""Option-letter logprob scoring for Qwen3-VL.

Confidence is the softmax over the logits of the option-letter tokens — {A,B,C,D}, or
{A,B,C,D,E} when the abstention option is offered — at the first generated position;
correctness is whether the argmax matches the gold letter. No text is generated. This is the
`multiple_choice`/`loglikelihood` protocol used by lmms-eval.

Reported as "model confidence (option-letter softmax)" — it is a proxy, not a calibrated
probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import INPUT_RES, MODEL_ID, OPTION_LETTERS, SAMPLE_FPS

SYSTEM = (
    "You are a wearable camera assistant. You have been observing the wearer's day through "
    "their glasses. Answer the multiple-choice question about what you saw."
)

PROMPT = (
    "{question}\n"
    "{options}\n"
    "Answer with the letter of the correct option only."
)


def option_block(choices: dict) -> str:
    """"A. …\\nB. …" for however many options this item has (four, or five with abstention)."""
    return "\n".join(f"{k.upper()}. {v}" for k, v in sorted(choices.items()))


@dataclass
class LetterScorer:
    model: object
    processor: object
    letter_ids: list[int]
    device: str
    input_res: int = INPUT_RES
    letters: tuple = tuple(OPTION_LETTERS)

    @classmethod
    def load(cls, model_id: str = MODEL_ID, dtype=torch.bfloat16, attn: str = "sdpa",
             input_res: int = INPUT_RES, letters=OPTION_LETTERS):
        """`letters` is the answer alphabet the softmax runs over — {A,B,C,D} normally, or
        {A,B,C,D,E} when the abstention option is offered (S4). It is a property of the
        *scorer*, not of the item, because the softmax denominator has to be fixed across
        every row in a records file for the confidences to be comparable."""
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map="cuda:0", attn_implementation=attn
        )
        model.eval()
        tok = processor.tokenizer
        ids = []
        for L in letters:
            enc = tok.encode(L, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(f"letter {L!r} is not a single token: {enc}")
            ids.append(enc[0])
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"option letters collide: {ids}")
        return cls(model, processor, ids, "cuda:0", input_res, tuple(letters))

    def build_inputs(self, images: list, question: str, choices: dict,
                     frame_offsets: list[int] | None = None):
        """images: list of PIL.Image forming the video; choices: {'a'..'d'(..'e'): str}.

        `frame_offsets` are the frames' positions on the 0.2 fps grid *relative to the first
        frame of this window*, so frame k sits at ``frame_offsets[k] * 5 s``. They must be
        supplied for the non-uniformly sampled arms; uniform 0.2 fps defaults to 0,1,2,...
        """
        from transformers.video_utils import VideoMetadata

        r = self.input_res
        images = [im if im.size == (r, r) else im.resize((r, r)) for im in images]
        n = len(images)
        idx = list(frame_offsets) if frame_offsets is not None else list(range(n))

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text",
                     "text": PROMPT.format(question=question, options=option_block(choices))},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Three processor defaults have to be overridden or the experiment measures the
        # wrong thing:
        #   do_sample_frames=False  the default resamples to fps=2 capped at 768 frames, so a
        #                           64-frame window would arrive as 6 frames;
        #   do_resize=False         the default fits frames to a *total* 25.2M-pixel budget,
        #                           making per-frame resolution shrink as Δt grows;
        #   video_metadata          Qwen3-VL writes "<t seconds>" markers into the prompt from
        #                           frames_indices/fps. Without metadata it assumes fps=24 and
        #                           would label a 60-minute window as ~16 seconds.
        meta = VideoMetadata(
            total_num_frames=n, fps=SAMPLE_FPS, frames_indices=idx,
            width=r, height=r, duration=(max(idx) + 1) / SAMPLE_FPS,
        )
        return self.processor(
            text=[text], videos=[images], return_tensors="pt", padding=True,
            do_resize=False, do_sample_frames=False, video_metadata=[meta],
        ).to(self.device)

    @torch.inference_mode()
    def score(self, images: list, question: str, choices: dict,
              frame_offsets: list[int] | None = None) -> dict:
        inputs = self.build_inputs(images, question, choices, frame_offsets)
        out = self.model(**inputs)
        logits = out.logits[0, -1].float()
        sel = logits[self.letter_ids]
        logprobs_full = torch.log_softmax(logits, dim=-1)[self.letter_ids]
        probs = torch.softmax(sel, dim=-1)
        p = probs.tolist()
        pred_i = int(max(range(len(p)), key=lambda i: p[i]))
        n_tok = int(inputs["input_ids"].shape[1])
        vis = int(inputs["video_grid_thw"].prod(dim=-1).sum().item() // 4) \
            if "video_grid_thw" in inputs else None
        return {
            "letter_probs": {L: round(v, 6) for L, v in zip(self.letters, p)},
            "letter_logprobs": {
                L: round(v, 4) for L, v in zip(self.letters, logprobs_full.tolist())
            },
            "pred": self.letters[pred_i],
            "confidence": round(p[pred_i], 6),
            "entropy": round(-sum(q * math.log(max(q, 1e-12)) for q in p), 6),
            "n_input_tokens": n_tok,
            "n_visual_tokens": vis,
        }
