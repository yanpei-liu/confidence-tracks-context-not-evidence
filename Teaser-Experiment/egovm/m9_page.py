"""M9 — render the shareable HTML page from m7_metrics.json + the headline figure.

Generated rather than hand-written so the page cannot drift from the numbers, and the
figure is inlined as a data URI because the artifact CSP blocks external hosts.

Artifact: artifacts/m9_teaser.html
"""
from __future__ import annotations

import base64
import json

from .config import ARTIFACTS, FIGURES, PRIOR_REF_DT_S


def pp(d, digits=1):
    if not d:
        return "n/a"
    return (f"{d['mean']*100:+.{digits}f} <span class='ci'>[{d['lo95']*100:+.{digits}f}, "
            f"{d['hi95']*100:+.{digits}f}]</span>")


def sig(d):
    return "excludes 0" if d and not (d["lo95"] < 0 < d["hi95"]) else "includes 0"


def main() -> None:
    m = json.loads((ARTIFACTS / "m7_metrics.json").read_text())
    env = json.loads((ARTIFACTS / "m0_env_report.json").read_text())
    a, v = m["all_questions"], m["vision_only_questions"]
    h, vh = a["headline"], v["headline"]
    ea, vea = a["evidence_awareness"], v["evidence_awareness"]
    tm = a["token_matched"]
    bins = a["bins"]["memory/main"]
    ctl = a["bins"]["control/main"]
    prior = a["bins"]["memory/prior"]["300"]
    dts = sorted(bins, key=int)

    contrasts = a["contrasts"]
    el = next(contrasts[k] for _, k in
              sorted((int(k.split("_")[-1][:-3]), k) for k in contrasts
                     if k.startswith("more_elapsed_time_"))[-1:])
    tk = contrasts["fewer_tokens"]
    # the prior contrast at the bin the evidence-awareness pair is defined on; with the prior
    # arm in every bin (S3) the key carries a suffix
    ev = contrasts.get("evidence_present") or contrasts[
        f"evidence_present_{PRIOR_REF_DT_S // 60}min"]
    dec = a["decomposition"]
    it = a["item_transitions"]
    disc = a["discrimination"]
    mem_auc = sorted((int(k.split("/")[-1]), val["auroc"]) for k, val in disc.items()
                     if k.startswith("memory/main/"))
    auc_lo, auc_hi = mem_auc[0][1], mem_auc[-1][1]
    auc_prior = next(val["auroc"] for k, val in disc.items() if k.startswith("memory/prior/"))

    # ordered as the argument runs, not by effect size: the contrasts that carry the claim are
    # the smallest numbers in the table
    def rank(key: str) -> tuple:
        order = ["more_elapsed_time_", "fewer_tokens", "evidence_present", "evidence_resolution",
                 "raw_sweep", "control_sweep", "question_kind_", "control_"]
        for i, pref in enumerate(order):
            if key == pref or key.startswith(pref):
                return (i, key)
        return (len(order), key)

    contrast_rows = "\n".join(
        f"<tr><td>{c['label']}</td><td>{pp(c['delta_accuracy'])}</td>"
        f"<td>{pp(c['delta_confidence'])}</td><td>{c['n_questions']}</td></tr>"
        for _, c in sorted(contrasts.items(), key=lambda kv: rank(kv[0])))

    fig_b64 = base64.b64encode((FIGURES / "headline.png").read_bytes()).decode()

    sweep_rows = "\n".join(
        f"<tr><td>{int(d)//60}</td><td>{bins[d]['n']}</td>"
        f"<td>{bins[d]['accuracy']*100:.1f}</td><td>{bins[d]['mean_confidence']*100:.1f}</td>"
        f"<td>{bins[d]['overconfidence']*100:+.1f}</td><td>{bins[d]['ece']:.2f}</td>"
        f"<td>{bins[d]['mean_frames']:.0f}</td>"
        f"<td>{bins[d]['mean_visual_tokens']:,.0f}</td>"
        f"<td>{ctl[d]['accuracy']*100:.1f}</td></tr>"
        for d in dts)

    robust_rows = "\n".join(
        f"<tr><td>{lbl}</td><td>{pp(x)}</td><td>{pp(y)}</td></tr>"
        for lbl, x, y in [
            ("accuracy, 1&rarr;60 min", h["delta_accuracy"], vh["delta_accuracy"]),
            ("confidence, 1&rarr;60 min", h["delta_confidence"], vh["delta_confidence"]),
            ("overconfidence gap", h["delta_gap"], vh["delta_gap"]),
            ("evidence vs none, accuracy", ea["delta_accuracy"], vea["delta_accuracy"]),
            ("evidence vs none, confidence", ea["delta_confidence"], vea["delta_confidence"]),
            ("&Delta;t 60 vs 15 min, tokens fixed",
             tm["tokmatch_evkeep"]["delta_accuracy"],
             v["token_matched"]["tokmatch_evkeep"]["delta_accuracy"]),
        ])

    type_rows = "\n".join(
        f"<tr><td>{t}</td>"
        f"<td>{d[min(d, key=int)]['accuracy']*100:.1f}</td>"
        f"<td>{d[max(d, key=int)]['accuracy']*100:.1f}</td>"
        f"<td>{d[min(d, key=int)]['mean_confidence']*100:.1f}</td>"
        f"<td>{d[max(d, key=int)]['mean_confidence']*100:.1f}</td></tr>"
        for t, d in a["by_type"].items() if len(d) >= 2)

    html = f"""<title>Does a streaming ego video-LLM know it forgot?</title>
<style>
  :root {{
    color-scheme: light;
    --ground: #f7f8fa;
    --panel: #ffffff;
    --ink: #10131a;
    --ink-2: #4a515e;
    --ink-3: #79818f;
    --rule: #dde2ea;
    --acc: #2a78d6;
    --conf: #c4521f;
    --floor: #4a3aa7;
    --flag: #a8320f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --ground: #10131a; --panel: #161a23; --ink: #eef1f6; --ink-2: #a3abba;
      --ink-3: #767e8c; --rule: #262b36; --acc: #6ba6ec; --conf: #f0895a;
      --floor: #9085e9; --flag: #f08a6c;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --ground: #10131a; --panel: #161a23; --ink: #eef1f6; --ink-2: #a3abba;
    --ink-3: #767e8c; --rule: #262b36; --acc: #6ba6ec; --conf: #f0895a;
    --floor: #9085e9; --flag: #f08a6c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--ground); color: var(--ink);
    font-family: Charter, "Bitstream Charter", "Sitka Text", Cambria, Georgia, serif;
    font-size: 17px; line-height: 1.62;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 78ch; margin: 0 auto; padding: 4rem 1.5rem 6rem; }}
  .mono {{ font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace; }}
  .eyebrow {{
    font-family: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
    font-size: .7rem; letter-spacing: .16em; text-transform: uppercase;
    color: var(--ink-3); margin: 0 0 1rem;
  }}
  h1 {{
    font-size: clamp(1.9rem, 4.4vw, 2.75rem); line-height: 1.12; margin: 0 0 1rem;
    text-wrap: balance; font-weight: 600; letter-spacing: -.015em;
  }}
  h2 {{
    font-size: 1.28rem; margin: 3.4rem 0 .9rem; font-weight: 600;
    letter-spacing: -.01em; text-wrap: balance;
    padding-top: .9rem; border-top: 1px solid var(--rule);
  }}
  h3 {{ font-size: 1rem; margin: 2rem 0 .5rem; font-weight: 600; }}
  p {{ margin: 0 0 1.05rem; }}
  .lede {{ font-size: 1.1rem; color: var(--ink-2); }}
  strong {{ font-weight: 600; }}
  .verdict {{
    background: var(--panel); border: 1px solid var(--rule);
    border-left: 3px solid var(--flag);
    padding: 1.2rem 1.35rem; margin: 2rem 0; border-radius: 2px;
  }}
  .verdict p:last-child {{ margin-bottom: 0; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: .8rem; margin: 2rem 0; }}
  .tile {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 2px;
           padding: .95rem 1rem; }}
  .tile .k {{ font-family: ui-monospace, Menlo, monospace; font-size: .68rem;
              letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); }}
  .tile .val {{ font-family: ui-monospace, Menlo, monospace; font-size: 1.5rem;
                font-weight: 600; margin: .3rem 0 .15rem; font-variant-numeric: tabular-nums; }}
  .tile .note {{ font-size: .78rem; color: var(--ink-2); line-height: 1.4; }}
  .t-acc .val {{ color: var(--acc); }}
  .t-conf .val {{ color: var(--conf); }}
  .t-gap .val {{ color: var(--ink); }}
  figure {{ margin: 2rem 0; }}
  figure img {{ width: 100%; height: auto; display: block;
                border: 1px solid var(--rule); border-radius: 2px; background: #fcfcfb; }}
  figcaption {{ font-size: .8rem; color: var(--ink-2); margin-top: .6rem; line-height: 1.5; }}
  .scroll {{ overflow-x: auto; margin: 1.2rem 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .85rem;
           font-variant-numeric: tabular-nums;
           font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
  th, td {{ text-align: right; padding: .42rem .6rem; border-bottom: 1px solid var(--rule);
            white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ color: var(--ink-3); font-weight: 500; font-size: .72rem;
              letter-spacing: .07em; text-transform: uppercase; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  .ci {{ color: var(--ink-3); font-size: .92em; }}
  ul {{ padding-left: 1.15rem; margin: 0 0 1.05rem; }}
  li {{ margin-bottom: .45rem; }}
  .foot {{ margin-top: 3.5rem; padding-top: 1.1rem; border-top: 1px solid var(--rule);
           font-size: .8rem; color: var(--ink-3); }}
  code {{ font-family: ui-monospace, Menlo, monospace; font-size: .86em;
          background: var(--panel); border: 1px solid var(--rule);
          padding: .08em .32em; border-radius: 2px; }}
</style>

<div class="wrap">
  <p class="eyebrow">Egocentric Vision Metamemory &middot; teaser experiment</p>
  <h1>Does a streaming ego video-LLM know it forgot?</h1>
  <p class="lede">Qwen3-VL-8B answering {bins[dts[0]]['n']} time-invariant EgoLifeQA questions at five
  recall delays over one 52-hour egocentric stream. The predicted answer was &ldquo;no &mdash; it stays
  confident as it forgets.&rdquo; What the controls actually show is stranger: its confidence
  <em>does</em> move, consistently, and never for a reason that has anything to do with what it
  can answer.</p>

  <div class="verdict">
    <p><strong>Confidence responds to the one thing that costs no accuracy.</strong> Hold the
    visual-token count and the evidence sampling fixed and let only elapsed time grow: accuracy
    moves {el['delta_accuracy']['mean']*100:+.1f} pp &mdash; nothing &mdash; while confidence drops
    {el['delta_confidence']['mean']*100:.1f} pp. Hold elapsed time fixed and cut the context to a
    quarter instead: accuracy moves {tk['delta_accuracy']['mean']*100:+.1f} pp and confidence
    {tk['delta_confidence']['mean']*100:+.1f}. The two measures are reading different variables.</p>
    <p><strong>And it barely notices whether it has the evidence.</strong> Swapping the evidence
    out of the context entirely, tokens held fixed, moves accuracy
    {ev['delta_accuracy']['mean']*100:+.1f} points and confidence only
    {ev['delta_confidence']['mean']*100:+.1f} &mdash; a
    {ev['delta_accuracy']['mean']/max(ev['delta_confidence']['mean'], 1e-9):.1f}&times;
    under-reaction to the largest accuracy effect in the experiment.</p>
    <p><strong>The originally predicted signature is absent.</strong> The overconfidence gap does
    not widen with &Delta;t ({h['delta_gap']['mean']*100:+.1f} pp
    <span class="ci">[{h['delta_gap']['lo95']*100:+.1f}, {h['delta_gap']['hi95']*100:+.1f}]</span>,
    {sig(h['delta_gap'])}). Both measures fall over the raw sweep &mdash; but for different
    reasons, which is what makes the sweep alone uninterpretable.</p>
  </div>

  <div class="tiles">
    <div class="tile t-conf"><div class="k">+45 min elapsed, context pinned</div>
      <div class="val">{el['delta_accuracy']['mean']*100:+.1f} acc /
        {el['delta_confidence']['mean']*100:+.1f} conf</div>
      <div class="note">confidence moves, accuracy does not &middot;
        {sig(el['delta_confidence'])} on confidence</div></div>
    <div class="tile t-acc"><div class="k">4&times; the context, &Delta;t pinned</div>
      <div class="val">{-tk['delta_accuracy']['mean']*100:+.1f} acc /
        {-tk['delta_confidence']['mean']*100:+.1f} conf</div>
      <div class="note">accuracy moves, confidence does not &middot;
        {sig(tk['delta_accuracy'])} on accuracy</div></div>
    <div class="tile t-gap"><div class="k">Evidence in context vs none</div>
      <div class="val">{ev['delta_accuracy']['mean']*100:+.1f} acc /
        {ev['delta_confidence']['mean']*100:+.1f} conf</div>
      <div class="note">both move, but confidence
        {ev['delta_accuracy']['mean']/max(ev['delta_confidence']['mean'], 1e-9):.1f}&times;
        too little</div></div>
  </div>

  <figure>
    <img src="data:image/png;base64,{fig_b64}"
         alt="Three panels. Left: accuracy and confidence against recall delay, once with context
              growing with the delay and once with context pinned at the 15-minute budget; the
              pinned accuracy line is flat while the pinned confidence line keeps falling. Middle:
              a scatter of change in accuracy against change in confidence for each manipulation,
              with a diagonal marking where confidence would track competence; every point is off
              the diagonal. Right: confidence change for items kept versus items lost, which are
              indistinguishable, and AUROC against delay, which is flat.">
    <figcaption><strong>A</strong> &mdash; two &Delta;t sweeps. With context growing, both measures
      fall. With context pinned at the 15-minute budget, accuracy is flat and confidence still
      falls. <strong>B</strong> &mdash; each manipulation placed by what it moves; the diagonal is
      where confidence would track competence, and nothing sits on it. <strong>C</strong> &mdash;
      the confidence drop is the same on items the model kept and items it lost, while its ability
      to <em>rank</em> its own answers is untouched by &Delta;t. All intervals are 95% cluster
      bootstraps resampling <em>questions</em>, since each question appears in every bin.</figcaption>
  </figure>

  <h2>What moves accuracy, and what moves confidence</h2>
  <div class="scroll"><table>
    <thead><tr><th>manipulation</th><th>&Delta; accuracy (pp)</th><th>&Delta; confidence (pp)</th>
      <th>n</th></tr></thead>
    <tbody>{contrast_rows}</tbody>
  </table></div>
  <p>The two token-matched arms are what make elapsed time and context length separately
  identifiable. The observed &Delta;t=15&rarr;60 min change in the <code>main</code> arm splits
  cleanly into the two: accuracy&rsquo;s decline is
  {dec['accuracy']['context_length']*100:+.1f} pp of context length and
  {dec['accuracy']['elapsed_time']*100:+.1f} pp of elapsed time, while confidence&rsquo;s is
  {dec['confidence']['elapsed_time']*100:+.1f} pp of elapsed time and
  {dec['confidence']['context_length']*100:+.1f} pp of context length.</p>

  <h2>A flat discount, not item knowledge</h2>
  <p>Pooling over adjacent &Delta;t steps and keeping only items the model had right at the earlier
  bin, confidence falls {it['delta_confidence_kept']['mean']*100:.1f} pp on the
  {it['n_kept']} items it kept and {it['delta_confidence_lost']['mean']*100:.1f} pp on the
  {it['n_lost']} it lost &mdash; a difference of {pp(it['difference'])}, {sig(it['difference'])}.
  The decline lands on everything.</p>
  <p>Its <em>ranking</em>, though, is intact: AUROC of confidence against correctness is
  {auc_lo:.3f} at 1 minute and {auc_hi:.3f} at 60 minutes, and drops to {auc_prior:.3f} only when
  the evidence is absent from the window. The usable signal is there &mdash; it just never reaches
  the confidence level.</p>

  <h2>The sweep</h2>
  <div class="scroll"><table>
    <thead><tr><th>&Delta;t (min)</th><th>n</th><th>acc %</th><th>conf %</th><th>gap pp</th>
      <th>ECE</th><th>frames</th><th>vis. tokens</th><th>control acc %</th></tr></thead>
    <tbody>{sweep_rows}</tbody>
  </table></div>
  <p>&Delta;t is elapsed <strong>stream</strong> time &mdash; video the model has actually observed
  &mdash; not wall-clock. The A1 recording has 31 gaps including six overnight breaks of 10&ndash;14 h,
  so a wall-clock delay would silently jump across a night's sleep.</p>

  <h2>Three things the controls establish</h2>
  <h3>1. The accuracy decline is context length, not elapsed time</h3>
  <p>Subsampling the 60-minute window down to the 15-minute token count removes the decline.
  <code>tokmatch_evkeep</code> is the arm that isolates elapsed time: it keeps every frame near the
  evidence and thins only the filler, holding token count <em>and</em> evidence resolution fixed.
  Accuracy there is {tm['tokmatch_evkeep']['delta_accuracy']['mean']*100:+.1f} pp
  <span class="ci">[{tm['tokmatch_evkeep']['delta_accuracy']['lo95']*100:+.1f},
  {tm['tokmatch_evkeep']['delta_accuracy']['hi95']*100:+.1f}]</span> &mdash; nothing. Confidence in
  the same arm falls {el['delta_confidence']['mean']*100:.1f} pp, which is the dissociation this
  page is about.</p>
  <h3>2. The no-memory control fails in the informative direction</h3>
  <p>A question answerable from the checkpoint instant alone was meant to stay flat and rule
  generic long-context degradation <em>out</em>. It falls from {ctl[dts[0]]['accuracy']*100:.1f}% to
  {ctl[dts[-1]]['accuracy']*100:.1f}% &mdash; further than the memory questions do. Present-moment
  perception degrades with context length more than memory does. The control has so far only run
  on <code>main</code> frames, so its own decline has not yet been token-matched.</p>
  <h3>3. The usable dynamic range is narrow</h3>
  <p>With no evidence in the window at all the model still scores {prior['accuracy']*100:.1f}%
  against a 25% chance floor, because these questions are about a shared house and its recurring
  people and objects. The whole sweep lives between that floor and
  {bins[dts[0]]['accuracy']*100:.1f}% at one minute.</p>

  <h2>Evidence awareness</h2>
  <p>Token-matched at {ea['mean_tokens_with_evidence']:,.0f} visual tokens and paired over
  {ea['n_questions']} questions, the only difference being whether the window contains the
  evidence:</p>
  <div class="scroll"><table>
    <thead><tr><th></th><th>no evidence</th><th>evidence</th><th>change (pp)</th></tr></thead>
    <tbody>
      <tr><td>accuracy</td><td>{ea['accuracy_without']*100:.1f}%</td>
        <td>{ea['accuracy_with_evidence']*100:.1f}%</td><td>{pp(ea['delta_accuracy'])}</td></tr>
      <tr><td>confidence</td><td>{ea['confidence_without']*100:.1f}%</td>
        <td>{ea['confidence_with_evidence']*100:.1f}%</td><td>{pp(ea['delta_confidence'])}</td></tr>
    </tbody>
  </table></div>
  <p>Having the evidence is worth
  {ea['delta_accuracy']['mean']/max(ea['delta_confidence']['mean'], 1e-9):.1f}&times; more accuracy
  than confidence. On top of that the model is uniformly overconfident at every horizon &mdash; gap
  {min(bins[d]['overconfidence'] for d in dts)*100:+.0f} to
  {max(bins[d]['overconfidence'] for d in dts)*100:+.0f} pp, ECE
  {min(bins[d]['ece'] for d in dts):.2f}&ndash;{max(bins[d]['ece'] for d in dts):.2f} &mdash;
  including when it has nothing to go on. That is the recalibration target, and it needs no
  forgetting to motivate it.</p>

  <h2>By question category</h2>
  <div class="scroll"><table>
    <thead><tr><th>category</th><th>acc @1 min</th><th>acc @60 min</th>
      <th>conf @1 min</th><th>conf @60 min</th></tr></thead>
    <tbody>{type_rows}</tbody>
  </table></div>

  <h2>Robustness: dropping audio-dependent questions</h2>
  <p>{a['n_records'] - v['n_records']} records come from questions flagged <code>need_audio</code>,
  which a vision-only model cannot answer in principle. Re-running everything without them
  ({vh['n_paired_questions']} paired questions) changes nothing that matters.</p>
  <div class="scroll"><table>
    <thead><tr><th>quantity</th><th>all questions</th><th>vision-only</th></tr></thead>
    <tbody>{robust_rows}</tbody>
  </table></div>

  <h2>Method</h2>
  <ul>
    <li><strong>Model.</strong> {env['model_hub_id']} (Apache 2.0), BF16, option-letter logprobs
      softmaxed over {{A,B,C,D}}; no text generated. Confidence is that max &mdash; a proxy, reported
      with ECE, never as a calibrated probability.</li>
    <li><strong>Questions.</strong> EgoLifeQA participant A1/Jake, the only released participant.
      Mitigation&nbsp;1 keeps the time-invariant categories (EntityLog, EventRecall, RelationMap);
      TaskMaster is dropped as prospective. Three further questions are excluded because their
      evidence points span more than 15 minutes, which leaves them without a single well-defined
      &Delta;t.</li>
    <li><strong>Evidence times.</strong> From the released <code>target_time</code> field, which is
      populated on all 500 items &mdash; contrary to the project's planning notes, so no
      AVW-derivation fallback dataset was needed.</li>
    <li><strong>Window.</strong> Frames at 0.2&nbsp;fps from 60&nbsp;s before the earliest evidence
      to the checkpoint, at a pinned {env['input_res']}&times;{env['input_res']}
      ({env['expected_tokens_per_frame']:.0f} visual tokens/frame, asserted constant in window
      length). Three Qwen3-VL preprocessing defaults had to be overridden or the sweep would have
      measured the wrong thing: frame resampling to 2&nbsp;fps, a total-pixel budget that shrinks
      frames as the window grows, and an assumed 24&nbsp;fps that would have labelled a 60-minute
      window as 16 seconds.</li>
    <li><strong>Statistics.</strong> Every interval is a cluster bootstrap over questions
      (5,000 resamples), because each question appears in all five bins.</li>
  </ul>

  <h2>What this does not settle</h2>
  <ul>
    <li>One participant, one model, one frame rate. N per bin is bounded by the single released
      EgoLife participant.</li>
    <li>&Delta;t tops out at 60 minutes. A gap that opens only over many hours would not appear
      here; the 44-hour horizons EGOSTREAM reports remain untested.</li>
    <li>The model sees a window around the evidence, not the full history the benchmark assumes,
      so the absolute accuracy level is not comparable to published EgoLifeQA numbers. The claims
      here are about slopes and contrasts.</li>
    <li>Control questions are auto-generated from the DenseCaption track and posed in Chinese;
      they are only ever compared against themselves across &Delta;t.</li>
  </ul>

  <p class="foot">{a['n_records']:,} scored (question, checkpoint, arm) records over
  {bins[dts[0]]['n']} questions &times; 5 delays &times; 4 arms, from a 37,401-frame bank covering
  51.95&nbsp;h of stream. EgoLife is S-Lab License 1.0, research use; no dataset content is
  redistributed here.</p>
</div>
"""
    out = ARTIFACTS / "m9_teaser.html"
    out.write_text(html)
    print(f"M9 -> {out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
