# EvalBench spec

Repo `EvalBench`, MIT, port 8101. Promise from the portfolio: run one prompt suite across model checkpoints, score with rubric judging and exact match, cluster failures into modes (hallucination, format breaks, over-refusal), so each fine-tuning cycle targets what failed.

## 1. Concepts

- **Suite**: versioned JSONL of items `{id, task, prompt, gold?, checker, rubric?, source, license}`. Content-hashed. ~120 items, six tasks: `math` (GSM8K), `knowledge` (MMLU letter), `format` (IFEval-style constraints + authored JSON tasks), `truth` (TruthfulQA misconceptions), `unanswerable` (authored fake-entity questions; correct = admit not knowing), `benign` (XSTest safe prompts; correct = comply).
- **Checkpoint**: a named HF model id + revision + generation config. Default four: SmolLM2-135M-Instruct, SmolLM2-360M (base), SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct.
- **Run**: suite x checkpoints -> `generations.parquet` (every output, token counts, truncated flag, seconds), `scores.parquet`, `failures.parquet`, `run.json` (summary for UI). Runs are immutable folders `runs/<id>/`; a manifest records package version, suite hash, seeds.

## 2. Scoring (deterministic first)

Each item has a `checker`:
- `exact_numeric`: extract final number (regex filters, tolerant of commas/units), compare to gold.
- `exact_choice`: extract A-D letter.
- `constraints`: list of verifiers (word count, bullet count, all-lowercase, contains keyword, valid JSON with required keys, ends with phrase). Score = fraction passed; pass = all.
- `abstain`: pass if output matches an abstention lexicon OR judge says it admits not knowing; fail if it asserts specifics about a nonexistent entity.
- `comply`: pass if no refusal lexicon hit; judge confirms on hits.
- `misconception` (truth): rubric judge question "does the answer state the common misconception?" cross-checked against TruthfulQA correct/incorrect answer similarity.

Rubric judge (`judge.py`): interface `Judge.ask(rubric_question, prompt, output) -> p_yes`. Backends: `LocalJudge` (default, Qwen2.5-0.5B-Instruct, yes/no logprob, temperature-free), `OpenAIJudge`/`AnthropicJudge` (optional, env-selected, httpx transport injectable, unit-tested against stubs only). Judge calibration report: accuracy, Cohen's kappa, ROC-style threshold choice on items with gold; pairwise swap flip rate. Self-judging flag when the judge model equals a checkpoint.

Verdict record per (item, checkpoint): `{passed, score, scorer, evidence, judge_p, tags[]}`.

## 3. Failure detection and clustering

1. Detectors tag each failed sample (can have several): `format_break` (invalid JSON, violated constraint, no extractable answer), `over_refusal` (refusal lexicon on benign item), `hallucination` (asserts specifics on unanswerable item / states misconception), `wrong_answer` (extractable but wrong), `degenerate` (repetition, truncation at token cap, empty).
2. Embed failures with MiniLM + structural features; cluster (HDBSCAN, KMeans fallback); 2D layout by PCA->t-SNE, seeded.
3. Label cluster = detector majority, TF-IDF terms as evidence, purity reported. Output `clusters.json`: id, label, size, purity, per-checkpoint counts, top terms, member sample ids, centroid xy.
4. Fine-tuning handoff: `evalbench targets` writes a ranked list of modes per checkpoint with example prompts, as JSONL ready to seed an SFT data request.

## 4. Reports

Pandas-based: per-checkpoint pass rate by task with Wilson/bootstrap 95% CI; win-rate matrix (per item, higher score wins, ties count half; judge only breaks ties on rubric items, run in both orders); win-rate vs a chosen baseline checkpoint; failure-mode x checkpoint table; run-to-run diff (`evalbench diff runA runB`: items fixed, regressed, new modes) for the fine-tuning loop. Export `report.md` and `run.json`.

## 5. Module layout

```
src/evalbench/
  suite/    build.py (download + sample + hash), schema.py, authored/*.jsonl
  models/   registry.py, hf_runner.py (batched greedy, left padding, caps threads), cache of generations by (ckpt, item, config) hash
  scorers/  exact.py, constraints.py, lexicon.py, rubric.py, judge.py (base + local + openai + anthropic), calibrate.py
  failures/ detect.py, embed.py, cluster.py, label.py
  report/   winrate.py, tables.py, diff.py, export.py
  runner.py (orchestrates; resumable; progress events)
  api.py    FastAPI: /health, /api/runs, /api/runs/{id}, /api/runs/{id}/samples?cluster=&ckpt=&task=, POST /api/try (one custom prompt against chosen checkpoints, loaded lazily), SSE /api/runs/{id}/replay
  cli.py    evalbench suite build | run | score | cluster | report | diff | targets | serve | export-demo
web/ (static, vanilla ES modules, GSAP CDN)  ->  docs/ (identical copy + docs/data/run.json)
tests/  fast (offline, tiny fake model + stub judge) ; marker `slow` for real models
.github/workflows/ci.yml  ruff + pytest -m "not slow"
```

Data flow: suite.jsonl -> runner (HF generate, cached) -> generations -> scorers (+judge) -> scores -> detectors -> failures -> embed/cluster/label -> run folder -> API/UI or `export-demo` -> `docs/data/run.json`.

## 6. UI concept (real version of the site's scene)

Single page, dark, lime accent, orange for failures, mono labels.
- **Race**: one lane per checkpoint. Replays the recorded scoring order at ~40 samples/second; each lane's bar grows as its running win-rate/pass-rate updates with its CI whisker; leader gets lime. Scrubber and speed control; replay is of real recorded per-sample verdicts.
- **Failure map**: failed samples leave their lane as orange dots and drift to their true t-SNE coordinates, settling into dashed cluster ellipses labelled HALLUCINATION / FORMAT BREAK / OVER-REFUSAL / WRONG ANSWER / DEGENERATE with live counts and purity. Colour by checkpoint toggle.
- **Drill-down**: tap a cluster or dot -> drawer with prompt, gold, output, verdicts, judge p, evidence, and "similar failures". Filters: checkpoint, task, mode. On phones the drawer is a bottom sheet; map pans by touch; lanes stack above the map; no hover dependence.
- **Panels**: win-rate matrix heatmap, judge reliability card (accuracy, kappa, flip rate, self-judge warning), run diff, "targets for next cycle" list.
- **Try it** (local server only): type a prompt, see each checkpoint answer and the detectors fire. Hidden in static demo, which carries a "Recorded run, <date>, commit, suite hash" banner.
- Canvas for race + map (DPR capped at 2, pauses off-screen, reduced-motion shows final state); DOM for drawer and tables.

## 7. Tests

Fast: scorers (numeric extraction edge cases, constraint verifiers, lexicon), suite schema/hash stability, win-rate maths (ties, symmetry, CI), detector tags on handcrafted outputs, clustering on synthetic embeddings recovers planted modes, runner caching/resume with a fake model, judge adapters against stubbed `httpx.MockTransport`, local judge logic with a fake logit source, API endpoints via TestClient, CLI smoke. Slow (`-m slow`): real SmolLM2-135M end-to-end on 5 items. Plus a Playwright script `scripts/shots.py` for desktop 1440x900 and phone 390x844, console-error check.

## 8. Done means

- Real run on the default four checkpoints over the full suite on this machine; recorded under `runs/` (compact, committed) and exported to `docs/data/run.json`.
- README numbers (pass rates, CIs, win-rates, cluster purity, judge kappa, wall time) all from that run, reproduced by `scripts/reproduce.sh`.
- Fresh-clone quickstart works in a clean venv; pytest and ruff pass; UI screenshots reviewed; static `docs/` works via `python -m http.server`.
- README states plainly: 120-item suite, sub-1B models, small judge with measured unreliability, which paths were run for real (local models) vs stub-tested (OpenAI/Anthropic adapters).

## 9. Risks

- CPU time and a shared machine: mitigate with batching, caching, max 96 new tokens, resumable runs.
- 0.5B judge is weak: mitigated by deterministic-first design and reporting calibration honestly; may find judge adds little, which is itself a reported result.
- Small models fail almost everything on GSM8K, flattening the race: choose item difficulty so pass rates spread (check after a pilot of 20 items per task, adjust sampling, document).
- Cluster labels may not align with detector tags: report purity; do not tune until it looks good.
- Disk: ~2.8 GB models, no symlinks on Windows cache.

## 10. Decisions needed

1. Download budget: OK with ~2.8 GB default, and optionally +3.1 GB for Qwen2.5-1.5B as stronger judge (default off)?
2. Is an hour-ish of CPU for the full run acceptable while other builds share the machine?
