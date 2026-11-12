# EvalBench research notes

Notes I made before building. Anything marked "measured" I checked myself on 2026-11-12; the rest comes from the linked sources.

## 1. The field

| Tool | What it does well | What I take from it | Where EvalBench differs |
|---|---|---|---|
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (EleutherAI, MIT) | Standard for academic benchmarks; YAML task configs; backend of the HF Open LLM Leaderboard | Tasks as declarative config with a named metric and a pinned version; `exact_match` with filters (regex extraction before comparing) | It reports one aggregate per task. EvalBench keeps every sample and asks why it failed. |
| [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) (UK AISI, MIT) | Clean dataset / solver / scorer split; log viewer; `model_graded_qa` and `match` scorers | The three-part task model, and logging every sample with its score and explanation | Inspect's viewer is a log browser. EvalBench's centre is the failure map. |
| [OpenAI evals](https://github.com/openai/evals) (MIT) | Registry of JSONL eval sets; basic match / includes / model-graded classes | JSONL sample format, "includes" vs "exact" distinction | Needs a hosted model for grading; mine runs offline. |
| [promptfoo](https://github.com/promptfoo/promptfoo) (MIT) | Matrix view of prompts x models, assertions per test case, CI friendly | Assertion-per-case idea, side-by-side matrix | Pass/fail table, no clustering of failures. |
| [DeepEval](https://github.com/confident-ai/deepeval) (Apache-2.0) | pytest-style LLM tests, G-Eval rubric metric | Rubric-as-criteria prompt (G-Eval style) | Metrics default to hosted judges. |
| [HELM](https://crfm.stanford.edu/helm/) (Stanford, Apache-2.0) | Holistic, many metrics per scenario, per-instance prompts and outputs released | Reporting several metrics per scenario, not one score; publishing per-instance data | Large scale; not designed for a fine-tuning loop on a laptop. |

EvalBench starts where these tools stop at "score per task". The next step I wanted is grouping the failing samples into modes so the next fine-tuning cycle has a target. Practitioner write-ups on error analysis ([Atla](https://www.atla-ai.com/post/automating-error-analysis), [Strick](https://alexstrick.com/posts/2025-05-23-error-analysis-to-find-failure-modes.html)) describe the same workflow by hand: read failures, label, cluster. Atla reports that embedding then non-parametric clustering gave more specific patterns than asking an LLM to cluster everything in one prompt, and scaled better. I will do that, with deterministic detectors supplying the labels so the modes are not an LLM's opinion. Taxonomy references: [ErrorAtlas](https://www.emergentmind.com/topics/erroratlas) (17 categories incl. inappropriate refusal; I only read a secondary summary), [system-level taxonomy of 15 failure modes](https://arxiv.org/pdf/2511.19933). A caution from [arXiv 2512.21422](https://arxiv.org/html/2512.21422v1): clustering regions of failure has not been shown to teach users reliably, so I must show real samples next to every cluster and not present labels as ground truth.

## 2. LLM-as-judge reliability (why the design is deterministic-first)

- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/html/2306.05685v4): documents position bias, verbosity bias, self-enhancement bias and weak reasoning/maths grading. GPT-4 agreed with humans about 85% (no ties) vs 81% human-human. Those results are for a frontier judge; a 0.5B judge will be much worse.
- [Prometheus 2](https://arxiv.org/abs/2405.01535) (EMNLP 2024): open evaluator models with direct-assessment (rubric) and pairwise modes. The smallest is 7B, too large for a CPU budget here; I borrow its rubric prompt format only.
- Consequences for the design:
  1. Wherever a deterministic check exists (exact match, regex, JSON schema, constraint verifiers, refusal lexicon) it is the source of truth.
  2. The small local judge is used only for narrow yes/no rubric questions, scored by token log-probabilities of "Yes"/"No" (no free-text parsing), run in both orders for pairwise comparisons, and ties are kept as ties.
  3. The judge is validated, not trusted: on every item with a deterministic gold label EvalBench reports judge accuracy, Cohen's kappa and position-flip rate. These numbers go in the README and UI, so the weakness of the judge is visible.
  4. Self-preference is real: the judge model is also one of the checkpoints. The UI flags that checkpoint's judged scores.
  5. Win-rates carry bootstrap confidence intervals; with ~120 prompts differences of a few points are noise and the UI should say so.

## 3. Models (licences read from the Hugging Face API, measured)

| Model | Licence | Size (safetensors) | Role |
|---|---|---|---|
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Apache-2.0 | 0.27 GB | checkpoint |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) (base) | Apache-2.0 | 0.72 GB | checkpoint (pre post-training) |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Apache-2.0 | 0.72 GB | checkpoint (after SFT/DPO) |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0 | 0.99 GB | checkpoint and default judge |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Apache-2.0 | 3.09 GB | optional stronger judge / 5th checkpoint |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 | 0.09 GB (already cached) | failure embeddings |

Rejected: Qwen2.5-3B-Instruct (licence is `other`, the Qwen research licence), Gemma-3 and Llama-3.2 (gated, custom licences). Qwen3 0.6B/1.7B/4B are Apache-2.0 but are reasoning-mode models whose think-tokens make CPU runs slow and outputs awkward to score.

The ladder is deliberately a real post-training story: base 360M -> instruct 360M is the same architecture before and after post-training, and the sizes 135M / 360M / 500M show scale. Total download for the default set is about 2.8 GB, within the 3 GB budget.

Measured CPU speed (torch 2.11 CPU, 4 threads, fp32, greedy, single prompt, includes warm-up, on a busy desktop): SmolLM2-360M-Instruct 8.6 tok/s, Qwen2.5-0.5B-Instruct 8.7 tok/s. Batched generation will be faster per token. The real wall time of the full run is in the README. A recorded run is committed so nobody has to wait.

## 4. Datasets (licences read from the Hub API, measured)

| Dataset | Licence | Use |
|---|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MIT | numeric exact match, reasoning failures |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | MIT | multiple choice, letter exact match |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | Apache-2.0 | prompts for verifiable format constraints (I implement the checks for a subset myself) |
| [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) | Apache-2.0 | misconception / hallucination probes |
| [XSTest](https://huggingface.co/datasets/natolambert/xstest-v2-copy) (Hub copy; original walledai copy is auto-gated) | CC-BY-4.0 | safe prompts that look unsafe, over-refusal |
| Authored by me | MIT | fake-entity questions (correct behaviour: say it does not know), JSON/format tasks, refusal-bait that is harmless |

Rejected: ARC and SQuAD v2 (CC-BY-SA share-alike), OpenBookQA (licence unknown). Datasets are fetched by script and sampled with a fixed seed into a versioned `suite.jsonl` with a content hash; only the authored items and the sampled suite (small, with attribution) are committed.

## 5. Clustering approach

- Embed each failing sample (prompt tail + output + task tag) with MiniLM. Add cheap structured features (length, truncated flag, has-JSON, refusal-lexicon hit) so format breaks and refusals separate from content errors.
- Cluster with scikit-learn 1.8 (`HDBSCAN` is in core; KMeans with silhouette-chosen k as fallback). UMAP and hdbscan are not installed; I will not add them. 2D layout for the UI via PCA then t-SNE (both in scikit-learn), seeded.
- Label each cluster by: (a) majority vote of the deterministic detector tags inside it; (b) nearest taxonomy seed description in embedding space as tie-break; (c) top TF-IDF terms as a human check. Report cluster purity against the detector tags, so the claim "clusters match modes" is a measured number.

## 6. Environment facts (measured)

transformers 5.7.0, sentence-transformers 5.4.1, scikit-learn 1.8.0, pandas 3.0.2, torch 2.11.0+cpu, 12 logical cores. Only MiniLM is cached so far. `huggingface_hub` 1.13; symlinks unsupported on this Windows setup (cache works, just bigger). `datasets` is not installed; the build will read Parquet/JSON via `huggingface_hub` + pandas instead of adding it.
