# ChronoTrack: Testing Order-Independent Temporal State Tracking in LLMs

## 1. The capability gap

Existing temporal-reasoning benchmarks (TimeQA, TempReason, StreamingQA, the bAbI
time-tracking tasks) overwhelmingly present events **in chronological order**. A model
can score well on these by learning a shortcut: *the last-mentioned state is the
current state*. That shortcut is right most of the time in naturally-written text,
but it is not temporal reasoning — it's recency-of-mention, not recency-of-time.

Real-world text routinely breaks this assumption: emails reference an update from
last week after describing today's meeting; contracts reference an amendment before
describing the original clause; witnesses recall events out of order. A model that
actually understands time should reconstruct the true sequence from explicit
timestamps regardless of narrative order. **ChronoTrack measures exactly this gap**,
isolating it from confounds like world knowledge, arithmetic, or multi-hop retrieval.

**Hypothesis under test:** open-source instruction-tuned models in the 7–8B class
will show a measurable accuracy drop when narrative order is scrambled relative to
chronological order, and this drop will grow with distractor load and with query
difficulty (state-at-time vs. final-state).

## 2. Task design

Each item is a short synthetic narrative describing a sequence of dated events
("On Tuesday, the backpack was handed over to Deo.") about one *target* entity
(location or possessor tracked over the week) optionally interleaved with events
about *distractor* entities that are irrelevant to the question.

Two independent variables are manipulated per item:

| Variable | Levels | What it tests |
|---|---|---|
| **Order mode** | chronological / scrambled | Whether the model relies on narrative position vs. actual date to determine recency |
| **Distractor load** | 0 / 2 / 4 distractor entities | Whether irrelevant interleaved events degrade tracking |

Two query types probe different reasoning depth:

- **final_state** — "At the end of the week, where was the X?" (needs: find the
  max-day update for the target entity)
- **state_at_time** — "As of Wednesday, where was the X?" (needs: find the most
  recent update *at or before* Wednesday — genuine interval reasoning, since the
  answer is often *not* the update nearest the question in the text)

Ground truth is computed programmatically from the underlying event objects, never
from the rendered text, so scoring is unambiguous exact string match (normalized:
lowercased, punctuation stripped).

A **self-consistency probe** accompanies every item: the same question is asked a
second time with a syntactic paraphrase ("where was the X" → "in what place was
the X"). Agreement between the two answers, independent of correctness, measures
whether a model's internal state representation is stable or whether it's
re-deriving (and sometimes flip-flopping on) the answer each time — a proxy for
the "logical consistency" capability adjacent to temporal reasoning.

## 3. Dataset

Procedurally generated (`generate_dataset.py`), fully reproducible via a fixed seed.

- **180 items**, balanced 2 (order) × 3 (distractor load) × 2 (query type) × 15 items/cell
- Entities drawn from fixed pools (12 objects, 12 locations, 8 people) to keep
  vocabulary difficulty constant across conditions — the manipulation is purely
  structural, not lexical
- Target entity has 3–4 dated events; each distractor entity has 2–3
- Sentence templates are varied (3 phrasings per event type) so models can't
  pattern-match on a single surface form

This design deliberately trades naturalism for **causal cleanliness**: because
order and distractor load are independently randomized and ground truth is
computed outside the text, any accuracy difference between conditions can be
attributed to the manipulated variable rather than confounds like vocabulary
frequency or sentence length.

## 4. Metrics

1. **Overall accuracy** per model (exact match against gold)
2. **Accuracy by condition** — the full model × order × distractor × query_type table
3. **Order-sensitivity gap** = accuracy(chronological) − accuracy(scrambled), the
   primary diagnostic for the capability gap this benchmark targets
4. **Distractor degradation** = accuracy(0 distractors) − accuracy(4 distractors)
5. **Query-depth gap** = accuracy(final_state) − accuracy(state_at_time)
6. **Self-consistency rate** = agreement rate between paraphrased query pairs

## 5. Experimental protocol

- **Models:** LLaMA 3 (8B-instruct), Mistral (7B-instruct), Phi-3 (mini-instruct),
  Gemma (7B-it) — all runnable locally via Ollama for a fair, identical-hardware
  comparison
- **Decoding:** temperature 0, max 30 tokens, single fixed system prompt across
  all models (see `eval_harness.py`) — zero-shot, no few-shot exemplars, to avoid
  leaking the answer format as an implicit reasoning strategy
- **Scoring:** programmatic exact match after normalization; answer is extracted
  from a constrained `Answer: <value>` format to avoid free-text parsing ambiguity
- **Repetitions:** the harness is deterministic (temp 0) so single-pass scoring is
  used; for a stricter study, run 3 seeds of the dataset and report mean ± std

## 6. How to run this yourself

```bash
# 1. Install Ollama and pull the four models
ollama pull llama3
ollama pull mistral
ollama pull phi3
ollama pull gemma

# 2. Generate the dataset (already done — chronotrack_dataset.jsonl is included)
python3 generate_dataset.py

# 3. Run the pilot (fast, ~20 items/model) or full run (180 items/model)
python3 eval_harness.py --dataset chronotrack_dataset.jsonl --limit 40
python3 eval_harness.py --dataset chronotrack_dataset.jsonl   # full run

# Outputs: results_raw.csv (per-item), results_summary.csv (aggregated),
# plus a printed summary table with the order-sensitivity gap highlighted.
```

The harness talks to any OpenAI-compatible `/v1/chat/completions` endpoint, so it
also works unmodified against vLLM, LM Studio, or a hosted API — just edit
`MODEL_CONFIGS` in `eval_harness.py`.

## 7. Results

*Not run in this environment* — the sandbox this benchmark was designed in has no
network access to model-hosting services (Ollama/HuggingFace are outside the
allowed domain list), so no live inference against LLaMA 3 / Mistral / Phi-3 /
Gemma was performed. Reporting fabricated numbers here would misrepresent the
models, so this section is intentionally left as a template — run the commands
above and drop your `results_summary.csv` output into the table below.

| Model | Overall Acc | Chrono Acc | Scrambled Acc | Order Gap | Distractor Gap (0→4) | Query Gap (final→state_at_time) | Self-Consistency |
|---|---|---|---|---|---|---|---|
| LLaMA 3 8B | | | | | | | |
| Mistral 7B | | | | | | | |
| Phi-3 mini | | | | | | | |
| Gemma 7B | | | | | | | |

**Suggested write-up once populated:** report the order-sensitivity gap as the
headline finding (this is the capability gap the benchmark targets), note which
model degrades least under scrambling (likely correlates with instruction-tuning
quality more than raw parameter count, worth checking against Phi-3's smaller
size), and flag any model with high accuracy but low self-consistency — that
combination indicates answer instability that raw accuracy alone would hide.

## 8. Limitations

- Synthetic template text is easier to parse than natural prose; treat this as a
  controlled diagnostic, not a proxy for real-document temporal QA
- English-only; the entity/location/people pools are Western-name-biased and
  should be diversified for a cultural-robustness follow-up
- 30-token answer budget assumes single-word/name answers; doesn't test
  explanatory temporal reasoning (e.g., "why" questions), only state retrieval
- No adversarial contradiction injection yet (e.g., two events claiming the same
  day) — a natural v2 extension for testing contradiction resolution specifically
