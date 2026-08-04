# Methodology

## Research question

How do different linguistic resources (bilingual pairs, trilingual triplets,
a dictionary, curriculum ordering) contribute to adapting a multilingual LLM
for machine translation into a low-resource language (Ekegusii) that the base
model's tokenizer was not trained on?

## Approach: controlled resource ablation

Rather than fine-tuning once and reporting a single number, the project trains
the SAME two base models (Qwen2.5-7B-Instruct, Llama-3.1-8B) under nine progressively
richer resource configurations (E0-E8, see `docs/experiments.md`), holding
everything else constant: QLoRA hyperparameters (`configs/training/*_qlora.yaml`),
random seed (`src.utils.seed.set_seed`), and -- critically -- the evaluation
set (`master_test.csv`, fixed across all nine).

This isolates the variable of interest: whatever score difference appears
between E1 and E4 can be attributed to the added trilingual signal, not to
confounds like a different test set or different training length.

## Fine-tuning method: QLoRA, not full fine-tuning

4-bit NF4 quantization (`bitsandbytes`) + LoRA adapters (rank 32, alpha 64,
targeting all attention/MLP projection matrices) via `peft`. This keeps
trainable parameters to a small fraction of the 8B base model, making it
feasible to train nine configurations x two models within an A100's memory
and time budget. See `src/models/qwen/qlora.py` / `src/models/llama/qlora.py`.

## Why two base models

Qwen2.5-7B-Instruct is explicitly multilingual-pretrained (Alibaba's own claim covers
23 languages, none of which is Ekegusii); Llama-3.1-8B is English-centric
with broader general capability. Comparing them under identical resource
conditions tests whether general multilingual pretraining transfers better
to an unseen low-resource language than raw scale/capability -- see
`src/tokenizer/compare.py`'s fertility comparison for the tokenizer-level
version of this question (notebook 04).

## Task formulation: instruction-tuned translation, not seq2seq

Both models are causal LMs, fine-tuned on prompt/response instruction pairs
(`src.task_generation.instruction_generator.InstructionTaskGenerator`) rather
than trained as dedicated encoder-decoder translation models. This matches
how these models are actually deployed (chat/instruction interfaces) and
lets the same checkpoint handle all six translation directions.

## Validity threats addressed

- **Data leakage**: `src.master_corpus.integrity.DataLeakageChecker` verifies
  zero concept-ID overlap across splits before any experiment runs.
- **Non-comparable test sets**: enforced structurally -- see `docs/reproducibility.md`.
- **Point-estimate false claims**: every experiment comparison that matters for
  the paper's conclusions goes through `PairedBootstrapTest`, not a bare mean difference.
