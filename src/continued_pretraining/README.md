# 📖 Continued Pretraining — Domain Adaptation

> *SFT changes behavior. DPO changes preferences. Continued pretraining changes what the model knows. The heaviest tool, most often over-reached-for, but the right answer for genuine domain adaptation.*

---

## When To Continue Pretraining (`when_to_continue_pretrain.py`)

**Legitimate:**
1. Genuine domain vocabulary/knowledge base lacks (medical, legal, financial, scientific, proprietary code).
2. Underrepresented language (adapting English-heavy Llama for Japanese, Vietnamese, Arabic).
3. Large corpus of high-signal domain text that can't be effectively used via RAG.
4. Different distributional target.

**Illegitimate:**
- "5GB of company docs" → RAG.
- "Doesn't know our product" → SFT + RAG.
- "Want it smarter at our task" → task fine-tuning.
- "Want it to know new events" → regularly-retrieved RAG or new base.
- "Have preference data" → DPO.

**Threshold:** <500M tokens of high-quality domain text → probably not warranted. RAG + SFT covers cheaper.

**Rule:** CPT is expensive, error-prone, and destroys capabilities if done wrong. Try alternatives first.

---

## Domain Adaptation Datasets (`domain_adaptation_datasets.py`)

**Characteristics of good CPT corpus:**
- **Volume:** 1B-100B tokens typical.
- **High signal-to-noise:** no boilerplate, low-quality translations.
- **Domain-representative:** full range of topics, formats, registers.
- **Diverse sources:** books, articles, structured data, dialogues.
- **Legally licensed** — provenance matters.

**Sources:** proprietary corpora (with license), domain-specific public (PubMed, CaseLaw, SEC EDGAR), filtered Common Crawl subsets, high-quality digital books.

**Preprocessing (heavy):** dedup (multiple levels), quality filter (perplexity, classifier), language ID, PII scrub, toxicity, length filter, format normalization, tokenize.

**Real-world attrition:** raw → training-ready is **50-90% loss**.

---

## Continued Pretraining Recipes (`continued_pretraining_recipes.py`)

**Different from SFT:**

**Learning rate:** 1e-5 to 1e-4 (vs SFT's 1e-4 to 5e-4). Model close to pretraining optimum.

**Data mixing (critical):** domain 70-95%, general (from original pretraining distribution) 5-30%. General data prevents catastrophic forgetting.

**Duration:** 1 epoch often enough for large corpora. Multiple epochs on smaller risks overfitting. Track token budget: "20B tokens total."

**Batch size:** large. Effective batch 4M-16M tokens. Multi-GPU/multi-node.

**Warmup:** 3-5% of steps. **Schedule:** cosine to ~10% of peak.

**Precision:** BF16. Full FT — PEFT generally not adequate for CPT.

**Checkpointing:** frequent (every 1-5%). Recovery from crashes common at this scale.

---

## Catastrophic Forgetting Mitigation (`catastrophic_forgetting_mitigation.py`)

CPT without care destroys base capabilities.

**Mitigations:**
1. **Data mixing (primary defense):** 5-30% general data.
2. **Lower LR:** 5-10× lower than SFT default.
3. **Regularization:** small weight decay.
4. **Rehearsal:** sample from original pretraining distribution. Requires having that data (often proprietary).
5. **EWC and related:** penalty for moving weights far from pretraining values. Rarely used at LLM scale.
6. **Evaluation-driven early stopping:** regression suite periodically. Stop when threshold exceeded.

**Every CPT project needs a regression suite. Non-negotiable.**

---

## Vocabulary Extension (`vocabulary_extension.py`)

For languages/domains with poorly-represented tokens, extending vocabulary can help.

**Problem:** Llama's tokenizer splits Japanese into bytes; each character becomes multiple tokens.

**Solution:**
1. Train small BPE/unigram tokenizer on domain corpus.
2. Merge new tokens into base tokenizer.
3. Add embedding rows for new tokens.
4. Initialize sensibly (mean of constituent old tokens, or small random).
5. Train with new tokens active (CPT with extended tokenizer).

**Cautions:** adds parameters; model must learn new tokens through training; irreversible in serving.

**When:** language adaptation where token efficiency matters (Japanese, Arabic, Thai, code). Skip for English domain adaptation.

---

## Evaluation of Domain Models (`evaluation_of_domain_models.py`)

**Layers:**
1. **Domain-specific benchmarks** (MedQA, PubMedQA, LegalBench, FinQA).
2. **Task-specific downstream** (extraction, classification, answer quality).
3. **General capability retention (regression):** MMLU, HellaSwag, ARC, HumanEval, GSM8K. Compare to base — how much dropped?
4. **Perplexity on domain-held-out.**
5. **Human evaluation** by domain experts.

**Ideal outcome:** domain metrics improve 10-40pp; general within ~5pp of base.

**Red flags:** general drops >10pp; unstable evals across checkpoints; domain up but human review bad.

---

## Continued Pretrain Case Studies (`continued_pretrain_case_studies.py`)

**Med-PaLM / Med-PaLM 2:** Google. CPT of PaLM on massive medical text + instruction tuning. State-of-the-art on medical Q&A; expert-level on tests.

**BloombergGPT:** financial LLM. Trained from scratch on 700B tokens finance + general. Interesting paper comparing from-scratch vs continued.

**Code Llama (from Llama-2):** 500B code tokens. Careful curation, long CPT, then SFT on code instructions.

**Chinese/Japanese/Korean Llama variants:** extensive CPT on target-language + vocabulary extension. Done poorly → worse; done well → production-viable.

**Enterprise patterns:** financial services adapting Llama for internal QA; healthcare for clinical documentation; legal for case law. 5-30B token corpus, careful mixing, 1-3 weeks, then SFT.

**Lessons:** small curated corpora often beat larger noisy; mixing general data preserves capabilities; eval infrastructure as important as training infrastructure; regular retraining as domain grows.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `when_to_continue_pretrain.py` | Legitimate vs alternatives |
| `domain_adaptation_datasets.py` | Corpus preparation |
| `continued_pretraining_recipes.py` | LR, data mix, duration |
| `catastrophic_forgetting_mitigation.py` | Data mix, LR, regression |
| `vocabulary_extension.py` | Adding domain tokens |
| `evaluation_of_domain_models.py` | Multi-layer eval |
| `continued_pretrain_case_studies.py` | Med-PaLM, BloombergGPT, Code Llama |

---

*Previous: [← Preference Optimization](../preference_optimization/README.md) · Next: [Training Infrastructure →](../training_infrastructure/README.md)*  ·  *Back to [main README](../../README.md)*
