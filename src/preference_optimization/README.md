# ⚖️ Preference Optimization — Alignment Training

> *SFT teaches the model to imitate. Preference optimization teaches the model to choose between two responses. That distinction is what makes modern chatbots behave the way they do.*

---

## RLHF Overview (`rlhf_overview.py`)

**Reinforcement Learning from Human Feedback** — the technique that made ChatGPT feel different from GPT-3.

**Three-stage pipeline:**

1. **SFT** — model learns to produce reasonable responses.
2. **Reward Model (RM) Training** — collect (prompt, A, B, human choice) triples. Train RM (SFT model + scalar head) to predict human preference.
3. **RL Optimization** — sample from SFT model, score with RM, update via PPO with KL constraint keeping outputs close to SFT.

**Why it was a big deal:** SFT teaches imitation. RLHF teaches preference-following.

**Why it fell out of favor:** complex. Three models. PPO is finicky. Expensive. Modern alternatives.

---

## Reward Modeling (`reward_modeling.py`)

RM turns pairs into a scalar.

**Architecture:** SFT base + scalar prediction head. Training data: `(prompt, chosen, rejected)`. Loss = Bradley-Terry preference:
```
L = -log(σ(r(chosen) - r(rejected)))
```

**Practical considerations:**
- Size typically matches SFT (7B, 13B, 70B). Smaller undertrain.
- 10K-100K preference pairs.
- Prone to overfitting.
- Distributional gap: RM trained on model M performs badly on model N. Retrain when generator changes.

**The RM's flaws become the aligned model's flaws** — any bias/gaming in the RM gets amplified by RL.

---

## PPO for LLMs (`ppo_for_llms.py`)

**Proximal Policy Optimization** — the RL algorithm originally used.

**Setup:** policy (being trained), value (predicts reward), reference (frozen SFT copy), reward model (frozen).

**Loop:** sample response from policy → score with RM → compute KL penalty vs reference → PPO update.

**PPO's clipping:** limits how much policy changes per step, preventing collapse.

**Why PPO is hard:** four models in memory; reward hacking; instability; compute (much more expensive than SFT); hyperparameter sensitivity.

**Frameworks:** `trl.PPOTrainer` handles machinery; still many failure modes.

**In modern practice:** PPO largely replaced by DPO for most fine-tunes.

---

## DPO Deep Dive (`dpo_deep_dive.py`)

**Direct Preference Optimization (Rafailov et al. 2023)** — the paper that made preference optimization mainstream.

**Insight:** you can skip the RM and RL loop entirely. The optimal policy under RLHF has a *closed-form relationship* to the SFT model and preference data. Turn that into a supervised loss.

**The DPO loss:**
```
L = -log σ(β · (log π_θ(chosen|x) - log π_ref(chosen|x))
             - β · (log π_θ(rejected|x) - log π_ref(rejected|x)))
```

Where:
- `π_θ` — policy being trained.
- `π_ref` — frozen reference (usually SFT starting point).
- `β` — controls how far policy can move from reference.

**What this achieves:** no RM. No RL algorithm. Just backprop. Two models in memory. Much more stable than PPO. Comparable or better quality on most benchmarks.

**Trade-offs:** different failure modes than explicit RM; less flexible for online preference; β sensitivity.

**The DPO revolution:** most preference-optimized open-source models from 2023 onwards use DPO or a variant.

---

## DPO Math Intuition (`dpo_math_intuition.py`)

The paper's derivation: under RLHF objective (maximize reward subject to KL constraint), optimal policy satisfies:
```
π*(y|x) ∝ π_ref(y|x) · exp(r(x, y) / β)
```

Solving for r(x, y):
```
r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + log Z(x)
```

**The trick:** in Bradley-Terry, only *differences* in reward matter. Z(x) cancels out:
```
P(chosen ≻ rejected) = σ(β · log(π*(chosen)/π_ref(chosen)) - β · log(π*(rejected)/π_ref(rejected)))
```

**Reward is implicit in the policy ratio.** Training on preferences directly optimizes log-likelihood of chosen relative to rejected, weighted by positions in the reference model.

**Intuition:** DPO tells the model "when I show chosen and rejected, increase chosen's probability relative to rejected, but stay close to the reference."

---

## IPO (`ipo.py`)

**IPO (Azar et al. 2023)** — fix to a subtle DPO issue.

**Problem:** DPO can overfit — `log(π_θ/π_ref)` can grow unbounded; model pushes chosen probabilities to 1 and rejected to 0, losing calibration.

**Fix:** replaces logistic loss with identity-based squared loss that stays bounded:
```
L = (log(π_θ(chosen)/π_ref(chosen)) - log(π_θ(rejected)/π_ref(rejected)) - β/2)²
```

**When:** if DPO shows loss going down but eval degrading (overfitting), try IPO.

---

## KTO (`kto.py`)

**KTO (Ethayarajh et al. 2024)** — preference optimization without pairs.

**Insight:** apply Kahneman-Tversky loss aversion. Data: `(prompt, response, is_desirable)` binary. Asymmetric loss penalizes undesirable more than rewarding desirable.

**Advantages:** point-wise data, not pair-wise. Works with unbalanced positive/negative.

**When:** thumbs-up/thumbs-down feedback rather than pair comparisons.

---

## ORPO (`orpo.py`)

**ORPO (Hong et al. 2024)** — combines SFT and preference in single training run.

**Insight:** normally SFT first, then DPO. ORPO adds preference term directly to SFT loss:
```
L = SFT_loss + λ · odds_ratio_penalty
```

**Advantages:** single stage; no reference model → less memory; competitive with SFT+DPO.

**When:** have preference data during SFT and want to skip separate DPO.

---

## SimPO (`simpo.py`)

**SimPO (Meng et al. 2024)** — simplifies DPO further.

**Insight:** reference model in DPO exists to prevent drift. Careful loss doesn't need it.

**SimPO loss:**
```
L = -log σ(β/|y_chosen| · log π_θ(chosen|x) - β/|y_rejected| · log π_θ(rejected|x) - γ)
```

No reference model. Length-normalized log-probs. Margin γ.

**Advantages:** single model in training. Halves memory vs DPO. Fast. Competitive quality.

---

## Choosing an Alignment Method (`choosing_alignment_method.py`)

**Already ran SFT?**
- Yes: DPO safe default. Memory-constrained: SimPO.
- No: consider ORPO (single-stage).

**Preference data type?**
- Pairs: DPO, IPO, SimPO.
- Binary: KTO.
- Scores/rankings: convert to pairs → DPO.

**Quality vs simplicity?**
- Simplest: DPO.
- Slight edge: IPO if DPO overfits, or careful PPO.
- Absolute best on frontier: well-tuned PPO.

**Memory constrained?** SimPO or ORPO (no reference).

**Cost:** PO ~1-3× SFT cost. PPO ~5-10× DPO.

**2026 default:** SFT → DPO with well-curated preferences → try IPO/KTO if DPO plateaus → only PPO with mature RM.

---

## Alignment Evaluation (`alignment_evaluation.py`)

**Failure mode:** DPO metrics improve on preference dataset. Users don't notice. Or quality drops elsewhere.

**Real evaluation:**
1. **AlpacaEval / MT-Bench** — automated LLM-judge on general quality.
2. **Task-specific golden set** — did the model get better at *your* task?
3. **Regression suite** — did base capabilities survive?
4. **Human eval on real prompts.**
5. **Length distribution** — aligned models often longer. Verify desired.
6. **Refusal behavior** — some methods increase refusals.
7. **Repetition** — alignment can amplify.

**A/B in production:** ultimate test.

**Common failures:** sycophancy, verbosity, over-refusal, style flattening, reward hacking.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `rlhf_overview.py` | Classical three-stage |
| `reward_modeling.py` | Training the RM |
| `ppo_for_llms.py` | The original method |
| `dpo_deep_dive.py` | The shift |
| `dpo_math_intuition.py` | Why it works without RM |
| `ipo.py` | Preventing DPO overfitting |
| `kto.py` | Point-wise preferences |
| `orpo.py` | Combined SFT + preference |
| `simpo.py` | No reference model |
| `choosing_alignment_method.py` | Decision framework |
| `alignment_evaluation.py` | Did it actually help? |

---

*Previous: [← SFT](../supervised_finetuning/README.md) · Next: [Continued Pretraining →](../continued_pretraining/README.md)*  ·  *Back to [main README](../../README.md)*
