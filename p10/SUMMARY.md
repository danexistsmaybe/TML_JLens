# P10 morning briefing: the Jacobian lens and the fitting-corpus question

*Read this first. About 15 minutes. Written overnight 2026-10-16 from the primary sources listed at the bottom.*

---

## 1. The one-paragraph version

A **lens** is a way to read an intermediate activation of a language model as a ranked list of vocabulary tokens ("what is this layer thinking about?"). The simplest one, the **logit lens**, just pushes the intermediate residual stream through the model's final norm and unembedding as if it were the last layer. That works in late layers but gives garbage early on, because earlier layers use different "coordinates". The **Jacobian lens (J-lens)**, introduced by Anthropic in July 2026, fixes this by first transporting the activation into final-layer coordinates with a matrix **J_ℓ = the average Jacobian ∂h_final/∂h_ℓ**, averaged over many contexts from a **fitting corpus**. The averaging is what makes it a statement about the model's *general* disposition to say a token rather than about one particular prompt. **P10 asks: since J_ℓ is an average over a corpus, does the choice of corpus change what the lens tells you, and can we pick or weight the corpus to get a better lens for a new domain?**

---

## 2. Background: lenses

All lenses share one template. Take the residual stream h_ℓ at layer ℓ and position t, map it somehow into final-layer space, then decode with the model's own head:

    readout(h_ℓ) = softmax( W_U · norm( T_ℓ(h_ℓ) ) )

| Lens | T_ℓ | How it's obtained | Known weakness |
|---|---|---|---|
| **Logit lens** (nostalgebraist, 2020) | identity | nothing to fit | Assumes every layer uses final-layer coordinates; noisy/uninterpretable in early-middle layers |
| **Tuned lens** (Belrose et al., 2023, arXiv 2303.08112) | learned affine map A_ℓh + b_ℓ | trained to minimize KL to the model's *final output distribution* | Correlational. Per the J-lens paper it "skips ahead" and shows the eventual answer instead of intermediate concepts; its early-layer advantage comes mostly from the bias term |
| **Jacobian lens** (Anthropic, 2026) | linear map J_ℓ h | *average Jacobian* ∂h_final/∂h_ℓ over a corpus | Only single-token concepts; noisy in first ~third of layers; depends on the fitting corpus ← **your project** |

Key contrast to keep straight for Q&A: the tuned lens is *trained* to predict the output; the J-lens is *computed* from the model's own derivatives. That makes the J-lens a statement about **first-order causal effect**, not prediction accuracy. In fact, the J-lens is the *worst* of the three at predicting the next token (paper, Fig. 55), and the authors consider that a feature, because it surfaces intermediate concepts instead of the answer.

---

## 3. The Jacobian lens in detail

### Definition
For each source layer ℓ:

    J_ℓ = E_{prompt, t, t' ≥ t} [ ∂h_final,t' / ∂h_ℓ,t ]     (a d_model × d_model matrix)

- **h_ℓ,t**: residual stream (output of block ℓ) at position t.
- **h_final,t'**: residual stream at the target layer, at a *current or later* position t'. A perturbation at t can affect every later position through attention, so the lens captures "things I might say now **or later**".
- **Expectation**: over source positions, target positions, and a corpus of prompts (paper default: **1000 sequences × 128 tokens** of pretraining-like text).

Readout: `lens(h_ℓ) = softmax(W_U · norm(J_ℓ h_ℓ))`. The rows of `W_U J_ℓ` are the **J-lens vectors**: one direction in residual space per vocabulary token. Logit lens = the special case J_ℓ = I.

### How it's actually computed (reference code `jlens/fitting.py`)
Computing a full Jacobian naively is expensive, so the reference implementation uses a trick:
1. One forward pass on the prompt, replicated `dim_batch` times.
2. For each output dimension i, inject a one-hot gradient e_i **at every valid target position simultaneously** and backprop to the source layer. The gradient at source position p is then Σ_{p'≥p} ∂h_final[p']_i / ∂h_ℓ[p]. That is one row of J.
3. Average those rows over valid source positions, giving one d×d matrix per prompt.
4. Average across prompts (per-element mean).

Cost: 1 forward + ⌈d_model / dim_batch⌉ backward passes per prompt. For Qwen2.5-0.5B (d_model = 896, 24 layers) that's cheap; see the pilot results for measured numbers.

**Conventions in the reference code you must match and report** (the course brief insists on this):
- "Layer ℓ" = output of decoder block ℓ (0-indexed).
- Target = output of the final block, **before** the final norm. The paper's Claude experiments used the penultimate layer instead (fewer artifacts). This is a documented choice.
- First **16 positions** are skipped (attention-sink artifacts), and the last position is skipped.
- Sum over target positions, then mean over source positions, then mean over prompts.
- Decode with `lm_head(final_norm(·))`.

### Design choices the paper ablates (Appendix "Methodological Details")
- **Gradient of what**: final vs penultimate target layer; frozen-QK (block gradients through attention patterns); self-only (t'=t) vs future-only (t'>t).
- **Aggregation**: mean vs median; drop outlier positions/prompts by norm.
- **Data amount**: J-lens beats logit and tuned lenses with as few as **10 prompts**; gains are modest after that.
- **Data distribution**: they only tried *restricting* the pretraining-like corpus and masking positions, and found "no meaningful improvement." **They did not study fitting on a genuinely different domain.** That is exactly P10's gap.

### What the paper uses it for
- **Reading**: top-k tokens; per-token probe (cosine with one J-lens vector); sparse decomposition of an activation into k ≈ 25 J-lens vectors.
- **Writing (interventions)**: steering `h ← h + α v_t`; ablation (project out v_t); **lens-coordinate swap**: with V = [v_s, v_t], c = V⁺h, set h' = h + V(σ(c) − c). This swaps concept s for t and leaves the orthogonal part alone. These are the basis for P10's required "intervention check".

### Evaluation methods from the paper you can borrow
- **pass@k AUC** on prompts with a *known intermediate* (e.g. "color of the planet fourth from the Sun" → intermediate **Mars**, answer **red**). Six sets: multihop, multilingual, order-of-operations, poetry, typo, association. These are public in the repo (`data/evaluations/`), and the course says to use them **for development only**.
- **Causal tests**: KL on the output when a lens vector is ablated; **swap success rate** (does the output flip to the swapped-in concept's answer?). J-lens directions have roughly 2× the ablation effect of logit/tuned-lens directions.
- **Inter-lens similarity** per layer: mean cosine between corresponding token vectors, symmetric KL between readouts, top-1 agreement. **This is directly reusable for comparing a WikiText-fit lens with a TinyStories-fit lens.**

---

## 4. The paper's broader claim (context for your intro slide)

*"Verbalizable Representations Form a Global Workspace in Language Models"* (Anthropic, transformer-circuits.pub, July 6 2026). Using the J-lens, the authors argue that LLMs have a small, privileged set of representations, the **J-space** (sparse nonnegative combinations of J-lens vectors), that behaves like a "global workspace" in the sense of cognitive science. They show it has five properties: verbal report, directed modulation, internal reasoning, flexible generalization, and selectivity. Structurally:
- It is workspace-like only in a **middle band of layers** (about 38%–92% of depth): early layers give noisy readouts ("sensory"), and the final layers just encode the next token ("motor").
- It is low capacity: around 25 active concepts, at most 10% of activation variance.
- It is used for alignment auditing (e.g. surfacing "evaluation awareness" or hidden objectives).

You do **not** need to engage with the consciousness framing. For P10, what matters is that the J-lens is presented as *the* readout of what a model is disposed to verbalize, and the authors' own data ablations never seriously tested **corpus dependence**. If a lens fit on Wikipedia reads TinyStories prompts differently than a lens fit on TinyStories, then "what the model is thinking" partly depends on the analyst's choice of fitting corpus. That's a trustworthiness question.

Course reminders (from the instructions §6): agreement with expectations ≠ faithfulness; a token readout does not establish intentions; scope claims to one small model.

---

## 5. The P10 problem, concretely

**Research question:** Which lens readouts transfer between fitting corpora, and can corpus weighting improve interpretation on a new prompt domain?

**Setup (from the brief):**
- Model: **Qwen2.5-0.5B** (24 layers, d_model 896), frozen. Two intermediate blocks.
- Corpora: **WikiText-2** (encyclopedic) and **TinyStories** (simple children's stories). 100 training windows each, matched sequence length and token budget, plus held-out validation/test windows per domain.
- Controlled prompts with known entities/properties and explicit target tokens, split by template/entity.

**You must implement yourselves:**
1. Corpus-conditioned estimator: J_ℓ fit on one corpus.
2. Corpus-weighted pooled estimator: J_ℓ = Σ_i w_i J_ℓ^(i) over windows from both corpora.
3. Readout evaluator.
4. Intervention check: small ± edits along lens directions; compare actual output change to the lens's prediction.

**Original component (suggested):** choose the weights w_i by how similar each fitting window is to the **deployment domain**, using frozen-model features (e.g. mean mid-layer activation, cosine to the deployment-domain centroid, softmax with temperature τ chosen on dev data). Compare at matched compute against:
- **Baselines:** logit lens, WikiText-only lens, TinyStories-only lens, uniform pooled lens.
- **Ablations:** no weighting; random weights; control for token frequency and length. Use repeated fitting-sample seeds.

**Metrics:** controlled-token retrieval (pass@k), cross-domain derivative/intervention prediction error, readout agreement, runtime, and sensitivity to sample count.

**Deliverable shape:** a **cross-domain evaluation matrix** (rows = lens fit on X, columns = evaluated on domain Y) plus a viewer comparing readouts for the same prompt.

**Plausible outcomes to mention in the talk:**
- *Expected:* lens matrices from the two corpora agree in late layers but diverge in middle layers. A lens fit in-domain retrieves controlled tokens better in-domain. Weighting recovers part of that gap.
- *Plausible failure:* J_ℓ is nearly corpus-invariant, consistent with the paper's "10 prompts suffice" finding, so weighting adds nothing. That is a legitimate negative result: "J-lens readouts are robust to fitting-corpus choice at this scale."

---

## 6. Glossary for Q&A

- **Residual stream**: the running vector each block reads from and adds to.
- **Unembedding W_U / final norm**: the head that turns the last residual into logits.
- **Jacobian ∂h_final/∂h_ℓ**: the matrix of first-order effects of perturbing layer ℓ on the final layer.
- **Cotangent / VJP**: the vector injected at the output in a backward pass. One backward pass with cotangent e_i gives row i of J.
- **J-lens vector**: row of W_U J_ℓ for one token. Its dot product with h approximates that token's lens logit.
- **J-space**: sparse nonnegative combinations of J-lens vectors.
- **Lens-coordinate swap**: c = V⁺h, h' = h + V(σ(c) − c).
- **pass@k**: fraction of known intermediates appearing in the top-k readout at any layer.
- **Attention sink**: early positions (especially the first token) soak up attention and have atypical activations. Hence skip-first-16.
- **Domain shift / importance weighting**: reweighting source samples toward a target domain. The weighted J-lens is this idea applied to the averaging step.

---

## 7. Sources
- Lindsey et al., *Verbalizable Representations Form a Global Workspace in Language Models*, transformer-circuits.pub/2026/workspace, July 6 2026. Read: Intro, Methods, Layer-band section, Appendix (Qualitative and Quantitative Comparisons, Methodological Details, Pseudocode), Limitations. Plain-text copy: `notes/workspace.txt`.
- Reference code: github.com/anthropics/jacobian-lens @ `581d398`. `fitting.py` and `lens.py` copied to `notes/raw/jlens_ref/` **for reading only**. Our estimator is written separately in `src/`.
- Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*, arXiv 2303.08112 (abstract plus the J-lens paper's comparison).
- Course brief: `../course-docs/Project_Descriptions.md` §P10.
