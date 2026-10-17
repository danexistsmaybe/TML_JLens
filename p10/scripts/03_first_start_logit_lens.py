"""First-start experiment from the brief: logit-lens readouts on ten matched-length windows
from each corpus, using one residual convention (block outputs; decode = lm_head(norm(.))).

Per layer and corpus, over valid positions (16..126) of 10 VAL windows:
  * top-1 agreement between the lens readout and the model's own top-1 prediction
  * KL(model || lens)
  * top-1 accuracy against the actual next token (model's own accuracy shown at the last layer)
Also dumps a qualitative table of top-5 readouts for one position per corpus.

Outputs: results/first_start_logit_lens.csv, results/first_start_examples.md,
         figures/fig1_logit_lens_by_corpus.png
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import P10, SKIP_FIRST, decode, load_manifest, load_model, residuals, window_ids  # noqa: E402

N_WIN = 10
model, tok = load_model()
L = model.config.num_hidden_layers
man = load_manifest()
rows, examples = [], []
for corpus in ["wikitext", "tinystories"]:
    ids = torch.tensor(window_ids(man, corpus, "val")[:N_WIN], device="cuda")
    acc = {l: dict(agree=0.0, kl=0.0, nexttok=0.0) for l in range(L)}
    n = 0
    for b in range(ids.shape[0]):
        x = ids[b:b + 1]
        res = residuals(model, x, range(L))
        pos = torch.arange(SKIP_FIRST, x.shape[1] - 1, device="cuda")
        logp_model = F.log_softmax(res["logits"][0, pos], -1)
        top_model = logp_model.argmax(-1)
        nxt = x[0, pos + 1]
        for l in range(L):
            lg = decode(model, res[l][0, pos])
            logp = F.log_softmax(lg, -1)
            acc[l]["agree"] += (lg.argmax(-1) == top_model).float().sum().item()
            acc[l]["kl"] += F.kl_div(logp, logp_model, log_target=True, reduction="sum").item()
            acc[l]["nexttok"] += (lg.argmax(-1) == nxt).float().sum().item()
        n += len(pos)
        if b == 0:
            p = 60
            ex = {"corpus": corpus, "context": tok.decode(x[0, p - 12:p + 1]), "next": tok.decode(x[0, p + 1])}
            for l in [4, 8, 12, 16, 20, 23]:
                top = decode(model, res[l][0, p]).topk(5).indices
                ex[l] = [tok.decode([t]) for t in top]
            examples.append(ex)
    for l in range(L):
        rows.append(dict(corpus=corpus, layer=l, n_positions=n,
                         top1_agree_with_model=acc[l]["agree"] / n, kl_model_lens=acc[l]["kl"] / n,
                         next_token_acc=acc[l]["nexttok"] / n))

(P10 / "results").mkdir(exist_ok=True)
with open(P10 / "results" / "first_start_logit_lens.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(P10 / "results" / "first_start_examples.md", "w") as f:
    f.write("# Logit-lens top-5 readouts at position 60 of the first val window\n\n")
    for ex in examples:
        f.write(f"**{ex['corpus']}** context: `...{ex['context']!r}` -> actual next `{ex['next']!r}`\n\n")
        f.write("| layer | top-5 |\n|---|---|\n")
        for l in [4, 8, 12, 16, 20, 23]:
            f.write(f"| {l} | {' · '.join(repr(t) for t in ex[l])} |\n")
        f.write("\n")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for corpus, c in [("wikitext", "#1f6fb4"), ("tinystories", "#d1621b")]:
    r = [x for x in rows if x["corpus"] == corpus]
    ax[0].plot([x["layer"] for x in r], [x["top1_agree_with_model"] for x in r], "o-", ms=3, color=c, label=corpus)
    ax[1].plot([x["layer"] for x in r], [x["kl_model_lens"] for x in r], "o-", ms=3, color=c, label=corpus)
ax[0].set(xlabel="layer (block output, 0-indexed)", ylabel="top-1 agreement with model", title="Logit lens: agreement with final output")
ax[1].set(xlabel="layer (block output, 0-indexed)", ylabel="KL(model || lens) [nats]", title="Logit lens: KL to final output", yscale="log")
for a in ax:
    a.grid(alpha=.3)
    a.legend()
fig.suptitle(f"Qwen2.5-0.5B, {N_WIN} val windows x 128 tokens per corpus, positions {SKIP_FIRST}-126", fontsize=9)
fig.tight_layout()
(P10 / "figures").mkdir(exist_ok=True)
fig.savefig(P10 / "figures" / "fig1_logit_lens_by_corpus.png", dpi=160)
for x in rows:
    if x["layer"] in (4, 8, 12, 16, 20, 23):
        print(x)
