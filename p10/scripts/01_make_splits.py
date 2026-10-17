"""Create the split manifest BEFORE any fitting (checklist step 3).

For each corpus we build a token stream per official split, cut it into non-overlapping
128-token windows, and sample with a fixed seed:
  fit  : 200 windows from the TRAIN split (lens-fitting pool; 100 per seed are drawn later)
  val  : 50 windows (development: weighting rule, temperatures, etc.)
  test : 50 windows (final evaluation only)

  WikiText-2 (wikitext-2-raw-v1): fit<-train, val<-validation, test<-test. Lines of each
  split are concatenated in order (never across splits); windows may cross article
  boundaries, like pretraining text.
  TinyStories: fit<-train shard 00000 (5,000 seeded stories), val/test<-the official
  validation split, partitioned BY STORY (seeded permutation, first half val, second half
  test) so no story contributes to both. Stories are joined with a blank line.

All windows are exactly 128 tokens (matched sequence length and token budget).
Writes manifests/splits.json (token ids + provenance + text hashes) and a readable preview.
Run: .venv/bin/python scripts/01_make_splits.py
"""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import MODEL_ID, P10  # noqa: E402  (sets HF_HOME)
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SEED, WIN = 0, 128
N = {"fit": 200, "val": 50, "test": 50}
tok = AutoTokenizer.from_pretrained(MODEL_ID)


def windows_from_text(text, n, rng, tag):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    starts = list(range(0, len(ids) - WIN + 1, WIN))
    pick = sorted(rng.sample(starts, n))
    out = []
    for s in pick:
        w = ids[s:s + WIN]
        out.append({"source": tag, "token_offset": s, "input_ids": w,
                    "sha256": hashlib.sha256(json.dumps(w).encode()).hexdigest()[:16]})
    return out, len(starts)


rng = random.Random(SEED)
man = {"seed": SEED, "window_tokens": WIN, "model": MODEL_ID, "tokenizer_add_special_tokens": False,
       "corpora": {}, "stream_window_counts": {}}

wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
man["corpora"]["wikitext"] = {}
for role, split in [("fit", "train"), ("val", "validation"), ("test", "test")]:
    text = "".join(wt[split]["text"])  # raw lines already end with "\n"
    man["corpora"]["wikitext"][role], cnt = windows_from_text(text, N[role], rng, f"wikitext/{split}")
    man["stream_window_counts"][f"wikitext/{split}"] = cnt

TS_FILES = {"train": "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
            "validation": "data/validation-00000-of-00001-869c898b519ad725.parquet"}
ts = load_dataset("roneneldan/TinyStories", data_files=TS_FILES)
train_idx = sorted(rng.sample(range(len(ts["train"])), 5000))
val_perm = list(range(len(ts["validation"])))
rng.shuffle(val_perm)
half = len(val_perm) // 2
parts = {"fit": ("train", train_idx), "val": ("validation", sorted(val_perm[:half])),
         "test": ("validation", sorted(val_perm[half:]))}
man["corpora"]["tinystories"] = {}
for role, (split, idx) in parts.items():
    text = "\n\n".join(ts[split][i]["text"].strip() for i in idx)
    man["corpora"]["tinystories"][role], cnt = windows_from_text(text, N[role], rng, f"tinystories/{split}/{role}")
    man["stream_window_counts"][f"tinystories/{split}/{role}"] = cnt
man["tinystories_files"] = TS_FILES
man["tinystories_story_counts"] = {r: len(i) for r, (_, i) in parts.items()}

# sanity: no window hash shared across roles within a corpus
for c, roles in man["corpora"].items():
    seen = {}
    for r, ws in roles.items():
        for w in ws:
            assert seen.setdefault(w["sha256"], r) == r, f"duplicate window across roles in {c}"

(P10 / "manifests").mkdir(exist_ok=True)
(P10 / "manifests" / "splits.json").write_text(json.dumps(man))
with open(P10 / "manifests" / "preview.md", "w") as f:
    f.write("# Window previews (first 2 per corpus/role, decoded)\n\n")
    for c, roles in man["corpora"].items():
        for r, ws in roles.items():
            for w in ws[:2]:
                f.write(f"## {c}/{r} offset {w['token_offset']}\n```\n{tok.decode(w['input_ids'])}\n```\n\n")
print({c: {r: len(v) for r, v in roles.items()} for c, roles in man["corpora"].items()})
print(man["stream_window_counts"])
