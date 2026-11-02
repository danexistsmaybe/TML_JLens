"""What did wider selection pick? Reads windows only (the confound check in EXPERIMENT_1.md)."""
import json, sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from src.corpus import load_windows
from src.features import candidate_windows
from src.model import MODEL_ID
tok = AutoTokenizer.from_pretrained(MODEL_ID)
sel = json.load(open("data/selected_windows.json"))
for corpus, idx in sel.items():
    ws = candidate_windows(corpus, idx)
    fit = {w.sha256 for w in load_windows(corpus, "fit")}
    offs = sorted(w.token_offset for w in ws)
    print(f"\n=== {corpus}: {len(ws)} chosen, {sum(w.sha256 in fit for w in ws)} already in fit; adjacent-offset pairs {sum(b - a == 128 for a, b in zip(offs, offs[1:]))}")
    print("offset range", offs[0], offs[-1])
    for w in ws[:3] + ws[500:502]:
        print("  --", repr(w.text(tok)[:220]))
