"""Swap success per template: which ones drive differences between lenses? Test results."""
import json
from pathlib import Path
import numpy as np
swaps = json.load(open("data/prompts/swap.json"))
root = Path("results/raw")
def rate(path, alpha):
    r = json.load(open(path))["swap"][alpha]
    n = max(x["template"] for x in r) + 1
    return [np.mean([x["pass"] for x in r if x["template"] == t]) for t in range(n)]
for exp, dom, lenses in [(1, "stories", ["uniform_pool_s0", "similarity_stories_s0", "oracle_stories", "oracle_wiki", "logit"]),
                         (1, "wiki", ["uniform_pool_s0", "similarity_wiki_s0", "oracle_wiki", "oracle_stories", "logit"]),
                         (2, "python", ["uniform_pool_s0", "similarity_python_s0", "oracle_python", "logit"])]:
    print("\n", dom, lenses)
    rows = {l: rate(root / f"exp{exp}/test/{dom}/{l}.json", "1.0") for l in lenses}
    for t, tmpl in enumerate(swaps[dom]["test"]):
        print(f"  {tmpl['template'][:52]!r:56s} trials {len(tmpl['trials']):3d} ", " ".join(f"{rows[l][t]:.2f}" for l in lenses))
    print("  alpha 2:", " ".join(f"{np.mean(rate(root / f'exp{exp}/test/{dom}/{l}.json', '2.0')):.2f}" for l in lenses))
