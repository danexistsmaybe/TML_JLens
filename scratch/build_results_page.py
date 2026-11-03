"""Builds the results page and results/REPORT.md from the test tables. Usage: scratch/build_results_page.py OUT.html"""
import base64, csv, html, sys
from pathlib import Path
R = Path("results"); T = R / "tables"; F = R / "figures"
def load(e):
    out = {}
    for r in csv.DictReader((T / f"exp{e}.csv").open()):
        if r["kind"] == "lens":
            out[r["domain"], r["lens"], r["metric"]] = (float(r["value"]), float(r["low"]), float(r["high"]))
    return out
COLS = [("jacobian_error", "Jacobian error", 3), ("pass10_target", "Pass@10", 2), ("pass10_control", "Control", 2), ("pass50_gap", "Pass@50 gap", 2),
        ("auc_gap", "AUC gap", 3), ("swap_1", "Swap α=1", 2), ("swap_2", "Swap α=2", 2), ("overlap", "Top-10 overlap", 3), ("sym_kl", "Sym. KL", 3), ("edit_error", "Edit error", 2)]
L12 = [("logit", "Logit lens", "base"), ("uniform_pool", "Uniform pool", "base"), ("random_weights", "Random weights", "ctrl"), ("random_n", "Random n", "ctrl"),
       ("similarity", "Similarity weighting", "ours"), ("top_n", "Top-n selection", "ours"), ("oracle:wiki", "Wiki oracle (1000)", "orc"), ("oracle:stories", "Stories oracle (1000)", "orc"),
       ("oracle500_in_pool", "Domain oracle (500)", "orc")]
L2 = L12[:6] + [("oracle:python", "Python oracle (1000)", "orc"), ("oracle500_half_b", "Python oracle (500)", "orc")]
L3 = [("logit", "Logit lens", "base"), ("oracle", "Uniform fit (1000)", "base"), ("random_weights_100", "Random weights, ESS 100", "ctrl"), ("random_n_100", "Random n = 100", "ctrl"),
      ("similarity_100", "Similarity, ESS 100", "ours"), ("top_n_100", "Top-n, n = 100", "ours"), ("random_weights_500", "Random weights, ESS 500", "ctrl"), ("random_n_500", "Random n = 500", "ctrl"),
      ("similarity_500", "Similarity, ESS 500", "ours"), ("top_n_500", "Top-n, n = 500", "ours"), ("wider_selection", "Wider selection (top 1000)", "ours")]
def table(t, domain, rows):
    cols = [c for c in COLS if any((domain, k, c[0]) in t for k, _, _ in rows)]
    head = "".join(f"<th>{c[1]}</th>" for c in cols)
    body = ""
    md = "| Lens | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for k, name, role in rows:
        cells, mdc = "", []
        for m, _, d in cols:
            v = t.get((domain, k, m))
            if v:
                cells += f'<td>{v[0]:.{d}f}<small>± {(v[2]-v[1])/2:.{d}f}</small></td>'; mdc.append(f"{v[0]:.{d}f} ± {(v[2]-v[1])/2:.{d}f}")
            else:
                cells += "<td class=na>–</td>"; mdc.append("–")
        body += f'<tr class={role}><th scope=row>{name}</th>{cells}</tr>'
        md += f"| {'**'+name+'**' if role=='ours' else name} | " + " | ".join(mdc) + " |\n"
    return f'<div class=tw><table><thead><tr><th>Lens</th>{head}</tr></thead><tbody>{body}</tbody></table></div>', md
def fig(name, alt):
    b = base64.b64encode((F / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(alt)}"></figure>'
t1, t2, t3 = load(1), load(2), load(3)
sections, md_tables = [], []
def add(title, intro, t, domains, rows, figure):
    s, m = f"<section><h2>{title}</h2><p>{intro}</p>{figure}", f"\n## {title}\n\n{intro}\n"
    for d in domains:
        h, tm = table(t, d, rows)
        s += f"<h3>Deployed on {d}</h3>{h}"; m += f"\n### Deployed on {d}\n\n{tm}"
    sections.append(s + "</section>"); md_tables.append(m)
add("Experiment 1: known domains", "The pool holds 500 wiki and 500 stories windows. The lens is deployed on one of them, and an unlabeled validation sample picks the windows. Settings: ESS 400 and n = 400 for both.",
    t1, ["wiki", "stories"], L12, fig("exp1_lenses.png", "Experiment 1 lenses across six metrics"))
add("Experiment 2: new domain", "The same pool, deployed on python, which it doesn't contain. Settings: ESS 25, n = 50. The python oracle is out of the method's reach and marks the ceiling.",
    t2, ["python"], L2, fig("exp2_lenses.png", "Experiment 2 lenses across six metrics"))
add("Experiment 3: within a domain", "No domain shift. Windows close to the domain's centroid against random windows from the same domain, at fixed budgets. Wider selection takes the top 1000 of a much larger candidate set. Readout agreement doesn't apply here, because the uniform fit is itself the reference.",
    t3, ["wiki", "stories", "python"], L3, fig("exp3_lenses.png", "Experiment 3 lenses across five metrics"))
FIND = [
 ("Known domains: the methods recover the domain's lens.", "Jacobian error reaches the oracle (wiki 0.499 vs 0.498, stories 0.349 vs 0.347, from 0.560 and 0.420 for the uniform pool). Top-10 overlap rises from about 0.65 to about 0.94, near the 0.95–0.97 that a separate 500-window oracle reaches."),
 ("Pass@k doesn't respond to the corpus.", "The methods leave the AUC gap unchanged in every domain. Even the two oracles mostly can't be told apart, except stories at pass@50. So this metric can't detect a gain here, and we should report it as uninformative about corpus choice."),
 ("Swaps go both ways.", "Similarity against the uniform pool: wiki +0.04, python +0.09, stories −0.15. On stories, the wiki oracle beats the stories oracle (0.74 vs 0.58), so a matching corpus can hurt. The loss is spread across most templates and mostly fades at α = 2. We have no explanation yet."),
 ("New domain: a real gain, far from the ceiling.", "On python, Jacobian error falls from 0.762 to 0.701 against an oracle at 0.474, and nearly all the weight lands on wiki windows."),
 ("Within a domain: selection doesn't beat random.", "At ESS or n of 100, selected windows have higher Jacobian error than random ones in all three domains (for example, wiki 0.524 vs 0.501). Wider selection is worse than the uniform fit on Jacobian error (wiki 0.542 vs 0.498) and shows no swap gain. Python's AUC gap rises slightly with selection (+0.014 to +0.017), which is the one small positive, and it may come from the digit controls."),
 ("Very few windows are enough.", "An oracle fit on 10 windows has nearly the same AUC gap as one fit on 1000 (wiki 0.656 vs 0.652), and Jacobian error levels off by about 50 windows. That may explain why selection has so little room to help."),
 ("Every J-lens beats the logit lens on the window metrics and on swaps.", "Those are Jacobian error, overlap and edit error. The AUC gap also favors the J-lens in all three domains, though on python the margin is narrower and the pass@10 gap doesn't separate them."),
]
CAVEATS = [
 "<b>Prompts are unreviewed.</b> The pass@k and swap columns could change after review. Judge prompts on their text, without per-prompt scores, and later changes get logged and rerun.",
 "<b>Stories read positions depart from the paper.</b> Each story has one added token: a pronoun where “ The” would ask the model to name the described thing, otherwise “ The”. Those are judgment calls for you to check. Python implied prompts read at the end of the code, which is closer to the paper.",
 "<b>Python evaluate prompts saturate pass@10.</b> Their digit controls are as plausible as the targets, so control rates are high there. The CSV has per-type rows: on test, implied targets reach 37–41% at pass@50 under J-lenses with 0% control, against 13% for the logit lens.",
 "<b>Edit error.</b> The main size moved from 10% to 5% of activation norm before test, because an exact Jacobian already shows 24–33% error at 10%. With real lenses the two sizes give nearly the same error, so this changed little.",
 "<b>Intervals.</b> Paired differences aren't corrected for multiple comparisons. The three seeds' pools overlap, so seed spread understates true variability. Swap intervals rest on 9 templates and are wide, especially on wiki, where templates mostly score all or nothing.",
 "<b>Wider selection is topical.</b> Wiki picks favor road and river articles, and 420 of 1000 sit next to another pick in the source. That may be why it does poorly.",
 "<b>Scope.</b> One small model, Qwen2.5-0.5B. Jacobians were computed under NVIDIA driver 595, and scoring ran under 580. Correctness checks passed, and a cached Jacobian matched a fresh one to within 2e-4.",
]
find_html = "".join(f"<li><b>{a}</b> {b}</li>" for a, b in FIND)
cav_html = "".join(f"<li>{c}</li>" for c in CAVEATS)
css = """
:root{--bg:#f7f8f6;--fg:#1c2321;--muted:#5b6662;--line:#d7dcd8;--panel:#ffffff;--accent:#1f5f8b;--ours:#e4eff7;--display:'Source Serif 4',Georgia,serif;--body:'Source Sans 3','Helvetica Neue',Arial,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151a19;--fg:#e7ebe9;--muted:#a3aeaa;--line:#343c3a;--panel:#1d2422;--accent:#7fb7dd;--ours:#20323e;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#151a19;--fg:#e7ebe9;--muted:#a3aeaa;--line:#343c3a;--panel:#1d2422;--accent:#7fb7dd;--ours:#20323e;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:16px/1.55 var(--body);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:32px 64px}
main{max-width:1080px;margin:0 auto;display:grid;gap:36px}
main>*,section{min-width:0}
h1,h2{font-family:var(--display);text-wrap:balance;line-height:1.2;margin:0}
h1{font-size:2rem}h2{font-size:1.45rem;border-top:1px solid var(--line);padding-top:24px}
h3{font-size:1rem;margin:22px 0 8px}
p,li{max-width:70ch}header p{color:var(--muted);margin:8px 0 0}
.lede{font-size:1.15rem;color:var(--fg)}
ol,ul{padding-left:22px;display:grid;gap:10px;margin:12px 0 0}
figure{margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:6px;padding:10px;overflow-x:auto}
figure img{display:block;width:100%;min-width:640px;height:auto}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{color:var(--muted);font-weight:600;vertical-align:bottom;white-space:normal;min-width:64px}
th:first-child{text-align:left;font-weight:500;position:sticky;left:0;background:var(--panel)}
tbody tr:last-child>*{border-bottom:0}
td small{display:block;color:var(--muted);font-size:.74rem}
tr.ours th,tr.ours td{background:var(--ours)}tr.ours th{font-weight:700}
tr.ctrl th,tr.ctrl td{color:var(--muted)}td.na{color:var(--muted)}
.note{color:var(--muted);font-size:.9rem}
"""
page = f"""<title>J-lens Corpus Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&family=Source+Serif+4:wght@600&display=swap">
<style>{css}</style>
<main>
<header><h1>Does the J-lens fitting corpus change the interpretation?</h1>
<p>CS 6501 P10. Preliminary test results for Qwen2.5-0.5B, with unreviewed prompts. Run 2026-11-02.</p></header>
<section><p class=lede>Weighting or selecting windows by similarity changes the lens and its readouts a lot, but changes little of what the lens retrieves. It recovers a known domain's lens from an unlabeled sample, helps somewhat on a new domain, and does nothing useful within a domain.</p>
{fig("methods_vs_uniform.png", "Similarity and top-n minus the uniform pool, by domain and metric")}
<h3>Main findings</h3><ol>{find_html}</ol>
<h3>Before relying on this</h3><ul>{cav_html}</ul></section>
{''.join(sections)}
<section><h2>Choosing settings and window count</h2><p>Settings came from validation Jacobian error. The large circles mark what test used. On wiki and stories, error climbs at 750 windows, once the other domain's windows enter.</p>
{fig("settings_choice.png", "Validation Jacobian error across ESS and n")}
<p>Own-domain lens fit on fewer windows, on test.</p>{fig("sample_count.png", "Jacobian error against number of windows")}</section>
<p class=note>Cells show the mean ± half the 95% bootstrap interval, resampling windows, prompts or templates. Seeded lenses are averaged over 3 seeds. Jacobian error, overlap, KL and edit error average layers 12 and 18. Pass@k takes the best layer in 9–22; gaps are target minus control. Edit error is the median relative error at 5% of activation norm. Lower is better for Jacobian error, control, KL and edit error. Paired differences and every layer are in results/tables/. Runtime: about 9.4 s per window for Jacobians on one GPU; once those are cached, fitting 53 lenses at 14 layers took 52 s.</p>
</main>"""
Path(sys.argv[1]).write_text(page)
md = "# Preliminary results (test, 2026-11-02)\n\nPrompts are unreviewed. Figures are in `figures/`; paired differences and per-layer numbers are in `tables/`.\n\n## Main findings\n\n" + "\n".join(f"{i+1}. **{a}** {b}" for i, (a, b) in enumerate(FIND)) + "\n\n## Before relying on this\n\n" + "\n".join("- " + c.replace("<b>", "**").replace("</b>", "**") for c in CAVEATS) + "\n" + "".join(md_tables)
(R / "REPORT.md").write_text(md)
