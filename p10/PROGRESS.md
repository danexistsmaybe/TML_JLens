# P10 overnight progress log
Started 2026-10-16 (night). Timestamped entries, newest at bottom.

- 2026-10-16 03:15 Created p10/ layout (src, scripts, manifests, results, figures, notes). Launching env setup subagent; reading papers in parallel.
- 2026-10-16 03:17 READ: J-lens paper (intro, methods, layer bands, appendix comparisons/ablations/pseudocode, limitations), reference repo fitting.py/lens.py/hooks.py, Tuned Lens abstract. Wrote SUMMARY.md.
- 2026-10-16 03:22 SPLITS: manifests/splits.json built (200 fit / 50 val / 50 test windows x 128 tok per corpus; seed 0).
- 2026-10-16 03:26 CORRECTNESS: PASS. Jacobian vs fp64 central finite differences relerr ~2e-8. Note: first attempt failed (relerr ~0.5) because HF Qwen2RMSNorm upcasts to fp32 internally, so the fp64 FD was really fp32; patched the norm to native dtype for the check only. Hook/decode conventions exact (0.0 error).
- 2026-10-16 03:26 FIRST START: logit lens on 10 val windows/corpus. Readouts are junk (recurring tokens like 'уницип', ' mắn') through ~layer 16, become sensible ~layer 20; curves nearly identical for WikiText vs TinyStories. figures/fig1_logit_lens_by_corpus.png
- 2026-10-16 03:27 TIMING PROBE: one 128-token window, all 23 source layers: ~8.7 s on TITAN RTX fp32 regardless of dim_batch (16: 6.2 GB peak, 64: 18.9 GB). Launching pilot fit: 80 fit windows/corpus (A/B halves of 40) + 20 val windows/corpus.
- 2026-10-16 03:29 ENV: uv 0.13.1 + Python 3.11.17 venv at p10/.venv with torch 2.6.0+cu124 (CUDA verified on TITAN RTX), HF_HOME=p10/hf_cache holds Qwen2.5-0.5B, wikitext-2-raw-v1 and TinyStories (1 train shard + validation); smoke test passed.
- 2026-10-16 03:59 PILOT FIT done: 200 windows, 9.0-9.4 s/window (all 23 source layers), peak 6.2 GB GPU. Per-window Jacobians in results/pilot/. Running analysis.
- 2026-10-16 04:03 ANALYSIS: cross-corpus top-1 readout agreement ~0.21 vs within-corpus ~0.79 at layers 6-12; in-domain lens 2-4x lower Jacobian error; TinyStories-fit lens reads 'obedient/caution' at L12-15 where WikiText-fit lens shows formatting tokens. J-lens beats logit lens on repo multihop (pass@10 .27 -> .38-.45). Wrote CHECKLIST.md and figures/fig4_headline.png. DONE with checklist steps 1-5.
