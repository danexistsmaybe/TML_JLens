# p10 Environment (ENV.md)

Set up 2026-10-16 on the UVA CS server (Ubuntu 22.04, no root).

## Tooling
- uv: `/u/qeq4kh/.local/bin/uv` (v0.13.1), installed via official `https://astral.sh/uv/install.sh`.
- System Python: `/usr/bin/python3` is 3.10 and has no pip/ensurepip. Not used.
- Project Python: CPython 3.11.17 (uv-managed download). Venv: `/u/qeq4kh/Claude/projects/cs6501-tml/p10/.venv`, interpreter `.venv/bin/python` (3.11.17).

## Commands used
```
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /u/qeq4kh/Claude/projects/cs6501-tml/p10
~/.local/bin/uv venv --python 3.11 .venv
~/.local/bin/uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
~/.local/bin/uv pip install --python .venv/bin/python transformers datasets accelerate numpy matplotlib pandas scipy tqdm safetensors huggingface_hub
~/.local/bin/uv pip freeze --python .venv/bin/python > requirements.txt
```
transformer_lens was NOT installed.

## Versions
- torch 2.6.0+cu124 (CUDA runtime 12.4, cuDNN 90100)
- transformers 5.20.0, huggingface_hub 2.2.0 (as resolved by uv), datasets 5.1.0, accelerate 1.15.0
- numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, matplotlib 3.11.2, safetensors 0.8.0, tqdm 4.70.1
- Full pinned list: `p10/requirements.txt` (73 lines, from `uv pip freeze`).
- GPU: NVIDIA TITAN RTX (PCI 65:00.0), 24 GB. The driver is not visible to `nvidia-smi` on this PATH, so the 595.x driver string was not verified directly.

## CUDA check
```
$ .venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
2.6.0+cu124 True NVIDIA TITAN RTX
```

## HF cache
- `HF_HOME=/u/qeq4kh/Claude/projects/cs6501-tml/p10/hf_cache` (must be exported for every HF call).
- Total size about 1.7 GB.

## Downloaded artifacts (revision shas)
| Artifact | Revision sha | Files |
|---|---|---|
| Qwen/Qwen2.5-0.5B (base model) | `060db6499f32faf8b98477b0a26969ef7d8b9987` | config.json, generation_config.json, merges.txt, model.safetensors, tokenizer.json, tokenizer_config.json, vocab.json (290 weight tensors loaded) |
| Salesforce/wikitext, config wikitext-2-raw-v1 (all splits) | `b08601e04326c79dfdd32d625aee71d232d685c3` | `wikitext-2-raw-v1/{train,validation,test}-00000-of-00001.parquet` (7.7 MB) |
| roneneldan/TinyStories | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | `data/validation-00000-of-00001-869c898b519ad725.parquet`, `data/train-00000-of-00004-2d5a1467fff1081b.parquet` (ONE train shard of 4), `README.md` |

Not downloaded from TinyStories: the other 3 train shards, the `TinyStoriesV2-GPT4-*` and `TinyStories-*.txt` files, and `TinyStories_all_data.tar.gz`. Download more with `snapshot_download(..., allow_patterns=[...])` if needed.

Snapshot paths:
- `hf_cache/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987`
- `hf_cache/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3`
- `hf_cache/hub/datasets--roneneldan--TinyStories/snapshots/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`

## Smoke test
Model loaded with `AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to("cuda")`. Prompt: "The capital of France is". Top-5 next tokens:
```
12095   ' Paris'   0.3156
32671   ' ______'  0.1146
30743   ' ____'    0.0635
1304    ' __'      0.0551
510     ':\n'      0.0513
```

## Problems and workarounds
- Unauthenticated HF Hub requests (warning only). Set `HF_TOKEN` if rate limits become a problem.
- A stray `cat >` command in my first download attempt waited on stdin and hung the chain for about 10 minutes. The downloads were rerun cleanly. No data was lost.
- `scripts/04_fit_pilot_lenses.py` (PID 197918) was already running in this project directory when I was setting up. It is not part of this setup, and I did not touch it. It shares the GPU, so GPU memory numbers from the smoke test may vary.
- No permission-classifier blocks occurred.
