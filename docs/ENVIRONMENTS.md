# Environments

All environments use Python 3.11 and were tested on NVIDIA A100 80 GB GPUs.

| Environment | Install | Used for |
| --- | --- | --- |
| base | `pip install -e .` | building the evaluation set, Typo-Evid rendering, the audit table |
| main | `pip install -e '.[main]'` | 16 of the 18 models |
| legacy | `pip install -e '.[legacy]'` | Qwen-VL-Chat and DeepSeek-VL-7B |
| generation | `pip install -e '.[generation]'` | Nat-Evid image generation |
| planning | `pip install -e '.[planning]'` | recomputing the Nat-Evid plans (optional) |

Keep `main` and `legacy` in separate virtual environments: they pin different
versions of `transformers`.

## Main

`.[main]` installs torch 2.10.0, torchvision 0.25.0, transformers 5.5.3 and
accelerate 1.13.0, tested with the CUDA 12.8 wheels.

## Legacy

`.[legacy]` installs torch 2.10.0 with transformers 4.32.0.

DeepSeek-VL-7B also needs the official `deepseek_vl` package at commit
`681bffb4519856ad27cc17531aacde31ddf6f1a7` with `compat/deepseek-vl-siglip.patch`
applied:

```bash
git clone https://github.com/deepseek-ai/DeepSeek-VL
cd DeepSeek-VL
git checkout 681bffb4519856ad27cc17531aacde31ddf6f1a7
git apply /path/to/fusionaudit/compat/deepseek-vl-siglip.patch
pip install --no-deps -e .
```

Qwen-VL-Chat imports `transformers_stream_generator`; the package ships a small
module with that name, so no extra install is needed.

## Nat-Evid

Generation calls the Gemini API (`gemini-3.1-flash-image-preview`) through
`google-genai` and reads the key from `GEMINI_API_KEY`.

`scripts/generate_nat_evid.py plan` recomputes the wrong answers in
`data/nat_evid/plans.jsonl` with INSTRUCTOR embeddings (`hkunlp/instructor-base`).
`.[planning]` installs InstructorEmbedding 1.0.1 and sentence-transformers 5.3.0.

## Memory

Pixel attacks backpropagate through the full model with gradient checkpointing.
When an image does not fit in GPU memory, the attack is computed on a copy scaled
to 0.75 and then 0.5 of the original size, and the perturbation is upsampled to
the original resolution.
