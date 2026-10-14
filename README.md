# FusionAudit

FusionAudit measures how easily a vision-language model can be misled through the image alone. It runs five
attacks on two visual pathways and reports, for each model, the share of correctly answered questions that the
attacks break.

Code of the NeurIPS 2026 paper *FusionAudit: Pathway-Conditioned Robustness Auditing for Native Multimodal
Models*. [Project page](https://jiamingzhang.netlify.app/fusionaudit/)

## Attacks

| Pathway | Attack | What it does |
| --- | --- | --- |
| Pixel perturbation | Vis-PGD | Pushes the vision-encoder features away from those of the clean image. |
| | End-PGD | Lowers the probability of the correct answer. |
| | MANTIS | Raises a wrong answer over the correct one at the first token where the two differ. |
| Rendered text | Typo-Evid | Writes a wrong answer on the image as a text overlay. |
| | Nat-Evid | Edits the image so that a wrong answer appears as part of it, such as a chart label or a sign. |

The pixel attacks are projected gradient descent on the RGB pixels of the original image, with an L&infin;
budget of 8/255, 40 steps of size 1/255 and a random start (seed 1).

## Models

18 open-weight models, loaded from the Hugging Face Hub at pinned revisions
([configs/models.json](configs/models.json)) and grouped by how deeply vision is fused into the language model:

| Level | Models |
| --- | --- |
| L1, alignment-era | SmolVLM2-2.2B, BLIP2-OPT-2.7B, InstructBLIP-Vicuna-7B, LLaVA-1.5-7B |
| L2, shallow fusion | Qwen-VL-Chat, Qwen2-VL-2B, Qwen2.5-VL-3B/7B, Gemma3-4B/12B, DeepSeek-VL-7B |
| L3, deep or early fusion | Qwen3-VL-2B/4B/8B, Qwen3.5-2B/4B/9B, Gemma4-E4B |

## Installation

Python 3.11.

```bash
pip install -e .            # data preparation and summaries
pip install -e '.[main]'    # inference and attacks
```

| Extra | Use |
| --- | --- |
| `main` | inference and attacks for 16 of the models (transformers 5.5.3) |
| `legacy` | Qwen-VL-Chat and DeepSeek-VL-7B (transformers 4.32.0), in a separate environment |
| `generation` | Nat-Evid image generation |
| `planning` | recomputing the Nat-Evid wrong answers |

See [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md). Gated models such as Gemma need a Hugging Face login
([docs/WEIGHTS.md](docs/WEIGHTS.md)).

## Data

**Evaluation set.** 1500 VQA questions, 500 each from [ChartQA](https://github.com/vis-nlp/ChartQA) (train),
[DocVQA](https://www.docvqa.org) (validation) and
[MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) (test). The repository holds an index
with the dataset row and image hash of every question. The script downloads the questions and images from the
Hugging Face Hub and checks them against the index:

```bash
python scripts/build_eval_set.py --index data/index.json \
    --out data/manifest.json --image-cache data/images
```

**Typo-Evid images**, rendered locally:

```bash
python scripts/render_typo_evid.py --manifest data/manifest.json --image-cache data/images \
    --out data/typo_evid --hashes data/typo_evid_hashes.json
```

**Nat-Evid images**, edited with the Gemini API. The wrong answer for each question is fixed in
`data/nat_evid/plans.jsonl`. This is the only step that needs an API key.

```bash
pip install -e '.[generation]'
export GEMINI_API_KEY=...
python scripts/generate_nat_evid.py generate --manifest data/manifest.json \
    --image-cache data/images --out data/nat_evid
```

## Running an audit

Run the clean pass first: the attacks use the questions the model answers correctly.

```bash
ARGS="--model Qwen/Qwen2.5-VL-3B-Instruct --manifest data/manifest.json"
ARGS="$ARGS --image-cache data/images --records runs/records"

python scripts/run_clean.py $ARGS
for a in vis_pgd end_pgd mantis; do
    python scripts/run_pixel_attacks.py $ARGS --attack $a
done
python scripts/run_rendered_eval.py $ARGS --condition typo_evid --images data/typo_evid
python scripts/run_rendered_eval.py $ARGS --condition nat_evid --images data/nat_evid/images

python scripts/summarize.py --manifest data/manifest.json \
    --records runs/records --out runs/results
```

Each run appends to `runs/records/<model>.jsonl` and resumes where it stopped. Options:

- `--limit N` or `--sample-ids FILE` runs part of the set; pass `--allow-incomplete` to `summarize.py` for such runs.
- `--eps`, `--steps`, `--alpha` and `--seed` change the pixel attacks.
- `--revision` loads another commit of the model.

## Outputs

`summarize.py` judges every answer again from the generated text and writes two files.

`main_table.csv` has one row per model:

| Column | Meaning |
| --- | --- |
| `clean_acc` | share of questions answered correctly without an attack |
| `vis_pgd` ... `nat_evid` | attack success rate: share of the correctly answered questions that the attack turns wrong |
| `delta_mantis` | share broken by MANTIS and by neither Vis-PGD nor End-PGD |
| `delta_nat_evid` | share broken by Nat-Evid and not by Typo-Evid |
| `rhat0` | share broken by at least one of Vis-PGD, End-PGD and Typo-Evid |
| `rhat_plus` | share broken by at least one of the five attacks |
| `delta_rhat_plus` | `rhat_plus` minus `rhat0` |

`level_summary.csv` averages the same columns over the models of each level.

## Released model outputs

`data/records/` holds the answers of the 18 models on the clean and attacked images used in the paper.
Summarizing them reproduces the paper's results without a GPU:

```bash
python scripts/summarize.py --manifest data/manifest.json --records data/records --out results
```

## Citation

```bibtex
@inproceedings{zhang2026fusionaudit,
  title     = {FusionAudit: Pathway-Conditioned Robustness Auditing
               for Native Multimodal Models},
  author    = {Zhang, Jiaming and Li, Zherui and Guo, Siqi and
               Dong, Junhao and Wang, Kun and Lim, Wei Yang Bryan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

Released under the [MIT License](LICENSE).
