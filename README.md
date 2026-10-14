# FusionAudit

Code for **FusionAudit: Pathway-Conditioned Robustness Auditing for Native Multimodal Models** (NeurIPS 2026).
[Project page](https://jiamingzhang.netlify.app/fusionaudit/)

FusionAudit attacks a vision-language model through the image and reports the share of its correct answers that
at least one attack breaks. Five attacks on two pathways:

- **Pixel perturbation** (L&infin; &le; 8/255): Vis-PGD, End-PGD, and our MANTIS.
- **Rendered text** (a wrong answer written into the image): Typo-Evid, and our Nat-Evid.

## Install

```bash
pip install -e .            # data and Table 1
pip install -e '.[main]'    # running the models
```

Qwen-VL-Chat and DeepSeek-VL-7B need a second environment: see [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md).

## Data

1500 questions, 500 each from [ChartQA](https://github.com/vis-nlp/ChartQA) (train),
[DocVQA](https://www.docvqa.org) (validation) and
[MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) (test). The repository ships their row
indices and image hashes; this downloads the questions and images and checks them against the index:

```bash
python scripts/build_eval_set.py --index data/index.json \
    --out data/manifest.json --image-cache data/images
```

Attack images:

```bash
# Typo-Evid, rendered locally and checked against the released hashes
python scripts/render_typo_evid.py --manifest data/manifest.json --image-cache data/images \
    --out data/typo_evid --hashes data/typo_evid_hashes.json

# Nat-Evid, edited with the Gemini API (wrong answers fixed in data/nat_evid/plans.jsonl)
pip install -e '.[generation]' && export GEMINI_API_KEY=...
python scripts/generate_nat_evid.py generate --manifest data/manifest.json \
    --image-cache data/images --out data/nat_evid
```

## Evaluation

Reproduce Table 1 from the released outputs of the 18 models (no GPU):

```bash
python scripts/summarize.py --manifest data/manifest.json --records data/records --out results
```

Audit a model. Run the clean pass first: the attacks use the questions the model answers correctly.

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

The 18 models and their pinned revisions are in [configs/models.json](configs/models.json)
(gated models: [docs/WEIGHTS.md](docs/WEIGHTS.md)); attack and decoding settings are in
[configs/audit.json](configs/audit.json).

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
