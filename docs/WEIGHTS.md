# Model weights

FusionAudit audits pretrained models and trains none. The 18 models are loaded
from the Hugging Face Hub by their repository id, at the commit listed in
`configs/models.json` (`revision`) and in `src/fusionaudit/registry.py`.

The run scripts load the pinned commit by default. `--revision <commit>` selects
another commit and `--revision main` the latest one. Each run writes the model
commit, the library versions and the manifest hash next to its outputs
(`<records>/<model>.<condition>.config.json`).

Gated repositories (for example the Gemma models) need a Hugging Face account
that has accepted the model license: run `huggingface-cli login` first. Every
model is used under the license of its owner.
