"""Resumable evaluation entry points: clean inference, pixel attacks, rendered-text evaluation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from .registry import MODEL_BY_ID, PIXEL_ATTACKS, TEXT_ATTACKS, safe_name


def load_samples(path):
    from .data.evalset import EvalSample
    rows = json.loads(Path(path).read_text())["samples"]
    samples = [EvalSample(**{k: r[k] for k in EvalSample.__dataclass_fields__}) for r in rows]
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample identifiers in manifest")
    if any(Path(sid).name != sid or sid in (".", "..") for sid in ids):
        raise ValueError("Sample identifiers must be plain filenames")
    return samples


def main(mode):
    ap = argparse.ArgumentParser(description=f"FusionAudit {mode} evaluation (resumable).")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", choices=sorted(MODEL_BY_ID), required=True)
    ap.add_argument("--records", type=Path, required=True, help="output directory for the model's records")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--revision", help="Model commit (default: the pinned revision in the registry; 'main' for the latest)")
    ap.add_argument("--image-cache", type=Path)
    ap.add_argument("--limit", type=int, help="only the first N manifest samples")
    ap.add_argument("--sample-ids", type=Path, help="only the sample ids listed in this file, one per line")
    if mode == "pixel":
        ap.add_argument("--attack", choices=PIXEL_ATTACKS, required=True)
        ap.add_argument("--eps", type=float, default=8/255)
        ap.add_argument("--alpha", type=float, default=1/255)
        ap.add_argument("--steps", type=int, default=40)
        ap.add_argument("--seed", type=int, default=1)
    if mode == "rendered":
        ap.add_argument("--condition", choices=TEXT_ATTACKS, required=True)
        ap.add_argument("--images", type=Path, required=True, help="Directory containing sample_id.png")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if mode == "pixel" and (not 0 <= args.eps <= 1 or args.alpha <= 0 or args.steps < 1):
        ap.error("Require 0 <= eps <= 1, alpha > 0, steps >= 1")
    condition = "clean" if mode == "clean" else (args.attack if mode == "pixel" else args.condition)
    samples = load_samples(args.manifest)
    selected = samples[:args.limit] if args.limit else samples
    if args.sample_ids:
        wanted = [line.strip() for line in args.sample_ids.read_text().splitlines() if line.strip()]
        known = {s.sample_id for s in selected}
        unknown = [sid for sid in wanted if sid not in known]
        if unknown:
            ap.error(f"{len(unknown)} sample ids are not in the manifest; first: {unknown[0]}")
        keep = set(wanted)
        selected = [s for s in selected if s.sample_id in keep]
    config = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "model_id": args.model, "revision": args.revision or MODEL_BY_ID[args.model].revision,
              "condition": condition, "max_new_tokens": 256,
              "do_sample": False}
    if mode == "pixel":
        config.update({k: getattr(args, k) for k in ("eps", "alpha", "steps", "seed")})
    if mode == "rendered":
        missing = [s.sample_id for s in selected if not (args.images / f"{s.sample_id}.png").is_file()]
        if missing:
            raise ValueError(f"Missing {len(missing)} edited images; first: {missing[0]}. Generate them before evaluation.")
        config["image_directory"] = str(args.images.resolve())
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    from .models.backends import load_study_backend
    from .models.inference import generate
    from .data.evalset import image_sha256, load_image
    from .judge import is_correct
    from .records import load_records, records_path
    from PIL import Image

    args.records.mkdir(parents=True, exist_ok=True)
    path = records_path(args.records, args.model)
    # Serialize all writers for one model. Different model files can run concurrently.
    with (args.records / f"{safe_name(args.model)}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = load_records(args.records, args.model) if path.exists() else {}
        done = existing.get(condition, {})
        meta_path = args.records / f"{safe_name(args.model)}.{condition}.config.json"
        if meta_path.exists():
            if json.loads(meta_path.read_text())["config"] != config:
                raise ValueError("Resume configuration differs. Use a new records directory.")
        elif done:
            raise ValueError("Existing outputs lack run provenance. Use a new records directory.")
        clean = existing.get("clean", {})
        if mode != "clean":
            baseline_meta = args.records / f"{safe_name(args.model)}.clean.config.json"
            if not baseline_meta.exists():
                raise ValueError("Clean baseline lacks provenance; run clean in a new records directory")
            if json.loads(baseline_meta.read_text())["config"]["manifest_sha256"] != config["manifest_sha256"]:
                raise ValueError("Clean baseline and attack manifest differ")
        if mode != "clean" and any(s.sample_id not in clean for s in selected):
            raise ValueError("Run clean evaluation for all selected samples first")
        todo = [s for s in selected if s.sample_id not in done and
                (mode != "pixel" or is_correct(clean[s.sample_id], s.expected_answer, s.dataset_id))]
        if not todo:
            print("All selected eligible samples already complete.")
            return 0
        backend = load_study_backend(args.model, args.device, revision=config["revision"])
        model, processor = backend.model, backend.processor
        import transformers
        provenance = {"config": config, "torch": torch.__version__, "transformers": transformers.__version__,
                      "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None)}
        if meta_path.exists():
            previous = json.loads(meta_path.read_text())
            if any(previous.get(k) != provenance.get(k) for k in ("torch", "transformers", "model_revision")):
                raise ValueError("Model revision or runtime changed during resume; use a new records directory")
        if not meta_path.exists():
            meta_path.write_text(json.dumps(provenance, indent=2) + "\n")
        with path.open("a", encoding="utf-8") as output:
            for i, s in enumerate(todo, 1):
                diagnostics = {}
                if mode == "rendered":
                    with Image.open(args.images / f"{s.sample_id}.png") as edited:
                        image = edited.convert("RGB")
                else:
                    image = load_image(s, args.image_cache)
                if mode == "pixel":
                    from .attacks.run import attack_image
                    result = attack_image(condition, model, processor, args.model, image, s,
                                          clean[s.sample_id], args.device, eps=args.eps, alpha=args.alpha,
                                          steps=args.steps, seed=args.seed)
                    image, diagnostics = result.adv_image, result.diagnostics
                    if diagnostics["max_u8_delta"] > round(args.eps * 255):
                        raise RuntimeError("Attack exceeded its RGB pixel budget")
                text = generate(model, processor, image, s.question, args.model, args.device, clean=mode == "clean")
                row = {"model_id": args.model, "sample_id": s.sample_id, "condition": condition,
                       "generated_text": text, "image_sha256": image_sha256(image)}
                if diagnostics:
                    row["diagnostics"] = diagnostics
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{i}/{len(todo)}] {s.sample_id} {condition}", flush=True)
    return 0
