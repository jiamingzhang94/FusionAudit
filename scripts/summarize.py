#!/usr/bin/env python3
"""Compute the main audit table (Table 1) from per-sample model outputs.

    python scripts/summarize.py --manifest data/manifest.json --records data/records --out results

Correctness is recomputed from the raw generated text. Writes main_table.csv (one
row per model) and level_summary.csv (level means), and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fusionaudit.metrics import audit_model, level_means  # noqa: E402
from fusionaudit.records import load_manifest, load_records, records_path  # noqa: E402
from fusionaudit.registry import ATTACK_LABELS, ATTACKS, MODELS  # noqa: E402

COLUMNS = ("clean_acc",) + ATTACKS + ("delta_mantis", "delta_nat_evid", "rhat0", "rhat_plus", "delta_rhat_plus")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="summarize records that do not cover every sample and attack")
    args = ap.parse_args()

    samples = load_manifest(args.manifest)
    audits = [audit_model(m.model_id, samples, load_records(args.records, m.model_id))
              for m in MODELS if records_path(args.records, m.model_id).exists()]
    if not audits:
        ap.error("no model record files found")
    if any(a.n_clean == 0 for a in audits):
        ap.error("a model has no clean-correct samples")
    warnings = [w for a in audits for w in a.warnings]
    if warnings and not args.allow_incomplete:
        ap.error(warnings[0] + " (pass --allow-incomplete to summarize partial records)")
    rows = [a.row() for a in audits]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "main_table.csv", rows)
    means = level_means(rows, COLUMNS)
    write_csv(args.out / "level_summary.csv", [{"level": lv, **v} for lv, v in means.items()])

    print(f"{'Model':24s} {'Clean':>6s} " + " ".join(f"{ATTACK_LABELS[a]:>9s}" for a in ATTACKS)
          + f" {'R^0':>6s} {'R^+':>6s}")
    for r in rows:
        print(f"{r['model_short']:24s} {r['clean_acc']:6.1f} " + " ".join(f"{r[a]:9.1f}" for a in ATTACKS)
              + f" {r['rhat0']:6.1f} {r['rhat_plus']:6.1f}")
    for lv, v in means.items():
        print(f"{'L%d mean (%d)' % (lv, v['n_models']):24s} {v['clean_acc']:6.1f} "
              + " ".join(f"{v[a]:9.1f}" for a in ATTACKS) + f" {v['rhat0']:6.1f} {v['rhat_plus']:6.1f}")
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
