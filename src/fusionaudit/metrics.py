"""Audit metrics: clean-correct ASR, pathway-complete risk, exclusive flips.

For a model with clean-correct set C, an attack a breaks S_a = {z in C : judge fails}.
ASR(a) = |S_a| / |C|; R^0 and R^+ are the union failure rates over the baseline
suite (Vis-PGD, End-PGD, Typo-Evid) and the strengthened suite (plus MANTIS and
Nat-Evid). Level means are unweighted averages over models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .judge import is_correct
from .records import Sample
from .registry import ATTACKS, BASELINE_SUITE, CLEAN, MODEL_BY_ID, PIXEL_ATTACKS, STRENGTHENED_SUITE


@dataclass
class ModelAudit:
    model_id: str
    n_total: int
    clean_correct: list[str]  # manifest order
    flips: dict[str, set[str]]
    warnings: list[str] = field(default_factory=list)

    @property
    def level(self) -> int:
        return MODEL_BY_ID[self.model_id].level

    @property
    def n_clean(self) -> int:
        return len(self.clean_correct)

    def rate(self, flipped: set[str]) -> float:
        return 100.0 * len(flipped) / self.n_clean

    def union(self, attacks: tuple[str, ...]) -> set[str]:
        out: set[str] = set()
        for a in attacks:
            out |= self.flips[a]
        return out

    def row(self) -> dict:
        r0 = self.union(BASELINE_SUITE)
        rp = self.union(STRENGTHENED_SUITE)
        return {
            "model_id": self.model_id,
            "model_short": MODEL_BY_ID[self.model_id].short,
            "level": self.level,
            "n_clean": self.n_clean,
            "clean_acc": 100.0 * self.n_clean / self.n_total,
            **{a: self.rate(self.flips[a]) for a in ATTACKS},
            "delta_mantis": self.rate(self.flips["mantis"] - self.union(("vis_pgd", "end_pgd"))),
            "delta_nat_evid": self.rate(self.flips["nat_evid"] - self.flips["typo_evid"]),
            "rhat0": self.rate(r0),
            "rhat_plus": self.rate(rp),
            "delta_rhat_plus": self.rate(rp - r0),
        }


def audit_model(model_id: str, samples: list[Sample], outputs: dict[str, dict[str, str]]) -> ModelAudit:
    """Judge one model's clean and attacked outputs on the evaluation set."""
    warnings: list[str] = []
    valid_ids = {s.sample_id for s in samples}
    if len(valid_ids) != len(samples):
        raise ValueError("Duplicate manifest sample identifiers")
    for condition, records in outputs.items():
        unknown = records.keys() - valid_ids
        if unknown:
            raise ValueError(f"{condition}: records outside the supplied manifest: {sorted(unknown)[:3]}")
    clean = outputs.get(CLEAN, {})
    clean_correct = [s.sample_id for s in samples
                     if is_correct(clean.get(s.sample_id, ""), s.expected_answer, s.dataset_id)]
    missing = sum(1 for s in samples if s.sample_id not in clean)
    if missing:
        warnings.append(f"{model_id}: {missing} samples have no clean output")
    by_id = {s.sample_id: s for s in samples}
    cc = set(clean_correct)
    flips: dict[str, set[str]] = {}
    for attack in ATTACKS:
        rows = outputs.get(attack, {})
        flips[attack] = {sid for sid, text in rows.items()
                         if sid in cc and not is_correct(text, by_id[sid].expected_answer, by_id[sid].dataset_id)}
        absent = len(cc - rows.keys())
        if absent:
            # A clean-correct example without an attacked output is counted as not broken.
            warnings.append(f"{model_id} {attack}: {absent} clean-correct samples have no output")
        if attack in PIXEL_ATTACKS and rows.keys() - cc:
            warnings.append(f"{model_id} {attack}: {len(rows.keys() - cc)} outputs outside the clean-correct set")
    # Clean accuracy is taken over the samples that have a clean output.
    return ModelAudit(model_id, len(samples) - missing, clean_correct, flips, warnings)


def level_means(rows: list[dict], keys: tuple[str, ...]) -> dict[int, dict]:
    out = {}
    for level in sorted({r["level"] for r in rows}):
        members = [r for r in rows if r["level"] == level]
        out[level] = {"n_models": len(members), **{k: float(np.mean([r[k] for r in members])) for k in keys}}
    return out
