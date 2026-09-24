"""Shared helpers for the Jev-vs-LLM AMR comparison pipeline.

Implements the "filter first in code, send only what the question needs;
judgment = model, facts = code" principle: the WHO catalogue lookup and the
construction of decision items happen deterministically in code; the model
(Jev or an LLM) only makes the judgment call on the pre-filtered evidence.
"""

import json
import os

DECISION_OPTIONS = {
    "resistant": "the supplied evidence supports calling this isolate RESISTANT to the drug",
    "no": "the supplied evidence supports calling this isolate NOT resistant (susceptible) to the drug",
    "unknown": "the evidence is insufficient or the mutation's association with resistance is uncertain; a resistance call cannot be made",
}


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def build_decision_items(sample, catalogue):
    """Filter sample mutations against the catalogue (facts = code).

    Returns a list of decision items, one per (mutation, catalogue entry) match.
    Non-catalogue mutations (e.g. katG 463, gyrA 384/668 polymorphisms) are
    dropped here and never shown to the model.
    """
    entries = catalogue["entries"]
    items = []
    dropped = []
    for mut in sample["mutations"]:
        matches = [
            e for e in entries
            if mut["gene"].lower() == e["gene"].lower()
            and mut["change"].lower() == e["change"].lower()
        ]
        if not matches:
            dropped.append(f"{mut['gene']} {mut['change']}")
        for e in matches:
            items.append({
                "item_id": "q%02d_%s_%s_%s" % (len(items), e["drug"], mut["gene"], mut["change"]),
                "drug": e["drug"],
                "gene": mut["gene"],
                "change": mut["change"],
                "nucleotide": mut.get("nucleotide"),
                "depth": mut.get("depth"),
                "alt_fraction": mut.get("alt_fraction"),
                "association": e["association"],
                "grading": e["grading"],
                "catalogue_id": e["id"],
            })
    return items, dropped


def evidence_state(sample_id, items):
    """The trimmed state sent to the model: only what the questions need."""
    return {
        "sample_id": sample_id,
        "evidence": [
            {
                "id": it["item_id"],
                "drug": it["drug"],
                "mutation": "%s %s (%s)" % (it["gene"], it["change"], it["nucleotide"]),
                "read_depth": it["depth"],
                "alt_fraction": it["alt_fraction"],
                "who_catalogue_association": it["association"],
                "who_grading": it["grading"],
            }
            for it in items
        ],
    }


def render_report(arm, sample_id, items, decisions, timing, extra_lines=None):
    """Rich text report with a per-item decision table and a Runtime row."""
    lines = []
    lines.append("=" * 88)
    lines.append("AMR prediction report — %s arm — sample %s" % (arm, sample_id))
    lines.append("=" * 88)
    lines.append("")
    lines.append("%-4s %-14s %-14s %-12s %-10s %-10s %s" % (
        "#", "Drug", "Mutation", "WHO class", "Decision", "Conf.", "Catalogue id"))
    lines.append("-" * 88)
    for i, (it, dec) in enumerate(zip(items, decisions), 1):
        lines.append("%-4d %-14s %-14s %-12s %-10s %-10s %s" % (
            i, it["drug"], "%s %s" % (it["gene"], it["change"]),
            it["association"].replace("resistance-associated", "Assoc-R").replace("uncertain", "Uncertain"),
            dec["decision"], dec.get("confidence", ""), it["catalogue_id"]))
    lines.append("-" * 88)
    n_res = sum(1 for d in decisions if d["decision"] == "resistant")
    n_no = sum(1 for d in decisions if d["decision"] == "no")
    n_unk = sum(1 for d in decisions if d["decision"] == "unknown")
    lines.append("Decisions: %d resistant / %d no / %d unknown (of %d items)" % (
        n_res, n_no, n_unk, len(items)))
    lines.append("")
    lines.append("Runtime:")
    for key, val in timing.items():
        lines.append("  %-18s %s" % (key, val))
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    lines.append("")
    return "\n".join(lines)


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def write_text(path, text):
    with open(path, "w") as fh:
        fh.write(text)
