#!/usr/bin/env python3
"""compare_models.py — compare Jev vs LLM arms per sample.

Writes model_comparison.txt with, per sample: agreement count, disagreeing
items, and a runtime line (Jev total vs LLM total + fold difference).

Usage:
  python3 compare_models.py --results-dir results --samples test_data1 test_data2
"""

import argparse
import os

from amr_common import load_json, write_text


def compare_sample(results_dir, sample_id):
    jev = load_json(os.path.join(results_dir, sample_id, "jev_raw.json"))
    llm = load_json(os.path.join(results_dir, sample_id, "llm_raw.json"))

    jev_by_id = {d["item_id"]: d for d in jev["decisions"]}
    llm_by_id = {d["item_id"]: d for d in llm["decisions"]}
    assert set(jev_by_id) == set(llm_by_id), "decision item sets differ between arms"

    agree, disagree = [], []
    for item_id in sorted(jev_by_id):
        j, l = jev_by_id[item_id], llm_by_id[item_id]
        label = "%s %s %s (%s)" % (j["drug"], j["gene"], j["change"], j["association"])
        if j["decision"] == l["decision"]:
            agree.append(label)
        else:
            disagree.append((label, j["decision"], l["decision"]))

    jev_t = jev["timing"]["total_s"]
    llm_t = llm["timing"]["total_s"]
    fold = llm_t / jev_t if jev_t > 0 else float("inf")

    lines = []
    lines.append("Sample: %s" % sample_id)
    lines.append("  Agreement: %d/%d (%.0f%%)" % (
        len(agree), len(jev_by_id), 100.0 * len(agree) / len(jev_by_id)))
    if disagree:
        lines.append("  Disagreements:")
        for label, jd, ld in disagree:
            lines.append("    - %s: Jev=%s, LLM=%s" % (label, jd, ld))
    else:
        lines.append("  Disagreements: none")
    lines.append("  Runtime: Jev %.1fs vs LLM %.1fs (%.0fx slower)" % (jev_t, llm_t, fold))
    lines.append("")
    return "\n".join(lines), len(agree), len(jev_by_id), fold


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--samples", nargs="+", default=["test_data1", "test_data2"])
    ap.add_argument("--output", default="model_comparison.txt",
                    help="output filename inside --results-dir")
    args = ap.parse_args()

    sections = []
    header = "=" * 72 + "\nJev vs LLM — AMR decision comparison (same evidence, model differs)\n" + "=" * 72 + "\n"
    for s in args.samples:
        section, _, _, _ = compare_sample(args.results_dir, s)
        sections.append(section)

    out = header + "\n".join(sections)
    out_path = os.path.join(args.results_dir, args.output)
    write_text(out_path, out)
    print(out)
    print("Written: %s" % out_path)


if __name__ == "__main__":
    main()
