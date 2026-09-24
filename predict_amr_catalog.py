#!/usr/bin/env python3
"""predict_amr_catalog.py — Jev (System One decision model) arm.

Filter first in code (WHO catalogue lookup), then send ONLY the filtered
evidence and 27 typed questions to Jev in ONE round trip. Jev returns typed
choices with calibrated probabilities; no parsing, no free text.

Timing recorded: timing.decision_call_s / attribution_s / total_s.

Usage:
  export JEV_API_KEY=jv_live_...
  python3 predict_amr_catalog.py --sample data/test_data1.json --outdir results/test_data1
  python3 predict_amr_catalog.py --sample data/test_data1.json --outdir results/test_data1 --mock
"""

import argparse
import os
import random
import time

import requests

from amr_common import (DECISION_OPTIONS, build_decision_items, evidence_state,
                        load_json, render_report, write_json, write_text)

JEV_URL = os.environ.get("JEV_API_URL", "https://jevtypesafeai.com/api/v1/decide")


def build_questions(items):
    """One typed choice question per decision item; Jev evaluates all in parallel."""
    questions = {}
    for it in items:
        questions[it["item_id"]] = {
            "type": "choice",
            "instructions": (
                "Based only on the supplied evidence for this isolate, what is the "
                "resistance decision for %s?" % it["drug"]),
            "criteria": DECISION_OPTIONS,
        }
    return questions


def call_jev(state, questions, api_key, max_retries=3):
    payload = {"model": "jev-latest", "state": state, "questions": questions}
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                JEV_URL,
                headers={"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"},
                json=payload, timeout=60)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout,
                requests.HTTPError) as err:
            last_err = err
            status = getattr(err.response, "status_code", None) if hasattr(err, "response") else None
            # retry transient network errors and 5xx; do NOT retry 4xx auth/validation errors
            if status is not None and status < 500:
                raise
            if attempt < max_retries:
                wait = 2 ** attempt
                print("Attempt %d/%d failed (%s); retrying in %ds..."
                      % (attempt, max_retries, err.__class__.__name__, wait))
                time.sleep(wait)
    raise last_err


def mock_jev(state, questions):
    """Deterministic stand-in for plumbing validation (no API key needed).

    Heuristic: resistance-associated evidence -> 'resistant'; uncertain
    evidence -> 'unknown' (Jev's documented conservative behavior).
    """
    time.sleep(0.35 + random.random() * 0.1)  # simulate one fast round trip
    answers = {}
    for ev in state["evidence"]:
        decision = "resistant" if ev["who_catalogue_association"] == "resistance-associated" else "unknown"
        answers[ev["id"]] = {
            "type": "choice", "choice": decision,
            "confidence": 0.97 if decision == "resistant" else 0.62,
            "probabilities": {k: (0.97 if k == decision else 0.015) for k in DECISION_OPTIONS},
        }
    return {"model": "jev-mock", "answers": answers,
            "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--catalogue", default="data/who_catalogue.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--mock", action="store_true", help="no API call; deterministic mock")
    args = ap.parse_args()

    t0 = time.perf_counter()
    sample = load_json(args.sample)
    catalogue = load_json(args.catalogue)
    items, dropped = build_decision_items(sample, catalogue)
    state = evidence_state(sample["sample_id"], items)
    questions = build_questions(items)
    filter_s = time.perf_counter() - t0

    # --- decision call (single round trip for ALL questions) ---
    if args.mock:
        resp = mock_jev(state, questions)
        decision_call_s = time.perf_counter() - t0 - filter_s
    else:
        api_key = os.environ.get("JEV_API_KEY")
        if not api_key:
            raise SystemExit("Set JEV_API_KEY (jv_live_...) or use --mock.")
        t1 = time.perf_counter()
        resp = call_jev(state, questions, api_key)
        decision_call_s = time.perf_counter() - t1

    # --- attribution: map typed answers back to drug/mutation items ---
    t2 = time.perf_counter()
    decisions = []
    for it in items:
        a = resp["answers"][it["item_id"]]
        decisions.append({
            "decision": a["choice"],
            "confidence": a.get("confidence"),
            "probabilities": a.get("probabilities"),
        })
    attribution_s = time.perf_counter() - t2
    total_s = time.perf_counter() - t0

    timing = {
        "filter_s": round(filter_s, 3),
        "decision_call_s": round(decision_call_s, 3),
        "attribution_s": round(attribution_s, 4),
        "total_s": round(total_s, 3),
    }

    os.makedirs(args.outdir, exist_ok=True)
    raw = {
        "arm": "jev",
        "sample_id": sample["sample_id"],
        "model": resp.get("model"),
        "n_questions": len(items),
        "dropped_non_catalogue_mutations": dropped,
        "state": state,
        "answers": resp["answers"],
        "usage": resp.get("usage"),
        "timing": timing,
        "decisions": [
            dict(item_id=it["item_id"], drug=it["drug"], gene=it["gene"],
                 change=it["change"], association=it["association"], **dec)
            for it, dec in zip(items, decisions)
        ],
    }
    write_json(os.path.join(args.outdir, "jev_raw.json"), raw)

    extra = [
        "Model: %s | Questions: %d in ONE call | Non-catalogue mutations dropped by code filter: %s"
        % (resp.get("model"), len(items), dropped or "none"),
        "API usage: %s" % resp.get("usage"),
    ]
    report = render_report("Jev", sample["sample_id"], items, decisions, timing, extra)
    write_text(os.path.join(args.outdir, "jev_report.txt"), report)
    print(report)


if __name__ == "__main__":
    main()
