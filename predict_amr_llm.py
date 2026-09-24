#!/usr/bin/env python3
"""predict_amr_llm.py — DeepSeek LLM arm (System Two baseline).

Same evidence, same 27 questions — but one sequential chat call per question
(27 calls per sample). Records per-drug seconds + total in the raw JSON.

Usage:
  export DEEPSEEK_API_KEY=sk-...
  python3 predict_amr_llm.py --sample data/test_data1.json --outdir results/test_data1 \
      --model deepseek-chat
  python3 predict_amr_llm.py --sample data/test_data1.json --outdir results/test_data1 --mock
"""

import argparse
import json
import os
import random
import time

import requests

from amr_common import (DECISION_OPTIONS, build_decision_items, evidence_state,
                        load_json, render_report, write_json, write_text)

DEEPSEEK_URL = os.environ.get("LLM_API_URL", "https://api.deepseek.com/chat/completions")
DEFAULT_LLM_MODEL = "Qwen/Qwen3.8-27B"

SYSTEM_PROMPT = (
    "You are an expert microbiologist interpreting Mycobacterium tuberculosis "
    "variant data for drug-resistance prediction. Answer with STRICT JSON only: "
    '{"decision": "resistant" | "no" | "unknown", "confidence": <float 0-1>}. '
    '"resistant" = the evidence supports a resistance call; "no" = the evidence '
    "supports calling the isolate not resistant; \"unknown\" = evidence is "
    "insufficient or the mutation-resistance association is uncertain.")


def evidence_block(it):
    return json.dumps({
        "drug": it["drug"],
        "mutation": "%s %s (%s)" % (it["gene"], it["change"], it["nucleotide"]),
        "read_depth": it["depth"],
        "alt_fraction": it["alt_fraction"],
        "who_catalogue_association": it["association"],
        "who_grading": it["grading"],
    }, indent=2)


def call_llm(it, model, api_key, base_url, max_retries=3):
    user_msg = (
        "Evidence for one drug-mutation decision:\n%s\n\n"
        "Question: based only on this evidence, what is the resistance decision "
        "for %s? (resistant / no / unknown)" % (evidence_block(it), it["drug"]))
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_msg}],
        "temperature": 0,
        "max_tokens": 1024,  # reasoning models spend tokens before the JSON
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                base_url,
                headers={"Authorization": "Bearer %s" % api_key, "Content-Type": "application/json"},
                json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout,
                requests.HTTPError) as err:
            last_err = err
            status = getattr(err.response, "status_code", None) if hasattr(err, "response") else None
            if status is not None and status < 500:
                raise
            if attempt < max_retries:
                wait = 2 ** attempt
                print("Attempt %d/%d failed (%s); retrying in %ds..."
                      % (attempt, max_retries, err.__class__.__name__, wait))
                time.sleep(wait)
    raise last_err


def mock_llm(it, latency_s):
    """Deterministic stand-in for plumbing validation (no API key needed).

    Heuristic: resistance-associated -> 'resistant'; uncertain -> 'no'
    (the overconfident free-text behavior the comparison is designed to expose).
    """
    time.sleep(latency_s)
    decision = "resistant" if it["association"] == "resistance-associated" else "no"
    return {
        "choices": [{"message": {"content": json.dumps(
            {"decision": decision,
             "confidence": 0.93 if decision == "resistant" else 0.71})}}],
        "model": "mock-llm",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--catalogue", default="data/who_catalogue.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default=DEFAULT_LLM_MODEL,
                    help="LLM model id (default: Qwen/Qwen3.8-27B)")
    ap.add_argument("--base-url", default=DEEPSEEK_URL,
                    help="OpenAI-compatible chat-completions endpoint URL")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-latency", type=float, default=8.0,
                    help="simulated seconds per call in mock mode (default 8)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    sample = load_json(args.sample)
    catalogue = load_json(args.catalogue)
    items, dropped = build_decision_items(sample, catalogue)
    state = evidence_state(sample["sample_id"], items)

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key and not args.mock:
        raise SystemExit("Set LLM_API_KEY or use --mock.")

    per_item = []
    decisions = []
    raw_answers = {}
    for it in items:
        t1 = time.perf_counter()
        if args.mock:
            resp = mock_llm(it, args.mock_latency)
        else:
            resp = call_llm(it, args.model, api_key, args.base_url)
        call_s = time.perf_counter() - t1

        msg = resp["choices"][0].get("message", {}) or {}
        # Reasoning models can return content=None (tokens exhausted by reasoning);
        # fall back to the reasoning text, which usually contains the JSON verdict.
        content = msg.get("content") or msg.get("reasoning") or ""
        try:
            parsed = json.loads(content)
            decision = parsed.get("decision", "unknown").lower()
            if decision not in DECISION_OPTIONS:
                decision = "unknown"
            conf = parsed.get("confidence")
        except (json.JSONDecodeError, AttributeError, TypeError):
            # last resort: scan the text for a bare decision keyword
            low = content.lower() if isinstance(content, str) else ""
            decision = next((k for k in ("resistant", "unknown", "no")
                             if '"decision": "%s"' % k in low or low.strip() == k), "unknown")
            conf = None

        per_item.append({"item_id": it["item_id"], "drug": it["drug"],
                         "gene": it["gene"], "change": it["change"],
                         "seconds": round(call_s, 2)})
        decisions.append({"decision": decision, "confidence": conf})
        raw_answers[it["item_id"]] = {"raw": content, "decision": decision,
                                      "confidence": conf, "seconds": round(call_s, 2)}
    total_s = time.perf_counter() - t0

    timing = {
        "n_calls": len(items),
        "mean_call_s": round(sum(p["seconds"] for p in per_item) / len(per_item), 2),
        "total_s": round(total_s, 2),
    }

    os.makedirs(args.outdir, exist_ok=True)
    raw = {
        "arm": "llm",
        "sample_id": sample["sample_id"],
        "model": args.model if not args.mock else "mock-llm",
        "n_calls": len(items),
        "dropped_non_catalogue_mutations": dropped,
        "state": state,
        "answers": raw_answers,
        "per_item_seconds": per_item,
        "timing": timing,
        "decisions": [
            dict(item_id=it["item_id"], drug=it["drug"], gene=it["gene"],
                 change=it["change"], association=it["association"], **dec)
            for it, dec in zip(items, decisions)
        ],
    }
    write_json(os.path.join(args.outdir, "llm_raw.json"), raw)

    extra = [
        "Model: %s | %d SEQUENTIAL chat calls | mean %.2fs/call"
        % (raw["model"], len(items), timing["mean_call_s"]),
        "Slowest calls: " + ", ".join(
            "%s %.1fs" % (p["item_id"], p["seconds"])
            for p in sorted(per_item, key=lambda x: -x["seconds"])[:3]),
    ]
    report = render_report("LLM", sample["sample_id"], items, decisions, timing, extra)
    write_text(os.path.join(args.outdir, "llm_report.txt"), report)
    print(report)


if __name__ == "__main__":
    main()
