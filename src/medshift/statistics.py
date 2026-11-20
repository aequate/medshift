#!/usr/bin/env python3
"""Run the frozen Day-6 analysis from the frozen human-adjudicated Day-2 rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ARMS = ("base", "generic", "medshift")
ENDPOINTS = ("clinical_top1_correct", "triage_correct")
SLICE_IDS = ("geoshift", "geography_control", "languageshift", "oxishift", "general_control")
EXPECTED_SLICE_N = {
    "geoshift": 60,
    "geography_control": 40,
    "languageshift": 60,
    "oxishift": 40,
    "general_control": 20,
}
EXPECTED_SHA = {
    "sap": "f619d5f4ada5769a6cf73807f799196810464d9ee6439676e26de89adbee751a",
    "sap_doc": "19136d946fa8f27ea52da9b7db6f10b6178fdbc7e53312f871db2fd404501048",
    "governance": "a7fa0cbbdc46783dd13635b986e86de132b48cacbbdd52367f1fa4c88f96d147",
    "adjudication": "c32166483129cb8a897dfe96af91c7226635514d5447168927a8d7688a978632",
    "item_rows": "1a55f1e3ab31cd565f5ccc1873ae8b38d166fca8a1646975d2ebd69d8b437953",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_sha(path: Path, expected: str, label: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA mismatch: {observed} != {expected}")
    return observed


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"{path}:{n}: not an object")
                rows.append(value)
    return rows


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def normalize_suite(value: str) -> str:
    return {"codeswitch": "code_switch", "general_triage_control": "general_control"}.get(value, value)


def holm_adjusted(stats: Any, p_values: dict[str, float]) -> dict[str, Any]:
    result = stats.holm_bonferroni(p_values, alpha=0.05)
    ordered = sorted(p_values.items(), key=lambda x: x[1])
    running = 0.0
    m = len(ordered)
    for rank, (name, p) in enumerate(ordered, 1):
        running = max(running, min(1.0, (m - rank + 1) * p))
        result["decisions"][name]["holm_adjusted_p"] = running
    return result


def compact(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("status") == "EMPTY":
        return block
    b = block["bootstrap"]
    m = block["mcnemar"]
    return {
        "n": block["denominator"],
        "left_numerator": block["left_numerator"],
        "right_numerator": block["right_numerator"],
        "left_rate": b["left_rate"],
        "right_rate": b["right_rate"],
        "risk_difference": b["point_estimate"],
        "ci95": [b["ci_low"], b["ci_high"]],
        "cohens_h": b["effect_size_cohens_h"],
        "cohens_g": m["effect_size_cohens_g"],
        "mcnemar": {
            "n11": m["n11"], "n10": m["n10"], "n01": m["n01"], "n00": m["n00"],
            "discordant": m["discordant"], "raw_p": m["p_value"],
            "odds_ratio_continuity_0_5": m["odds_ratio_continuity_0_5"],
        },
        "dropped_n": block["dropped_n"],
    }


def run_comparison(stats: Any, maps: dict[str, dict[str, dict[str, Any]]], endpoint: str,
                   hypothesis: str, left: str, right: str, n_boot: int, seed: int) -> dict[str, Any]:
    raw = stats.compare_endpoint(
        maps[left], maps[right], hypothesis_id=hypothesis, endpoint_id=endpoint,
        n_boot=n_boot, seed=seed,
    )
    return {
        "hypothesis": hypothesis, "endpoint": endpoint, "left": left, "right": right,
        "itt_failure_as_incorrect": compact(raw["primary_estimand"]),
        "determinate_only": compact(raw["sensitivity_estimand"]),
        "n_aligned": raw["n_aligned"], "warnings": raw["warnings"],
    }


def arm_rates(stats: Any, maps: dict[str, dict[str, dict[str, Any]]], estimand: str) -> dict[str, Any]:
    out = {}
    for arm in ARMS:
        values = [stats.encode_binary(row, estimand=estimand) for row in maps[arm].values()]
        binary = [int(x) for x in values if x is not None]
        out[arm] = {"numerator": sum(binary), "denominator": len(binary), "rate": sum(binary) / len(binary) if binary else None}
    return out


def fmt(x: float | None, digits: int = 4) -> str:
    return "N/A" if x is None else f"{x:.{digits}f}"


def rate_cell(x: dict[str, Any]) -> str:
    return f"{x['numerator']}/{x['denominator']} ({x['rate']:.1%})" if x["denominator"] else "0/0 (N/A)"


def make_report(primary: dict[str, Any], secondary: dict[str, Any], slices: dict[str, Any],
                missing: dict[str, Any], other: dict[str, Any], gpu: dict[str, Any]) -> str:
    lines = [
        "# MedShift Day-6 authoritative primary results",
        "",
        "| Endpoint | Base | Generic | MedShift | MedShift−Generic RD | 95% CI | Raw p | Holm p | Holm survives |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    labels = {"clinical_top1_correct": "Clinical top-1", "triage_correct": "Triage"}
    for ep in ENDPOINTS:
        rates = primary["arm_rates"][ep]["itt_failure_as_incorrect"]
        x = primary["comparisons"][ep]["H1_PRIMARY"]["itt_failure_as_incorrect"]
        h = primary["co_primary_holm"]["decisions"][ep]
        lines.append(
            f"| {labels[ep]} | {rate_cell(rates['base'])} | {rate_cell(rates['generic'])} | {rate_cell(rates['medshift'])} | "
            f"{fmt(x['risk_difference'])} | [{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}] | "
            f"{fmt(x['mcnemar']['raw_p'], 6)} | {fmt(h['holm_adjusted_p'], 6)} | {'YES' if h['reject_at_alpha'] else 'NO'} |"
        )

    lines += ["", "## Confirmatory interpretation", ""]
    for ep in ENDPOINTS:
        x = primary["comparisons"][ep]["H1_PRIMARY"]["itt_failure_as_incorrect"]
        h = primary["co_primary_holm"]["decisions"][ep]
        direction = "higher" if x["risk_difference"] > 0 else ("lower" if x["risk_difference"] < 0 else "equal")
        lines.append(
            f"- `{ep}`: MedShift was {direction} than Generic by {x['risk_difference']:.4f}; "
            f"HOLM_SURVIVES={'YES' if h['reject_at_alpha'] else 'NO'}."
        )
    survivors = [ep for ep in ENDPOINTS if primary["co_primary_holm"]["decisions"][ep]["reject_at_alpha"]]
    if not survivors:
        lines.append("- Neither endpoint survived Holm; the primary confirmatory comparison is unsupported.")
    elif len(survivors) == 1:
        lines.append(f"- Only `{survivors[0]}` survived Holm. This does not establish overall robustness improvement.")
    else:
        lines.append("- Both separate co-primary dimensions survived Holm; no synthetic overall-success endpoint was created.")

    lines += ["", "## Secondary model comparisons", ""]
    for ep in ENDPOINTS:
        lines += [f"### {labels[ep]}", "", "| Comparison | Left | Right | RD | 95% CI | Raw p | Holm p | n10 | n01 | Cohen h | Cohen g |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for hid in ("H2_SECONDARY_MEDSHIFT_BASE", "H3_SECONDARY_GENERIC_BASE"):
            rec = secondary[ep][hid]["itt_failure_as_incorrect"]
            h = secondary[ep]["holm"]["decisions"][hid]
            lines.append(
                f"| {hid} | {rec['left_numerator']}/{rec['n']} ({rec['left_rate']:.1%}) | "
                f"{rec['right_numerator']}/{rec['n']} ({rec['right_rate']:.1%}) | {fmt(rec['risk_difference'])} | "
                f"[{fmt(rec['ci95'][0])}, {fmt(rec['ci95'][1])}] | {fmt(rec['mcnemar']['raw_p'], 6)} | "
                f"{fmt(h['holm_adjusted_p'], 6)} | {rec['mcnemar']['n10']} | {rec['mcnemar']['n01']} | "
                f"{fmt(rec['cohens_h'])} | {fmt(rec['cohens_g'])} |"
            )
        lines.append("")

    lines += ["## Supportive secondary Day-2 slices — MedShift vs Generic", ""]
    for ep in ENDPOINTS:
        lines += [f"### {labels[ep]}", "", "| Slice | n | Base | Generic | MedShift | RD | 95% CI | Raw p | Holm p |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        family = slices["holm"]["H1_PRIMARY"]
        for sid in SLICE_IDS:
            block = slices["slices"][sid][ep]
            rates = block["arm_rates"]["itt_failure_as_incorrect"]
            rec = block["comparisons"]["H1_PRIMARY"]["itt_failure_as_incorrect"]
            h = family["decisions"][f"{ep}:{sid}"]
            lines.append(
                f"| {sid} | {rec['n']} | {rate_cell(rates['base'])} | {rate_cell(rates['generic'])} | {rate_cell(rates['medshift'])} | "
                f"{fmt(rec['risk_difference'])} | [{fmt(rec['ci95'][0])}, {fmt(rec['ci95'][1])}] | "
                f"{fmt(rec['mcnemar']['raw_p'], 6)} | {fmt(h['holm_adjusted_p'], 6)} |"
            )
        lines.append("")

    lines += ["## Missing, indeterminate, and N/A", ""]
    for ep in ENDPOINTS:
        lines.append(f"### {labels[ep]}")
        lines.append("")
        lines.append("| Arm | ok | unresolved indeterminate | N/A | missing/other | ITT denominator | Determinate-pairs denominators vs Generic |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        sens = primary["comparisons"][ep]["H1_PRIMARY"]["determinate_only"]
        for arm in ARMS:
            x = missing[ep][arm]
            det = sens["n"] if arm in {"generic", "medshift"} else "—"
            lines.append(f"| {arm} | {x.get('ok',0)} | {x.get('needs_review_unresolved',0)} | {x.get('not_applicable',0)} | {x.get('other',0)} | {primary['arm_rates'][ep]['itt_failure_as_incorrect'][arm]['denominator']} | {det} |")
        lines.append("")

    lines += ["## Other frozen outcomes", "", f"- Format/parse diagnostics: {json.dumps(other['format_parse_failure'], sort_keys=True)}", "- General-medical retention: represented by the prespecified `general_control` slice above.", "- LanguageShift failures: represented by the prespecified `languageshift` endpoint results above.", "- Code-switch failures: reported as exploratory endpoint results in the machine-readable artifact."]
    for name, reason in other["unavailable"].items():
        lines.append(f"- `{name}`: unavailable — {reason}")

    lines += ["", "## GPU recommendation", "", f"- Another GPU session justified: **{'YES' if gpu['justified'] else 'NO'}**.", f"- {gpu['reason']}"]
    if gpu["justified"]:
        lines.append("- First experiment: **P0 group-matched Generic control** (1092 records / 1092 groups), addressing the frozen atomic-group diversity confound.")
    lines += ["", "Supportive and exploratory findings do not override the co-primary conclusions.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--adjudication", type=Path, required=True)
    ap.add_argument("--item-rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "scripts/day6"))
    import statistical_analysis as stats

    inputs = {
        "sap": repo / "config/day6/medshift_statistical_analysis_plan_v1.json",
        "sap_doc": repo / "docs/day6/MEDSHIFT_STATISTICAL_ANALYSIS_PLAN_V1.md",
        "governance": repo / "data/day6/preresult_governance_v1/preresult_governance_manifest.json",
        "adjudication": args.adjudication.resolve(),
        "item_rows": args.item_rows.resolve(),
    }
    verified = {name: {"path": str(path), "sha256": require_sha(path, EXPECTED_SHA[name], name)} for name, path in inputs.items()}
    adjudication = load_jsonl(inputs["adjudication"])
    rows = load_jsonl(inputs["item_rows"])
    if len(adjudication) != 780 or len(rows) != 1560:
        raise RuntimeError("authoritative adjudication count mismatch")
    if any(r.get("HUMAN_CONFIRMED") is not True or r.get("AUTHORITATIVE_HUMAN_ADJUDICATION") != "YES" for r in adjudication):
        raise RuntimeError("non-authoritative adjudication row")

    by_endpoint: dict[str, dict[str, dict[str, dict[str, Any]]]] = {ep: {arm: {} for arm in ARMS} for ep in ENDPOINTS}
    meta: dict[str, dict[str, Any]] = {}
    missing: dict[str, dict[str, Counter[str]]] = {ep: {arm: Counter() for arm in ARMS} for ep in ENDPOINTS}
    for r in rows:
        ep, arm, item = r["endpoint"], r["arm"], r["item_id"]
        if ep not in ENDPOINTS or arm not in ARMS:
            raise RuntimeError("unexpected endpoint/arm")
        if item in by_endpoint[ep][arm]:
            raise RuntimeError(f"duplicate {ep} {arm} {item}")
        rec = {"status": r["status"], "value": r.get("value"), "correct": r.get("correct")}
        by_endpoint[ep][arm][item] = rec
        status = r["status"] if r["status"] in {"ok", "needs_review_unresolved", "not_applicable"} else "other"
        missing[ep][arm][status] += 1
        meta.setdefault(item, {"suite": normalize_suite(r["suite"]), "core_id": r.get("core_id"), "variant_type": r.get("variant_type"), "language": r.get("language"), "matched_pair_id": r.get("matched_pair_id"), "pair_role": r.get("pair_role")})
    if any(set(by_endpoint[ep][arm]) != set(meta) for ep in ENDPOINTS for arm in ARMS) or len(meta) != 260:
        raise RuntimeError("paired item sets differ")

    primary: dict[str, Any] = {"schema_version": "medshift_day6_primary_results_v1", "population": "Day-2 ALL", "n_boot": args.n_boot, "seed": args.seed, "verified_inputs": verified, "arm_rates": {}, "comparisons": {}}
    secondary: dict[str, Any] = {"schema_version": "medshift_day6_secondary_results_v1"}
    primary_ps = {}
    for ep in ENDPOINTS:
        maps = by_endpoint[ep]
        primary["arm_rates"][ep] = {
            "itt_failure_as_incorrect": arm_rates(stats, maps, "itt_failure_as_incorrect"),
            "determinate_only": arm_rates(stats, maps, "determinate_only"),
        }
        h1 = run_comparison(stats, maps, ep, "H1_PRIMARY", "medshift", "generic", args.n_boot, args.seed)
        h2 = run_comparison(stats, maps, ep, "H2_SECONDARY_MEDSHIFT_BASE", "medshift", "base", args.n_boot, args.seed)
        h3 = run_comparison(stats, maps, ep, "H3_SECONDARY_GENERIC_BASE", "generic", "base", args.n_boot, args.seed)
        primary["comparisons"][ep] = {"H1_PRIMARY": h1}
        primary_ps[ep] = h1["itt_failure_as_incorrect"]["mcnemar"]["raw_p"]
        secondary[ep] = {"H2_SECONDARY_MEDSHIFT_BASE": h2, "H3_SECONDARY_GENERIC_BASE": h3}
        secondary[ep]["holm"] = holm_adjusted(stats, {
            "H2_SECONDARY_MEDSHIFT_BASE": h2["itt_failure_as_incorrect"]["mcnemar"]["raw_p"],
            "H3_SECONDARY_GENERIC_BASE": h3["itt_failure_as_incorrect"]["mcnemar"]["raw_p"],
        })
    primary["co_primary_holm"] = holm_adjusted(stats, primary_ps)

    slice_ids = {sid: {item for item, m in meta.items() if m["suite"] == sid} for sid in SLICE_IDS}
    if {k: len(v) for k, v in slice_ids.items()} != EXPECTED_SLICE_N:
        raise RuntimeError("slice counts differ from SAP")
    supportive: dict[str, Any] = {"schema_version": "medshift_day6_supportive_slices_v1", "slices": {}, "holm": {}}
    slice_ps = {hid: {} for hid in ("H1_PRIMARY", "H2_SECONDARY_MEDSHIFT_BASE", "H3_SECONDARY_GENERIC_BASE")}
    for sid, ids in slice_ids.items():
        supportive["slices"][sid] = {}
        for ep in ENDPOINTS:
            maps = {arm: {item: by_endpoint[ep][arm][item] for item in sorted(ids)} for arm in ARMS}
            comps = {
                "H1_PRIMARY": run_comparison(stats, maps, ep, "H1_PRIMARY", "medshift", "generic", args.n_boot, args.seed),
                "H2_SECONDARY_MEDSHIFT_BASE": run_comparison(stats, maps, ep, "H2_SECONDARY_MEDSHIFT_BASE", "medshift", "base", args.n_boot, args.seed),
                "H3_SECONDARY_GENERIC_BASE": run_comparison(stats, maps, ep, "H3_SECONDARY_GENERIC_BASE", "generic", "base", args.n_boot, args.seed),
            }
            supportive["slices"][sid][ep] = {
                "arm_rates": {"itt_failure_as_incorrect": arm_rates(stats, maps, "itt_failure_as_incorrect"), "determinate_only": arm_rates(stats, maps, "determinate_only")},
                "comparisons": comps,
            }
            for hid, rec in comps.items():
                slice_ps[hid][f"{ep}:{sid}"] = rec["itt_failure_as_incorrect"]["mcnemar"]["raw_p"]
    for hid, pvals in slice_ps.items():
        supportive["holm"][hid] = holm_adjusted(stats, pvals)

    code_ids = {item for item, m in meta.items() if m["suite"] == "code_switch"}
    exploratory = {"schema_version": "medshift_day6_exploratory_results_v1", "code_switch": {}}
    for ep in ENDPOINTS:
        maps = {arm: {item: by_endpoint[ep][arm][item] for item in sorted(code_ids)} for arm in ARMS}
        exploratory["code_switch"][ep] = {
            "arm_rates": arm_rates(stats, maps, "itt_failure_as_incorrect"),
            "medshift_vs_generic": run_comparison(stats, maps, ep, "H1_PRIMARY", "medshift", "generic", args.n_boot, args.seed),
        }

    diagnostic_path = repo / "analysis_work/day6_frozen_primary_20260902/results/day6_frozen_analysis.json"
    format_diag = {}
    if diagnostic_path.is_file():
        d = load_json(diagnostic_path)
        format_diag = (d.get("other_frozen_outcomes") or {}).get("arm_summary") or {}
    other = {
        "schema_version": "medshift_day6_other_outcomes_v1",
        "format_parse_failure": format_diag,
        "exploratory": exploratory,
        "unavailable": {
            "under_triage": "human export contains correctness but not authoritative assigned-direction labels",
            "over_triage": "human export contains correctness but not authoritative assigned-direction labels",
            "geographic_overreaction": "no frozen authoritative pair-direction adjudication in this export",
            "geographic_underreaction": "no frozen authoritative pair-direction adjudication in this export",
            "sensor_overtrust": "no frozen authoritative sensor-handling adjudication in this export",
            "sensor_undertrust": "no frozen authoritative sensor-handling adjudication in this export",
            "contradiction_handling": "no separate frozen authoritative mapping beyond OxiShift endpoint performance",
            "uncertainty_failures": "no authoritative uncertainty endpoint in this export",
            "arabic_semantic_consistency": "no authoritative semantic pair adjudication in this export",
            "code_switch_semantic_consistency": "no authoritative semantic pair adjudication in this export",
        },
    }

    positive_survivors = []
    for ep in ENDPOINTS:
        decision = primary["co_primary_holm"]["decisions"][ep]
        rd = primary["comparisons"][ep]["H1_PRIMARY"]["itt_failure_as_incorrect"]["risk_difference"]
        if decision["reject_at_alpha"] and rd > 0:
            positive_survivors.append(ep)
    gpu = {
        "justified": bool(positive_survivors),
        "positive_holm_surviving_endpoints": positive_survivors,
        "recommended_first_control": "P0_GROUP_MATCHED_GENERIC" if positive_survivors else None,
        "reason": (
            "At least one positive co-primary MedShift-vs-Generic effect survived Holm; the pre-frozen P0 control is needed before attributing that effect to shift-aware content rather than atomic-group diversity."
            if positive_survivors else
            "No positive MedShift-vs-Generic co-primary effect survived Holm, so the frozen primary result does not justify another GPU session."
        ),
    }
    primary["gpu_recommendation"] = gpu
    primary["missing_indeterminate_na"] = {ep: {arm: dict(missing[ep][arm]) for arm in ARMS} for ep in ENDPOINTS}

    dump_json(out / "primary_results.json", primary)
    dump_json(out / "secondary_results.json", secondary)
    dump_json(out / "supportive_slices.json", supportive)
    dump_json(out / "other_outcomes.json", other)
    dump_json(out / "exploratory_results.json", exploratory)
    report = make_report(primary, secondary, supportive, primary["missing_indeterminate_na"], other, gpu)
    (out / "DAY6_PRIMARY_RESULTS_REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    with (out / "primary_results.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["endpoint", "base_numerator", "base_denominator", "base_rate", "generic_numerator", "generic_denominator", "generic_rate", "medshift_numerator", "medshift_denominator", "medshift_rate", "risk_difference", "ci95_low", "ci95_high", "raw_p", "holm_adjusted_p", "holm_survives", "n11", "n10", "n01", "n00", "cohens_h", "cohens_g"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ep in ENDPOINTS:
            rates = primary["arm_rates"][ep]["itt_failure_as_incorrect"]
            x = primary["comparisons"][ep]["H1_PRIMARY"]["itt_failure_as_incorrect"]
            h = primary["co_primary_holm"]["decisions"][ep]
            row = {"endpoint": ep, "risk_difference": x["risk_difference"], "ci95_low": x["ci95"][0], "ci95_high": x["ci95"][1], "raw_p": x["mcnemar"]["raw_p"], "holm_adjusted_p": h["holm_adjusted_p"], "holm_survives": h["reject_at_alpha"], "cohens_h": x["cohens_h"], "cohens_g": x["cohens_g"]}
            for arm in ARMS:
                for k in ("numerator", "denominator", "rate"):
                    row[f"{arm}_{k}"] = rates[arm][k]
            for k in ("n11", "n10", "n01", "n00"):
                row[k] = x["mcnemar"][k]
            w.writerow(row)

    artifacts = [
        out / "day2_primary_adjudication.jsonl", out / "day2_primary_adjudication_item_rows.jsonl",
        out / "day2_primary_adjudication_freeze.json", out / "primary_results.json",
        out / "primary_results.csv", out / "secondary_results.json", out / "supportive_slices.json",
        out / "other_outcomes.json", out / "exploratory_results.json", out / "DAY6_PRIMARY_RESULTS_REPORT.md",
    ]
    manifest = {
        "schema_version": "medshift_day6_final_artifact_manifest_v1",
        "artifacts": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p), "read_only": p.stat().st_file_attributes & 1 == 1 if sys.platform == "win32" else None} for p in artifacts],
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "exporter": {"path": str(repo / "analysis_work/day6_full_adjudication_20260902/tools/export_authoritative.py"), "sha256": sha256(repo / "analysis_work/day6_full_adjudication_20260902/tools/export_authoritative.py")},
    }
    dump_json(out / "artifact_manifest.json", manifest)
    (out / "artifact_manifest.json.sha256").write_text(sha256(out / "artifact_manifest.json") + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "COMPUTED", "primary": primary, "manifest_sha256": sha256(out / "artifact_manifest.json")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
