"""Figure 2: v1 diagnosis-versus-triage endpoint tradeoff (frozen v1 data only).

Generates ``paper/figures/v1_endpoint_tradeoff.pdf`` (vector) plus a PNG
preview and a provenance sidecar from the two canonical frozen v1 artifacts:

- ``analysis_work/day6_frozen_primary_20260902/final_results_rereview_20260902/
  primary_results.json`` (Day-6 authoritative rereview: arm rates, H1 paired
  contrasts, Holm decisions), and
- ``day6_output/p0_final_archive/P0_FINAL_FREEZE.json`` (P0 group-matched
  control: ITT contrasts and Holm decisions).

The script is fail-closed: it validates the artifact schema and re-derives every
displayed quantity from the frozen integer counts. If any expected count, rate,
risk difference, confidence bound, or Holm p-value does not match the frozen
values embedded below, it raises and writes nothing. Only the two v1 artifacts
above are read; no DEV or FINAL artifact is touched.

Usage (repository root):
    python paper/figures/plot_v1_endpoints.py \
        [--primary PATH] [--p0 PATH] [--outdir paper/figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import style  # noqa: E402  (shared figure style; sets matplotlib backend)

PRIMARY_DEFAULT = REPO / (
    "analysis_work/day6_frozen_primary_20260902/final_results_rereview_20260902/"
    "primary_results.json"
)
P0_DEFAULT = REPO / "day6_output/p0_final_archive/P0_FINAL_FREEZE.json"
OUTPUT_STEM = "v1_endpoint_tradeoff"

N_ITEMS = 260

# Frozen values this figure is allowed to display. Any mismatch with the
# canonical artifacts is an error, not a reason to redraw.
EXPECTED = {
    "itt": {
        # arm -> endpoint -> (numerator, denominator)
        "base": {"clinical_top1_correct": (155, 260), "triage_correct": (150, 260)},
        "generic": {"clinical_top1_correct": (163, 260), "triage_correct": (151, 260)},
        "medshift": {"clinical_top1_correct": (149, 260), "triage_correct": (187, 260)},
    },
    "h1": {
        # endpoint -> (RD, ci_low, ci_high, holm_p); RD/CI in proportion units
        "clinical_top1_correct": (-0.05384615384615388, -0.10769230769230775, 0.0, 0.07590472948910143),
        "triage_correct": (0.1384615384615384, 0.08461538461538465, 0.1923076923076923, 8.078643478626191e-07),
    },
    "p0": {
        "clinical_top1_correct": (
            149, 167, -0.06923076923076932, -0.12692307692307692, -0.011538461538461497,
            0.024746035842440797,
        ),
        "triage_correct": (
            187, 164, 0.08846153846153848, 0.038461538461538436, 0.13846153846153852,
            0.0021770764129485087,
        ),
    },
}

ARM_LABELS = {"base": "Base", "generic": "Generic", "medshift": "MedShift"}
ENDPOINT_TITLES = {
    "clinical_top1_correct": "Diagnostic Top-1",
    "triage_correct": "Triage",
}
ENDPOINT_COLORS = {
    "clinical_top1_correct": style.COLORS["top1"],
    "triage_correct": style.COLORS["triage"],
}


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"fail-closed validation failed: {message}")


def load_and_validate_primary(path: Path) -> dict:
    """Validate the frozen Day-6 primary artifact and extract figure data."""
    with Path(path).open(encoding="utf-8") as handle:
        art = json.load(handle)
    rates = art["arm_rates"]
    out: dict = {"path": str(Path(path).as_posix())}
    for arm, endpoints in EXPECTED["itt"].items():
        for endpoint, (num, den) in endpoints.items():
            cell = rates[endpoint]["itt_failure_as_incorrect"][arm]
            _require(cell["denominator"] == den, f"primary {arm}/{endpoint} denominator")
            _require(cell["numerator"] == num, f"primary {arm}/{endpoint} numerator")
            _require(_close(cell["rate"], num / den), f"primary {arm}/{endpoint} rate")
    out["rates"] = {
        arm: {
            endpoint: 100.0 * num / den
            for endpoint, (num, den) in endpoints.items()
        }
        for arm, endpoints in EXPECTED["itt"].items()
    }
    out["counts"] = {arm: dict(endpoints) for arm, endpoints in EXPECTED["itt"].items()}
    comparisons = art["comparisons"]
    decisions = art["co_primary_holm"]["decisions"]
    out["h1"] = {}
    for endpoint, (rd, lo, hi, holm) in EXPECTED["h1"].items():
        comp = comparisons[endpoint]["H1_PRIMARY"]["itt_failure_as_incorrect"]
        dec = decisions[endpoint]
        _require(_close(comp["risk_difference"], rd), f"H1 {endpoint} RD")
        _require(_close(comp["ci95"][0], lo), f"H1 {endpoint} CI low")
        _require(_close(comp["ci95"][1], hi), f"H1 {endpoint} CI high")
        _require(_close(dec["holm_adjusted_p"], holm), f"H1 {endpoint} Holm p")
        _require(
            _close(comp["risk_difference"], comp["left_numerator"] / comp["n"]
                   - comp["right_numerator"] / comp["n"]),
            f"H1 {endpoint} RD not consistent with frozen counts",
        )
        out["h1"][endpoint] = {
            "rd_pp": 100.0 * rd, "ci_pp": [100.0 * lo, 100.0 * hi],
            "holm_p": holm, "reject": bool(dec["reject_at_alpha"]),
        }
    return out


def load_and_validate_p0(path: Path) -> dict:
    """Validate the frozen P0 artifact and extract figure data."""
    with Path(path).open(encoding="utf-8") as handle:
        art = json.load(handle)
    itt = art["frozen_itt_results"]
    out: dict = {"path": str(Path(path).as_posix())}
    out["p0"] = {}
    for endpoint, (ms, ctrl, rd, lo, hi, holm) in EXPECTED["p0"].items():
        cell = itt[endpoint]
        _require(cell["medshift_numerator"] == ms, f"P0 {endpoint} medshift numerator")
        _require(
            cell["generic_group_matched_numerator"] == ctrl,
            f"P0 {endpoint} control numerator",
        )
        _require(cell["n"] == N_ITEMS, f"P0 {endpoint} denominator")
        _require(_close(cell["paired_risk_difference"], rd), f"P0 {endpoint} RD")
        _require(_close(cell["ci95_low"], lo), f"P0 {endpoint} CI low")
        _require(_close(cell["ci95_high"], hi), f"P0 {endpoint} CI high")
        _require(_close(cell["holm_adjusted_p"], holm), f"P0 {endpoint} Holm p")
        _require(
            _close(cell["paired_risk_difference"], ms / float(cell["n"]) - ctrl / float(cell["n"])),
            f"P0 {endpoint} RD not consistent with frozen counts",
        )
        out["p0"][endpoint] = {
            "rd_pp": 100.0 * rd, "ci_pp": [100.0 * lo, 100.0 * hi],
            "holm_p": holm, "reject": bool(cell["holm_reject_at_alpha"]),
            "medshift": ms, "control": ctrl, "n": cell["n"],
        }
    return out


def build_figure(primary: dict, p0: dict):
    """Build the two-panel endpoint tradeoff figure from validated data."""
    import matplotlib.pyplot as plt

    style.apply_style()
    endpoints = ["clinical_top1_correct", "triage_correct"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.1, 3.1), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    # ---- Panel A: absolute ITT rates by arm (point estimates only; no rate
    # CIs are frozen, so none are drawn).
    ax_a.set_title("A  Absolute ITT rates", loc="left", fontsize=style.FONT_SIZES["base"],
                   fontweight="bold")
    y_pos = {"clinical_top1_correct": 1.0, "triage_correct": 0.0}
    for endpoint in endpoints:
        for arm in ("base", "generic", "medshift"):
            rate = primary["rates"][arm][endpoint]
            ax_a.plot(
                rate, y_pos[endpoint] + (0.16 * (1 - ["base", "generic", "medshift"].index(arm))),
                linestyle="none",
                marker=style.MARKERS[arm],
                color=style.COLORS[arm],
                label=ARM_LABELS[arm] if endpoint == "clinical_top1_correct" else None,
                clip_on=False,
            )
            num, den = primary["counts"][arm][endpoint]
            ax_a.annotate(
                f"{num}/{den}", (rate, y_pos[endpoint] + 0.16 * (1 - ["base", "generic", "medshift"].index(arm))),
                textcoords="offset points", xytext=(7, -2.5),
                fontsize=style.FONT_SIZES["smallest"], color=style.COLORS["ink"],
            )
    ax_a.set_yticks([1.0, 0.0], [style.ENDPOINT_LABELS["top1"], style.ENDPOINT_LABELS["triage"]])
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(-0.55, 1.55)
    ax_a.set_xlabel("Correct (%)")
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.grid(axis="x")
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncols=1, columnspacing=0.9,
                handletextpad=0.3, borderaxespad=0.0)

    # ---- Panel B: paired risk differences with frozen unadjusted item-paired
    # bootstrap 95% CIs; H1 (original Generic) then P0 (group-matched Generic);
    # Top-1 above triage within each estimand.
    ax_b.set_title("B  Paired risk differences", loc="left", fontsize=style.FONT_SIZES["base"],
                   fontweight="bold")
    rows = [
        ("h1", "clinical_top1_correct", 3.0),
        ("h1", "triage_correct", 2.0),
        ("p0", "clinical_top1_correct", 1.0),
        ("p0", "triage_correct", 0.0),
    ]
    estimand_label = {"h1": "H1 vs Generic (equal record)", "p0": "P0 vs group-matched Generic"}
    for source, endpoint, y in rows:
        data = primary["h1"][endpoint] if source == "h1" else p0["p0"][endpoint]
        lo, hi = data["ci_pp"]
        marker = "o" if source == "h1" else "^"
        ax_b.errorbar(
            data["rd_pp"], y, xerr=[[data["rd_pp"] - lo], [hi - data["rd_pp"]]],
            fmt=marker, color=ENDPOINT_COLORS[endpoint], ecolor=ENDPOINT_COLORS[endpoint],
            markersize=5.5, linewidth=1.4, capsize=3, clip_on=False,
        )
        if source == "h1" and endpoint == "clinical_top1_correct":
            p_text = "p=0.0759 (n.s.; CI includes 0)"
        elif data["holm_p"] < 0.001:
            mantissa, exponent = f"{data['holm_p']:.1e}".split("e")
            p_text = f"p={mantissa}$\\times$10$^{{{int(exponent)}}}$"
        else:
            p_text = f"p={data['holm_p']:.4f}"
        star_free = f"{data['rd_pp']:+.2f} pp, Holm-adjusted {p_text}"
        ax_b.annotate(
            star_free, (hi, y), textcoords="offset points", xytext=(8, -2.5),
            fontsize=style.FONT_SIZES["smallest"], color=style.COLORS["ink"],
        )
    ax_b.axvline(0, color=style.COLORS["reference"], linewidth=0.9, linestyle="--", zorder=0)
    ax_b.axhline(1.5, color=style.COLORS["grid"], linewidth=0.7)
    ax_b.set_yticks([3.0, 2.0, 1.0, 0.0])
    ax_b.set_yticklabels(
        [
            "H1: Diagnostic Top-1\nvs Generic",
            "H1: Triage\nvs Generic",
            "P0: Diagnostic Top-1\nvs group-matched Generic",
            "P0: Triage\nvs group-matched Generic",
        ],
        fontsize=style.FONT_SIZES["small"],
    )
    ax_b.set_xlim(-16, 40)
    ax_b.set_ylim(-0.75, 3.6)
    ax_b.set_xlabel("Risk difference, MedShift $-$ comparator (percentage points;\npositive favors MedShift)",
                    fontsize=style.FONT_SIZES["small"])
    ax_b.set_xticks([-10, 0, 10, 20, 30])
    ax_b.grid(axis="x")
    ax_b.spines[["top", "right", "left"]].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    ax_b.annotate(
        "unadjusted item-paired bootstrap 95% CIs\n(v1 resampling unit: items, not cores)",
        (0.02, 0.02), xycoords="axes fraction", ha="left", va="bottom",
        fontsize=style.FONT_SIZES["smallest"], color=style.COLORS["reference"],
    )

    fig.tight_layout(w_pad=2.2)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--primary", type=Path, default=PRIMARY_DEFAULT)
    parser.add_argument("--p0", type=Path, default=P0_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=HERE)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args(argv)

    primary = load_and_validate_primary(args.primary)
    p0 = load_and_validate_p0(args.p0)
    fig = build_figure(primary, p0)

    out_pdf = args.outdir / f"{OUTPUT_STEM}.pdf"
    style.save_figure(fig, out_pdf, png_preview=not args.no_png)
    style.write_provenance(
        out_pdf,
        Path(__file__),
        {"primary_results": args.primary, "p0_final_freeze": args.p0},
        extra={"figure_label": "fig:v1-endpoint-tradeoff", "endpoint_terms": "Diagnostic Top-1 / Triage"},
    )
    print(f"wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
