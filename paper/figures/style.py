"""Shared publication style for MedShift manuscript figures.

Design rules (paper/PUBLICATION_FIGURE_TABLE_SPEC.md, "Caption rules" and the
visual design system in the publication-figure task):
- vector PDF output, optional PNG preview;
- publication-safe typography (serif, matching the LaTeX text);
- readable at single-column width and in grayscale: color is never the only
  carrier of meaning (marker shapes and line styles are redundant encodings);
- colorblind-safe palette (Okabe-Ito subset);
- no decorative gradients, no 3D, no dashboard styling;
- figure text sized for 10 pt article text at a full \\linewidth (~6.5 in).

All figure scripts import this module; no script may set its own rcParams.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Okabe-Ito colorblind-safe palette (subset used by MedShift figures).
COLORS = {
    "top1": "#D55E00",      # vermillion  (diagnostic Top-1 endpoint)
    "triage": "#0072B2",    # blue        (triage endpoint)
    "base": "#000000",      # black       (Base arm)
    "generic": "#009E73",   # bluish green (Generic arm)
    "medshift": "#CC79A7",  # reddish purple (MedShift arm)
    "reference": "#555555", # neutral reference lines
    "grid": "#DDDDDD",
    "ink": "#1A1A1A",
}

MARKERS = {
    "base": "o",
    "generic": "s",
    "medshift": "D",
}

# Endpoint labels are fixed manuscript terminology.
ENDPOINT_LABELS = {
    "top1": "Diagnostic Top-1",
    "triage": "Triage",
}

FONT_SIZES = {
    "smallest": 7,
    "small": 8,
    "base": 9,
    "label": 10,
}


def apply_style() -> None:
    """Apply the shared rcParams. Call once at the top of every figure script."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "font.size": FONT_SIZES["base"],
            "mathtext.fontset": "cm",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.labelsize": FONT_SIZES["label"],
            "axes.linewidth": 0.8,
            "axes.titlesize": FONT_SIZES["base"],
            "axes.unicode_minus": False,
            "xtick.labelsize": FONT_SIZES["base"],
            "ytick.labelsize": FONT_SIZES["base"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": FONT_SIZES["small"],
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 5,
            "errorbar.capsize": 3,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed (used for provenance records)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(
    output_pdf: Path,
    script_path: Path,
    inputs: dict[str, Path],
    extra: dict | None = None,
) -> Path:
    """Write a JSON provenance sidecar next to a generated figure.

    The sidecar records the output path, the generating script, the SHA-256 of
    every input artifact, and the Python/matplotlib versions, so each figure in
    the manuscript can be traced to its inputs (paper/VISUAL_TRACEABILITY.md).
    """
    record = {
        "schema": "medshift_paper_figure_provenance_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(script_path).name,
        "script_sha256": sha256_file(Path(script_path)),
        "output": Path(output_pdf).name,
        "inputs": {
            name: {"path": str(Path(path).as_posix()), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
        "python_version": sys.version.split()[0],
        "matplotlib_version": matplotlib.__version__,
        "platform": platform.platform(),
    }
    if extra:
        record["extra"] = extra
    sidecar = Path(output_pdf).with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def save_figure(fig, output_pdf: Path, png_preview: bool = True) -> Path:
    """Save vector PDF (primary) and an optional PNG preview next to it."""
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    if png_preview:
        fig.savefig(output_pdf.with_suffix(".png"), bbox_inches="tight")
    return output_pdf
