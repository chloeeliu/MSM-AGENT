from __future__ import annotations
from pathlib import Path
import json

def write_report(run_dir: Path, cfg: dict, metrics: dict) -> None:
    figs = {
        "tica_density": "figs/tica_density_hexbin.png",
        "free_energy": "figs/free_energy.png",
        "occupancy": "figs/occupancy_hist.png",
        "its": "figs/its_curve.png",
        "macro": "figs/macro_overlay.png",
    }

    grade = metrics["grade"]
    md = []
    md.append(f"# MSM Agent MVP Report\n")
    md.append(f"**Grade:** `{grade['label']}`\n")
    if grade["fail_reasons"]:
        md.append("## Fail reasons\n")
        for r in grade["fail_reasons"]:
            md.append(f"- {r}\n")
    if grade["warn_reasons"]:
        md.append("## Warnings\n")
        for r in grade["warn_reasons"]:
            md.append(f"- {r}\n")

    md.append("## Data summary\n")
    d = metrics["data"]
    md.append(f"- trajectories: {d['n_trajs']}\n")
    md.append(f"- total_time_ns (approx): {d['total_time_ns']:.2f}\n")
    md.append(f"- dt_ns_effective: {d['dt_ns_effective']:.4f}\n")

    md.append("## Key metrics\n")
    md.append(f"- tiny_frac: {grade['tiny_frac']:.3f}\n")
    md.append(f"- avg_out_degree: {grade['avg_out_degree']:.2f}\n")
    md.append(f"- ITS rel_std_max(top-k): {grade['rel_std_max']:.3f}\n")

    md.append("## Suggestions\n")
    for s in metrics["suggestions"]:
        md.append(f"- {s}\n")

    md.append("## Figures\n")
    for k, rel in figs.items():
        p = run_dir / rel
        if p.exists():
            md.append(f"### {k}\n")
            md.append(f"![{k}]({rel})\n")

    (run_dir / "report.md").write_text("".join(md))