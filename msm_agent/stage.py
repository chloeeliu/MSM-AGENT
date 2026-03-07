#msm_agent/msm_agent/stage.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import glob
import json
import time

import numpy as np
from msmbuilder.decomposition import tICA

# ===== Adjust these imports to your real project layout =====
from msm_agent.pipeline_w_mdtraj import (
    _make_run_dir,
    _load_feature,
    _save_intermediate,
)

from msm_agent.metrics import compute_tica_its
from msm_agent.plots import (
    plot_its_curve,
    plot_tica_density_hexbin,
)


# ----------------------------
# JSON / IO helpers
# ----------------------------
def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------
# Shared helpers
# ----------------------------
def load_features_from_run_dir(run_dir: str | Path) -> List[np.ndarray]:
    run_dir = Path(run_dir)
    feature_dir = run_dir / "features"
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    files = sorted(glob.glob(str(feature_dir / "*.npy")))
    if not files:
        raise FileNotFoundError(f"No feature .npy files found in: {feature_dir}")

    return [np.load(f) for f in files]


def build_stage1_summary(
    cfg: Dict[str, Any],
    run_dir: Path,
    n_trajs: int,
    traj_lens: List[int],
    feature_dims: List[Optional[int]],
    dt_ps_effective: float,
) -> str:
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})

    lines = [
        "Stage 1 completed: data loading and featurization finished.",
        f"Run dir: {run_dir}",
        f"Data kind: {data_cfg.get('kind', 'NA')}",
        f"Number of trajectories: {n_trajs}",
        f"Effective timestep: {dt_ps_effective:.4f} ps",
        f"Feature type: {feat_cfg.get('type', 'NA')}",
        f"Feature selection: {feat_cfg.get('selection', 'NA')}",
        f"Atom selection: {feat_cfg.get('atom_selection', 'NA')}",
    ]
    if traj_lens:
        lines.append(
            f"Trajectory length range (frames): min={min(traj_lens)}, max={max(traj_lens)}"
        )
    uniq_dims = sorted({d for d in feature_dims if d is not None})
    if uniq_dims:
        lines.append(f"Feature dimension(s): {uniq_dims}")

    lines += [
        f"Saved features: {run_dir / 'features'}",
        "",
        "Please review whether this featurization is acceptable. You can keep it or modify feature-related config and rerun Stage 1.",
    ]
    return "\n".join(lines)


def build_stage2_summary(
    cfg: Dict[str, Any],
    run_dir: Path,
    lag_list: List[int],
    dt_ns: float,
    plot_path: Path,
) -> str:
    tica_cfg = cfg.get("tica", {})
    lines = [
        "Stage 2 completed: tICA lag scan finished.",
        f"Run dir: {run_dir}",
        f"Scan lag list (frames): {lag_list}",
        f"Scan n_components: {tica_cfg.get('n_components', 'NA')}",
        f"Effective timestep: {dt_ns:.6f} ns",
        f"ITS plot: {plot_path}",
        "",
        "Please review the ITS curve. You can keep the current scan settings, change scan parameters, or set final tICA parameters for Stage 3.",
    ]
    return "\n".join(lines)


def build_stage3_summary(
    run_dir: Path,
    selected_lag_time: int,
    selected_n_components: int,
    tica_shapes: List[List[int]],
    density_plot_path: Optional[Path],
) -> str:
    lines = [
        "Stage 3 completed: final tICA fit finished.",
        f"Run dir: {run_dir}",
        f"Selected lag_time (frames): {selected_lag_time}",
        f"Selected n_components: {selected_n_components}",
        f"Saved tICA trajectories: {run_dir / 'tica_trajs'}",
        f"tICA trajectory shapes: {tica_shapes}",
    ]
    if density_plot_path is not None:
        lines.append(f"Density plot: {density_plot_path}")

    lines += [
        "",
        "Please review the final tICA embedding. If it looks reasonable, you can proceed to clustering in the next phase.",
    ]
    return "\n".join(lines)


# ----------------------------
# Stage 1
# ----------------------------
def run_stage1_featurization(cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = _make_run_dir(cfg)
    ensure_dir(run_dir / "figs")

    features, dt_ps_effective = _load_feature(cfg, run_dir)

    traj_lens = [len(x) for x in features]
    feature_dims = [
        int(x.shape[1]) if getattr(x, "ndim", None) == 2 else None
        for x in features
    ]

    manifest = {
        "stage": "stage1",
        "run_dir": str(run_dir),
        "dt_ps_effective": float(dt_ps_effective),
        "dt_ns_effective": float(dt_ps_effective / 1000.0),
        "n_trajs": len(features),
        "traj_lens": traj_lens,
        "feature_dims": feature_dims,
        "feature_dir": str(run_dir / "features"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(manifest, run_dir / "stage1_manifest.json")

    summary = build_stage1_summary(
        cfg=cfg,
        run_dir=run_dir,
        n_trajs=len(features),
        traj_lens=traj_lens,
        feature_dims=feature_dims,
        dt_ps_effective=float(dt_ps_effective),
    )

    return {
        "ok": True,
        "stage": "stage1",
        "run_dir": str(run_dir),
        "summary": summary,
        "feature_dir": str(run_dir / "features"),
        "manifest_path": str(run_dir / "stage1_manifest.json"),
        "plot_path": None,
    }


# ----------------------------
# Stage 2
# ----------------------------
def run_stage2_tica_scan(cfg: Dict[str, Any], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")

    manifest_path = run_dir / "stage1_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Stage 1 manifest not found: {manifest_path}. Please run Stage 1 first."
        )

    stage1_manifest = read_json(manifest_path)
    features = load_features_from_run_dir(run_dir)
    dt_ps_effective = float(stage1_manifest["dt_ps_effective"])
    dt_ns = dt_ps_effective / 1000.0

    tica_cfg = cfg["tica"]
    lag_min = int(tica_cfg["lag_time_frames_range"][0])
    lag_max = int(tica_cfg["lag_time_frames_range"][1])
    grid_size = int(tica_cfg.get("lag_time_frames_grid_size", 20))
    n_components = int(tica_cfg["n_components"])

    lag_list = np.linspace(lag_min, lag_max, num=grid_size, dtype=int)
    lag_list = np.unique(lag_list)

    tica_its = compute_tica_its(
        features=features,
        lag_list=lag_list,
        n_components=n_components,
        dt_ns=dt_ns,
    )

    plot_path = run_dir / "figs" / "tica_its_curve.png"
    plot_its_curve(
        tica_its,
        outpath=plot_path,
        top_k=int(cfg["gates"]["plateau_k"]),
    )

    manifest = {
        "stage": "stage2",
        "run_dir": str(run_dir),
        "dt_ns": float(dt_ns),
        "tica_lag_list_frames": lag_list.tolist(),
        "n_components": n_components,
        "plot_path": str(plot_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(manifest, run_dir / "stage2_manifest.json")

    summary = build_stage2_summary(
        cfg=cfg,
        run_dir=run_dir,
        lag_list=lag_list.tolist(),
        dt_ns=dt_ns,
        plot_path=plot_path,
    )

    return {
        "ok": True,
        "stage": "stage2",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": str(plot_path),
        "manifest_path": str(run_dir / "stage2_manifest.json"),
    }


# ----------------------------
# Stage 3
# ----------------------------
def run_stage3_tica_fit(cfg: Dict[str, Any], run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    ensure_dir(run_dir / "figs")

    manifest_path = run_dir / "stage1_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Stage 1 manifest not found: {manifest_path}. Please run Stage 1 first."
        )

    features = load_features_from_run_dir(run_dir)

    tica_cfg = cfg["tica"]
    selected_lag_time = tica_cfg.get("selected_lag_time", None)
    selected_n_components = tica_cfg.get("selected_n_components", None)

    if selected_lag_time is None:
        raise ValueError(
            "cfg['tica']['selected_lag_time'] is required for Stage 3. "
            "Please set it first, for example after reviewing Stage 2."
        )

    if selected_n_components is None:
        selected_n_components = int(tica_cfg["n_components"])

    selected_lag_time = int(selected_lag_time)
    selected_n_components = int(selected_n_components)

    tica_model = tICA(
        lag_time=selected_lag_time,
        n_components=selected_n_components,
    )
    tics = tica_model.fit_transform(features)

    _save_intermediate(tics, run_dir / "tica_trajs")

    txx = np.concatenate(tics, axis=0)
    density_plot_path: Optional[Path] = None
    if txx.ndim == 2 and txx.shape[1] >= 2:
        density_plot_path = run_dir / "figs" / "tica_density_hexbin.png"
        plot_tica_density_hexbin(
            txx[:, 0],
            txx[:, 1],
            outpath=density_plot_path,
            gridsize=int(cfg["plots"]["gridsize"]),
        )

    tica_shapes = [list(x.shape) for x in tics]

    manifest = {
        "stage": "stage3",
        "run_dir": str(run_dir),
        "selected_lag_time": selected_lag_time,
        "selected_n_components": selected_n_components,
        "tica_shapes": tica_shapes,
        "tica_traj_dir": str(run_dir / "tica_trajs"),
        "plot_path": str(density_plot_path) if density_plot_path else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(manifest, run_dir / "stage3_manifest.json")

    summary = build_stage3_summary(
        run_dir=run_dir,
        selected_lag_time=selected_lag_time,
        selected_n_components=selected_n_components,
        tica_shapes=tica_shapes,
        density_plot_path=density_plot_path,
    )

    return {
        "ok": True,
        "stage": "stage3",
        "run_dir": str(run_dir),
        "summary": summary,
        "plot_path": str(density_plot_path) if density_plot_path else None,
        "manifest_path": str(run_dir / "stage3_manifest.json"),
    }