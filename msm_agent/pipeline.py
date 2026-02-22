from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np

from msmbuilder.dataset import dataset
from msmbuilder.example_datasets import FsPeptide
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.preprocessing import RobustScaler
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from msmbuilder.lumping import PCCAPlus

from msm_agent.metrics import (
    compute_occupancy_stats,
    compute_transition_sparsity,
    compute_its_table,
    plateau_metric,
    grade_run,
    suggest_fixes,
)
from msm_agent.plots import (
    plot_tica_density_hexbin,
    plot_free_energy,
    plot_occupancy_hist,
    plot_its_curve,
    plot_macro_overlay,
)
from msm_agent.report import write_report

def _make_run_dir(cfg: dict) -> Path:
    base = Path(cfg["run"]["output_dir"])
    base.mkdir(parents=True, exist_ok=True)
    name = cfg["run"]["run_name"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _load_xyz(cfg: dict):
    kind = cfg["data"]["kind"]
    stride = int(cfg["data"]["stride"])

    if kind == "fs_peptide":
        fs = FsPeptide(verbose=False)
        fs.cache()
        xyz = dataset(fs.data_dir + "/*.xtc", topology=fs.data_dir + "/fs-peptide.pdb", stride=stride)
        # original example says 50 ps/frame before stride
        dt_ps = float(cfg["data"]["dt_ps_per_frame"]) * stride
        return xyz, dt_ps
    elif kind == "xtc_glob":
        xtc_glob = cfg["data"]["xtc_glob"]
        top = cfg["data"]["topology"]
        if not xtc_glob or not top:
            raise ValueError("xtc_glob and topology required for kind=xtc_glob")
        xyz = dataset(xtc_glob, topology=top, stride=stride)
        dt_ps = float(cfg["data"]["dt_ps_per_frame"]) * stride
        return xyz, dt_ps
    else:
        raise ValueError(f"Unknown data.kind={kind}")

def run_mvp(cfg: dict) -> str:
    run_dir = _make_run_dir(cfg)
    (run_dir / "figs").mkdir(exist_ok=True)

    # --- load data
    xyz, dt_ps_effective = _load_xyz(cfg)
    dt_ns = dt_ps_effective / 1000.0  # ps -> ns
    n_trajs = len(xyz)
    traj_lens = [len(x) for x in xyz]

    # --- featurize
    featurizer = DihedralFeaturizer(types=cfg["features"]["dihedral_types"])
    diheds = xyz.fit_transform_with(featurizer, str(run_dir / "diheds"), fmt="dir-npy")

    # --- scale
    scaler = RobustScaler()
    scaled = diheds.fit_transform_with(scaler, str(run_dir / "scaled_diheds"), fmt="dir-npy")

    # --- tICA
    tica_cfg = cfg["tica"]
    tica_model = tICA(lag_time=int(tica_cfg["lag_time_frames"]), n_components=int(tica_cfg["n_components"]))
    tica_model = scaled.fit_with(tica_model)
    tica_trajs = scaled.transform_with(tica_model, str(run_dir / "ticas"), fmt="dir-npy")
    txx = np.concatenate(tica_trajs)  # (N_total, tica_dim)

    # plots: density
    plot_tica_density_hexbin(
        txx[:, 0], txx[:, 1],
        outpath=run_dir / "figs" / "tica_density_hexbin.png",
        gridsize=int(cfg["plots"]["gridsize"]),
    )

    # --- clustering
    cl_cfg = cfg["clustering"]
    clusterer = MiniBatchKMeans(n_clusters=int(cl_cfg["n_clusters"]), random_state=int(cfg["run"]["seed"]))
    clustered_trajs = tica_trajs.fit_transform_with(clusterer, str(run_dir / "kmeans"), fmt="dir-npy")

    # occupancy & sparsity checks need "assignments" in 1D
    micro_assign = np.concatenate([np.asarray(t).reshape(-1) for t in clustered_trajs])

    occ_stats = compute_occupancy_stats(micro_assign, n_clusters=int(cl_cfg["n_clusters"]))
    plot_occupancy_hist(
        occ_stats["occupancies"],
        outpath=run_dir / "figs" / "occupancy_hist.png",
        min_occupancy=int(cfg["gates"]["min_occupancy"]),
    )

    # --- MSM fits across lag list (for ITS)
    msm_cfg = cfg["msm"]
    lag_list = [int(x) for x in msm_cfg["lag_time_frames_list"]]
    its = compute_its_table(
        clustered_trajs=clustered_trajs,
        lag_list=lag_list,
        n_timescales=int(msm_cfg["n_timescales"]),
        ergodic_cutoff=float(msm_cfg["ergodic_cutoff"]),
        dt_ns=dt_ns,
    )

    plot_its_curve(
        its,
        outpath=run_dir / "figs" / "its_curve.png",
        top_k=int(cfg["gates"]["plateau_k"]),
    )

    # choose a "primary" MSM (use median lag or smallest passing lag; MVP picks lag_list[1] if exists)
    primary_lag = lag_list[1] if len(lag_list) > 1 else lag_list[0]
    msm = MarkovStateModel(
        lag_time=int(primary_lag),
        n_timescales=int(msm_cfg["n_timescales"]),
        ergodic_cutoff=float(msm_cfg["ergodic_cutoff"]),
    ).fit(clustered_trajs)

    # transition sparsity on primary MSM: approximate from counts if available; fallback: from assignments sequence
    sparsity = compute_transition_sparsity(clustered_trajs, n_states=len(msm.state_labels_))

    # free energy weighted by stationary populations
    # map each frame’s microstate to its population weight
    assignments = clusterer.partial_transform(txx)
    assignments = msm.partial_transform(assignments)
    if isinstance(assignments, list):
        assignments = assignments[0]
    assignments = np.asarray(assignments).reshape(-1)
    w = np.asarray(msm.populations_[assignments], dtype=float).reshape(-1)

    plot_free_energy(
        x=txx[:, 0], y=txx[:, 1], weights=w,
        outpath=run_dir / "figs" / "free_energy.png",
        bins=int(cfg["plots"]["bins"]),
    )

    # macro overlay (optional in MVP)
    try:
        pcca = PCCAPlus.from_msm(msm, n_macrostates=4)
        plot_macro_overlay(
            x=txx[:, 0], y=txx[:, 1], weights=w,
            centers=clusterer.cluster_centers_[msm.state_labels_],
            macro_labels=pcca.microstate_mapping_[msm.state_labels_],
            outpath=run_dir / "figs" / "macro_overlay.png",
            bins=int(cfg["plots"]["bins"]),
        )
    except Exception:
        pass

    # --- gates
    plat = plateau_metric(its, top_k=int(cfg["gates"]["plateau_k"]))
    grade = grade_run(cfg, occ_stats, sparsity, plat)
    suggestions = suggest_fixes(cfg, occ_stats, sparsity, plat)

    # --- write artifacts
    metrics = {
        "data": {
            "n_trajs": n_trajs,
            "traj_len_frames_min": int(min(traj_lens)),
            "traj_len_frames_max": int(max(traj_lens)),
            "dt_ns_effective": dt_ns,
            "total_frames": int(sum(traj_lens)),
            "total_time_ns": float(sum(traj_lens) * dt_ns),
        },
        "occupancy": occ_stats,
        "sparsity": sparsity,
        "its": its,
        "plateau": plat,
        "grade": grade,
        "suggestions": suggestions,
        "primary_msm_lag_frames": int(primary_lag),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_report(run_dir, cfg, metrics)

    # save config snapshot
    with open(run_dir / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    return str(run_dir)