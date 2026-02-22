from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
from sklearn.preprocessing import RobustScaler
import mdtraj as md

from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from msmbuilder.lumping import PCCAPlus

from msm_agent.metrics import (
    compute_occupancy_stats,
    compute_tica_its,
    compute_transition_sparsity,
    compute_msm_its,
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
import glob, os, itertools
from functools import partial


def _make_run_dir(cfg: dict) -> Path:
    base = Path(cfg["run"]["output_dir"])
    base.mkdir(parents=True, exist_ok=True)
    name = cfg["run"]["run_name"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _find_featurizer(frame, feature_selection, atom_selection):
    if atom_selection is None or atom_selection.upper() == "CA": # default to CA
        atom_slice = frame.topology.select("name CA")
    elif atom_selection.upper() == "BACKBONE":
        atom_slice = frame.topology.select("backbone")
    elif atom_selection.upper() == "HEAVY":
        atom_slice = frame.topology.select("not element H")
    else:
        try:
            atom_slice = frame.topology.select(atom_selection)
        except Exception:
            raise ValueError(f"Accepted atom selections are : CA, BACKBONE, HEAVY, or a custom selection string compatible with mdtraj's topology.select syntax. Got: {atom_selection}")

    if feature_selection in ["distances", "displacements"]:
        pairs = list(itertools.combinations(atom_slice, 2))
        return partial(getattr(md, f"compute_{feature_selection}"), atom_pairs=pairs)
    elif feature_selection == "neighbors":
        return partial(getattr(md, f"compute_{feature_selection}"), cutoff=5, query_indices=atom_slice)
    else:
        fun_list = []
        for angle_feature in feature_selection:
            fun_list.append(partial(getattr(md, f"compute_{angle_feature}")))
        return fun_list if len(fun_list) >0 else None

def _transform_data(featurizer, traj):
    if isinstance(featurizer, list):
        features = [f(traj) for f in featurizer]
        return np.concatenate(features, axis=1)
    else:
        return featurizer(traj)
    
def _load_feature(cfg: dict, run_dir: Path):
    kind = cfg["data"]["kind"]
    assert kind in ["xtc", "dcd", "trr"], f"Unsupported data.kind: {kind}. Supported: xtc, dcd, trr"

    data_dir = cfg["data"]["dir"]
    top = cfg["data"]["topology"]
    stride = int(cfg["data"].get("stride", 1))
    feature_type = cfg["features"]["type"]
    feature_selection = cfg["features"]["selection"] # list of angles or single distacne type
    atom_selection = cfg["features"].get("atom_selection", None)
    prepossed_dir = cfg["data"].get("load_preprocessed_dir", None)

    if not data_dir or not top:
        raise ValueError("Both data_dir and topology are required for kind=xtc,dcd,trr")

    files = list(glob.glob(os.path.join(data_dir, f"*.{kind}")))
    loaded_features = []
    if prepossed_dir is not None:
        print("Features already exist, loading from disk...")
        for file in files:
            feature_file = Path(prepossed_dir) / (Path(file).stem + ".npy")
            try:
                loaded_features.append(np.load(feature_file))
            except Exception as e:
                raise ValueError(f"Error loading feature file {feature_file}: {e}")
    else:
        frame = md.load(top)
        if feature_type == "angle":
            assert len(feature_selection) > 1, "Must specify at least two angle types"
            assert all(angle_feature in ["phi", "psi", "chi1", "chi2", "chi3", "chi4", "omega"] for angle_feature in feature_selection), "Unsupported angle type"
        elif feature_type == "distance":
            assert feature_selection in ["distances", "displacements", "neighbors"], "Unsupported distance type"
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        featurizer = _find_featurizer(frame, feature_selection, atom_selection)
        assert featurizer is not None, f"Could not find featurizer for selection: {feature_selection}, {atom_selection}"

        (run_dir / "features").mkdir(exist_ok=True)
        for file in files:
            traj = md.load(file, top=top, stride=stride)
            processed_feature = _transform_data(featurizer,traj)
            out_file = str(run_dir / "features" / (Path(file).stem + ".npy"))
            np.save(out_file, processed_feature)
            loaded_features.append(processed_feature)

    dt_ps = float(cfg["data"]["saving_interval"]) * stride
    return loaded_features, dt_ps

def _save_intermediate(data, out_path: Path):
    out_path.mkdir(exist_ok=True)
    feature_dir = out_path.parent / "features"
    file_name = glob.glob(str(feature_dir / "*.npy"))
    for i,file in enumerate(file_name):
        name = Path(file).stem
        np.save(out_path / f"{name}.npy", data[i])

def run_mvp(cfg: dict) -> str:
    run_dir = _make_run_dir(cfg)
    (run_dir / "figs").mkdir(exist_ok=True)

    # --- load data and featurize
    features, dt_ps_effective = _load_feature(cfg, run_dir)
    dt_ns = dt_ps_effective / 1000.0  # ps -> ns
    n_trajs = len(features)
    traj_lens = [len(x) for x in features]

    # --- tICA
    tica_cfg = cfg["tica"]
    tica_lag_list = np.linspace(int(tica_cfg["lag_time_frames_range"][0]),int(tica_cfg["lag_time_frames_range"][1]),num=20, dtype=int)
    tica_its = compute_tica_its(features, tica_lag_list, n_components=int(tica_cfg["n_components"]), dt_ns=dt_ns)
    # visualize tICA ITS curve to choose lag time for clustering, not implemented afterwards
    plot_its_curve(
        tica_its,
        outpath=run_dir / "figs" / "tica_its_curve.png",
        top_k=int(cfg["gates"]["plateau_k"]),
    )

    tica_model = tICA(lag_time=int(tica_cfg["lag_time_frames"]), n_components=int(tica_cfg["n_components"]))
    tics = tica_model.fit_transform(features)
    _save_intermediate(tics, run_dir / "tica_trajs")
   
    # plots: density
    txx = np.concatenate(tics, axis=0)
    plot_tica_density_hexbin(
        txx[:, 0], txx[:, 1],
        outpath=run_dir / "figs" / "tica_density_hexbin.png",
        gridsize=int(cfg["plots"]["gridsize"]),
    )

    # --- clustering
    cl_cfg = cfg["clustering"]
    clusterer = MiniBatchKMeans(n_clusters=int(cl_cfg["n_clusters"]), random_state=int(cfg["run"]["seed"]))
    clustered_trajs = clusterer.fit_transform(tics)
    _save_intermediate(clustered_trajs, run_dir / "clustered_trajs")

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
        reversible_type=msm_cfg["reversible_type"],
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