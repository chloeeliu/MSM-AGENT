from __future__ import annotations
import numpy as np
from msmbuilder.decomposition import tICA
from msmbuilder.msm import MarkovStateModel

def compute_occupancy_stats(assign_1d: np.ndarray, n_clusters: int) -> dict:
    assign_1d = np.asarray(assign_1d).reshape(-1)
    occ = np.bincount(assign_1d, minlength=n_clusters)
    occ_sorted = np.sort(occ)
    return {
        "n_clusters": int(n_clusters),
        "n_used": int(np.sum(occ > 0)),
        "tiny_frac": float(np.mean(occ < 10)),  # threshold refined in grade_run
        "occupancies": occ.tolist(),
        "occupancy_min": int(occ.min()),
        "occupancy_median": float(np.median(occ)),
        "occupancy_p90": float(np.percentile(occ, 90)),
    }

def compute_transition_sparsity(clustered_trajs, n_states: int) -> dict:
    # quick proxy: count unique outgoing transitions from each state using adjacent frames
    out_sets = [set() for _ in range(n_states)]
    total_edges = 0
    for traj in clustered_trajs:
        a = np.asarray(traj).reshape(-1)
        for i in range(len(a) - 1):
            s = int(a[i])
            t = int(a[i+1])
            if s < n_states and t < n_states:
                out_sets[s].add(t)
    out_degrees = np.array([len(s) for s in out_sets], dtype=float)
    return {
        "n_states": int(n_states),
        "avg_out_degree": float(out_degrees.mean()),
        "median_out_degree": float(np.median(out_degrees)),
        "min_out_degree": float(out_degrees.min()),
        "p10_out_degree": float(np.percentile(out_degrees, 10)),
    }

def compute_tica_its(features, lag_list, n_components: int, dt_ns: float) -> dict:
    # returns per-lag implied timescales in ns
    table = {}
    for lag in lag_list:
        tica = tICA(lag_time=lag, n_components=int(n_components))
        tica.fit(features)
        ts = np.asarray(tica.timescales_, dtype=float) # [1, n_components]
        table[str(lag*dt_ns)] = ts * dt_ns.tolist()
    return table 

def compute_msm_its(clustered_trajs, lag_list, n_timescales: int, reversible_type: str, ergodic_cutoff: float, dt_ns: float) -> dict:
    # returns per-lag implied timescales in ns
    table = {}
    for lag in lag_list:
        msm = MarkovStateModel(lag_time=int(lag), n_timescales=int(n_timescales), reversible_type=reversible_type, ergodic_cutoff=float(ergodic_cutoff))
        msm.fit(clustered_trajs)
        ts = np.asarray(msm.timescales_, dtype=float)
        table[str(lag*dt_ns)] = ts * dt_ns.tolist()
    return table

def plateau_metric(its: dict, top_k: int) -> dict:
    lags = its["lag_frames"]
    arr = []
    for lag in lags:
        ts = np.asarray(its["timescales_ns"][str(lag)], dtype=float)
        arr.append(ts[:top_k])
    M = np.vstack(arr)  # shape (n_lags, top_k)
    # relative variation per timescale index across lags
    mean = np.mean(M, axis=0) + 1e-12
    rel_std = np.std(M, axis=0) / mean
    return {
        "top_k": int(top_k),
        "rel_std": rel_std.tolist(),
        "rel_std_max": float(np.max(rel_std)),
    }

def grade_run(cfg: dict, occ: dict, sparsity: dict, plat: dict) -> dict:
    g = cfg["gates"]
    min_occ = int(g["min_occupancy"])
    max_tiny_frac = float(g["max_tiny_state_frac"])
    min_out = float(g["min_avg_out_degree"])
    plateau_rel = float(g["plateau_rel_var"])

    occupancies = np.asarray(occ["occupancies"], dtype=float)
    tiny_frac = float(np.mean(occupancies < min_occ))
    rel_std_max = float(plat["rel_std_max"])
    avg_out = float(sparsity["avg_out_degree"])

    # MVP grading rules
    fail_reasons = []
    warn_reasons = []

    if avg_out < min_out:
        fail_reasons.append(f"avg_out_degree<{min_out:.1f} (sparse transitions)")
    if rel_std_max > plateau_rel * 2:
        fail_reasons.append(f"ITS not stable (rel_std_max={rel_std_max:.2f})")

    if tiny_frac > max_tiny_frac:
        warn_reasons.append(f"too many tiny states: tiny_frac={tiny_frac:.2f} (> {max_tiny_frac:.2f})")
    if rel_std_max > plateau_rel and not fail_reasons:
        warn_reasons.append(f"ITS only weakly stable (rel_std_max={rel_std_max:.2f})")

    if fail_reasons:
        label = "FAIL"
    elif warn_reasons:
        label = "WARN"
    else:
        label = "PASS"

    return {"label": label, "fail_reasons": fail_reasons, "warn_reasons": warn_reasons,
            "tiny_frac": tiny_frac, "rel_std_max": rel_std_max, "avg_out_degree": avg_out}

def suggest_fixes(cfg: dict, occ: dict, sparsity: dict, plat: dict) -> list[str]:
    g = cfg["gates"]
    min_occ = int(g["min_occupancy"])
    plateau_rel = float(g["plateau_rel_var"])

    occupancies = np.asarray(occ["occupancies"], dtype=float)
    tiny_frac = float(np.mean(occupancies < min_occ))
    rel_std_max = float(plat["rel_std_max"])
    avg_out = float(sparsity["avg_out_degree"])

    suggestions = []

    if tiny_frac > float(g["max_tiny_state_frac"]):
        suggestions.append("Reduce n_clusters (e.g., 100→50) OR increase data/stride to reduce sparsity.")
    if avg_out < float(g["min_avg_out_degree"]):
        suggestions.append("Increase MSM lag_time (use larger τ) to improve Markovianity and reduce transition noise.")
    if rel_std_max > plateau_rel:
        suggestions.append("Try increasing tICA components (4→6/8) and/or increasing MSM lag list to find plateau.")
        suggestions.append("If still unstable, consider adding contact features (dihedral-only may miss slow modes).")

    if not suggestions:
        suggestions.append("Model passes MVP gates. Next: CK test + macrostate interpretation + rates/TPT (full agent).")

    return suggestions