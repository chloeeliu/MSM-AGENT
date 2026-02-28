from __future__ import annotations
import numpy as np
from pathlib import Path
from msmbuilder.decomposition import tICA
from msmbuilder.msm import MarkovStateModel
from msm_agent.ck_test import remaining_probability_from_model, remaining_probability_from_data, \
    get_data_standard_error, get_model_standard_error, evaluate_ck_pass, plot_ck_test

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

def compute_transition_sparsity(clustered_trajs, n_states: int, lagtimes: list) -> list[dict]:
    # go throgh all lagtimes in list and calculate the sparsity
    # quick proxy: count unique outgoing transitions from each state using adjacent frame
    sparsity = []
    for lagtime in lagtimes:
        out_count = {}
        for traj in clustered_trajs:
            a = np.asarray(traj).reshape(-1)
            for i in range(len(a) - lagtime):
                s = int(a[i])
                t = int(a[i+lagtime])
                if s != t:
                    if s not in out_count.keys():
                        out_count[s] = 1
                    else:
                        out_count[s] += 1
        out_count_list = list(out_count.values())
        sparsity.append({
            "lagtime": int(lagtime),
            "disconnected": n_states - int(len(out_count_list)),
            "avg_out_degree": float(np.mean(out_count_list)) if len(out_count_list) > 0 else 0.0,
            "median_out_degree": float(np.median(out_count_list)) if len(out_count_list) > 0 else 0.0,
            "min_out_degree": float(np.min(out_count_list)) if len(out_count_list) > 0 else 0.0,
            "p10_out_degree": float(np.percentile(out_count_list, 10)) if len(out_count_list) > 0 else 0.0,
        })
    return sparsity

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

def its_plateau_metric(its: dict, top_k: int, threshold: float = 0.1, last_step: int = 4) -> dict:
    lagtimes = []
    timescales = []
    for key, val in its.items():
        lagtimes.append(int(key)) # [num of lag]
        timescales.append(np.asarray(val[:top_k], dtype=float)) # [num of lag, top k]
        if len(val) < top_k:
            raise ValueError(f"ITS for lag {key} has only {len(val)} timescales, less than top_k={top_k}")
    # plateau check
    assert len(lagtimes) > last_step, f"Need at least {last_step+1} lag times for plateau check, but got {len(lagtimes)}"
    d_lag = np.diff(lagtimes) 
    d_ts = np.diff(timescales, axis=0) # [num of lag - 1, top k]
    rel_d_ts = np.abs(d_ts / (d_lag.reshape(-1, 1) + 1e-12)) # [num of lag - 1, top k]
    plateaued = rel_d_ts[-last_step:,] < threshold
    return {
        "top_k": int(top_k),    
        "last_step": int(last_step),
        "plateaued": [p.all() for p in plateaued],
    }

def ck_test(mdl, clustered_trajs, num_states: int, out_dir: Path, plot_only: bool = True, n_steps: int = 4, window: int = 1000) -> dict:
    # returns CK test results for each step up to n_steps
    # here we use the Chapman-Kolmogorov test implemented in msmbuilder, which compares the predicted transition probabilities with the observed ones at multiple lag times.
    # Note: this is a more stringent test than just checking if the implied timescales are stable, as it checks the full transition matrix.                    # Population of each state
    pop_sort = sorted(range(len(mdl.populations_)), key=lambda k: mdl.populations_[k])
    prob_model = remaining_probability_from_model(num_states, n_steps, len(mdl.state_labels_), mdl.transmat_, pop_sort)
    state_flag, prob_data = remaining_probability_from_data(num_states, n_steps, mdl.lag_time_, pop_sort, clustered_trajs)
    data_se = get_data_standard_error(num_states, n_steps, state_flag, [window]*num_states)
    plot_ck_test(num_states, n_steps, prob_data, data_se, prob_model , out_dir / "CK_test.png")
    if not plot_only:
        model_se = get_model_standard_error(num_states, n_steps, clustered_trajs, mdl)
        eval_results = evaluate_ck_pass(prob_model, prob_data, model_se, data_se)

    return eval_results

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