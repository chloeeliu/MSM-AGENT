# MSM Agent MVP Report
**Grade:** `WARN`
## Warnings
- ITS only weakly stable (rel_std_max=0.53)
## Data summary
- trajectories: 28
- total_time_ns (approx): 14000.00
- dt_ns_effective: 0.5000
## Key metrics
- tiny_frac: 0.000
- avg_out_degree: 7.63
- ITS rel_std_max(top-k): 0.534
## Suggestions
- Try increasing tICA components (4→6/8) and/or increasing MSM lag list to find plateau.
- If still unstable, consider adding contact features (dihedral-only may miss slow modes).
## Figures
### tica_density
![tica_density](figs/tica_density_hexbin.png)
### free_energy
![free_energy](figs/free_energy.png)
### occupancy
![occupancy](figs/occupancy_hist.png)
### its
![its](figs/its_curve.png)
### macro
![macro](figs/macro_overlay.png)
