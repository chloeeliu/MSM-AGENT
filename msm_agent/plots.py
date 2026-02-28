from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_tica_density_hexbin(x, y, outpath, gridsize=120):
    plt.figure()
    hb = plt.hexbin(x, y, gridsize=gridsize, bins="log", mincnt=1)
    plt.xlabel("tIC 1")
    plt.ylabel("tIC 2")
    cb = plt.colorbar(hb)
    cb.set_label("log10(count)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def _free_energy_2d(x, y, weights=None, bins=90, eps=1e-12):
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights, density=False)
    P = H / (H.sum() + eps)
    F = -np.log(P + eps)
    F -= np.nanmin(F[np.isfinite(F)])
    return F.T, xedges, yedges

def plot_free_energy(x, y, weights, outpath, bins=90):
    F, xedges, yedges = _free_energy_2d(x, y, weights=weights, bins=bins)
    plt.figure()
    mesh = plt.pcolormesh(xedges, yedges, F, shading="auto")
    plt.xlabel("tIC 1")
    plt.ylabel("tIC 2")
    plt.colorbar(mesh, label="Free energy (arb.)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_occupancy_hist(occupancies, outpath, min_occupancy=10):
    occ = np.asarray(occupancies, dtype=float)
    plt.figure()
    plt.hist(occ, bins=50)
    plt.axvline(min_occupancy, linestyle="--")
    plt.xlabel("Microstate occupancy (#frames)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_its_curve(its: dict, outpath, top_k: int = 3):
    lags = []
    timescales = []
    for key, value in its.items():
        lags.append(int(key))
        timescales.append(np.asarray(value[:top_k], dtype=float))
    plt.figure(figsize=(12,8))
    plt.semilogy(lags, timescales, marker="o")
    plt.xlabel("Lag time (ns)")
    plt.ylabel("Implied timescales (ns)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_macro_overlay(x, y, weights, centers, macro_labels, outpath, bins=90):
    F, xedges, yedges = _free_energy_2d(x, y, weights=weights, bins=bins)
    plt.figure()
    mesh = plt.pcolormesh(xedges, yedges, F, shading="auto")
    plt.xlabel("tIC 1")
    plt.ylabel("tIC 2")
    plt.colorbar(mesh, label="Free energy (arb.)")
    plt.scatter(centers[:,0], centers[:,1], c=macro_labels, s=40, cmap="tab10", zorder=3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()