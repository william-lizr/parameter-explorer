"""Generate sample parameter-sweep CSVs in samples/ for testing the explorer.

Each file tests a different shape of data:

  stuart_landau_sweep.csv   3 params on a full grid, timeseries + FC matrix results
  hopf_bifurcation_2d.csv   2 params, dense grid -> clean 3D surface out of the box
  ou_random_search.csv      4 params, random (not grid) samples -> scatter / PCA
  montbrio_mpr.csv          4 params, bistable regions, firing-rate timeseries
  avalanche_criticality.csv 3 params, power-law exponents, avalanche-size histogram
  disinfo_agents.csv        text (categorical) params, repeated seeds, adoption curves

The metrics are made-up functions with plausible shapes, not real simulations.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "samples"


def js(a, nd=3):
    """Store an array in one CSV cell as a JSON list."""
    return json.dumps(np.round(np.asarray(a, dtype=float), nd).tolist())


def toy_fc(n, coupling, noise):
    """A symmetric toy FC matrix: stronger coupling -> higher off-diagonal values."""
    fc = coupling + (1 - coupling) * np.eye(n) + rng.normal(0, noise, (n, n))
    return np.clip((fc + fc.T) / 2, -1, 1)


def save(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT / name, index=False)
    print(f"  {name:28s} {len(df):5d} rows × {len(df.columns)} cols")


# ── 1. Stuart-Landau whole-brain sweep (3 params, full grid) ──────────────────

def stuart_landau():
    t = np.arange(100)
    rows = []
    for K, omega, sigma in itertools.product(
            np.linspace(0.01, 0.5, 12), np.linspace(0.01, 0.15, 10), [0.01, 0.05, 0.1]):
        fc_corr = 0.5 * np.exp(-((K - 0.2)**2) / 0.02) * np.exp(-((omega - 0.08)**2) / 0.002)
        ts = np.sin(2 * np.pi * omega * t) * (1 + K) + rng.normal(0, sigma * 5, t.size)
        rows.append({
            "K": round(K, 4), "omega": round(omega, 4), "sigma": sigma,
            "fc_corr": round(float(np.clip(fc_corr + rng.normal(0, 0.03), -1, 1)), 4),
            "criticality": round(float(1 / (1 + np.exp(-(K * 8 - 2))) + rng.normal(0, 0.05)), 4),
            "mean_firing_rate": round(float(abs(omega * 100 + K * 20 + rng.normal(0, 1))), 4),
            "mean_signal": js(ts),
            "fc_matrix": js(toy_fc(8, K, sigma)),
        })
    save("stuart_landau_sweep.csv", rows)


# ── 2. Hopf normal form: bifurcation param a × global coupling G ──────────────

def hopf():
    rows = []
    for a, G in itertools.product(np.linspace(-0.2, 0.2, 25), np.linspace(0, 1.0, 25)):
        # Best FC fit sits just below the bifurcation (a slightly < 0) at medium G.
        fit = 0.55 * np.exp(-((a + 0.02)**2) / 0.004 - ((G - 0.45)**2) / 0.06)
        sync = 1 / (1 + np.exp(-(G * 10 - 4 + a * 20)))
        meta = 0.25 * np.exp(-((G - 0.4)**2) / 0.03) * (1 - abs(a) * 3)
        rows.append({
            "a": round(a, 4), "G": round(G, 4),
            "fc_fit_r": round(float(fit + rng.normal(0, 0.015)), 4),
            "synchrony": round(float(np.clip(sync + rng.normal(0, 0.02), 0, 1)), 4),
            "metastability": round(float(max(meta + rng.normal(0, 0.01), 0)), 4),
        })
    save("hopf_bifurcation_2d.csv", rows)


# ── 3. Linear OU model, random search (not a grid) ────────────────────────────

def ou_random():
    rows = []
    for _ in range(600):
        g = rng.uniform(0.05, 0.95)      # global coupling
        tau = rng.uniform(0.5, 5.0)      # time constant
        noise = rng.uniform(0.01, 0.3)
        density = rng.choice([0.1, 0.2, 0.3])  # SC threshold
        # Stability: coupling times spectral radius must stay below 1.
        lam_max = g * (1 + density * 3)
        stable = lam_max < 1
        fit = (0.5 * np.exp(-((g - 0.7)**2) / 0.05) * np.exp(-((tau - 2)**2) / 4)
               if stable else 0.0)
        rows.append({
            "g": round(g, 4), "tau": round(tau, 3), "noise": round(noise, 4), "sc_density": density,
            "fc_edge_r": round(float(fit + rng.normal(0, 0.02 + noise * 0.05)), 4),
            "max_eigenvalue": round(float(lam_max - 1), 4),
            "fc_mean": round(float(np.clip(g * 0.6 + rng.normal(0, 0.03), 0, 1)), 4),
        })
    save("ou_random_search.csv", rows)


# ── 4. Montbrió-Pazó-Roxin mean field ─────────────────────────────────────────

def montbrio():
    t = np.linspace(0, 40, 160)
    rows = []
    for eta, J, Delta, I in itertools.product(
            np.linspace(-10, 0, 11), np.linspace(5, 25, 9), [0.5, 1.0, 2.0], [0.0, 3.0]):
        drive = eta + I + J * 0.3
        # Rough regimes: low-activity, high-activity, oscillating in between.
        osc = np.exp(-((drive + 1.5)**2) / (2 * Delta)) * (J > 10)
        r_mean = 0.05 + 2 / (1 + np.exp(-drive)) / (np.pi * Delta)
        freq = 0.05 + 0.02 * J / 10
        r_ts = r_mean + osc * r_mean * np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.01, t.size)
        rows.append({
            "eta_bar": round(eta, 3), "J": round(J, 3), "Delta": Delta, "I_ext": I,
            "r_mean": round(float(r_mean + rng.normal(0, 0.01)), 4),
            "v_mean": round(float(-2 + drive * 0.15 + rng.normal(0, 0.05)), 4),
            "osc_power": round(float(osc + abs(rng.normal(0, 0.01))), 4),
            "r_timeseries": js(np.clip(r_ts, 0, None)),
        })
    save("montbrio_mpr.csv", rows)


# ── 5. Neuronal avalanches / criticality ──────────────────────────────────────

def avalanches():
    bins = np.logspace(0, 3, 25)
    rows = []
    for m, N, p_ext in itertools.product(
            np.linspace(0.8, 1.2, 17), [100, 300, 1000], [1e-4, 1e-3, 1e-2]):
        dist = abs(m - 1)
        tau = 1.5 + dist * 4 + rng.normal(0, 0.03)        # size exponent, 1.5 at criticality
        alpha = 2.0 + dist * 5 + rng.normal(0, 0.05)      # duration exponent
        kappa = 1.0 - dist * 3 + rng.normal(0, 0.03) - p_ext * 20
        # Size histogram: power law with a cutoff set by N and distance to criticality.
        counts = bins ** -tau * np.exp(-bins / (N * (1 - min(dist * 4, 0.95))))
        counts = counts / counts.sum() * 1e4 * (1 + rng.normal(0, 0.05, bins.size))
        rows.append({
            "branching_m": round(m, 4), "N_neurons": N, "p_ext": p_ext,
            "tau_size": round(tau, 4), "alpha_duration": round(alpha, 4),
            "kappa": round(float(kappa), 4),
            "scaling_err": round(float(abs((alpha - 1) / (tau - 1) - 2) + rng.normal(0, 0.02)), 4),
            "size_hist": js(np.clip(counts, 0, None), 2),
        })
    save("avalanche_criticality.csv", rows)


# ── 6. LLM agents spreading disinformation (text params, repeated seeds) ──────

def disinfo():
    models = {"llama3-8b": 1.0, "mistral-7b": 1.15, "qwen2-7b": 0.9}
    nets = {"erdos_renyi": 1.0, "small_world": 1.2, "scale_free": 1.5}
    steps = np.arange(50)
    rows = []
    for model, net, n_agents, skeptic, seed in itertools.product(
            models, nets, [50, 100, 200], [0.0, 0.1, 0.2, 0.3, 0.4], range(3)):
        rate = 0.25 * models[model] * nets[net] * (1 - skeptic * 1.8)
        final = np.clip(0.9 * (1 - np.exp(-rate * 4)) + rng.normal(0, 0.04), 0, 1)
        curve = final / (1 + np.exp(-(steps - 20 / max(rate, 0.05) ** 0.5) * rate))
        peak = int(np.argmax(np.diff(curve))) if final > 0.05 else 0
        rows.append({
            "llm": model, "network": net, "n_agents": n_agents, "skeptic_frac": skeptic, "seed": seed,
            "final_believers": round(float(final), 4),
            "time_to_peak": peak,
            "polarization": round(float(np.clip(0.5 * skeptic + final * 0.3 + rng.normal(0, 0.03), 0, 1)), 4),
            "adoption_curve": js(curve),
        })
    save("disinfo_agents.csv", rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print(f"Writing samples to {OUT}")
    for make in (stuart_landau, hopf, ou_random, montbrio, avalanches, disinfo):
        make()
