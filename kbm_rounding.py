"""Rundung der SDP-Lösung (Goemans, Williamson): eine Zufallshyperebene r trennt die Einheitsvektoren v_i; s_i = sign(v_i · r).

Die Kante {i, j} wird mit Wahrscheinlichkeit θ_ij/π geschnitten (θ_ij = arccos(X_ij) der Winkel der beiden Vektoren), der **erwartete Schnitt ist exakt** Σ_{i<j} w_ij arccos(X_ij)/π.
Mit α_GW = min_{0<θ≤π} (2/π) θ/(1 − cos θ) = 0,87856… ist er mindestens α_GW mal der SDP-Wert (jede Kante: θ/π ≥ α_GW (1 − cos θ)/2).
Dazu der exakte Schnitt durch Aufzählen aller 2^(n−1) Vorzeichen (Knoten 0 fest), vektorisiert, für n ≤ 20."""

from __future__ import annotations

import math

import numpy as np

import kbm_graphs as G
import kbm_sdp as S

EXACT_MAX = 20


def alpha_gw() -> float:
    """min_{0<θ≤π} (2/π) θ/(1 − cos θ): auf einem feinen Gitter und dann um das Minimum verfeinert."""
    th = np.linspace(1e-3, math.pi, 400001)
    f = (2.0 / math.pi) * th / (1.0 - np.cos(th))
    k = int(np.argmin(f))
    lo, hi = th[max(k - 1, 0)], th[min(k + 1, len(th) - 1)]
    th2 = np.linspace(lo, hi, 20001)
    return float(np.min((2.0 / math.pi) * th2 / (1.0 - np.cos(th2))))


ALPHA_GW = 0.878567


def expected_rounding(W: np.ndarray, X: np.ndarray) -> float:
    """E[Schnitt] = Σ_{i<j} w_ij arccos(X_ij)/π."""
    iu = np.triu_indices(len(W), 1)
    return float(np.sum(W[iu] * np.arccos(np.clip(X[iu], -1.0, 1.0))) / math.pi)


def round_values(W: np.ndarray, V: np.ndarray, count: int, rng: np.random.Generator):
    """`count` Zufallshyperebenen: die Schnittwerte und der beste Schnitt (Vorzeichenvektor)."""
    n = len(W)
    R = rng.normal(size=(V.shape[1], count))
    signs = np.where(V @ R >= 0.0, 1.0, -1.0)                                  # (n, count)
    values = 0.25 * (W.sum() - np.einsum("ik,ij,jk->k", signs, W, signs))
    k = int(np.argmax(values))
    return values, signs[:, k].copy()


def best_of_k_curve(values: np.ndarray, ks) -> list:
    """Mittelwert des Maximums über Blöcke von k der gegebenen Schnittwerte (Schätzung des Erwartungswerts des besten von k)."""
    out = []
    for k in ks:
        m = len(values) // k
        out.append(float(np.max(values[: m * k].reshape(m, k), axis=1).mean()))
    return out


def exact_maxcut(W: np.ndarray):
    """Der größte Schnitt und ein Vorzeichenvektor; Aufzählung aller 2^(n−1) Schnitte, vektorisiert (n ≤ EXACT_MAX)."""
    n = len(W)
    if n > EXACT_MAX:
        raise ValueError(f"die Aufzählung ist auf n ≤ {EXACT_MAX} begrenzt")
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    idx = np.arange(2 ** (n - 1), dtype=np.int64)
    S = np.ones((len(idx), n))
    for i in range(1, n):
        S[:, i] = 1 - 2 * ((idx >> (i - 1)) & 1)
    val = 0.25 * (W.sum() - np.einsum("ni,ij,nj->n", S, W, S))
    k = int(np.argmax(val))
    return float(val[k]), S[k].copy()
