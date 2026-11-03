"""Rang und Speicher: was die Theorie über p sagt, was die Faktorisierung an Speicher spart, und das Zertifikat von Rakoto Endor und Waldspurger.

Theorie (Boumal, Voroninski, Bandeira): ist p(p+1)/2 > m (m Nebenbedingungen, hier m = n: Xᵢᵢ = 1), so ist für **fast alle** Kostenmatrizen jeder Punkt erster und zweiter Ordnung global optimal.
Zertifikat (Rakoto Endor, Waldspurger, Satz 2.2): hat die MaxCut-SDP eine Lösung x xᵀ mit x ∈ {±1}ⁿ, ist L = ddiag(C x xᵀ) − C PSD mit λ₂(L) > 0 (C = −W), und gilt p > λₙ(L)/λ₂(L), so ist jeder Punkt zweiter Ordnung
global optimal. Bei bipartiten, zusammenhängenden Graphen mit Zweifärbung x ist x xᵀ optimal (alle Kanten geschnitten, Wert = Σ w) und L = D + W (die vorzeichenlose Laplace-Matrix)."""

from __future__ import annotations

import numpy as np

import kbm_graphs as G

BYTES = 8                                                # je double


def bytes_matrix(n: int) -> int:
    """Speicher der n×n-Matrix X."""
    return BYTES * n * n


def bytes_factor(n: int, p: int) -> int:
    """Speicher des Faktors V ∈ ℝⁿˣᵖ."""
    return BYTES * n * p


def human_bytes(b: float) -> str:
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= size:
            return f"{b / size:.3g} {unit}".replace(".", ",")
    return f"{b:.0f} B"


def rew_certificate(W: np.ndarray):
    """Das Zertifikat für einen bipartiten, zusammenhängenden Graphen: {"x", "lam2", "lamn", "kappa", "p_min"} (p_min = kleinstes ganzes p > κ), sonst None
    (nicht bipartit, nicht zusammenhängend oder ohne Kanten: dann gilt der Satz in dieser Form nicht)."""
    n = len(W)
    if n < 2 or G.total_weight(W) == 0.0 or not G.is_connected(W):
        return None
    x = G.bipartition(W)
    if x is None:
        return None
    C = -W
    L = np.diag(np.diag(C @ np.outer(x, x))) - C
    ev = np.linalg.eigvalsh(L)
    if ev[0] < -1e-9 or ev[1] <= 1e-9:
        return None
    kappa = float(ev[-1] / ev[1])
    return {"x": x, "lam2": float(ev[1]), "lamn": float(ev[-1]), "kappa": kappa, "p_min": int(np.floor(kappa)) + 1}
