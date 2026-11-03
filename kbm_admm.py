"""ADMM für die MaxCut-SDP in Matrixform (Stück 9 auf dem PSD-Kegel dieser Aufgabe): die Gegenprobe zu Burer–Monteiro.

  min ⟨C, X⟩  mit C_ij = w_ij/2 (i ≠ j), X_ii = 1, X ⪰ 0.   Spaltung X = Z; X-Schritt: Diagonale auf 1 setzen, Z-Schritt: **Projektion auf den PSD-Kegel** (Eigenzerlegung, O(n³) je Iteration).
Das ist die Eigenzerlegung, die Burer–Monteiro vermeidet. Der allgemeine Kegel-ADMM aus Stück 9 hätte n(n+1)/2 Variablen; die Matrixform ist die gleiche Methode ohne den Umweg über svec."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import kbm_graphs as G


@dataclass(frozen=True, eq=False)
class AdmmResult:
    X: np.ndarray
    value: float              # Wert der Relaxation: ½ (Σ w − Σ_{i<j} w_ij X_ij)
    iterations: int
    converged: bool
    seconds: float


def admm(W: np.ndarray, rho: float = 1.0, tol: float = 1e-7, max_iter: int = 5000) -> AdmmResult:
    n = len(W)
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    C = np.triu(W, 1)
    C = (C + C.T) / 2.0                                   # ⟨C, X⟩ = Σ_{i<j} w_ij X_ij
    X = np.eye(n)
    Z = np.eye(n)
    U = np.zeros((n, n))
    t0 = time.perf_counter()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        X = Z - U - C / rho
        np.fill_diagonal(X, 1.0)
        Zold = Z
        M = X + U
        ev, Q = np.linalg.eigh((M + M.T) / 2.0)
        Z = (Q * np.clip(ev, 0.0, None)) @ Q.T
        U = U + X - Z
        r = np.linalg.norm(X - Z)
        d = rho * np.linalg.norm(Z - Zold)
        if r <= tol * max(1.0, np.linalg.norm(X)) and d <= tol * max(1.0, np.linalg.norm(U) * rho):
            converged = True
            break
    value = 0.5 * (G.total_weight(W) - float(np.sum(np.triu(W, 1) * X)))
    return AdmmResult(Z, value, it, converged, time.perf_counter() - t0)
