"""Die SDP-Relaxation von MaxCut (Goemans, Williamson 1995) als Kegelprogramm des Kerns.

  MaxCut:   max Σ_{i<j} w_ij (1 − s_i s_j)/2,  s ∈ {±1}ⁿ.
  Relaxation: s_i s_j → X_ij mit X ⪰ 0, X_ii = 1 (X ist die Gram-Matrix von Einheitsvektoren v_i):   max Σ_{i<j} w_ij (1 − X_ij)/2.
Im Format des Kerns:  min cᵀx  mit  s = b − A x ∈ PSD,  x = die Außerdiagonale (x_ij = X_ij, i < j), c_ij = w_ij;  s = svec(X) = b − A x mit b = svec(I) und A_{pos(ij), ij} = −√2.
SDP-Wert (obere Schranke des Schnitts) = (Σ w − min cᵀx)/2.
Dual: Z = smat(z) ⪰ 0 mit Z_ij = w_ij/2 außerhalb der Diagonale (Aᵀz = −c); die Schranke (Σ w + tr Z)/2 ist ein **bewiesener** oberer Wert für den Schnitt, ohne ihn zu kennen.
Der Löser ist der primal-duale innere Punkt aus Stück 6; er liefert den Mittelpunkt der Menge optimaler Lösungen (nicht die Ecke mit kleinstem Rang)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import kegel_kern as K
import kbm_graphs as G
import kbm_solver as PD


@dataclass(frozen=True, eq=False)
class SdpResult:
    X: np.ndarray                # Gram-Matrix der Einheitsvektoren
    Z: np.ndarray                # PSD-Dualvariable
    value: float                 # SDP-Wert: obere Schranke des Schnitts
    dual_bound: float            # (Σ w + tr Z)/2
    status: str
    iterations: int
    seconds: float
    eig: np.ndarray              # Eigenwerte von X, absteigend
    z_min_eig: float             # kleinster Eigenwert von Z (≥ 0 bis auf Rundung)
    z_offdiag: float             # größte Abweichung der Außerdiagonale von Z von w_ij/2
    n: int

    @property
    def rank(self) -> int:
        return int(np.sum(self.eig > 1e-6))

    @property
    def gap(self) -> float:
        return self.dual_bound - self.value


def build(W: np.ndarray):
    """Das Kegelprogramm und die Paare (i, j), i < j, in der Reihenfolge der Variablen."""
    n = len(W)
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    iu0, iu1 = np.triu_indices(n)
    pos = {(int(a), int(b)): p for p, (a, b) in enumerate(zip(iu0, iu1))}
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    A = np.zeros((len(iu0), len(pairs)))
    b = np.zeros(len(iu0))
    for col, (i, j) in enumerate(pairs):
        A[pos[(i, j)], col] = -K.SQ2
    for i in range(n):
        b[pos[(i, i)]] = 1.0
    c = np.array([W[i, j] for i, j in pairs], float)
    return K.ConeProblem(c, A, b, (("s", n),), np.zeros(len(pairs))), pairs


def solve(W: np.ndarray, tol: float = 1e-9) -> SdpResult:
    n = len(W)
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    if not np.allclose(W, W.T) or np.any(np.diag(W) != 0.0) or np.any(W < 0.0):
        raise ValueError("W muss symmetrisch sein, Null auf der Diagonale und nichtnegative Gewichte haben")
    total = G.total_weight(W)
    if total == 0.0:                                                             # keine Kanten: X = I, der Schnitt ist 0
        return SdpResult(np.eye(n), np.zeros((n, n)), 0.0, 0.0, "optimal", 0, 0.0, np.ones(n), 0.0, 0.0, n)
    prob, pairs = build(W)
    t0 = time.perf_counter()
    res = PD.solve_pd(prob, tol=tol, max_iter=100)
    sec = time.perf_counter() - t0
    X = np.eye(n)
    for (i, j), v in zip(pairs, res.x):
        X[i, j] = X[j, i] = v
    Z = K.smat(res.z, n)
    off = Z - np.diag(np.diag(Z))
    dev = float(np.abs(off - np.triu(W, 1) / 2.0 - np.tril(W, -1) / 2.0).max())
    return SdpResult(X, Z, (total - res.value) / 2.0, (total + float(np.trace(Z))) / 2.0, res.status, res.iterations, sec, np.sort(np.linalg.eigvalsh(X))[::-1], float(np.linalg.eigvalsh(Z).min()), dev, n)


def vectors(X: np.ndarray) -> np.ndarray:
    """Zerlegung X = V Vᵀ (Eigenzerlegung, negative Eigenwerte auf 0): die Zeilen von V sind die Einheitsvektoren v_i (bis auf Rundung)."""
    ev, Q = np.linalg.eigh(X)
    return Q * np.sqrt(np.clip(ev, 0.0, None))
