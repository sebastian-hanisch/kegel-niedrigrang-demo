"""Burer–Monteiro für die MaxCut-SDP: X = V Vᵀ mit V ∈ ℝⁿˣᵖ, Zeilen der Länge 1, statt der n×n-Matrix X ⪰ 0.

  SDP:   max Σ_{i<j} w_ij (1 − X_ij)/2,  X ⪰ 0, X_ii = 1.
  Mit X = V Vᵀ erfüllt jedes V die Bedingung X ⪰ 0 von selbst; X_ii = 1 heißt ‖v_i‖ = 1 (n Punkte auf der Einheitssphäre im ℝᵖ).
  Das Programm ist nicht mehr konvex: für zu kleines p kann der Löser in einer Falle enden (p = 1 ist die Flip-Suche der lokalen Optima).

Löser: Blockkoordinatenabstieg („mixing method“, Wang, Chang, Kolter 2017; Aktualisierung hier selbst hergeleitet). Bei festen übrigen Zeilen minimiert v_i die Größe ⟨g_i, v_i⟩
mit g_i = Σ_j w_ij v_j; nach Cauchy–Schwarz ist das Minimum bei v_i = −g_i/‖g_i‖ (g_i = 0: Zeile bleibt). Keine Schrittweite, keine freien Parameter.

Zertifikat aus V: am Fixpunkt gilt ‖g_i‖ v_i = −g_i, also Z V = 0 für Z = Diag(‖g_i‖/2) + W/2 (die Dualmatrix des SDP aus Stück 12). Ist Z ⪰ 0, so ist Z dual zulässig,
die Schranke (Σ w + tr Z)/2 ist gleich dem Wert von V Vᵀ, und V Vᵀ ist **global optimal**. Ist Z nicht PSD, zeigt ein Eigenvektor zu einem negativen Eigenwert eine Abstiegsrichtung (Falle oder Sattel)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

import kbm_graphs as G

MAX_SWEEPS = 2000


@dataclass(frozen=True, eq=False)
class BmResult:
    V: np.ndarray              # (n, p), Zeilen der Länge 1
    value: float               # Wert der Relaxation bei X = V Vᵀ: ½ Σ_{i<j} w_ij (1 − ⟨v_i, v_j⟩)
    sweeps: int
    converged: bool
    seconds: float
    history: np.ndarray        # Wert vor dem ersten Sweep und nach jedem Sweep
    n: int
    p: int


def objective(W: np.ndarray, V: np.ndarray) -> float:
    """Wert der Relaxation bei X = V Vᵀ: ½ (Σ w − Σ_{i<j} w_ij ⟨v_i, v_j⟩)."""
    return 0.5 * (G.total_weight(W) - 0.5 * float(np.sum(W * (V @ V.T))))


def random_rows(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    V = rng.normal(size=(n, p))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def mixing(W: np.ndarray, p: int, rng: np.random.Generator, tol: float = 1e-8, max_sweeps: int = MAX_SWEEPS, V0: np.ndarray | None = None) -> BmResult:
    """Blockkoordinatenabstieg aus einem Zufallsstart (oder V0). Abbruch, wenn sich der Wert in einem Sweep um höchstens tol·max(1, |Wert|) ändert."""
    n = len(W)
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    if p < 1:
        raise ValueError("der Rang p muss mindestens 1 sein")
    if V0 is None:
        V = random_rows(n, p, rng)
    else:
        V = np.array(V0, float)
        if V.shape != (n, p):
            raise ValueError("V0 hat die falsche Form")
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-300)
    nbrs = [np.flatnonzero(W[i]) for i in range(n)]
    wts = [W[i, nbrs[i]] for i in range(n)]
    t0 = time.perf_counter()
    prev = objective(W, V)
    history = [prev]
    converged = False
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        for i in range(n):
            if len(nbrs[i]) == 0:
                continue
            g = wts[i] @ V[nbrs[i]]
            ng = float(np.linalg.norm(g))
            if ng > 1e-14:
                V[i] = -g / ng
        cur = objective(W, V)
        history.append(cur)
        if abs(cur - prev) <= tol * max(1.0, abs(cur)):
            converged = True
            break
        prev = cur
    return BmResult(V, history[-1], sweeps, converged, time.perf_counter() - t0, np.array(history), n, p)


def dual_matrix(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Z(V) = Diag(‖g_i‖/2) + W/2 mit g = W V."""
    return np.diag(0.5 * np.linalg.norm(W @ V, axis=1)) + 0.5 * W


def certificate(W: np.ndarray, V: np.ndarray) -> dict:
    """Kleinster Eigenwert von Z(V) und die Schranke (Σ w + tr Z)/2. Z(V) ⪰ 0 (bis auf Rundung) ⇒ V Vᵀ ist global optimal."""
    Z = dual_matrix(W, V)
    lam = float(np.linalg.eigvalsh(Z)[0])
    return {"lam_min": lam, "bound": 0.5 * (G.total_weight(W) + float(np.trace(Z)))}


def cert_threshold(tol: float) -> float:
    """Schwelle für „Zertifikat ja“: λ_min(Z) ≥ −10·√tol. Ein global optimaler, aber bei Toleranz tol beendeter Lauf hat λ_min bis etwa −3,8·√tol (obere Hüllkurve der Messung, Gewicht 1, n = 20 und 30; typisch deutlich weniger);
    in den 144 Läufen von G(20; 0,3) lagen die Fallen bei λ_min ≤ −3,8·10⁻², die Optima bei ≥ −6,2·10⁻³. Bei tol ≥ 10⁻⁴ ist die Schwelle wertlos (Fallen und ungenaue Läufe überlappen): die App bietet nur tol ≤ 10⁻⁶ an."""
    return 10.0 * tol ** 0.5


def is_certified(cert: dict, tol: float) -> bool:
    return cert["lam_min"] >= -cert_threshold(tol)


def global_gap(value: float) -> float:
    """Wie weit unter dem zertifizierten Referenzwert ein Lauf noch als „Optimum erreicht“ zählt: max(5·10⁻³, 2·10⁻⁵·Wert). Messung (n = 20 und 30, Gewicht 1, tol ≤ 10⁻⁶): Läufe im Optimum liegen ≤ 1,5·10⁻³ darunter,
    Fallen ≥ 3·10⁻² darunter; der relative Teil gilt für große Werte. Er zählt aber auch eine echte Falle als Optimum: n = 320, Rang 8, Seed 3 endet an einem Fixpunkt 0,029 unter dem Optimum (relativ 3·10⁻⁶)."""
    return max(5e-3, 2e-5 * abs(value))


def rank_for_theory(n: int) -> int:
    """Kleinstes p mit p(p+1)/2 > n (Boumal, Voroninski, Bandeira: für fast alle Kostenmatrizen reicht das; die Diagonalbedingungen sind n)."""
    p = 1
    while p * (p + 1) // 2 <= n:
        p += 1
    return p


def sqrt_rank(n: int) -> int:
    """⌈√(2n)⌉: der Rang, den die Phase-0-Zeitmessung verwendet."""
    return int(np.ceil(np.sqrt(2.0 * n)))


def reference(W: np.ndarray, rng: np.random.Generator, tol: float = 1e-10, tries: int = 3) -> dict:
    """Ein numerisch zertifiziert optimaler Wert ohne SDP-Löser: Läufe mit p = rank_for_theory(n) + 1 (Zufallsstarts), der erste mit Zertifikat zählt. Ohne Zertifikat: der beste Lauf, certified = False."""
    n = len(W)
    p = min(n, rank_for_theory(n) + 1)
    best = None
    for _ in range(tries):
        r = mixing(W, p, rng, tol=tol)
        c = certificate(W, r.V)
        ok = is_certified(c, tol)
        if best is None or r.value > best["value"]:
            best = {"value": r.value, "bound": c["bound"], "lam_min": c["lam_min"], "certified": ok, "p": p, "sweeps": r.sweeps, "V": r.V}
        if ok:
            return {"value": r.value, "bound": c["bound"], "lam_min": c["lam_min"], "certified": True, "p": p, "sweeps": r.sweeps, "V": r.V}
    return best
