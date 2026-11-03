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
    in den 144 Läufen von G(20; 0,3) lagen die Optima bei λ_min ≥ −1,1·10⁻³, die Fallen bei ≤ −6,1·10⁻³ (die schwächste ist der Rang-2-Fixpunkt auf Seed 3, die meisten liegen bei ≤ −3,8·10⁻²). Bei tol ≥ 10⁻⁴ ist die Schwelle wertlos (Fallen und ungenaue Läufe überlappen): die App bietet nur tol ≤ 10⁻⁶ an."""
    return 10.0 * tol ** 0.5


def is_certified(cert: dict, tol: float) -> bool:
    return cert["lam_min"] >= -cert_threshold(tol)


GAP_FACTOR = 100.0       # erlaubter Überschuss in Einheiten Toleranz · max(1, |Wert|), siehe gap_allowance
RHO_MAX = 0.999          # Deckel für die geschätzte Konvergenzrate des Restabstiegs


def tail_estimate(history) -> float:
    """Was ein bei Toleranz tol beendeter Lauf noch gewinnen würde: geometrische Reihe aus den letzten beiden Sweeps, d_letzt · ρ/(1 − ρ) mit ρ = d_letzt/d_vorletzt (ρ höchstens RHO_MAX).
    Ohne zwei positive Zuwächse (ein Sweep, Wert ändert sich nicht mehr) ist der Rest 0. Ein Fixpunkt hat keinen Rest, ein langsam konvergierender Lauf einen großen."""
    d = np.diff(np.asarray(history, float))
    if len(d) < 2 or d[-1] <= 0.0 or d[-2] <= 0.0:
        return 0.0
    rho = min(float(d[-1] / d[-2]), RHO_MAX)
    return float(d[-1]) * rho / (1.0 - rho)


def gap_allowance(run: BmResult, tol: float) -> float:
    """Wie weit unter dem zertifizierten Referenzwert ein Lauf noch als „Optimum erreicht“ zählt: der geschätzte Rest seines Abstiegs plus GAP_FACTOR · tol · max(1, |Wert|).
    Messung: 2 786 Läufe auf G(n, p) mit Gewicht 1 (n = 10 bis 320, p = 0,1 bis 0,5, Rang 1 bis 26, tol 10⁻⁶ bis 10⁻¹²); Wahrheit: der Lauf, bei 10⁻¹⁴ bis 8 000 Sweeps fortgesetzt, erreicht den Referenzwert oder bleibt darunter.
    Der Überschuss (Abstand − Rest)/(tol · max(1, |Wert|)) ist bei tol 10⁻⁸ für 661 Läufe, die das Optimum erreichen, höchstens 37 (ohne einen auf einem Plateau stehengebliebenen Lauf), für 248 Läufe an einem Fixpunkt unter dem Optimum
    mindestens 11, bis auf zwei weitere (41 und 63; alle Rang 6 bei n = 150) mindestens 133 (n = 320, Rang 8: 326; Rang 2, Seed 3: 863); bei 10⁻¹⁰ höchstens 0 gegen mindestens 3·10⁴, bei 10⁻¹² höchstens 0 gegen mindestens 8·10⁶.
    GAP_FACTOR = 100 liegt zwischen 37 und 133. Bei tol = 10⁻⁶ überlappen die Gruppen (Optima bis 48, Fixpunkte ab 0,9): dort trennt das Kriterium nicht."""
    return tail_estimate(run.history) + GAP_FACTOR * tol * max(1.0, abs(run.value))


def reached_optimum(run: BmResult, ref_value: float, tol: float) -> bool:
    """„Optimum erreicht“: der Lauf ist konvergiert und liegt höchstens gap_allowance unter dem zertifizierten Referenzwert. Ein Fixpunkt unter dem Optimum (Rang 2, Seed 3: 3,5·10⁻⁴; n = 320, Rang 8: 0,029)
    hat keinen Rest und fällt durch. Gilt für tol ≤ 10⁻⁸; bei tol = 10⁻⁶ liegt ein Fixpunkt 3,5·10⁻⁴ unter dem Optimum noch innerhalb der Unschärfe eines grob beendeten Laufs (Rang 2, Seed 3: Überschuss 9, gute Läufe bis 48).
    Fixpunkte, die weniger als rund 100 Toleranzen unter dem Optimum liegen, erkennt es auch bei 10⁻⁸ nicht (3 von 248 Fixpunkt-Läufen); dort entscheidet das Zertifikat."""
    return bool(run.converged and ref_value - run.value <= gap_allowance(run, tol))


def rank_for_theory(n: int) -> int:
    """Kleinstes p mit p(p+1)/2 > n (Boumal, Voroninski, Bandeira: für fast alle Kostenmatrizen reicht das; die Diagonalbedingungen sind n)."""
    p = 1
    while p * (p + 1) // 2 <= n:
        p += 1
    return p


def sqrt_rank(n: int) -> int:
    """⌈√(2n)⌉: der Rang, den die Phase-0-Zeitmessung verwendet."""
    return int(np.ceil(np.sqrt(2.0 * n)))


REF_POLISH_TOL = 1e-14


def reference(W: np.ndarray, rng: np.random.Generator, tol: float = 1e-10, tries: int = 3) -> dict:
    """Ein numerisch zertifiziert optimaler Wert ohne SDP-Löser: Läufe mit p = rank_for_theory(n) + 1 (Zufallsstarts), der erste mit Zertifikat zählt. Ohne Zertifikat: der beste Lauf, certified = False.
    Ein zertifizierter Lauf wird bei Toleranz REF_POLISH_TOL weitergeführt (ohne neuen Zufallsstart): bei tol = 10⁻¹⁰ liegt der Wert zwar auf 10⁻⁶ genau, der Punkt V aber nicht
    (der erwartete Schnitt der Rundung hängt am Punkt: 37,6011 aus dem Lauf bei 10⁻¹⁰, 37,6025 aus dem polierten)."""
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
            q = mixing(W, p, rng, tol=REF_POLISH_TOL, V0=r.V)
            cq = certificate(W, q.V)
            if q.value >= r.value and is_certified(cq, tol):
                return {"value": q.value, "bound": cq["bound"], "lam_min": cq["lam_min"], "certified": True, "p": p, "sweeps": r.sweeps + q.sweeps, "V": q.V}
            return {"value": r.value, "bound": c["bound"], "lam_min": c["lam_min"], "certified": True, "p": p, "sweeps": r.sweeps, "V": r.V}
    return best
