"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

import kbm_admm as A
import kbm_bm as B
import kbm_constants as C
import kbm_graphs as G
import kbm_rank as K
import kbm_rounding as R
import kbm_sdp as S


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    p: float = C.DEFAULT_P
    seed: int = C.DEFAULT_SEED
    rank: int = C.DEFAULT_RANK
    tol: float = C.DEFAULT_TOL
    graph1: str = C.DEFAULT_GRAPH1
    rank1: int = C.DEFAULT_RANK1


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def p_text(p: float) -> str:
    return de(p, 1)


def num(x: float) -> str:
    """Ein Schnittwert: ganze Zahlen ohne Nachkommastellen, sonst vier."""
    return de(x, 0) if abs(x - round(x)) < 5e-5 else de(x, 4)


def ms(seconds: float) -> str:
    """Zeit in Millisekunden (unter 10 s) oder Sekunden."""
    return f"{de(seconds, 1)} s" if seconds >= 10.0 else f"{de(seconds * 1000.0, 0)} ms"


def tol_text(tol: float) -> str:
    return sci(tol, 0)


def graph_name(s: Settings) -> str:
    return f"G({s.n}, {p_text(s.p)})"


def _rng(seed: int, kind: str) -> np.random.Generator:
    return np.random.default_rng(seed + C.SEED_OFFSET[kind])


def _sub(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


# ----------------------------------------------------------------------------- Ergebnis: ein Lauf auf dem Graphen der Seitenleiste

def main_view(s: Settings) -> dict:
    """Burer–Monteiro mit Rang s.rank auf G(n, p) (Seed): der Lauf, sein Zertifikat, und der zertifizierte Referenzwert eines Laufs mit größerem Rang."""
    W = G.gnp(s.n, s.p, s.seed)
    rank = min(s.rank, s.n)
    r = B.mixing(W, rank, _rng(s.seed, "main"), tol=s.tol)
    cert = B.certificate(W, r.V)
    ref = B.reference(W, _rng(s.seed, "reference"))
    gap = ref["value"] - r.value
    return {"W": W, "rank": rank, "value": r.value, "sweeps": r.sweeps, "converged": r.converged, "seconds": r.seconds, "history": r.history, "V": r.V, "lam_min": cert["lam_min"], "bound": cert["bound"],
            "certified": B.is_certified(cert, s.tol), "ref": ref, "gap": gap, "global": gap <= B.global_gap(ref["value"]), "edges": len(G.edges(W)), "total": G.total_weight(W),
            "theory_rank": B.rank_for_theory(s.n), "sqrt_rank": math.sqrt(2.0 * s.n)}


# ----------------------------------------------------------------------------- Experiment 1: ein Lauf, wenige Punkte

def small_graph(name: str) -> np.ndarray:
    return {"fuenfeck": G.cycle(5), "k5": G.complete(5), "k6": G.complete(6), "petersen": G.petersen()}[name]


def small_view(graph: str, rank: int, seed: int) -> dict:
    """Ein kleiner Graph, Rang 1 bis 3: die Punkte auf Gerade, Kreis oder Kugel, der Verlauf des Werts, das Zertifikat; dazu die Erfolgsquote über 20 Starts."""
    W = small_graph(graph)
    n = len(W)
    with threadpool_limits(1):
        sdp = S.solve(W, 1e-9)
    r = B.mixing(W, rank, _rng(seed, "small"), tol=1e-10)
    cert = B.certificate(W, r.V)
    rng = _rng(seed + 1, "small")
    wins = 0
    for _ in range(20):
        q = B.mixing(W, rank, rng, tol=1e-10)
        wins += sdp.value - q.value <= B.global_gap(sdp.value)
    pts = np.zeros((n, 3))
    pts[:, :rank] = r.V
    return {"graph": graph, "W": W, "n": n, "rank": rank, "sdp": sdp.value, "value": r.value, "gap": sdp.value - r.value, "global": sdp.value - r.value <= B.global_gap(sdp.value), "points": pts, "history": r.history,
            "sweeps": r.sweeps, "lam_min": cert["lam_min"], "certified": B.is_certified(cert, 1e-10), "success_rate": wins / 20.0, "exact": R.exact_maxcut(W)[0]}


# ----------------------------------------------------------------------------- Experiment 2 und 3: Rang und Zertifikat

def scan_view(s: Settings) -> dict:
    """SCAN_GRAPHS Zufallsgraphen G(n', p) (n' = min(n, SCAN_N_MAX)), je Rang SCAN_STARTS Zufallsstarts: Erfolg (Wert nahe dem zertifizierten Referenzwert) und Zertifikat."""
    n = min(s.n, C.SCAN_N_MAX)
    rng = _rng(s.seed, "scan")
    ranks = [p for p in C.SCAN_RANKS if p <= n]
    runs = []                                          # (rank, lam_min, gap, global, certified)
    refs = 0
    for g in range(C.SCAN_GRAPHS):
        W = G.gnp(n, s.p, s.seed + g)
        if G.total_weight(W) == 0.0:
            continue
        ref = B.reference(W, rng)
        refs += bool(ref["certified"])
        for p in ranks:
            for _ in range(C.SCAN_STARTS):
                r = B.mixing(W, p, rng, tol=s.tol)
                c = B.certificate(W, r.V)
                gap = ref["value"] - r.value
                runs.append((p, c["lam_min"], gap, gap <= B.global_gap(ref["value"]), B.is_certified(c, s.tol)))
    arr = np.array(runs, float).reshape(-1, 5)                                       # leer, wenn alle Graphen keine Kante haben
    rate, cert_rate = {}, {}
    for p in ranks:
        m = arr[:, 0] == p
        rate[p] = float(arr[m, 3].mean()) if m.any() else 0.0
        cert_rate[p] = float(arr[m, 4].mean()) if m.any() else 0.0
    good = [p for p in ranks if rate[p] >= 0.95]
    return {"n": n, "ranks": ranks, "rate": rate, "cert_rate": cert_rate, "runs": arr, "smallest": good[0] if good else None, "graphs": C.SCAN_GRAPHS, "starts": C.SCAN_STARTS,
            "refs_certified": refs, "sqrt_rank": math.sqrt(2.0 * n), "theory_rank": B.rank_for_theory(n)}


def certificate_table(sc: dict) -> dict:
    """Zählung der Läufe: Zertifikat ja/nein gegen Erfolg ja/nein."""
    a = sc["runs"]
    glob, cert = a[:, 3] == 1.0, a[:, 4] == 1.0
    return {"yes_global": int(np.sum(cert & glob)), "yes_trap": int(np.sum(cert & ~glob)), "no_global": int(np.sum(~cert & glob)), "no_trap": int(np.sum(~cert & ~glob)), "total": len(a),
            "lam_global_min": float(a[glob, 1].min()) if glob.any() else float("nan"), "lam_trap_max": float(a[~glob, 1].max()) if (~glob).any() else float("nan")}


# ----------------------------------------------------------------------------- Experiment 4: Zeit gegen n

def speed_rows(s: Settings) -> list:
    """Je n: Burer–Monteiro (p = ⌈√(2n)⌉, Toleranz der Seitenleiste), ADMM in Matrixform (n ≤ ADMM_N_MAX) und primal-dualer innerer Punkt (n ≤ PD_N_MAX) auf demselben Graphen G(n, p)."""
    rows = []
    with threadpool_limits(1):
        for n in C.SPEED_NS:
            W = G.gnp(n, s.p, s.seed)
            p = min(n, B.sqrt_rank(n))
            r = B.mixing(W, p, _rng(s.seed, "main"), tol=s.tol)
            row = {"n": n, "p": p, "bm_seconds": r.seconds, "bm_sweeps": r.sweeps, "bm_value": r.value, "bm_converged": r.converged, "admm_seconds": None, "admm_iterations": None, "admm_value": None,
                   "pd_seconds": None, "pd_value": None}
            if n <= C.ADMM_N_MAX:
                a = A.admm(W, tol=C.ADMM_TOL)
                row.update(admm_seconds=a.seconds, admm_iterations=a.iterations, admm_value=a.value)
            if n <= C.PD_N_MAX:
                sd = S.solve(W, 1e-9)
                row.update(pd_seconds=sd.seconds, pd_value=sd.value)
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- Experiment 5: Rundung, Speicher, Zertifikat vor dem Lauf

def rounding_view(main: dict, seed: int) -> dict:
    """Die Zufallshyperebene wirkt auf die Zeilen des zertifizierten Referenz-V (n × p), ohne n×n-Matrix: Erwartungswert aus X = V Vᵀ, Mittel von HYPER_SAMPLES Hyperebenen, bester Schnitt."""
    W, V = main["W"], main["ref"]["V"]
    rng = _rng(seed, "round")
    values, _ = R.round_values(W, V, C.HYPER_SAMPLES, rng)
    exact = R.exact_maxcut(W)[0] if len(W) <= R.EXACT_MAX else None
    return {"value": main["ref"]["value"], "expected": R.expected_rounding(W, V @ V.T), "mean": float(values.mean()), "stderr": float(values.std() / math.sqrt(len(values))), "best": float(values.max()), "exact": exact,
            "p": V.shape[1], "n": len(W)}


def memory_rows() -> list:
    rows = []
    for n in C.MEMORY_NS:
        p = B.sqrt_rank(n)
        rows.append({"n": n, "p": p, "x_bytes": K.bytes_matrix(n), "v_bytes": K.bytes_factor(n, p), "factor": K.bytes_matrix(n) / K.bytes_factor(n, p)})
    return rows


REW_CASES = (("K₅,₅", "k", 5, 5, 1.0, 0), ("K₁₀,₁₀", "k", 10, 10, 1.0, 0), ("B(10, 10, ½) Seed 1", "r", 10, 10, 0.5, 1), ("B(15, 15, ½) Seed 0", "r", 15, 15, 0.5, 0), ("B(20, 20, ½) Seed 0", "r", 20, 20, 0.5, 0),
             ("B(20, 20, 0,3) Seed 0", "r", 20, 20, 0.3, 0), ("B(20, 20, 0,3) Seed 2", "r", 20, 20, 0.3, 2), ("B(30, 30, 0,2) Seed 0", "r", 30, 30, 0.2, 0))


def rew_rows(seed: int = C.DEFAULT_SEED) -> list:
    """Das Zertifikat von Rakoto Endor und Waldspurger (p > λₙ/λ₂) an bipartiten Graphen gegen den kleinsten Rang, mit dem 10 von 10 Zufallsstarts den Optimalwert (alle Kanten geschnitten) erreichen."""
    rows = []
    rng = _rng(seed, "rew")
    for name, kind, a, b, pe, sd in REW_CASES:
        W = G.bipartite(a, b) if kind == "k" else G.random_bipartite(a, b, pe, sd)
        cert = K.rew_certificate(W)
        if cert is None:
            continue
        total = G.total_weight(W)
        found = None
        for p in (1, 2, 3, 4):
            wins = sum(total - B.mixing(W, p, rng, tol=1e-8).value <= B.global_gap(total) for _ in range(10))
            if wins >= 10:
                found = p
                break
        rows.append({"name": name, "n": len(W), "edges": len(G.edges(W)), "lam2": cert["lam2"], "lamn": cert["lamn"], "kappa": cert["kappa"], "p_cert": cert["p_min"], "p_found": found})
    return rows


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_main(s: Settings, m: dict) -> tuple:
    ref = m["ref"]
    head = (f"{graph_name(s)} (Seed {s.seed}, {m['edges']} Kanten), Rang p = {m['rank']}: der Lauf endet nach {m['sweeps']} Sweeps bei **{num(m['value'])}**; der zertifizierte Referenzwert (Rang {ref['p']}) ist "
            f"**{num(ref['value'])}**" + ("" if ref["certified"] else " (ohne Zertifikat: der beste von drei Läufen)") + ". ")
    if m["global"]:
        tail = f"Der Lauf hat den Optimalwert erreicht (Abstand {sci(max(m['gap'], 0.0))}). "
    else:
        tail = f"Der Lauf endet {de(m['gap'], 4)} **unter dem Optimum**: eine Falle, ein für diesen Graphen zu kleiner Rang oder (bei grober Toleranz) noch nicht konvergiert. "
    cert = (f"Zertifikat aus V: kleinster Eigenwert von Z(V) {sci(m['lam_min'])}, " + ("**ja**, V Vᵀ ist global optimal" if m["certified"] else "**nein** (Falle, Sattel oder zu grobe Toleranz)") + ".")
    ok = m["global"] and m["certified"]
    kind = "success" if ok else ("warning" if m["certified"] != m["global"] else "info")
    return kind, head + tail + cert


def verdict_small(c: dict, rank: int) -> tuple:
    name = C.GRAPH1_NAMES[c["graph"]]
    shape = {1: "auf der Geraden (Vorzeichen ±1)", 2: "auf dem Kreis", 3: "auf der Kugel"}[rank]
    text = (f"{name}, Rang {rank}: die {c['n']} Punkte liegen {shape}; der Lauf endet bei {num(c['value'])}, die SDP-Schranke ist {num(c['sdp'])}, der exakte größte Schnitt {num(c['exact'])}. "
            + ("Der Lauf hat die Schranke erreicht" if c["global"] else "Der Lauf endet unter der Schranke (Falle, oder der Rang genügt für diesen Graphen nicht)") + f"; über 20 Zufallsstarts erreichen **{de(100.0 * c['success_rate'], 0)} %** die Schranke."
            + (" Rang 1 kann nur Vorzeichen ±1 darstellen: das ist die Flip-Suche, ihr Optimum ist ein ganzzahliger Schnitt." if rank == 1 else ""))
    return ("success" if c["global"] else "info"), text


def verdict_scan(sc: dict, s: Settings) -> tuple:
    rates = ", ".join(f"p = {p}: {de(100.0 * sc['rate'][p], 0)} %" for p in sc["ranks"])
    sm = sc["smallest"]
    text = (f"{sc['graphs']} Zufallsgraphen G({sc['n']}, {p_text(s.p)}) mit je {sc['starts']} Zufallsstarts je Rang, Anteil der Läufe, die den zertifizierten Optimalwert erreichen: {rates}. "
            + (f"Ab Rang **{sm}** gelingen mindestens 95 % der Läufe" if sm is not None else f"Bis Rang {sc['ranks'][-1]} erreicht kein Rang 95 %")
            + f"; die Theorie fordert für fast alle Graphen p(p+1)/2 > n, also p ≥ {sc['theory_rank']} (√(2n) = {de(sc['sqrt_rank'], 1)}). Das ist eine Messung an Zufallsgraphen, keine Garantie: unterhalb der Theorie-Schwelle gibt es Fälle mit Fallen.")
    return "info", text


def verdict_certificate(t: dict, s: Settings) -> tuple:
    text = (f"{t['total']} Läufe: Zertifikat ja und Optimum erreicht {t['yes_global']}, Zertifikat ja aber Falle **{t['yes_trap']}** (dürfte nie vorkommen), Zertifikat nein und Optimum erreicht {t['no_global']} "
            f"(Toleranz {tol_text(s.tol)}: der Lauf endete zu grob für die Schwelle), Zertifikat nein und Falle {t['no_trap']}. Kleinster Eigenwert von Z(V) bei Optima {sci(t['lam_global_min'])}, größter bei Fallen "
            f"{sci(t['lam_trap_max'])}: die beiden Gruppen liegen weit auseinander.")
    return ("success" if t["yes_trap"] == 0 and t["lam_global_min"] > t["lam_trap_max"] else "warning"), text


def verdict_speed(rows: list, s: Settings) -> tuple:
    parts = []
    for r in rows:
        txt = f"n = {r['n']}: Burer–Monteiro {ms(r['bm_seconds'])} ({r['bm_sweeps']} Sweeps)"
        if r["admm_seconds"] is not None:
            txt += f", ADMM {ms(r['admm_seconds'])} ({r['admm_iterations']} Iterationen)"
        if r["pd_seconds"] is not None:
            txt += f", innerer Punkt {ms(r['pd_seconds'])}"
        parts.append(txt)
    last = [r for r in rows if r["admm_seconds"] is not None][-1]
    bigger = rows[-1]
    text = ("; ".join(parts) + f". Bei n = {last['n']} braucht ADMM {de(last['admm_seconds'] / last['bm_seconds'], 1)}-mal so lange wie Burer–Monteiro; bei n = {bigger['n']} läuft nur noch Burer–Monteiro ({ms(bigger['bm_seconds'])}). "
            f"Alle Zeiten sind Python auf diesem Rechner (Toleranz {tol_text(s.tol)}, Rang ⌈√(2n)⌉); sie hängen vom Rechner und von der Toleranz ab.")
    return "info", text


def verdict_rounding(r: dict) -> tuple:
    text = (f"Die Zufallshyperebene wirkt direkt auf die {r['n']} Zeilen von V (Rang {r['p']}), ohne die n×n-Matrix: SDP-Wert {num(r['value'])}, erwarteter Schnitt aus X = V Vᵀ **{de(r['expected'], 4)}**, Mittel von {C.HYPER_SAMPLES} Hyperebenen "
            f"{de(r['mean'], 4)} ± {de(r['stderr'], 4)}, bester Schnitt {num(r['best'])}" + (f", exakter Schnitt {num(r['exact'])}" if r["exact"] is not None else "") + ".")
    return "info", text


def verdict_memory(mem: list, rew: list) -> tuple:
    big = mem[-1]
    n_text = f"{big['n']:,}".replace(",", ".")
    text = (f"Bei n = {n_text} Knoten braucht X {K.human_bytes(big['x_bytes'])}, der Faktor V mit p = {big['p']} nur {K.human_bytes(big['v_bytes'])} (reine Arithmetik, 8 Byte je Zahl). "
            + (f"Das Zertifikat von Rakoto Endor und Waldspurger fordert an {len(rew)} bipartiten Graphen p > λₙ/λ₂ = {de(min(r['kappa'] for r in rew), 1)} bis {de(max(r['kappa'] for r in rew), 1)}, "
               f"also p ≥ {min(r['p_cert'] for r in rew)} bis {max(r['p_cert'] for r in rew)}; gebraucht haben die Läufe p = {min(r['p_found'] for r in rew if r['p_found'])} bis {max(r['p_found'] for r in rew if r['p_found'])}: hinreichend, aber vorsichtig." if rew else ""))
    return "info", text
