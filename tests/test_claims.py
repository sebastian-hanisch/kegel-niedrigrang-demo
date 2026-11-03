"""Jede Zahl aus README.md, den Hilfetexten der Presets und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Die Werte sind deterministisch bis auf die Rundung der BLAS-Routinen und die Abbruchtoleranz: Bänder. Zeiten, Sweepzahlen, der Status des Lösers und alle Quoten über Zufallsstarts nur in weiten Bändern
(Lehre aus Stück 9 bis 12: Zeiten und Status nie festschreiben)."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

import kbm_bm as B
import kbm_constants as C
import kbm_evaluation as ev
import kbm_graphs as G
import kbm_rank as K
import kbm_rounding as R
import kbm_sdp as S

S0 = ev.Settings()


@pytest.fixture(scope="module")
def main():
    return ev.main_view(S0)


@pytest.fixture(scope="module")
def scan():
    return ev.scan_view(S0)


def test_the_default_run(main):
    """README / Hilfe: G(20, 0,3), Seed 3, 51 Kanten, Rang 4: Wert 40,4393 (= Referenzwert des Rangs 7), Zertifikat ja, rund 100 Sweeps, Abstand unter 10⁻⁴, kleinster Eigenwert um −10⁻⁵."""
    assert main["edges"] == 51 and main["value"] == pytest.approx(40.4393, abs=5e-4) and main["ref"]["value"] == pytest.approx(40.4393, abs=5e-4) and main["ref"]["p"] == 7 and main["ref"]["certified"]
    assert main["certified"] and main["global"] and abs(main["gap"]) < 1e-4 and -1e-3 < main["lam_min"] <= 0.0 and 40 <= main["sweeps"] <= 250
    assert ev.verdict_main(S0, main)[0] == "success" and main["theory_rank"] == 6 and main["sqrt_rank"] == pytest.approx(math.sqrt(40.0))


def test_the_reference_value_is_the_sdp_value_of_the_stuck_six_solver(main):
    assert main["ref"]["value"] == pytest.approx(S.solve(main["W"]).value, abs=1e-4)


def test_preset_rank_one(main):
    """Hilfe: Rang 1 bleibt nach wenigen Sweeps bei 36 stehen (Referenzwert 40,4393), Zertifikat nein, kleinster Eigenwert −0,99; das Fünfeck mit Rang 1 verfehlt die Schranke in allen 20 Starts."""
    m = ev.main_view(ev.Settings(**C.PRESETS["Rang 1: die Flip-Suche"]))
    assert m["value"] == 36.0 and m["sweeps"] <= 10 and m["lam_min"] == pytest.approx(-0.99, abs=2e-2) and not m["certified"] and not m["global"] and m["ref"]["value"] == pytest.approx(40.4393, abs=5e-4)
    c = ev.small_view("fuenfeck", 1, 3)
    assert c["success_rate"] == 0.0 and c["value"] == 4.0 and c["sdp"] == pytest.approx(4.5225, abs=5e-5)


def test_preset_rank_two_is_a_trap_on_seed_five():
    """Hilfe: Seed 5, Rang 2: 48,4133 gegen Referenz 48,9295 (Abstand 0,5162), Zertifikat nein; mit Rang 3 Abstand unter 10⁻⁴."""
    m = ev.main_view(ev.Settings(**C.PRESETS["Rang 2: eine Falle (Seed 5)"]))
    assert m["edges"] == 66 and m["value"] == pytest.approx(48.4133, abs=5e-4) and m["ref"]["value"] == pytest.approx(48.9295, abs=5e-4) and m["gap"] == pytest.approx(0.5162, abs=2e-3) and not m["certified"] and not m["global"]
    m3 = ev.main_view(ev.Settings(seed=5, rank=3))
    assert abs(m3["gap"]) < 1e-4 and m3["certified"]


def test_preset_theory_rank_and_the_smallest_rank_on_the_default_graph():
    """Hilfe: für n = 20 fordert die Theorie p ≥ 6; der Lauf erreicht 40,4393 mit Zertifikat; gemessen genügt auf diesem Graphen Rang 3."""
    assert B.rank_for_theory(20) == 6
    m = ev.main_view(ev.Settings(**C.PRESETS["Rang nach der Theorie (p(p+1)/2 > n)"]))
    assert m["value"] == pytest.approx(40.4393, abs=5e-4) and m["certified"] and m["global"]
    m3 = ev.main_view(ev.Settings(rank=3))
    assert m3["certified"] and m3["global"]


def test_preset_large_graph():
    """Hilfe: n = 160, 3 830 Kanten, Rang 8: 2329,7279, Referenz (Rang 19) 2329,7284, Zertifikat ja; V hat 160·8 Zahlen statt 160² für X."""
    m = ev.main_view(ev.Settings(**C.PRESETS["Großer Graph (n = 160)"]))
    assert m["edges"] == 3830 and m["value"] == pytest.approx(2329.7279, abs=5e-4) and m["ref"]["value"] == pytest.approx(2329.7284, abs=2e-3) and m["ref"]["p"] == 19 and m["certified"] and m["global"]
    assert K.bytes_factor(160, 8) // 8 == 1280 and K.bytes_matrix(160) // 8 == 25600


def test_preset_strict_tolerance(main):
    """Hilfe: Toleranz 10⁻¹²: 342 Sweeps statt rund 100 (hier: mehr als doppelt so viele), Wert 40,4393, kleinster Eigenwert um −1,6·10⁻⁷."""
    m = ev.main_view(ev.Settings(**C.PRESETS["Strenge Toleranz (10⁻¹²)"]))
    assert m["sweeps"] > 2 * main["sweeps"] and 250 <= m["sweeps"] <= 450 and m["value"] == pytest.approx(40.4393, abs=5e-4) and -1e-5 < m["lam_min"] <= 1e-9 and m["gap"] < 1e-6 and m["certified"]


def test_the_small_graphs_and_the_ranks():
    """README: Fünfeck Rang 2: 4,5225 (alle 20 Starts), Rang 1: 4; K₅ Rang 1: 6 gegen 6,25, Rang 2: 6,25; K₆ schon Rang 1: 9 = Schranke; Petersen: Rang 1 12, Rang 2 12,375, Rang 3 12,4641, Rang 4 12,5 (Schranke 12,5)."""
    r = {(g, k): ev.small_view(g, k, 3) for g in C.GRAPH1_CHOICES for k in C.RANK1_CHOICES}
    assert r[("fuenfeck", 2)]["value"] == pytest.approx(4.5225, abs=5e-4) and r[("fuenfeck", 2)]["success_rate"] == 1.0 and r[("fuenfeck", 3)]["success_rate"] == 1.0
    assert r[("k5", 1)]["value"] == 6.0 and r[("k5", 1)]["sdp"] == pytest.approx(6.25, abs=1e-6) and r[("k5", 1)]["success_rate"] == 0.0 and r[("k5", 2)]["value"] == pytest.approx(6.25, abs=1e-6) and r[("k5", 2)]["success_rate"] == 1.0
    assert r[("k6", 1)]["value"] == 9.0 and r[("k6", 1)]["success_rate"] == 1.0 and r[("k6", 1)]["sdp"] == pytest.approx(9.0, abs=1e-6)
    assert r[("petersen", 1)]["value"] == 12.0 and r[("petersen", 2)]["value"] == pytest.approx(12.375, abs=1e-3) and r[("petersen", 3)]["value"] == pytest.approx(12.4641, abs=1e-3)
    assert all(r[("petersen", k)]["success_rate"] == 0.0 for k in (1, 2, 3)) and all(r[("petersen", k)]["sdp"] == pytest.approx(12.5, abs=1e-6) for k in (1, 2, 3))
    for key, c in r.items():
        assert c["exact"] == {"fuenfeck": 4.0, "k5": 6.0, "k6": 9.0, "petersen": 12.0}[key[0]]


def test_petersen_needs_rank_four():
    """App-Text und README: Petersen mit Rang 3 endet in 40 Starts nie bei der Schranke 12,5 (höchstens 12,4641); mit Rang 4 und 5 erreicht jeder Start sie."""
    W = G.petersen()
    rng = np.random.default_rng(0)
    assert max(B.mixing(W, 3, rng, tol=1e-10).value for _ in range(40)) == pytest.approx(12.4641, abs=1e-3)
    for p in (4, 5):
        assert all(12.5 - B.mixing(W, p, rng, tol=1e-10).value < 5e-3 for _ in range(20))
    assert C.RANK1_CHOICES == (1, 2, 3)


def test_the_rank_scan_on_the_default_graph(scan):
    """README: 3 Graphen G(20, 0,3), je Rang 8 Starts: Rang 1 nie, Rang 2 um 30 %, ab Rang 3 mindestens 95 %; Zertifikat dort fast immer; kleinster Rang mit 95 % ist 3 (Band 3 bis 4); die Theorie fordert p ≥ 6."""
    assert scan["n"] == 20 and scan["graphs"] == 3 and scan["starts"] == 8 and scan["rate"][1] == 0.0 and 0.05 <= scan["rate"][2] <= 0.6 and scan["rate"][3] >= 0.95 and scan["smallest"] in (3, 4)
    assert scan["theory_rank"] == 6 and all(scan["cert_rate"][p] <= scan["rate"][p] + 1e-12 for p in scan["ranks"]) and scan["cert_rate"][6] >= 0.9 and scan["refs_certified"] == 3
    assert len(scan["runs"]) == 3 * 6 * 8


def test_the_smallest_rank_grows_slowly_with_n():
    """README: kleinster Rang mit mindestens 95 % auf G(n, 0,3): n = 10 → 3, n = 20 → 3, n = 30 → 4 (Band je ±1), weit unter √(2n)."""
    got = {n: ev.scan_view(ev.Settings(n=n))["smallest"] for n in (10, 20, 30)}
    assert got[10] in (2, 3, 4) and got[20] in (3, 4) and got[30] in (3, 4, 5)
    assert all(got[n] < math.sqrt(2 * n) for n in got) and got[30] >= got[10]


def test_the_certificate_separates_optima_and_traps(scan):
    """README: 144 Läufe; Zertifikat ja und Falle: 0 (nie); die kleinsten Eigenwerte der Optima und die größten der Fallen liegen getrennt (−6·10⁻³ gegen −4·10⁻²: Band)."""
    t = ev.certificate_table(scan)
    assert t["total"] == 144 and t["yes_global"] + t["yes_trap"] + t["no_global"] + t["no_trap"] == 144 and t["yes_trap"] == 0
    assert t["yes_global"] >= 60 and t["no_trap"] >= 20 and t["lam_global_min"] > t["lam_trap_max"] and t["lam_trap_max"] < -0.01 and t["lam_global_min"] > -0.02
    assert ev.verdict_certificate(t, S0)[0] == "success"


def test_a_trap_has_a_strongly_negative_eigenvalue_over_many_graphs():
    """README: Fallen haben λ_min(Z(V)) ≤ −0,06 (n = 20 und 30, Toleranz ≤ 10⁻⁶), Optima ≥ −4·10⁻³ bei Toleranz 10⁻⁸ (hier Toleranz 10⁻¹⁰: ≥ −10⁻³)."""
    rng = np.random.default_rng(3)
    glob, trap = [], []
    for n, pe in ((20, 0.5), (30, 0.3)):
        for seed in range(2):
            W = G.gnp(n, pe, seed)
            ref = B.reference(W, rng)["value"]
            for p in (1, 2, 3, 4):
                for _ in range(4):
                    r = B.mixing(W, p, rng, tol=1e-10)
                    lam = B.certificate(W, r.V)["lam_min"]
                    (glob if ref - r.value <= B.global_gap(ref) else trap).append(lam)
    assert min(glob) > -1e-3 and max(trap) < -0.03 and len(glob) > 10 and len(trap) > 10


def test_the_time_series(main):
    """README: Burer–Monteiro, ADMM und innerer Punkt auf demselben Graphen; ab n = 40 ist Burer–Monteiro schneller als ADMM (Band: Faktor 1,5 bis 200), alle Verfahren treffen denselben Wert; ADMM nur bis 80, innerer Punkt nur bis 30."""
    rows = {r["n"]: r for r in ev.speed_rows(S0)}
    assert list(rows) == list(C.SPEED_NS)
    for n, r in rows.items():
        assert r["bm_converged"] and r["bm_sweeps"] < B.MAX_SWEEPS and r["p"] == B.sqrt_rank(n)
        assert (r["admm_seconds"] is not None) == (n <= C.ADMM_N_MAX) and (r["pd_seconds"] is not None) == (n <= C.PD_N_MAX)
        if r["admm_seconds"] is not None:
            assert r["admm_value"] == pytest.approx(r["bm_value"], rel=2e-3)
        if r["pd_seconds"] is not None:
            assert r["pd_value"] == pytest.approx(r["bm_value"], abs=5e-3 * max(1.0, r["bm_value"] / 40.0))
    for n in (40, 80):
        assert 1.5 <= rows[n]["admm_seconds"] / rows[n]["bm_seconds"] <= 200.0
    assert rows[320]["bm_seconds"] < 120.0


def test_the_tolerance_decides_the_time():
    """README: G(80, 0,2) Seed 3, Rang 13: Toleranz 10⁻⁶ / 10⁻⁸ / 10⁻¹⁰ / 10⁻¹² braucht 34 / 59 / 85 / 113 Sweeps (Band: immer mehr, letzte mehr als das 2,5-fache der ersten), der Abstand zum Wert bei Toleranz 10⁻¹⁴ fällt."""
    W = G.gnp(80, 0.2, 3)
    sweeps, dist = [], []
    adm = B.mixing(W, 13, np.random.default_rng(0), tol=1e-14, max_sweeps=4000).value
    for tol in (1e-6, 1e-8, 1e-10, 1e-12):
        r = B.mixing(W, 13, np.random.default_rng(0), tol=tol)
        sweeps.append(r.sweeps)
        dist.append(abs(adm - r.value))
    assert sweeps == sorted(sweeps) and 5 <= sweeps[0] and sweeps[-1] <= 3000 and sweeps[-1] > 2.5 * sweeps[0] and dist[0] > dist[-1] and dist[-1] < 1e-5


def test_the_rounding_acts_on_v_alone(main):
    """README / App: Rang-7-Faktor des Standardgraphen: SDP-Wert 40,4393, erwarteter Schnitt 37,6011, Mittel von 2 000 Hyperebenen innerhalb von 4 Standardfehlern, bester Schnitt = exakter Schnitt 39."""
    r = ev.rounding_view(main, S0.seed)
    assert r["value"] == pytest.approx(40.4393, abs=5e-4) and r["expected"] == pytest.approx(37.6011, abs=1e-3) and abs(r["mean"] - r["expected"]) < 4 * r["stderr"] and r["best"] == 39.0 and r["exact"] == 39.0 and r["p"] == 7 and r["n"] == 20
    assert r["expected"] >= R.ALPHA_GW * r["value"] - 1e-6 and r["expected"] == pytest.approx(R.expected_rounding(main["W"], S.solve(main["W"]).X), abs=0.05)   # README: der erwartete Schnitt hängt an der Optimallösung (37,6011 aus V gegen 37,6025 aus dem Mittelpunkt des inneren Punkts)
    assert r["expected"] != pytest.approx(R.expected_rounding(main["W"], S.solve(main["W"]).X), abs=1e-6)
    assert ev.rounding_view(ev.main_view(ev.Settings(n=30)), 3)["exact"] is None


def test_the_memory_table():
    """README: n = 100 / 1 000 / 10 000 / 100 000 mit p = ⌈√(2n)⌉ = 15 / 45 / 142 / 448: X 80 kB / 8 MB / 800 MB / 80 GB, V 12 kB / 360 kB / 11,4 MB / 358 MB; Faktor n/p."""
    rows = ev.memory_rows()
    assert [r["n"] for r in rows] == [100, 1000, 10000, 100000] and [r["p"] for r in rows] == [15, 45, 142, 448]
    assert [K.human_bytes(r["x_bytes"]) for r in rows] == ["80 kB", "8 MB", "800 MB", "80 GB"] and [K.human_bytes(r["v_bytes"]) for r in rows] == ["12 kB", "360 kB", "11,4 MB", "358 MB"]
    assert all(r["factor"] == pytest.approx(r["n"] / r["p"]) for r in rows)


def test_the_certificate_before_the_run_is_sufficient_but_cautious():
    """README: acht bipartite Graphen: κ = 2 (K_{a,a}) bis 19; das Zertifikat fordert p ≥ 3 bis 20; gebraucht haben 10 von 10 Starts Rang 1 oder 2; Verhältnis gefordert/gebraucht 1,5 bis 10."""
    rows = ev.rew_rows()
    assert len(rows) == 8 and rows[0]["name"] == "K₅,₅" and rows[0]["kappa"] == pytest.approx(2.0) and rows[1]["kappa"] == pytest.approx(2.0)
    assert min(r["kappa"] for r in rows) == pytest.approx(2.0) and 18.0 <= max(r["kappa"] for r in rows) <= 20.0
    assert [r["p_cert"] for r in rows if r["name"].startswith("K")] == [3, 3] and all(r["p_found"] in (1, 2) for r in rows)
    ratios = [r["p_cert"] / r["p_found"] for r in rows]
    assert 1.5 <= min(ratios) and max(ratios) <= 10.5 and all(r["p_cert"] > r["kappa"] for r in rows)
    bm = lambda W, p: B.mixing(W, p, np.random.default_rng(1), tol=1e-10).value                       # noqa: E731
    W = G.random_bipartite(15, 15, 0.5, 0)
    assert G.total_weight(W) - bm(W, K.rew_certificate(W)["p_min"]) < 1e-4


def test_k_n_needs_only_rank_two():
    """README: K_n mit p = 2: n²/4 (regelmäßiges Vieleck), mit p = 1 für ungerade n nur (n² − 1)/4."""
    rng = np.random.default_rng(0)
    for n in (5, 6, 7, 9):
        W = G.complete(n)
        assert max(B.mixing(W, 2, rng, tol=1e-12).value for _ in range(5)) == pytest.approx(n * n / 4.0, abs=1e-6)
        one = max(B.mixing(W, 1, rng, tol=1e-12).value for _ in range(8))
        assert one == pytest.approx(n * n / 4.0 if n % 2 == 0 else (n * n - 1) / 4.0, abs=1e-6)


def test_the_quoted_figures_of_the_cited_papers_are_not_recomputed_here():
    """Die Zitate (Speicher O(d + r n/ζ), über 20 Millionen Knoten, 10¹⁴ Einträge) stehen in README und App als Zitat mit Quelle; sie werden nicht gemessen."""
    root = Path(__file__).resolve().parent.parent
    app = (root / "app.py").read_text(encoding="utf-8")
    assert "d + r n/ζ" in app and "über 20 Millionen Knoten" in app and "hier nicht gemessen" in app and "die Demo implementiert das Verfahren nicht und misst nichts daran" in app


def test_the_kernel_is_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2 bis 12: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-maxcut-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()


def test_rank_two_on_the_default_seed_is_near_the_optimum_without_a_certificate():
    """README: Rang 2, Seed 3: Wert 40,4390 (3·10⁻⁴ unter dem Optimum), kleinster Eigenwert um −6·10⁻³, kein Zertifikat; das Verdikt ist eine Warnung."""
    s = ev.Settings(rank=2)
    m = ev.main_view(s)
    assert m["value"] == pytest.approx(40.4390, abs=5e-4) and 1e-5 < m["gap"] < 2e-3 and m["global"] and not m["certified"] and -2e-2 < m["lam_min"] < -1e-3
    kind, text = ev.verdict_main(s, m)
    assert kind == "warning" and "**nein**" in text and "erreicht" in text


def test_the_certificate_threshold_follows_the_tolerance():
    """README: Optima haben λ_min ≥ −10·√tol für tol = 10⁻⁶, 10⁻⁸, 10⁻¹⁰ (hier Stichprobe n = 20), Fallen liegen bei Toleranz 10⁻⁶ unter −0,03."""
    rng = np.random.default_rng(11)
    W = G.gnp(20, 0.5, 2)
    ref = B.reference(W, rng)["value"]
    for tol in (1e-6, 1e-8, 1e-10):
        lams = []
        traps = []
        for p in (1, 2, 4, 6):
            for _ in range(4):
                r = B.mixing(W, p, rng, tol=tol)
                lam = B.certificate(W, r.V)["lam_min"]
                (lams if ref - r.value <= B.global_gap(ref) else traps).append(lam)
        assert lams and min(lams) >= -B.cert_threshold(tol) * 1.5, tol
        assert all(l < -0.03 for l in traps) and traps


def test_a_large_graph_ends_within_the_relative_gap_but_beyond_the_fixed_one():
    """README: n = 320, Rang 8, Seed 3: der Abstand zum Referenzwert ist rund 0,03 (relativ 3·10⁻⁶): größer als die feste Schwelle 5·10⁻³, aber ein Rest des Schwanzes, keine Falle."""
    m = ev.main_view(ev.Settings(n=320, rank=8))
    assert m["gap"] > 5e-3 and m["gap"] / m["ref"]["value"] < 1e-5 and m["global"]


def test_the_time_series_favours_burer_monteiro_already_at_small_n():
    rows = {r["n"]: r for r in ev.speed_rows(S0)}
    for n in (10, 20, 40, 80):
        assert rows[n]["admm_seconds"] > 1.2 * rows[n]["bm_seconds"], n


def test_more_readme_numbers_of_the_rounding_and_the_large_graph(main):
    r = ev.rounding_view(main, S0.seed)
    assert r["mean"] == pytest.approx(37.6095, abs=0.03) and r["stderr"] == pytest.approx(0.0337, abs=0.005)
    m = ev.main_view(ev.Settings(**C.PRESETS["Großer Graph (n = 160)"]))
    assert -1e-3 < m["lam_min"] < 0.0 and m["sweeps"] < 400


def test_the_random_bipartite_certificates_in_the_readme():
    rows = ev.rew_rows()
    random_rows = [r for r in rows if not r["name"].startswith("K")]
    assert min(r["kappa"] for r in random_rows) == pytest.approx(5.0, abs=0.1) and max(r["kappa"] for r in random_rows) == pytest.approx(19.0, abs=0.1)
    assert min(r["p_cert"] for r in random_rows) == 6 and max(r["p_cert"] for r in random_rows) == 20 and min(r["n"] for r in random_rows) == 20 and max(r["n"] for r in random_rows) == 60
