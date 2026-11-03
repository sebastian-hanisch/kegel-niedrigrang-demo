"""Verdikte (reine Funktionen): Werte von Hand gesetzt, Art und Text geprüft."""

import numpy as np

import kbm_constants as C
import kbm_evaluation as ev

S0 = ev.Settings()


def _main(**kw):
    ref = {"value": 40.4393, "p": 7, "certified": True}
    base = {"rank": 4, "value": 40.4393, "sweeps": 107, "gap": 1e-5, "global": True, "certified": True, "lam_min": -1.7e-5, "ref": ref, "edges": 51}
    base.update(kw)
    return base


def test_main_verdict_kinds_follow_the_two_flags():
    assert ev.verdict_main(S0, _main())[0] == "success"
    kind, text = ev.verdict_main(S0, _main(**{"global": False, "certified": False, "gap": 0.5162, "value": 48.4133}))
    assert kind == "info" and "unter dem Optimum" in text and "0,5162" in text and "**nein**" in text
    assert ev.verdict_main(S0, _main(certified=False))[0] == "warning"                         # Optimum erreicht, aber kein Zertifikat (zu grobe Toleranz)
    assert ev.verdict_main(S0, _main(**{"global": False}))[0] == "warning"                       # Zertifikat ja, aber unter dem Optimum: dürfte nie vorkommen


def test_main_verdict_names_the_reference_and_its_certificate():
    kind, text = ev.verdict_main(S0, _main())
    assert "G(20, 0,3) (Seed 3, 51 Kanten), Rang p = 4" in text and "**40,4393**" in text and "Rang 7" in text and "**ja**" in text and "107 Sweeps" in text
    text2 = ev.verdict_main(S0, _main(ref={"value": 40.4393, "p": 7, "certified": False}))[1]
    assert "ohne Zertifikat" in text2 and "ohne Zertifikat" not in text


def test_small_verdict_for_each_rank():
    c = {"graph": "fuenfeck", "n": 5, "sdp": 4.5225, "value": 4.0, "exact": 4.0, "global": False, "success_rate": 0.0}
    kind, text = ev.verdict_small(c, 1)
    assert kind == "info" and "Fünfeck C₅, Rang 1" in text and "auf der Geraden" in text and "0 %" in text and "Flip-Suche" in text and "unter der Schranke" in text
    c2 = dict(c, value=4.5225, **{"global": True}, success_rate=1.0)
    kind2, text2 = ev.verdict_small(c2, 2)
    assert kind2 == "success" and "auf dem Kreis" in text2 and "100 %" in text2 and "Flip-Suche" not in text2
    assert "auf der Kugel" in ev.verdict_small(c2, 3)[1]


def test_scan_verdict_with_and_without_a_good_rank():
    sc = {"graphs": 3, "starts": 8, "n": 20, "ranks": [1, 2, 3], "rate": {1: 0.0, 2: 0.29, 3: 1.0}, "smallest": 3, "theory_rank": 6, "sqrt_rank": 6.32}
    kind, text = ev.verdict_scan(sc, S0)
    assert kind == "info" and "p = 1: 0 %" in text and "p = 2: 29 %" in text and "Ab Rang **3**" in text and "p ≥ 6" in text and "6,3" in text and "keine Garantie" in text
    sc2 = dict(sc, smallest=None, rate={1: 0.0, 2: 0.0, 3: 0.5})
    assert "Bis Rang 3 erreicht kein Rang 95 %" in ev.verdict_scan(sc2, S0)[1]


def test_certificate_verdict_flags_a_wrong_certificate():
    t = {"yes_global": 94, "yes_trap": 0, "no_global": 9, "no_trap": 41, "total": 144, "lam_global_min": -6e-3, "lam_trap_max": -0.04}
    kind, text = ev.verdict_certificate(t, S0)
    assert kind == "success" and "144 Läufe" in text and "Falle **0**" in text and "94" in text
    assert ev.verdict_certificate(dict(t, yes_trap=2), S0)[0] == "warning"
    assert ev.verdict_certificate(dict(t, lam_trap_max=-1e-4), S0)[0] == "warning"             # die Gruppen überlappen


def test_speed_and_memory_and_rounding_verdicts():
    rows = [{"n": 10, "bm_seconds": 0.003, "bm_sweeps": 17, "admm_seconds": 0.026, "admm_iterations": 275, "pd_seconds": 0.055},
            {"n": 80, "bm_seconds": 0.2, "bm_sweeps": 138, "admm_seconds": 1.2, "admm_iterations": 898, "pd_seconds": None},
            {"n": 320, "bm_seconds": 1.1, "bm_sweeps": 122, "admm_seconds": None, "admm_iterations": None, "pd_seconds": None}]
    kind, text = ev.verdict_speed(rows, S0)
    assert kind == "info" and "n = 80: Burer–Monteiro 200 ms (138 Sweeps), ADMM 1200 ms (898 Iterationen)" in text and "6,0-mal" in text and "bei n = 320 läuft nur noch Burer–Monteiro (1100 ms)" in text
    mem = [{"n": 100000, "p": 448, "x_bytes": 80_000_000_000, "v_bytes": 358_400_000, "factor": 223.2}]
    rew = [{"kappa": 2.0, "p_cert": 3, "p_found": 1}, {"kappa": 19.0, "p_cert": 20, "p_found": 2}]
    text = ev.verdict_memory(mem, rew)[1]
    assert "100.000 Knoten" in text and "80 GB" in text and "358 MB" in text and "2,0 bis 19,0" in text and "p ≥ 3 bis 20" in text and "p = 1 bis 2" in text
    assert "Rakoto" not in ev.verdict_memory(mem, [])[1]
    r = {"value": 40.4393, "expected": 37.6011, "mean": 37.6095, "stderr": 0.0337, "best": 39.0, "exact": 39.0, "p": 7, "n": 20}
    text = ev.verdict_rounding(r)[1]
    assert "**37,6011**" in text and "37,6095 ± 0,0337" in text and "exakter Schnitt 39" in text and "Rang 7" in text
    assert "exakter Schnitt" not in ev.verdict_rounding(dict(r, exact=None))[1]


def test_formatting_helpers():
    assert ev.de(-1.5, 1) == "−1,5" and ev.num(36.0) == "36" and ev.num(40.43931) == "40,4393" and ev.sci(8.8e-6) == "8,8·10⁻⁶" and ev.sci(0.0) == "0,0"
    assert ev.ms(0.0034) == "3 ms" and ev.ms(1.5) == "1500 ms" and ev.ms(12.34) == "12,3 s" and ev.tol_text(1e-8) == "1·10⁻⁸"
    assert ev.p_text(0.3) == "0,3" and ev.graph_name(S0) == "G(20, 0,3)"


def test_every_constant_choice_is_a_valid_setting():
    for n in C.NS:
        assert ev.Settings(n=n).n == n
    assert C.DEFAULT_N in C.NS and C.DEFAULT_RANK in C.RANK_CHOICES and C.DEFAULT_TOL in C.TOL_CHOICES and C.DEFAULT_RANK1 in C.RANK1_CHOICES and max(C.TOL_CHOICES) <= 1e-6
    assert np.isclose(C.DEFAULT_TOL, 1e-8)
