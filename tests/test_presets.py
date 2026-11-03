"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import kbm_constants as C
import kbm_evaluation as ev
import kbm_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: G(20, 0,3), Rang 4"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_choices():
    for preset in C.PRESETS.values():
        assert preset["n"] in C.NS and preset["p"] in C.P_CHOICES and preset["rank"] in C.RANK_CHOICES and preset["tol"] in C.TOL_CHOICES and preset["graph1"] in C.GRAPH1_CHOICES and preset["rank1"] in C.RANK1_CHOICES and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings() and next(iter(C.PRESETS)) == DEFAULT


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def test_the_preset_values_differ_from_the_default_where_the_help_says_so():
    d = C.PRESETS[DEFAULT]
    assert C.PRESETS["Rang 1: die Flip-Suche"]["rank"] == 1 and C.PRESETS["Rang 2: eine Falle (Seed 5)"]["rank"] == 2 and C.PRESETS["Rang 2: eine Falle (Seed 5)"]["seed"] == 5
    assert C.PRESETS["Rang nach der Theorie (p(p+1)/2 > n)"]["rank"] == 6 and C.PRESETS["Großer Graph (n = 160)"]["n"] == 160 and C.PRESETS["Großer Graph (n = 160)"]["rank"] == 8 and C.PRESETS["Strenge Toleranz (10⁻¹²)"]["tol"] == 1e-12
    assert all(C.PRESETS[name][k] == d[k] for name in C.PRESETS for k in ("p", "graph1") if name != DEFAULT)


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "30"
    at.query_params["p"] = "0.2"
    at.query_params["rank"] = "3"
    at.query_params["tol"] = "1e-06"
    at.query_params["graph1"] = "petersen"
    at.query_params["rank1"] = "7"             # kein erlaubter Wert: bleibt Standard
    at.run()
    assert not at.exception
    assert at.session_state["n_select"] == 30 and at.session_state["p_select"] == 0.2 and at.session_state["rank_select"] == 3 and at.session_state["tol_select"] == 1e-6
    assert at.session_state["graph1_select"] == "petersen" and at.session_state["rank1_select"] == C.DEFAULT_RANK1
    bad = AppTest.from_file(APP, default_timeout=300)
    bad.query_params["n"] = "31"
    bad.query_params["seed"] = "abc"
    bad.query_params["tol"] = "0.1"
    bad.run()
    assert not bad.exception and bad.session_state["n_select"] == C.DEFAULT_N and bad.sidebar.number_input(key="seed_input").value == C.DEFAULT_SEED and bad.session_state["tol_select"] == C.DEFAULT_TOL
