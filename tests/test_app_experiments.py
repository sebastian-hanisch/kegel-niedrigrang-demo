"""Pfade der App, die der Rauchtest nicht erreicht: die fünf Experimente im Seiteninhalt, PDF, jeder Rang, jeder kleine Graph, Eckfälle (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kbm_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _widget(at, key):
    if key == "seed_input":
        return at.sidebar.number_input(key=key)
    if key == "graph1_select":
        return at.sidebar.selectbox(key=key)
    return at.sidebar.select_slider(key=key)


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        _widget(at, key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Wert des Laufs"] == "40,4393" and values["Referenzwert"] == "40,4393" and values["Zertifikat"] == "ja" and "Abstand" in values
    text = _texts(at)
    for part in ("G(20, 0,3) (Seed 3, 51 Kanten), Rang p = 4: der Lauf endet nach", "Fünfeck C₅, Rang 2: die 5 Punkte liegen auf dem Kreis", "3 Zufallsgraphen G(20, 0,3) mit je 8 Zufallsstarts je Rang",
                 "144 Läufe: Zertifikat ja und Optimum erreicht", "Die Zufallshyperebene wirkt direkt auf die 20 Zeilen von V", "Bei n = 100.000 Knoten braucht X 80 GB", "Alle Zeiten sind Python auf diesem Rechner"):
        assert part in text, part


def test_the_five_experiments_the_section_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings and "📦 Abschnitt: Frank–Wolfe und Skizze" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) == 6 and not at.error


def test_the_tables_have_the_expected_rows():
    at = _app()
    frames = [d.value for d in at.dataframe]
    assert len(frames[0]) == 5 and len(frames[1]) == 2 and len(frames[2]) == len(C.SPEED_NS) and len(frames[3]) == 5 and len(frames[4]) == len(C.MEMORY_NS) and len(frames[5]) == 8
    assert list(frames[2]["Burer–Monteiro"]) and frames[2]["innerer Punkt"].tolist().count("–") == len([n for n in C.SPEED_NS if n > C.PD_N_MAX])
    assert frames[4]["Knoten n"].tolist()[-1] == "100.000" and frames[5]["bipartiter Graph"].tolist()[0] == "K₅,₅"


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("rank", C.RANK_CHOICES)
def test_every_rank_renders(rank):
    at = _app(rank_select=rank)
    assert not at.error and any("Rang p = " + str(rank) in s.value for s in list(at.success) + list(at.info) + list(at.warning))


@pytest.mark.parametrize("graph", C.GRAPH1_CHOICES)
@pytest.mark.parametrize("rank1", C.RANK1_CHOICES)
def test_every_small_graph_and_rank_renders(graph, rank1):
    at = _app(graph1_select=graph, rank1_select=rank1)
    assert not at.error and any(C.GRAPH1_NAMES[graph] + f", Rang {rank1}" in s.value for s in list(at.success) + list(at.info))


def test_the_extreme_numbers_of_nodes_and_the_tolerances_render():
    for n in (C.NS[0], C.NS[-1]):
        at = _app(n_select=n)
        assert not at.error and not at.exception
        assert {m.label: m.value for m in at.metric}["Zertifikat"] in ("ja", "nein")
    for tol in (C.TOL_CHOICES[0], C.TOL_CHOICES[-1]):
        at = _app(tol_select=tol)
        assert not at.error and not at.exception
