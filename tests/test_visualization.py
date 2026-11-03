"""Abbildungen: Achsen gesperrt (2D), Spuren stimmen mit den Zahlen überein."""

import numpy as np
import pytest

import kbm_bm as B
import kbm_constants as C
import kbm_evaluation as ev
import kbm_visualization as V

S0 = ev.Settings(n=12)


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.fixture(scope="module")
def main():
    return ev.main_view(S0)


@pytest.fixture(scope="module")
def scan():
    return ev.scan_view(S0)


def test_convergence_chart_shows_the_gap_to_the_reference(main):
    fig = V.build_convergence(main["history"], main["ref"]["value"], "Titel")
    assert _locked(fig) and fig.layout.yaxis.type == "log" and fig.layout.title.text == "Titel"
    y = np.array(fig.data[0].y)
    assert len(y) == len(main["history"]) and np.all(y >= 1e-12) and y[0] == pytest.approx(main["ref"]["value"] - main["history"][0]) and np.all(np.diff(y) <= 1e-9)    # der Abstand fällt monoton


def test_point_chart_for_ranks_one_two_and_three():
    for graph, rank in (("fuenfeck", 1), ("fuenfeck", 2), ("k5", 3)):
        c = ev.small_view(graph, rank, 3)
        fig = V.build_points(c)
        n = c["n"]
        if rank == 3:
            names = [t.type for t in fig.data]
            assert names == ["surface", "scatter3d", "scatter3d"] and len(fig.data[2].x) == n
            P = np.stack([fig.data[2].x, fig.data[2].y, fig.data[2].z], axis=1)
        elif rank == 2:
            assert _locked(fig) and fig.layout.yaxis.scaleanchor == "x"
            P = np.stack([fig.data[2].x, fig.data[2].y], axis=1)
        else:
            assert _locked(fig) and sum(len(t.x) for t in fig.data[1:]) == n
            P = None
        if P is not None:
            assert np.allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-9)                       # die Punkte liegen auf Kreis bzw. Kugel
            assert np.allclose(P @ P.T, c["points"][:, :rank] @ c["points"][:, :rank].T, atol=1e-6)   # und tragen die Gram-Matrix, die gezeichnet werden soll


def test_the_pentagon_points_carry_the_sdp_solution():
    """Zurückgerechnet aus den gezeichneten Punkten: Gram-Matrix des Fünfecks mit Rang 2 hat Nachbarwinkel 144° (cos = −0,809)."""
    c = ev.small_view("fuenfeck", 2, 3)
    P = c["points"][:, :2]
    X = P @ P.T
    assert np.allclose(np.diag(X), 1.0, atol=1e-8)
    assert np.allclose([X[i, (i + 1) % 5] for i in range(5)], -(1 + 5 ** 0.5) / 4, atol=2e-3)       # cos(4π/5) = −(1 + √5)/4 = −0,809
    assert c["value"] == pytest.approx(0.5 * (5.0 - sum(X[i, (i + 1) % 5] for i in range(5))), abs=1e-6) and c["value"] == pytest.approx(4.5225, abs=5e-4)


def test_rate_chart(scan):
    fig = V.build_rate(scan)
    assert _locked(fig) and [t.name for t in fig.data] == ["erreichen den Optimalwert", "mit Zertifikat"]
    assert list(fig.data[0].x) == scan["ranks"] and list(fig.data[0].y) == pytest.approx([100.0 * scan["rate"][p] for p in scan["ranks"]]) and list(fig.data[1].y) == pytest.approx([100.0 * scan["cert_rate"][p] for p in scan["ranks"]])
    assert len(fig.layout.shapes) == (1 if scan["ranks"][0] <= scan["theory_rank"] <= scan["ranks"][-1] else 0)


def test_lambda_chart(scan):
    fig = V.build_lambda(scan, S0.tol)
    assert _locked(fig) and fig.layout.yaxis.type == "log" and [t.name for t in fig.data] == ["Optimum erreicht", "unter dem Optimum"]
    assert len(fig.data[0].x) + len(fig.data[1].x) == len(scan["runs"]) and len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y0 == pytest.approx(B.cert_threshold(S0.tol))
    assert np.median(fig.data[0].y) < fig.layout.shapes[0].y0 < np.median(fig.data[1].y)               # die Schwelle liegt zwischen den mittleren Werten der beiden Gruppen


def test_speed_chart_has_one_line_per_method_and_skips_missing_points():
    rows = [{"n": 10, "bm_seconds": 0.01, "admm_seconds": 0.05, "pd_seconds": 0.1}, {"n": 40, "bm_seconds": 0.02, "admm_seconds": 0.4, "pd_seconds": None}, {"n": 160, "bm_seconds": 0.3, "admm_seconds": None, "pd_seconds": None}]
    fig = V.build_speed(rows)
    assert _locked(fig) and [t.name for t in fig.data] == ["Burer–Monteiro", "ADMM (Eigenzerlegung)", "innerer Punkt (Stück 6)"] and [len(t.x) for t in fig.data] == [3, 2, 1]
    assert fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert V.lock_axes(V.go.Figure()).layout.xaxis.fixedrange is True and C.COLORS["bm"]
