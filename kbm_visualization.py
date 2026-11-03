"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben (die Kugel in Experiment 1 ist 3D und drehbar)."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import kbm_bm as B
import kbm_constants as C
import kbm_graphs as G


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def log_axis(fig, values, title: str):
    """Logarithmische y-Achse mit Beschriftung nur bei Zehnerpotenzen (10⁻⁶ statt 1µ, 7e+0, 6e+0 …); der Bereich umfasst alle Werte."""
    v = np.asarray([x for x in values if x > 0], float)
    lo, hi = (math.floor(math.log10(v.min())) - 0.2, math.ceil(math.log10(v.max())) + 0.1) if len(v) else (-1.0, 1.0)
    fig.update_yaxes(title_text=title, type="log", dtick=1, exponentformat="power", showexponent="all", range=[lo, max(hi, lo + 1.0)])
    return fig


# ----------------------------------------------------------------------------- Ergebnis und Experiment 1

def build_convergence(history: np.ndarray, ref_value: float, title: str = "") -> go.Figure:
    """Abstand des Werts zum zertifizierten Referenzwert gegen die Sweeps (logarithmisch; Abstände unter 10⁻¹² werden bei 10⁻¹² gezeichnet)."""
    gap = np.maximum(ref_value - np.asarray(history, float), 1e-12)
    fig = go.Figure(go.Scatter(x=list(range(len(gap))), y=gap, mode="lines", name="Abstand zum Referenzwert", line=dict(color=C.COLORS["bm"], width=3), hovertemplate="Sweep %{x}: %{y:.2e}<extra></extra>"))
    fig.update_xaxes(title_text="Sweeps")
    log_axis(fig, gap, "Referenzwert − Wert")
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return _base(fig)


def _edge_lines(W: np.ndarray, P: np.ndarray):
    xs, ys, zs = [], [], []
    for i, j, _ in G.edges(W):
        xs.extend([P[i, 0], P[j, 0], None])
        ys.extend([P[i, 1], P[j, 1], None])
        zs.extend([P[i, 2], P[j, 2], None])
    return xs, ys, zs


def build_points(c: dict) -> go.Figure:
    """Die Zeilen von V als Punkte: Rang 1 auf der Geraden (±1), Rang 2 auf dem Einheitskreis, Rang 3 auf der Einheitskugel; die Kanten des Graphen grau."""
    P, W, rank = c["points"], c["W"], c["rank"]
    labels = [str(i) for i in range(len(P))]
    if rank == 3:
        fig = go.Figure()
        u, v = np.meshgrid(np.linspace(0, 2 * math.pi, 30), np.linspace(0, math.pi, 15))
        fig.add_trace(go.Surface(x=np.cos(u) * np.sin(v), y=np.sin(u) * np.sin(v), z=np.cos(v), opacity=0.12, showscale=False, hoverinfo="skip", colorscale=[[0, "#bbbbbb"], [1, "#bbbbbb"]]))
        xs, ys, zs = _edge_lines(W, P)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#aaaaaa", width=2), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text", text=labels, textposition="top center", marker=dict(size=6, color=C.COLORS["bm"]), name="Zeilen von V",
                                   hovertemplate="Knoten %{text}<extra></extra>"))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="cube"), showlegend=False)
        return fig
    fig = go.Figure()
    if rank == 2:
        t = np.linspace(0, 2 * math.pi, 200)
        fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#dddddd", width=1), hoverinfo="skip", showlegend=False))
        xs, ys, _ = _edge_lines(W, P)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#aaaaaa", width=1.5), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers+text", text=labels, textposition="top center", marker=dict(size=13, color=C.COLORS["bm"]), name="Zeilen von V", hovertemplate="Knoten %{text}<extra></extra>"))
        fig.update_xaxes(range=[-1.4, 1.4], visible=False)
        fig.update_yaxes(range=[-1.4, 1.4], visible=False, scaleanchor="x", scaleratio=1)
    else:
        fig.add_trace(go.Scatter(x=[-1.3, 1.3], y=[0, 0], mode="lines", line=dict(color="#dddddd", width=2), hoverinfo="skip", showlegend=False))
        side = P[:, 0] >= 0
        for flag, color, name in ((True, C.COLORS["bm"], "Seite +1"), (False, "#b08a2e", "Seite −1")):
            idx = [i for i in range(len(P)) if bool(side[i]) == flag]
            fig.add_trace(go.Scatter(x=[P[i, 0] for i in idx], y=[0.02 * (k % 5 - 2) for k, i in enumerate(idx)], mode="markers+text", text=[str(i) for i in idx], textposition="top center", marker=dict(size=14, color=color),
                                     name=name, hovertemplate="Knoten %{text}<extra></extra>"))
        fig.update_xaxes(range=[-1.4, 1.4], visible=False)
        fig.update_yaxes(range=[-0.5, 0.5], visible=False)
    return _base(fig, height=360)


# ----------------------------------------------------------------------------- Experiment 2 und 3

def build_rate(sc: dict) -> go.Figure:
    ranks = sc["ranks"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ranks, y=[100.0 * sc["rate"][p] for p in ranks], name="erreichen den Optimalwert", marker_color=C.COLORS["bm"], hovertemplate="p = %{x}: %{y:.0f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=ranks, y=[100.0 * sc["cert_rate"][p] for p in ranks], mode="lines+markers", name="mit Zertifikat", line=dict(color="#b08a2e", width=3), hovertemplate="p = %{x}: %{y:.0f} %<extra></extra>"))
    th = sc["theory_rank"]
    if ranks[0] <= th <= ranks[-1]:
        fig.add_vline(x=th, line=dict(color=C.COLORS["theory"], dash="dot"), annotation_text="p(p+1)/2 > n", annotation_position="top left")
    fig.update_xaxes(title_text="Rang p", tickvals=ranks)
    fig.update_yaxes(title_text="Anteil der Läufe (%)", range=[0, 105])
    return _base(fig)


def build_lambda(sc: dict, tol: float) -> go.Figure:
    a = sc["runs"]
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for flag, color, name in ((1.0, C.COLORS["global"], "Optimum erreicht"), (0.0, C.COLORS["trap"], "unter dem Optimum")):
        m = a[:, 3] == flag
        fig.add_trace(go.Scatter(x=a[m, 0] + rng.uniform(-0.25, 0.25, int(m.sum())), y=np.maximum(-a[m, 1], 1e-12), mode="markers", name=name, marker=dict(size=8, color=color, opacity=0.7),
                                 hovertemplate="p = %{x:.0f}: −λ = %{y:.2e}<extra></extra>"))
    fig.add_hline(y=B.cert_threshold(tol), line=dict(color=C.COLORS["theory"], dash="dash"), annotation_text="Schwelle", annotation_position="top left")
    fig.update_xaxes(title_text="Rang p", tickvals=sc["ranks"])
    log_axis(fig, np.maximum(-a[:, 1], 1e-12), "−λ_min(Z(V))")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 4

def build_speed(rows: list) -> go.Figure:
    fig = go.Figure()
    for key, name, color in (("bm_seconds", "Burer–Monteiro", C.COLORS["bm"]), ("admm_seconds", "ADMM (Eigenzerlegung)", C.COLORS["admm"]), ("pd_seconds", "innerer Punkt (Stück 6)", C.COLORS["pd"])):
        pts = [(r["n"], r[key]) for r in rows if r[key] is not None]
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=name, line=dict(color=color, width=3), hovertemplate="n = %{x}: %{y:.3f} s<extra></extra>"))
    fig.update_xaxes(title_text="Knoten n", type="log", tickvals=[r["n"] for r in rows], ticktext=[str(r["n"]) for r in rows])
    log_axis(fig, [r[k] for r in rows for k in ("bm_seconds", "admm_seconds", "pd_seconds") if r[k] is not None], "Zeit (s)")
    return _base(fig)
