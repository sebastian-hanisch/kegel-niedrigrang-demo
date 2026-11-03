"""Niedriger Rang: Burer–Monteiro - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 13 der Reihe „Konvexe und konische Optimierung“: derselbe PSD-Kegel wie in Stück 12, aber ohne den Kegel. Statt der n×n-Matrix X ⪰ 0 rechnet man mit einem Faktor V (n×p), X = V Vᵀ; die Bedingung X ⪰ 0 ist dann von selbst erfüllt.
Die Demo zeigt, wie klein der Rang p sein darf, was zu kleiner Rang kostet (Fallen), wie ein Zertifikat aus V Optimalität beweist und wann die Faktorisierung schneller und kleiner ist als ADMM und der innere Punkt.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kbm_constants as C
from kbm_evaluation import (
    Settings,
    certificate_table,
    de,
    graph_name,
    main_view,
    memory_rows,
    ms,
    num,
    p_text,
    rew_rows,
    rounding_view,
    scan_view,
    sci,
    small_view,
    speed_rows,
    tol_text,
    verdict_certificate,
    verdict_main,
    verdict_memory,
    verdict_rounding,
    verdict_scan,
    verdict_small,
    verdict_speed,
)
from kbm_layout import dunkles_design, kopf_css
from kbm_pdf_export import build_pdf
from kbm_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kbm_rank import human_bytes
from kbm_visualization import build_convergence, build_lambda, build_points, build_rate, build_speed

st.set_page_config(page_title="Niedriger Rang: Burer–Monteiro – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _main(n, p, seed, rank, tol):
    return main_view(Settings(n=n, p=p, seed=seed, rank=rank, tol=tol))


@st.cache_data(show_spinner=False)
def _small(graph, rank, seed):
    return small_view(graph, rank, seed)


@st.cache_data(show_spinner=False)
def _scan(n, p, seed, tol):
    return scan_view(Settings(n=n, p=p, seed=seed, tol=tol))


@st.cache_data(show_spinner=False)
def _speed(p, seed, tol):
    return speed_rows(Settings(p=p, seed=seed, tol=tol))


@st.cache_data(show_spinner=False)
def _round(n, p, seed, rank, tol):
    return rounding_view(_main(n, p, seed, rank, tol), seed)


@st.cache_data(show_spinner=False)
def _memory():
    return memory_rows()


@st.cache_data(show_spinner=False)
def _rew(seed):
    return rew_rows(seed)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kbm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📉 Niedriger Rang: Burer–Monteiro")
st.markdown(
    """
**Man spart den Kegel, indem man die Matrix nicht aufstellt.** Das MaxCut-SDP aus Stück 12 verlangt `X ⪰ 0` mit `Xᵢᵢ = 1`: `n²` Zahlen und eine Eigenwertrechnung je Iteration. Setzt man `X = V Vᵀ` mit `V ∈ ℝⁿˣᵖ`, ist `X ⪰ 0` von selbst
erfüllt, `Xᵢᵢ = 1` heißt, dass jede Zeile von `V` Länge 1 hat, und man rechnet mit `n·p` Zahlen. Der Preis: das Programm ist **nicht mehr konvex**, und bei zu kleinem Rang `p` kann der Löser in einer **Falle** enden (bei `p = 1` ist es die Flip-Suche
der lokalen Optima). Die Demo misst, wie klein `p` sein darf, und zeigt ein **Zertifikat**, das aus dem Ergebnis `V` allein beweist, ob ein Lauf das Optimum der Relaxation erreicht hat. Dazu der Vergleich mit ADMM (Stück 9) und dem inneren Punkt (Stück 6),
die Rundung direkt aus `V` und ein Abschnitt zu Frank–Wolfe und Skizze. Stück 13 der Reihe „Konvexe und konische Optimierung“; der Strang Relaxation endet hier.

Weiter unten: **fünf Experimente** (**🔬**), der Abschnitt **📦 Frank–Wolfe und Skizze**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Die Aufgabe:** dieselbe wie in Stück 12: `max Σ wᵢⱼ (1 − Xᵢⱼ)/2` mit `X ⪰ 0`, `Xᵢᵢ = 1` für einen Zufallsgraphen `G(n, p)`.
2. **Die Faktorisierung:** `X = V Vᵀ`, die Zeilen `vᵢ` von `V` liegen auf der Einheitssphäre im `ℝᵖ`. Bei `p = 1` sind das Vorzeichen `±1` (ein Schnitt), bei `p = 2` Punkte auf einem Kreis, bei `p = 3` auf einer Kugel.
3. **Der Löser:** Blockkoordinatenabstieg („mixing method“): eine Zeile nach der anderen wird auf `vᵢ = −gᵢ/‖gᵢ‖` mit `gᵢ = Σⱼ wᵢⱼ vⱼ` gesetzt, das ist die exakte Minimierung über `vᵢ`. Keine Schrittweite, keine freien Parameter.
4. **Das Zertifikat:** am Fixpunkt ist `Z(V) = Diag(‖gᵢ‖/2) + W/2` die Dualmatrix aus Stück 12. Ist sie positiv semidefinit, ist `V Vᵀ` am exakten Fixpunkt **global optimal**: ein Beweis aus `V` allein. Im Lauf ist das numerisch zertifiziert: der Wert liegt höchstens `n·|λ_min|/2` unter dem Optimum. Ist `Z(V)` nicht positiv semidefinit, zeigt ein Eigenvektor eine Abstiegsrichtung.
5. **Der Referenzwert:** ein Lauf mit größerem Rang (`p(p+1)/2 > n`, für fast alle Kostenmatrizen reicht das), dessen Zertifikat gilt, liefert einen numerisch zertifizierten Optimalwert **ohne SDP-Löser**.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Zufallsgraph (Ergebnis, Experiment 2 bis 5)**")
    n = st.select_slider("Knoten n", options=C.NS, key="n_select", help="Der Zufallsgraph G(n, p): bis n = 30 für die Rang-Reihe (Experiment 2 und 3), bis n = 320 für den Lauf und die Zeitreihe.")
    p = st.select_slider("Kantenwahrscheinlichkeit p", options=C.P_CHOICES, key="p_select", format_func=p_text, help="Jede der n(n−1)/2 möglichen Kanten ist mit dieser Wahrscheinlichkeit vorhanden (Gewicht 1).")
    seed = st.number_input("Zufalls-Seed des Graphen", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuer Graph", width="stretch", on_click=randomize_seed)
    rank = st.select_slider("Rang p des Faktors V", options=C.RANK_CHOICES, key="rank_select", help="Spalten von V (Ergebnis): zu klein, und der Lauf kann in einer Falle enden.")
    tol = st.select_slider("Toleranz", options=C.TOL_CHOICES, key="tol_select", format_func=tol_text,
                           help="Abbruch, wenn sich der Wert in einem Sweep um höchstens Toleranz · Wert ändert. Strenger heißt mehr Sweeps; das Zertifikat braucht höchstens 10⁻⁶ (gröber trennt es Fallen und ungenaue Läufe nicht).")
    st.markdown("**Experiment 1**")
    graph1 = st.selectbox("Kleiner Graph", C.GRAPH1_CHOICES, key="graph1_select", format_func=C.GRAPH1_NAMES.get, help="Fünf bis zehn Knoten: die Punkte (Zeilen von V) sind sichtbar.")
    rank1 = st.select_slider("Rang in Experiment 1", options=C.RANK1_CHOICES, key="rank1_select", help="1: Gerade (Vorzeichen), 2: Kreis, 3: Kugel.")

sync_query_params({"n_select": int(n), "p_select": float(p), "seed_input": int(seed), "rank_select": int(rank), "tol_select": float(tol), "graph1_select": graph1, "rank1_select": int(rank1)})

settings = Settings(int(n), float(p), int(seed), int(rank), float(tol), graph1, int(rank1))
main = _main(settings.n, settings.p, settings.seed, settings.rank, settings.tol)
ref = main["ref"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("Wert des Laufs", num(main["value"]), help="Wert der Relaxation bei X = V Vᵀ nach dem Abbruch.")
m2.metric("Referenzwert", num(ref["value"]), help=f"Ein Lauf mit Rang {ref['p']} (p(p+1)/2 > n) mit Zertifikat: der numerisch zertifizierte Optimalwert der Relaxation." if ref["certified"] else f"Ein Lauf mit Rang {ref['p']}, ohne Zertifikat (bester von drei).")
m3.metric("Abstand", sci(max(main["gap"], 0.0)), help="Referenzwert minus Wert des Laufs.")
m4.metric("Zertifikat", "ja" if main["certified"] else "nein", help="Ist Z(V) positiv semidefinit (Schwelle 10·√Toleranz), ist V Vᵀ numerisch zertifiziert optimal: der Wert liegt höchstens n·|λ_min|/2 unter dem Optimum.")
m5.metric("Sweeps", str(main["sweeps"]), help=f"Zeit {ms(main['seconds'])} (hängt vom Rechner ab).")
st.plotly_chart(build_convergence(main["history"], ref["value"], f"{graph_name(settings)}, Rang {main['rank']}: Abstand zum Referenzwert"), width="stretch", key="main_chart")
st.caption("Der Abstand des Werts zum zertifizierten Referenzwert nach jedem Sweep (logarithmisch): lineare Konvergenz (jede weitere Stelle kostet etwa gleich viele Sweeps); endet der Lauf in einer Falle, bleibt der Abstand stehen.")

if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, main))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="niedriger-rang.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie klein darf der Rang sein, und woher weiß man, dass der Lauf das Optimum erreicht hat?")
verdict(verdict_main(settings, main))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ein Lauf, wenige Punkte")
small = _small(settings.graph1, settings.rank1, settings.seed)
st.caption("Die Zeilen von `V` als Punkte auf Gerade, Kreis oder Kugel (Rang 1, 2, 3) und der Verlauf des Werts. Das Fünfeck mit Rang 2 liegt auf dem Kreis im Winkel 144°; Rang 1 kann nur Vorzeichen darstellen und bleibt unter der Schranke.")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_points(small), width="stretch", key="points_chart")
with c2:
    st.plotly_chart(build_convergence(small["history"], small["sdp"], f"{C.GRAPH1_NAMES[small['graph']]}, Rang {small['rank']}: Abstand zur Schranke"), width="stretch", key="small_chart")
st.dataframe(pd.DataFrame([{"Kennzahl": "Wert des Laufs", "Wert": num(small["value"])}, {"Kennzahl": "SDP-Schranke (Stück 6)", "Wert": num(small["sdp"])}, {"Kennzahl": "exakter größter Schnitt", "Wert": num(small["exact"])},
                           {"Kennzahl": "Starts, die die Schranke erreichen (von 20)", "Wert": f"{de(100.0 * small['success_rate'], 0)} %"}, {"Kennzahl": "kleinster Eigenwert von Z(V)", "Wert": sci(small["lam_min"])}]), hide_index=True, width="stretch")
verdict(verdict_small(small, settings.rank1))

st.markdown("---")

st.subheader("🔬 Experiment 2: Wie klein darf der Rang sein?")
scan = _scan(settings.n, settings.p, settings.seed, settings.tol)
st.caption(f"{scan['graphs']} Zufallsgraphen G({scan['n']}, {p_text(settings.p)}), je Rang {scan['starts']} Zufallsstarts: Anteil der Läufe, die den zertifizierten Optimalwert erreichen, und Anteil mit Zertifikat. Die gestrichelte Linie ist der Rang, den die Theorie für fast alle Kostenmatrizen fordert (Zufallsgraphen mit Gewicht 1 fallen nicht unter diese Aussage).")
st.plotly_chart(build_rate(scan), width="stretch", key="rate_chart")
verdict(verdict_scan(scan, settings))

st.markdown("---")

st.subheader("🔬 Experiment 3: Das Zertifikat trennt Optima und Fallen")
table = certificate_table(scan)
st.caption("Dieselben Läufe: der kleinste Eigenwert von `Z(V)` je Lauf (logarithmisch, positiv aufgetragen als −λ). In dieser Stichprobe liegen Optima nahe null, Fallen weit darüber; die Schwelle liegt dazwischen und hängt an der Toleranz.")
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(build_lambda(scan, settings.tol), width="stretch", key="lambda_chart")
with c4:
    st.dataframe(pd.DataFrame([{"": "Zertifikat ja", "Optimum erreicht": table["yes_global"], "unter dem Optimum": table["yes_trap"]}, {"": "Zertifikat nein", "Optimum erreicht": table["no_global"], "unter dem Optimum": table["no_trap"]}]),
                 hide_index=True, width="stretch")
verdict(verdict_certificate(table, settings))

st.markdown("---")

st.subheader("🔬 Experiment 4: Burer–Monteiro, ADMM und innerer Punkt")
speed = _speed(settings.p, settings.seed, settings.tol)
st.caption(f"Derselbe Graph G(n, {p_text(settings.p)}) für wachsendes n: Burer–Monteiro mit Rang ⌈√(2n)⌉ und der Toleranz der Seitenleiste, ADMM in Matrixform (Stück 9; Eigenzerlegung je Iteration, nur bis n = {C.ADMM_N_MAX}) und der innere Punkt aus Stück 6 (nur bis n = {C.PD_N_MAX}).")
st.plotly_chart(build_speed(speed), width="stretch", key="speed_chart")
st.dataframe(pd.DataFrame([{"n": r["n"], "Rang": str(r["p"]), "Burer–Monteiro": ms(r["bm_seconds"]), "Sweeps": r["bm_sweeps"], "ADMM": ms(r["admm_seconds"]) if r["admm_seconds"] is not None else "–",
                            "Iterationen": str(r["admm_iterations"]) if r["admm_iterations"] is not None else "–", "innerer Punkt": ms(r["pd_seconds"]) if r["pd_seconds"] is not None else "–"} for r in speed]), hide_index=True, width="stretch")
verdict(verdict_speed(speed, settings))

st.markdown("---")

st.subheader("🔬 Experiment 5: Rundung aus V, Speicher und das Zertifikat vor dem Lauf")
rnd = _round(settings.n, settings.p, settings.seed, settings.rank, settings.tol)
mem = _memory()
rew = _rew(settings.seed)
st.caption("Die Zufallshyperebene aus Stück 12 braucht nur die Zeilen von `V` (hier der Referenz-Faktor), nicht die n×n-Matrix. Darunter der Speicher von `X` und `V`, und das Zertifikat von Rakoto Endor und Waldspurger an bipartiten Graphen.")
st.dataframe(pd.DataFrame([{"Kennzahl": "SDP-Wert (Referenz)", "Wert": num(rnd["value"])}, {"Kennzahl": "erwarteter Schnitt Σ w arccos(⟨vᵢ, vⱼ⟩)/π", "Wert": de(rnd["expected"], 4)},
                           {"Kennzahl": f"Mittel von {C.HYPER_SAMPLES} Hyperebenen", "Wert": f"{de(rnd['mean'], 4)} ± {de(rnd['stderr'], 4)}"}, {"Kennzahl": "bester Schnitt der Stichprobe", "Wert": num(rnd["best"])},
                           {"Kennzahl": "exakter größter Schnitt", "Wert": num(rnd["exact"]) if rnd["exact"] is not None else "–"}]), hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"Knoten n": f"{r['n']:,}".replace(",", "."), "Rang ⌈√(2n)⌉": str(r["p"]), "X (n×n)": human_bytes(r["x_bytes"]), "V (n×p)": human_bytes(r["v_bytes"]), "Faktor": de(r["factor"], 0)} for r in mem]), hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"bipartiter Graph": r["name"], "n": r["n"], "Kanten": r["edges"], "λ₂": de(r["lam2"], 2), "λₙ": de(r["lamn"], 2), "κ = λₙ/λ₂": de(r["kappa"], 1), "Zertifikat fordert p ≥": str(r["p_cert"]),
                           "gebraucht": str(r["p_found"]) if r["p_found"] is not None else ">4"} for r in rew]), hide_index=True, width="stretch")
verdict(verdict_rounding(rnd))
verdict(verdict_memory(mem, rew))

st.markdown("---")

st.subheader("📦 Abschnitt: Frank–Wolfe und Skizze")
st.markdown(
    """
Die Faktorisierung ist **ein** Weg, den Speicher von `n²` auf `n·p` zu senken. Ein anderer ist **Frank–Wolfe auf der Spur-Ball-Menge** (CGAL) mit einer **Skizze** der Lösung: man hält nie die ganze Matrix `X`, nur eine Zufallsprojektion ihres Bildes, und gewinnt am Ende eine
Näherung mit niedrigem Rang daraus (Yurtsever, Tropp, Fercoq, Udell, Cevher, *Scalable semidefinite programming*, SIAM Journal on Mathematics of Data Science 3 (1), 2021). Nach Satz 1.4 der Arbeit braucht das Verfahren Speicher `O(d + r n/ζ)` (`d` Nebenbedingungen, `r` Rang,
`ζ` Genauigkeit), und die Autoren geben ein MaxCut-SDP mit über 20 Millionen Knoten an, dessen Matrixvariable über 10¹⁴ Einträge hätte. Das sind Zitate aus dem gelesenen Anfang der Arbeit; **die Demo implementiert das Verfahren nicht und misst nichts daran.**

| | Speicher | Garantie | Preis |
|---|---|---|---|
| ADMM / innerer Punkt (Experiment 4) | `n²` | konvex, global | Eigenzerlegung je Iteration, `O(n³)` |
| Burer–Monteiro (diese Demo) | `n·p` | nicht konvex; Zertifikat aus `V` | Fallen bei zu kleinem `p` |
| Frank–Wolfe mit Skizze (zitiert) | `d + r n/ζ` | konvex; Näherung mit Rang `r` | langsame Konvergenz (hier nicht gemessen) |
    """
)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein kleiner Rang genügt** | Gemessen an Zufallsgraphen genügt ein Rang unter `√(2n)` (Experiment 2; bei `n` = 10 nur knapp, bei 20 und 30 etwa die Hälfte). Die Theorie von Boumal, Voroninski und Bandeira gilt für *fast alle* Kostenmatrizen (Graphen mit Gewicht 1 sind eine Nullmenge, über die sie nichts sagt); unterhalb ihrer Schwelle gibt es nach Rakoto Endor und Waldspurger (Zitat) Kostenmatrizen positiven Maßes mit nichtoptimalen kritischen Punkten. Der Graph Petersen braucht Rang 4: mit Rang 3 endet jeder Lauf unter der Schranke (Experiment 1, nur Rang 1 bis 3 wählbar). | – |
| **Das Zertifikat beweist Optimalität** | Es gilt für einen Fixpunkt: ist `Z(V)` positiv semidefinit, ist `V Vᵀ` global optimal. Bei einem noch nicht konvergierten Lauf ist `λ_min` leicht negativ; der Wert liegt dann höchstens `n·(−λ_min)/2` unter dem Optimum, und bei grober Toleranz (10⁻⁶) und dünnen größeren Graphen kann das Zertifikat auch Läufen „ja“ geben, die noch ein Stück darunter liegen. Die Schwelle `10·√Toleranz` ist gemessen (Gewicht 1, `n` = 20 und 30), bei Toleranz 10⁻⁴ und gröber trennt sie nicht mehr. Das Zertifikat braucht eine Eigenwertrechnung der `n×n`-Matrix `Z(V)`: der Speichergewinn gilt für den Lauf, nicht für die Prüfung (matrixfreie Eigenwertsuche wäre ein eigener Baustein, hier nicht gebaut). | – |
| **Der Blockkoordinatenabstieg endet im Optimum, wenn es eines gibt** | Er endet an Fixpunkten; ein Sattel ist ein Fixpunkt, den das Zertifikat verneint. Die Demo sagt deshalb „Falle oder Sattel“ und prüft keine Hesse-Matrix. | – |
| **Burer–Monteiro ist schneller** | Für diese Messung (Python, MaxCut-Typ, Zeitreihe in Experiment 4): ja, mit großem Abstand bei großen `n`, und die Zeit hängt an der Toleranz (lineare Konvergenz: jede weitere Stelle kostet weitere Sweeps). Mit kompilierten Kernen und einer anderen Genauigkeit wäre der Faktor ein anderer; Zeiten hängen vom Rechner ab und werden nicht getestet. | – |
| **Alle SDPs sind so einfach wie MaxCut** | Hier stehen nur Diagonalbedingungen `Xᵢᵢ = 1`: jede Zeile von `V` bekommt eine geschlossene Aktualisierung. Allgemeine SDPs mit `m` Gleichungen brauchen das Originalverfahren (Augmented Lagrangian, Burer und Monteiro 2003) oder Riemannsche Verfahren; nicht gebaut. | – |
| **Das Zertifikat vor dem Lauf** | Der Satz von Rakoto Endor und Waldspurger fordert eine Rang-1-Lösung `x xᵀ` (hier: bipartite, zusammenhängende Graphen) und `p > λₙ/λ₂`. Er ist hinreichend (so die Arbeit) und in dieser Messung um ein Vielfaches vorsichtig (Rang 2 genügte in allen Starts, Rang 1 je nach Graph in 67 bis 100 %); seine Schärfe (die konstruierten Gegenbeispiele der Arbeit) wurde nicht nachgebaut. | – |
| **Frank–Wolfe mit Skizze** | Nur als Abschnitt mit Zitaten; kein Code, keine Messung. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Faktorisierung.** Die MaxCut-SDP aus Stück 12 ist $\max \sum_{i<j} w_{ij}\tfrac{1 - X_{ij}}{2}$ mit $X \succeq 0$, $X_{ii} = 1$. Mit $X = V V^\top$, $V \in \mathbb{R}^{n \times p}$ und Zeilen $v_i$ wird daraus
$$\max_{\|v_i\| = 1} \; \tfrac12 \Big(\sum_{i<j} w_{ij} - \sum_{i<j} w_{ij}\,\langle v_i, v_j\rangle\Big),$$
ein Programm über dem Produkt von $n$ Sphären $S^{p-1}$. Bei $p = 1$ sind das die Vorzeichen $\pm 1$: der Schnitt selbst.

**Die Aktualisierung.** Bei festen übrigen Zeilen ist der Zielwert in $v_i$ gleich $\text{const} - \tfrac12\langle g_i, v_i\rangle$ mit $g_i = \sum_j w_{ij} v_j$. Nach Cauchy–Schwarz ist $\langle g_i, v_i\rangle \ge -\|g_i\|$ mit Gleichheit bei $v_i = -g_i/\|g_i\|$. Ein Sweep setzt alle Zeilen nacheinander so; der Wert wächst monoton. Für $p = 1$ ist das $s_i \leftarrow -\operatorname{sign}\big(\sum_j w_{ij} s_j\big)$, die Flip-Suche.

**Das Zertifikat.** Am Fixpunkt gilt $\|g_i\|\, v_i = -g_i$, also $Z V = 0$ für $Z = \mathrm{Diag}(\|g_i\|/2) + \tfrac12 W$ (zeilenweise: $\tfrac12\|g_i\| v_i + \tfrac12 g_i = 0$). Ist $Z \succeq 0$, so ist $Z$ dual zulässig (Stück 12), die Dualschranke $\tfrac12(\sum_{i<j} w_{ij} + \operatorname{tr} Z)$ ist gleich dem Wert von $V V^\top$,
und nach schwacher Dualität ist $V V^\top$ optimal für die Relaxation. Ist $Z \not\succeq 0$, zeigt ein Eigenvektor zu einem negativen Eigenwert eine Richtung, in der der Wert wächst.

**Der Rang.** Boumal, Voroninski und Bandeira (Satz 1.4): gilt $p(p+1)/2 > m$ (hier $m = n$ Nebenbedingungen $X_{ii} = 1$) und sind die $A_i V$ linear unabhängig (hier $A_i V = e_i v_i^\top$, unabhängig solange alle $v_i \ne 0$; eigene Überlegung), so ist für **fast alle** Kostenmatrizen jeder Punkt erster
und zweiter Ordnung global optimal. Für MaxCut etwa $p > \sqrt{2n}$; für alle Kostenmatrizen genügt $p > n/2$ (Abschnitt 5 der Arbeit, hier nur über die Einleitung von Rakoto Endor und Waldspurger belegt).

**Das Zertifikat vor dem Lauf (Rakoto Endor, Waldspurger, Satz 2.2).** Ist $x x^\top$ mit $x \in \{\pm 1\}^n$ eine Lösung, $C = -W$, $L = \mathrm{ddiag}(C x x^\top) - C$ positiv semidefinit mit $\lambda_2(L) > 0$ und gilt $p > \lambda_n(L)/\lambda_2(L)$, so ist jeder Punkt zweiter Ordnung global optimal.
Für zusammenhängende bipartite Graphen mit Zweifärbung $x$ ist $L = D + W$ (vorzeichenlose Laplace-Matrix).

**Speicher.** $X$: $8n^2$ Byte; $V$: $8np$ Byte.

**Literatur.** S. Burer, R. D. C. Monteiro: *A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization*, Mathematical Programming, Ser. B 95, 2003, S. 329–357 (nur das Abstract geprüft).
N. Boumal, V. Voroninski, A. S. Bandeira: *Deterministic guarantees for Burer–Monteiro factorizations of smooth semidefinite programs*, Communications on Pure and Applied Mathematics (arXiv 1804.02008; Satz 1.4 am Text geprüft).
F. Rakoto Endor, I. Waldspurger: *Benign landscape for Burer–Monteiro factorizations of MaxCut-type semidefinite programs*, arXiv 2411.03103 (Satz 2.2 und Proposition 2.3 am Text geprüft).
P.-W. Wang, W.-C. Chang, J. Z. Kolter: *The Mixing method: coordinate descent for low-rank semidefinite programming*, arXiv 1706.00476 (Abstract geprüft; die Aktualisierung ist hier selbst hergeleitet).
A. Yurtsever, J. A. Tropp, O. Fercoq, M. Udell, V. Cevher: *Scalable semidefinite programming*, SIAM Journal on Mathematics of Data Science 3 (1), 2021, S. 171–200 (Satz 1.4 am Text geprüft). Beweise der Arbeiten nicht nachgelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
