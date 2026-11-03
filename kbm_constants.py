"""Konstanten: Regler-Grenzen, feste Reihen der Experimente, Farben und Presets."""

NS = (10, 20, 30, 40, 60, 80, 120, 160, 240, 320)         # Knotenzahlen des Reglers
DEFAULT_N = 20
P_CHOICES = (0.1, 0.2, 0.3, 0.5)                          # Kantenwahrscheinlichkeit
DEFAULT_P = 0.3
RANK_CHOICES = (1, 2, 3, 4, 5, 6, 8, 12)                  # Rang p des Faktors V
DEFAULT_RANK = 4
TOL_CHOICES = (1e-6, 1e-8, 1e-10, 1e-12)                  # Abbruchtoleranz des Blockkoordinatenabstiegs (relative Änderung des Werts je Sweep)
DEFAULT_TOL = 1e-8
GRAPH1_CHOICES = ("fuenfeck", "k5", "k6", "petersen")     # kleine Graphen für Experiment 1
GRAPH1_NAMES = {"fuenfeck": "Fünfeck C₅", "k5": "vollständig K₅", "k6": "vollständig K₆", "petersen": "Petersen-Graph"}
DEFAULT_GRAPH1 = "fuenfeck"
RANK1_CHOICES = (1, 2, 3)                                 # Rang in Experiment 1 (Punkte auf Gerade, Kreis, Kugel)
DEFAULT_RANK1 = 2
SEED_MAX, DEFAULT_SEED = 999999, 3

SCAN_N_MAX = 30                                           # größtes n der Rang-Reihe (Experiment 2 und 3)
SCAN_GRAPHS = 3                                           # Zufallsgraphen (fortlaufende Seeds)
SCAN_STARTS = 8                                           # Zufallsstarts je Graph und Rang
SCAN_RANKS = (1, 2, 3, 4, 5, 6)
SPEED_NS = (10, 20, 40, 80, 160, 320)                     # Experiment 4
ADMM_N_MAX = 80                                           # ADMM nur bis hierhin (Eigenzerlegung je Iteration, 4 s bei n = 80)
PD_N_MAX = 30                                             # primal-dualer innerer Punkt (Stück 6) nur bis hierhin
ADMM_TOL = 1e-7
HYPER_SAMPLES = 2000                                      # Hyperebenen in Experiment 5
MEMORY_NS = (100, 1000, 10000, 100000)
SEED_OFFSET = {"main": 1000, "reference": 2000, "small": 3000, "scan": 4000, "round": 5000, "rew": 6000}

COLORS = {"bm": "#2F6B65", "admm": "#b08a2e", "pd": "#4c78a8", "trap": "#c0504d", "global": "#2F6B65", "theory": "#888888", "cut": "#c0504d", "uncut": "#bbbbbb"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

_BASE = {"n": DEFAULT_N, "p": DEFAULT_P, "seed": DEFAULT_SEED, "rank": DEFAULT_RANK, "tol": DEFAULT_TOL, "graph1": DEFAULT_GRAPH1, "rank1": DEFAULT_RANK1}
PRESETS = {
    "Standardfall: G(20, 0,3), Rang 4": dict(_BASE),
    "Rang 1: die Flip-Suche": dict(_BASE, rank=1, rank1=1),
    "Rang 2: eine Falle (Seed 5)": dict(_BASE, rank=2, seed=5),
    "Rang nach der Theorie (p(p+1)/2 > n)": dict(_BASE, rank=6),
    "Großer Graph (n = 160)": dict(_BASE, n=160, rank=8),
    "Strenge Toleranz (10⁻¹²)": dict(_BASE, tol=1e-12),
}
PRESET_HELP = {
    "Standardfall: G(20, 0,3), Rang 4": "Ein Zufallsgraph mit zwanzig Knoten und 51 Kanten, Rang 4: der Lauf endet nach rund 100 Sweeps bei 40,4393, dem zertifizierten Referenzwert (Rang 7); das Zertifikat sagt ja (kleinster Eigenwert von Z(V) −1,7·10⁻⁵).",
    "Rang 1: die Flip-Suche": "Rang 1 sind Vorzeichen: der Lauf bleibt nach wenigen Sweeps bei 36 stehen, der Referenzwert ist 40,4393; das Zertifikat sagt nein (kleinster Eigenwert −0,99). Im Experiment 1 verfehlt das Fünfeck mit Rang 1 die Schranke in allen 20 Starts.",
    "Rang 2: eine Falle (Seed 5)": "Seed 5: mit Rang 2 endet der Lauf bei 48,4133, der Referenzwert ist 48,9295 (Abstand 0,5162); das Zertifikat sagt nein. Mit Rang 3 erreicht derselbe Graph den Referenzwert (Seed 5, Rang 3: Abstand unter 10⁻⁴).",
    "Rang nach der Theorie (p(p+1)/2 > n)": "Für n = 20 fordert die Theorie p(p+1)/2 > 20, also p ≥ 6: der Lauf erreicht 40,4393 mit Zertifikat. Gemessen genügt auf diesem Graphen schon Rang 3.",
    "Großer Graph (n = 160)": "160 Knoten und 3 830 Kanten, Rang 8: der Lauf endet bei 2329,7279, der Referenzwert (Rang 19) ist 2329,7284; das Zertifikat sagt ja. Der Faktor V hat 160·8 Zahlen statt 160² für X.",
    "Strenge Toleranz (10⁻¹²)": "Dieselbe Aufgabe wie der Standardfall, aber mit Toleranz 10⁻¹²: 342 Sweeps statt rund 100 für denselben Wert 40,4393; der Abstand zum Referenzwert fällt unter 10⁻⁶, der kleinste Eigenwert von Z(V) auf −1,6·10⁻⁷.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["n"] in NS and p["p"] in P_CHOICES and p["rank"] in RANK_CHOICES and p["tol"] in TOL_CHOICES and p["graph1"] in GRAPH1_CHOICES and p["rank1"] in RANK1_CHOICES for p in PRESETS.values())
