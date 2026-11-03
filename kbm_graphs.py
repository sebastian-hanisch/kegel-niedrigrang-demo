"""Graphen als symmetrische Gewichtsmatrix W (Null auf der Diagonale, W_ij = Gewicht der Kante {i, j}, 0 = keine Kante) und der Wert eines Schnitts.

Ein Schnitt ist ein Vorzeichenvektor s ∈ {±1}ⁿ (die zwei Seiten); sein Wert ist die Summe der Gewichte der Kanten zwischen den Seiten:
  cut(s) = Σ_{i<j} w_ij (1 − s_i s_j)/2 = ¼ (Σ_{i,j} W_ij − sᵀ W s).
Familien: Zufallsgraph G(n, p), Kreis, vollständiger Graph, vollständig bipartiter Graph, zufälliger bipartiter Graph, Baum, Gitter, Rad, Petersen-Graph."""

from __future__ import annotations

import math

import numpy as np


def _sym(A: np.ndarray) -> np.ndarray:
    A = np.triu(A, 1)
    return A + A.T


def edges(W: np.ndarray) -> list:
    n = len(W)
    return [(i, j, float(W[i, j])) for i in range(n) for j in range(i + 1, n) if W[i, j] != 0.0]


def total_weight(W: np.ndarray) -> float:
    return float(np.triu(W, 1).sum())


def cut_value(W: np.ndarray, s: np.ndarray) -> float:
    s = np.asarray(s, float)
    return 0.25 * float(W.sum() - s @ W @ s)


def gnp(n: int, p: float, seed: int, weights: str = "eins") -> np.ndarray:
    """Zufallsgraph: jede der n(n−1)/2 Kanten mit Wahrscheinlichkeit p; Gewichte 1 oder (weights = "zufall") gleichverteilt in (0, 1]. Deterministisch aus (n, p, seed, weights)."""
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p muss in [0, 1] liegen")
    rng = np.random.default_rng(seed)
    present = np.triu(rng.random((n, n)) < p, 1).astype(float)
    if weights == "zufall":
        present = present * (1.0 - np.triu(rng.random((n, n)), 1))
    elif weights != "eins":
        raise ValueError(f"unbekannte Gewichte {weights!r}")
    return _sym(present)


def cycle(n: int) -> np.ndarray:
    if n < 3:
        raise ValueError("ein Kreis hat mindestens drei Knoten")
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = W[(i + 1) % n, i] = 1.0
    return W


def complete(n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    return 1.0 - np.eye(n)


def bipartite(a: int, b: int) -> np.ndarray:
    if a < 1 or b < 1:
        raise ValueError("beide Seiten brauchen einen Knoten")
    W = np.zeros((a + b, a + b))
    W[:a, a:] = 1.0
    return W + W.T


def random_bipartite(a: int, b: int, p: float, seed: int) -> np.ndarray:
    """Zufälliger bipartiter Graph: Seiten der Größe a und b, jede der a·b Kanten zwischen den Seiten mit Wahrscheinlichkeit p; deterministisch aus (a, b, p, seed)."""
    if a < 1 or b < 1:
        raise ValueError("beide Seiten brauchen einen Knoten")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p muss in [0, 1] liegen")
    rng = np.random.default_rng(seed)
    W = np.zeros((a + b, a + b))
    for i in range(a):
        for j in range(b):
            if rng.random() < p:
                W[i, a + j] = W[a + j, i] = 1.0
    return W


def bipartition(W: np.ndarray):
    """Eine Zweifärbung s ∈ {±1}ⁿ, bei der jede Kante zwei Farben verbindet (Knoten ohne Kante: +1), oder None, wenn der Graph nicht bipartit ist."""
    n = len(W)
    s = np.zeros(n)
    for start in range(n):
        if s[start] != 0.0:
            continue
        s[start] = 1.0
        stack = [start]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(W[i]):
                if s[j] == 0.0:
                    s[j] = -s[i]
                    stack.append(int(j))
                elif s[j] == s[i]:
                    return None
    return s


def is_connected(W: np.ndarray) -> bool:
    n = len(W)
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(W[i]):
            if int(j) not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return len(seen) == n


def tree(n: int, seed: int = 0) -> np.ndarray:
    """Zufallsbaum: Knoten i > 0 hängt an einen zufälligen früheren Knoten."""
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(1, n):
        j = int(rng.integers(0, i))
        W[i, j] = W[j, i] = 1.0
    return W


def grid(rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    if n < 2:
        raise ValueError("mindestens zwei Knoten")
    W = np.zeros((n, n))
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                W[i, i + 1] = W[i + 1, i] = 1.0
            if r + 1 < rows:
                W[i, i + cols] = W[i + cols, i] = 1.0
    return W


def wheel(n: int) -> np.ndarray:
    """Rad: Nabe 0 und ein Kreis aus den Knoten 1 … n − 1, Nabe mit allen verbunden (n ≥ 4)."""
    if n < 4:
        raise ValueError("ein Rad hat mindestens vier Knoten")
    W = np.zeros((n, n))
    for i in range(1, n):
        W[0, i] = W[i, 0] = 1.0
        j = i % (n - 1) + 1
        W[i, j] = W[j, i] = 1.0
    return W


def petersen() -> np.ndarray:
    W = np.zeros((10, 10))
    for i in range(5):
        for a, b in ((i, (i + 1) % 5), (i, i + 5), (i + 5, (i + 2) % 5 + 5)):
            W[a, b] = W[b, a] = 1.0
    return W


def circle_layout(n: int) -> np.ndarray:
    """Knoten gleichmäßig auf dem Einheitskreis (n, 2)."""
    ang = 2.0 * math.pi * np.arange(n) / n + math.pi / 2.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def brute_force_maxcut(W: np.ndarray) -> float:
    """Der größte Schnitt durch einfaches Aufzählen aller 2ⁿ Vorzeichen (für Tests an kleinen Graphen, n ≤ 12)."""
    n = len(W)
    if n > 12:
        raise ValueError("zu groß für die einfache Aufzählung")
    best = 0.0
    for mask in range(2 ** n):
        s = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
        best = max(best, cut_value(W, s))
    return best
