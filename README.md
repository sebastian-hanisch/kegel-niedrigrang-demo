# Niedriger Rang: Burer–Monteiro – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-niedrigrang-demo.streamlit.app/)** -->

Stück 13 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Dieses Stück schließt den Strang **Relaxation** ab und ist die **Gegenprobe zu Stück 9**: dasselbe MaxCut-SDP wie in Stück 12, aber **ohne den Kegel**.

## Kernfrage

**Wie klein darf der Rang sein, und woher weiß man, dass ein Lauf das Optimum der Relaxation erreicht hat?** Das MaxCut-SDP verlangt `X ⪰ 0` mit `Xᵢᵢ = 1`: `n²` Zahlen und eine Eigenzerlegung je Iteration. Setzt man `X = V Vᵀ` mit `V ∈ ℝⁿˣᵖ`, ist `X ⪰ 0` von selbst erfüllt, `Xᵢᵢ = 1` heißt, dass jede Zeile von `V` Länge 1 hat, und man rechnet mit `n·p` Zahlen. Der Preis: das Programm ist **nicht mehr konvex**, und bei zu kleinem Rang `p` endet der Löser in einer **Falle** (bei `p = 1` ist es die Flip-Suche der lokalen Optima). Gemessen (Blockkoordinatenabstieg, Zufallsgraphen `G(n, 0,3)`): `G(20, 0,3)` erreicht mit Rang 1 und mit Rang 2 nie das Optimum (Rang 2 endet an Fixpunkten knapp darunter, ohne Zertifikat); ab **Rang 3** erreichten alle 24 Läufe das Optimum (Toleranz 10⁻⁸). „Optimum erreicht“ heißt dabei: der Lauf ist konvergiert und liegt höchstens so weit unter dem zertifizierten Referenzwert, wie sein Abstieg noch gewinnen würde, plus `100·Toleranz·Wert`. Die Theorie von Boumal, Voroninski und Bandeira fordert für *fast alle Kostenmatrizen* `p(p+1)/2 > n`, hier `p ≥ 6` (Zufallsgraphen mit Gewicht 1 fallen nicht unter diese Aussage). Ein **Zertifikat** aus dem Ergebnis allein trennt Optima und Fallen: am Fixpunkt ist `Z(V) = Diag(‖gᵢ‖/2) + W/2` die Dualmatrix aus Stück 12, und ist sie positiv semidefinit, ist `V Vᵀ` am exakten Fixpunkt **global optimal**. Im Lauf ist das numerisch zertifiziert: der Wert liegt höchstens `n·|λ_min|/2` unter dem Optimum der Relaxation (Standardlauf: 1,8·10⁻⁴). In 144 Läufen auf `G(20, 0,3)` gab es **kein** Zertifikat bei einer Falle; bei grober Toleranz (10⁻⁶) und dünnen größeren Graphen kann das Zertifikat auch Läufen „ja“ geben, die noch ein Stück unter dem Optimum liegen (8 von 722 Läufen einer breiteren Stichprobe, 0,006 bis 0,008 darunter).

Das Beispiel ist hier wieder der **Graph**: ein Zufallsgraph `G(n, p)` (Gewicht 1), dazu kleine Graphen (Fünfeck, `K₅`, `K₆`, Petersen) für die Anschauung.

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 bis 12, unverändert): `ConeProblem` (PSD-Kegel), `svec`/`smat`; hier nur über den SDP-Löser als Gegenprobe gebraucht.
- **Löser Burer–Monteiro** (`kbm_bm.py`): Blockkoordinatenabstieg („mixing method“, Wang, Chang, Kolter): `vᵢ ← −gᵢ/‖gᵢ‖` mit `gᵢ = Σⱼ wᵢⱼ vⱼ`, die exakte Minimierung über eine Zeile (Cauchy–Schwarz, selbst hergeleitet); Dualmatrix `Z(V)`, Zertifikat und Schwelle, zertifizierter Referenzwert, Rang nach der Theorie.
- **ADMM in Matrixform** (`kbm_admm.py`): Stück 9 auf diesem Kegel: Diagonale auf 1 setzen, Projektion auf den PSD-Kegel per Eigenzerlegung. Der allgemeine Kegel-ADMM aus Stück 9 hätte `n(n+1)/2` Variablen und wäre nicht vergleichbar.
- **Innerer Punkt** (`kbm_solver.py`, `kbm_cones.py`, `kbm_sdp.py`): primal-dualer Löser aus Stück 6 und das SDP aus Stück 12 als Gegenprobe und für die Zeitreihe.
- **Rang und Speicher** (`kbm_rank.py`): Speicher von `X` und `V`, das Zertifikat von Rakoto Endor und Waldspurger für bipartite Graphen.
- **Graphen** (`kbm_graphs.py`): wie in Stück 12, dazu zufällige bipartite Graphen, Zweifärbung, Zusammenhang. **Rundung** (`kbm_rounding.py`): der Erwartungswert `Σ w arccos(⟨vᵢ, vⱼ⟩)/π`, Zufallshyperebenen, exakter Schnitt für `n ≤ 20`.
- **Kein Eingriff in den Kern:** alles Neue steht in `kbm_*.py`; BLAS-Threads laufen für die Zeitreihe mit einem Thread (`threadpoolctl`).

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) **ein Lauf, wenige Punkte**: die Zeilen von `V` auf Gerade, Kreis oder Kugel (Rang 1 bis 3) für Fünfeck, `K₅`, `K₆`, Petersen, der Verlauf des Werts, die Erfolgsquote über 20 Starts; (2) **wie klein darf der Rang sein**: 3 Zufallsgraphen `G(n', p)` (`n' = min(n, 30)`), je Rang 1 bis 6 acht Zufallsstarts, Anteil der Läufe im Optimum und mit Zertifikat; (3) **das Zertifikat**: der kleinste Eigenwert von `Z(V)` je Lauf, Zählung Zertifikat ja/nein gegen Optimum ja/nein; (4) **Zeit gegen `n`**: Burer–Monteiro (Rang `⌈√(2n)⌉`), ADMM (bis `n = 80`) und innerer Punkt (bis `n = 30`) auf demselben Graphen; (5) **Rundung aus `V`, Speicher und das Zertifikat vor dem Lauf**: die Zufallshyperebene ohne `n×n`-Matrix, die Speichertabelle, das Zertifikat von Rakoto Endor und Waldspurger an acht bipartiten Graphen. Dazu ein Abschnitt zu **Frank–Wolfe mit Skizze** (nur Text mit Zitaten). Regler: Knoten `n` (10 bis 320), Kantenwahrscheinlichkeit `p` (0,1 bis 0,5), Seed, Rang `p`, Toleranz (10⁻⁶ bis 10⁻¹²), kleiner Graph und Rang für Experiment 1; ein Schnellstart lädt sechs Szenarien, die Adresszeile spiegelt die Konfiguration. Der **Referenzwert** ist ein Lauf mit Rang `p(p+1)/2 > n` plus eins und gültigem Zertifikat, danach bei Toleranz 10⁻¹⁴ poliert (der Wert ist schon bei 10⁻¹⁰ auf 10⁻⁶ genau, der Punkt `V` erst nach dem Polieren): ein numerisch zertifizierter Optimalwert **ohne SDP-Löser**; die Tests prüfen ihn gegen den Löser aus Stück 6 und gegen cvxpy.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung und Rechner, steht die gemessene Zahl trotzdem hier). Zeiten, Sweepzahlen, der Status des Lösers und alle Quoten über Zufallsstarts werden nur in weiten Bändern getestet. Standardgraph: `G(20, 0,3)`, Seed 3, 51 Kanten, Rang 4, Toleranz 10⁻⁸.

| Frage | Befund |
|---|---|
| Der Standardlauf | Wert **40,4393**, gleich dem zertifizierten Referenzwert (Rang 7); 107 Sweeps, Abstand 9·10⁻⁶, kleinster Eigenwert von `Z(V)` −1,7·10⁻⁵, **Zertifikat ja** |
| Rang 1 (Flip-Suche) | Der Lauf bleibt nach 3 Sweeps bei **36**, Referenzwert 40,4393; Zertifikat nein (kleinster Eigenwert −0,99). Fünfeck mit Rang 1: 4 statt 4,5225 in allen 20 Starts |
| Rang 2, Seed 5 (`G(20, 0,3)`, 66 Kanten) | Wert **48,4133**, Referenzwert 48,9295 (Abstand 0,5162), Zertifikat nein; mit Rang 3 erreicht derselbe Graph den Referenzwert (Abstand 2·10⁻⁷, Zertifikat ja) |
| Rang 2, Seed 3 | Wert 40,4390, also 3,5·10⁻⁴ unter dem Optimum, **kein Zertifikat** (kleinster Eigenwert −6·10⁻³): Rang 2 endet hier an einem Fixpunkt knapp unter dem Optimum (mit Toleranz 10⁻¹⁰, 10⁻¹² und bei Fortsetzung bis 10⁻¹⁴ bleibt der Abstand gleich; von 30 Zufallsstarts enden 26 an diesem Fixpunkt, 4 an größeren Fallen 0,58 und 1,40 darunter, keiner im Optimum): fast optimal, aber **nicht** „Optimum erreicht“ (erlaubt wären 4,1·10⁻⁵ bei Toleranz 10⁻⁸); Rang 3 genügt. Der Text sagt dazu „unter dem Optimum“ (bei Toleranz 10⁻⁶ liegt der Fixpunkt innerhalb der Unschärfe des Kriteriums, siehe Grenzen) |
| Der Rang nach der Theorie | `n = 20`: `p(p+1)/2 > 20` gibt `p ≥ 6`, `√(2n) = 6,3`; der Lauf mit Rang 6 erreicht 40,4393 mit Zertifikat. Gemessen genügt auf diesem Graphen Rang 3 |
| Erfolgsquote nach Rang, `G(20, 0,3)`, 3 Graphen × 8 Starts | Rang 1 / 2 / 3 / 4 / 5 / 6: **0 / 0 / 100 / 100 / 100 / 100 %** (Optimum erreicht: höchstens der Rest des Abstiegs plus `100·Toleranz·Wert` unter dem Referenzwert), mit Zertifikat 0 / 0 / 92 / 100 / 100 / 100 %. Kleinster Rang mit mindestens 95 %: `n` = 10 → 3, `n` = 20 → 3, `n` = 30 → 4 (Band ±1), unter `√(2n)` = 4,5 / 6,3 / 7,7 (bei `n` = 10 nur knapp, bei 20 und 30 etwa die Hälfte) |
| Kleine Graphen (Schranke / Rang 1 / 2 / 3) | Fünfeck: 4,5225 / 4 / 4,5225 / 4,5225 (Rang 2: 20 von 20 Starts); `K₅`: 6,25 / 6 / 6,25 / 6,25; `K₆`: 9 / 9 / 9 / 9 (schon Rang 1 genügt, die Schranke ist exakt); Petersen: 12,5 / 12 / 12,375 / 12,4641, und mit **Rang 4 und 5** erreicht jeder Start 12,5 (40 Starts mit Rang 3: höchstens 12,4641) |
| `K_n` | Mit Rang 2 genügt das regelmäßige `n`-Eck (`Σ vᵢ = 0`): Wert `n²/4`; mit Rang 1 nur `n²/4` (gerade `n`) oder `(n² − 1)/4` (ungerade `n`) |
| Das Zertifikat, 144 Läufe (`G(20, 0,3)`, Ränge 1 bis 6, Toleranz 10⁻⁸) | Zertifikat ja und Optimum: 94; Zertifikat ja und Falle: **0**; Zertifikat nein und Optimum erreicht: 2 (knapp an der Zertifikat-Schwelle); Zertifikat nein und Falle: 48 (darunter 7 Läufe am Rang-2-Fixpunkt 3,5·10⁻⁴ unter dem Optimum). Kleinster Eigenwert bei den Optima ≥ −1,1·10⁻³, bei den Fallen ≤ −6,1·10⁻³ (der Rang-2-Fixpunkt; die übrigen Fallen meist ≤ −3,8·10⁻²). Die Schwelle ist `10·√Toleranz` (Gewicht 1, `n` = 20 und 30; bei Toleranz 10⁻⁴ und gröber überlappen Fallen und ungenaue Läufe) |
| Große Aufgabe | `G(160, 0,3)`, Seed 3, 3 830 Kanten, Rang 8: Wert **2329,7279**, Referenzwert (Rang 19) 2329,7284, 124 Sweeps, Zertifikat ja (kleinster Eigenwert −2·10⁻⁴). `V` hat `160·8 = 1 280` Zahlen, `X` hat 25 600 |
| Die Toleranz | Standardgraph mit Toleranz 10⁻¹²: **342 Sweeps** statt 107, derselbe Wert 40,4393, kleinster Eigenwert −1,6·10⁻⁷. `G(80, 0,2)`, Rang 13: Toleranz 10⁻⁶ / 10⁻⁸ / 10⁻¹⁰ / 10⁻¹² braucht 34 / 59 / 85 / 113 Sweeps (lineare Konvergenz) |
| Zeit gegen `n` (Entwicklungsrechner, Rang `⌈√(2n)⌉`, Toleranz 10⁻⁸; Größenordnung, nicht getestet) | `n` = 10 / 20 / 40 / 80 / 160 / 320: Burer–Monteiro rund 3 / 40 / 30 / 210 / 410 / 1 100 ms; ADMM 26 / 160 / 500 / 1 300 ms (bis 80); innerer Punkt 55 / 520 ms (bis 30). Getestet wird nur: Burer–Monteiro ist ab `n = 40` um mindestens das 1,5-Fache schneller als ADMM (gemessen 6- bis 20-fach), alle Verfahren treffen denselben Wert |
| Rundung aus `V` | Rang-7-Faktor des Standardgraphen: SDP-Wert 40,4393, **erwarteter Schnitt 37,6025** (aus `X = V Vᵀ`, `V` des bei Toleranz 10⁻¹⁴ polierten Referenzlaufs), Mittel von 2 000 Hyperebenen 37,614 ± 0,0336, bester Schnitt 39 = exakter Schnitt. Aus dem Mittelpunkt des inneren Punkts (Stück 12) folgt derselbe Wert 37,6025. **Der erwartete Schnitt hängt vom Punkt `X` ab, nicht vom Zielwert:** der bei Toleranz 10⁻¹⁰ beendete Lauf gäbe 37,6011 (ein Artefakt der Referenztoleranz, der Wert 40,4393 ist derselbe), 37,6023 bei 10⁻¹², 37,6025 bei 10⁻¹⁴; schon 10⁻⁷ Restfehler im Wert verschiebt ihn um 10⁻³ |
| Speicher (8 Byte je Zahl; Rang `⌈√(2n)⌉`) | `n` = 100 / 1 000 / 10 000 / 100 000, `p` = 15 / 45 / 142 / 448: `X` 80 kB / 8 MB / 800 MB / 80 GB; `V` 12 kB / 360 kB / 11,4 MB / 358 MB |
| Das Zertifikat vor dem Lauf (Rakoto Endor, Waldspurger), 8 bipartite Graphen | `p > κ = λₙ/λ₂` der vorzeichenlosen Laplace-Matrix: `K_{a,a}`: κ = 2 → `p ≥ 3`; Zufallsbipartite `n` = 20 bis 60: κ = 5,0 bis 19,0 → `p ≥ 6` bis 20. Mit **Rang 2** erreichten in 300 Starts je Graph 100 % das Optimum, mit Rang 1 je nach Graph 67 bis 100 % (Verhältnis gefordert zu gebraucht bei Rang 2: 1,5 bis 10): die Bedingung ist hinreichend (so die Arbeit), in dieser Messung aber vorsichtig |
| Gegen cvxpy | Ein **eigener** Aufbau des SDP in cvxpy gibt mit Clarabel denselben Wert wie der zertifizierte Referenzwert (10⁻⁴) für C₅, Petersen, `K₄`, `G(10, ½)`, `G(16, ½)` und `G(20, 0,3)`, mit SCS für `G(12, ½)`; die Schranke aus `Z(V)` ist gleich dem cvxpy-Optimum; ein Lauf mit Rang 3 auf Petersen liegt mehr als 0,03 darunter; das Matrix-ADMM trifft Clarabel auf 10⁻⁴ |

## Befunde und Korrekturen gegenüber dem Plan

- **Burer–Monteiro verliert nicht bei `n = 40`.** Die Phase-0-Messung (515 ms gegen 138 ms für ADMM) lief mit Toleranz 10⁻¹² und stieß an die Sweep-Grenze; mit Toleranz 10⁻⁸ ist es schon ab `n = 10` schneller. Die Zeit hängt stark an der Toleranz, deshalb ist sie ein Regler.
- **Der Schwanz ist kürzer als in der Phase 0** (dort 25 / 135 / 568 / 1 606 Sweeps für `G(80, 0,2)` mit Rang 13; gebaut 34 / 59 / 85 / 113 mit einem anderen Graphen und Start). Der Text sagt „lineare Konvergenz“ und nennt die gebauten Zahlen.
- **Die Zertifikat-Schwelle ist `10·√Toleranz`**, nicht eine feste Zahl (Plan: Schwelle zwischen 10⁻⁵ und 10⁻²): ein bei Toleranz `tol` beendeter Lauf im Optimum hat `λ_min` bis etwa `−3,8·√tol` (obere Hüllkurve der Messung für `tol` = 10⁻⁶ bis 10⁻¹², Gewicht 1; typisch deutlich weniger).
- **„Optimum erreicht“ ist der Rest des Abstiegs plus `100·Toleranz·Wert`.** Die erste Fassung (`max(5·10⁻³, 2·10⁻⁵·Wert)` unter dem Referenzwert) zählte zwei echte Nicht-Optima als Optimum: den Rang-2-Fixpunkt auf Seed 3 (3,5·10⁻⁴ darunter) und `n = 320`, Rang 8 (Fixpunkt 0,029 darunter, relativ 3·10⁻⁶, `λ_min` −3,5·10⁻²; Rang 12, 16 und 26 erreichen den Referenzwert). Das jetzige Kriterium verlangt einen konvergierten Lauf, dessen Abstand zum Referenzwert höchstens der geschätzte Rest seines Abstiegs (geometrische Reihe aus den letzten beiden Sweeps) plus `100·Toleranz·max(1, Wert)` ist: ein Fixpunkt hat keinen Rest, ein langsam konvergierender Lauf einen großen. Der Faktor 100 ist gemessen (2 786 Läufe auf `G(n, p)`, `n` = 10 bis 320, Rang 1 bis 26, Wahrheit: die bei 10⁻¹⁴ fortgesetzte Rechnung): bei Toleranz 10⁻⁸ ist der Überschuss (Abstand − Rest)/(Toleranz·Wert) bei 661 Läufen, die das Optimum erreichen, höchstens 37, bei 248 Läufen an einem Fixpunkt darunter mindestens 11 und, bis auf zwei weitere (41 und 63), mindestens 133 (`n = 320`, Rang 8: 326; Rang 2, Seed 3: 863); bei 10⁻¹⁰ und 10⁻¹² höchstens 0 gegen mindestens 3·10⁴.
- **Der Rang 2 auf Seed 3 ist ein eigener Fall** (ein Fixpunkt 3,5·10⁻⁴ unter dem Optimum, kein Zertifikat), nicht im Plan; das Verdikt sagt dort „unter dem Optimum“ (in der ersten Fassung war es eine Warnung „bis auf die Schwelle am Optimalwert“, und die Erfolgsquote von Rang 2 war 29 %).
- **Der Petersen-Graph braucht Rang 4** (nicht im Plan, an der Graphenreihe beim Bau gemessen: 12 / 12,375 / 12,4641 / 12,5 für Rang 1 bis 4). Experiment 1 bietet nur Rang 1 bis 3 (Gerade, Kreis, Kugel); die Aussage steht im Text und im Test.
- **Das Zertifikat vor dem Lauf ist um den Faktor 1,5 bis 10 vorsichtig** (Plan: 3 bis 20): `K_{a,a}` fordert `p ≥ 3`, gebraucht wurden 1 bis 2 (Rang 2 genügte in allen Starts).
- **Der Erfolg der Rang-Reihe hängt an wenigen Graphen:** drei Graphen mal acht Starts statt acht Graphen mal zehn Starts der Phase 0 (die Reihe muss live bleiben, rund 3 s); Quoten sind Bänder, keine Zahlen für die Theorie.
- **Der Referenzwert kommt ohne SDP-Löser** (nicht im Plan): ein zertifizierter Lauf mit größerem Rang. Der SDP-Löser aus Stück 6 liefert nur die Zeitreihe (`n ≤ 30`), Experiment 1 und die Gegenproben in den Tests.
- **Quellen:** geprüft am 2026-11-03. **Am Volltext (ar5iv):** Boumal, Voroninski, Bandeira, *Deterministic guarantees for Burer–Monteiro factorizations of smooth semidefinite programs*, Communications on Pure and Applied Mathematics (arXiv 1804.02008): Satz 1.4, `p(p+1)/2 > rank A` (ersetzbar durch `m`), Annahme 1.1; Rakoto Endor, Waldspurger, *Benign landscape for Burer–Monteiro factorizations of MaxCut-type semidefinite programs*, arXiv 2411.03103: Satz 2.2 und Proposition 2.3; Yurtsever, Tropp, Fercoq, Udell, Cevher, *Scalable semidefinite programming*, SIAM Journal on Mathematics of Data Science 3 (1), 2021, S. 171–200 (arXiv 1912.02949): Speicher `O(d + r n/ζ)` (Satz 1.4), über 20 Millionen Knoten. **Nur über Abstract oder Einleitung:** Burer, Monteiro, Mathematical Programming, Ser. B 95, 2003, S. 329–357 (Abstract); Wang, Chang, Kolter, arXiv 1706.00476 (Abstract; die Aktualisierung ist hier selbst hergeleitet); die Aussagen „Korollar 5.9 und 5.11 der Arbeit von Boumal u. a.“ (Rang `≈ √(2n)` für fast alle, `n/2` für alle Kostenmatrizen) sowie „Kostenmatrizen positiven Maßes mit Fallen unterhalb `√(2n)`“ (Waldspurger, Waters) und „Fehler mit Maß null bis `n/2`“ (O'Carroll u. a.) stehen als **Zitat** in der Einleitung von Rakoto Endor und Waldspurger. **Eigene Überlegung, nicht zitiert:** dass Annahme 1.1 für die Diagonalbedingungen gilt (`Aᵢ V = eᵢ vᵢᵀ`, linear unabhängig, solange alle `vᵢ ≠ 0`). **Nicht im gelesenen Text:** das Wort „Nyström“ und die Form `S = X Ω` bei SketchyCGAL; die Demo behauptet sie nicht. Beweise der Arbeiten nicht nachgelesen. <!-- lint-ok: dezimalpunkt -->
- **Abgrenzung:** `lokale-optima-demo` (Komplexitätsklassen-Linie) behandelt MaxCut mit Flip-Nachbarschaft; hier ist Rang 1 genau diese Flip-Suche, und der Befund ist, dass **eine zusätzliche Dimension die Fallen auflöst**. Keine Zahlen werden übernommen.

## Ehrliche Grenzen

- **Nur MaxCut-Typ** (Diagonalbedingungen `Xᵢᵢ = 1`, Gewichte 1): jede Zeile hat eine geschlossene Aktualisierung. Allgemeine SDPs mit `m` Gleichungen brauchen das Originalverfahren (Augmented Lagrangian) oder Riemannsche Verfahren; nicht gebaut.
- **Die Messung des kleinsten Rangs ist eine Messung an Zufallsgraphen** `G(n, p)` mit `n ≤ 30` in der App; die Theorie gilt für *fast alle* Kostenmatrizen, und unterhalb ihrer Schwelle gibt es nach Zitat Mengen positiven Maßes mit Fallen. „Der Rang darf klein sein“ ist hier ein Befund, kein Satz.
- **Das Zertifikat prüft einen Fixpunkt.** Es braucht eine Eigenwertrechnung der `n×n`-Matrix `Z(V)`: der Speichergewinn gilt für den Lauf, nicht für die Prüfung. Die Schwelle ist an die Toleranz gebunden und gemessen für Gewicht 1, `n` = 20 und 30 (und bei den größeren Graphen der Tests); bei Toleranz 10⁻⁴ und gröber trennt sie nicht (die App bietet nur 10⁻⁶ und strenger an).
- **„Optimum erreicht“ trennt bei Toleranz 10⁻⁶ nicht, und nicht jeden Fixpunkt:** bei 10⁻⁶ überlappen die Gruppen (Optima bis Überschuss 48, Fixpunkte ab 0,9; 31 von 244 gemessenen Fixpunkten zählen als Optimum, darunter der Rang-2-Fixpunkt auf Seed 3 mit Überschuss 9), auch das Zertifikat trennt dort nicht mehr; die App bietet 10⁻⁶ an, sagt es aber im Hilfetext. Bei 10⁻⁸ bleiben drei von 248 gemessenen Fixpunkten unerkannt (Rang 6 bei `n` = 150, 1,0·10⁻⁴ bis 3,3·10⁻³ unter dem Optimum, Überschuss 11, 41 und 63): dort sagt das Zertifikat „nein“, und das Verdikt wird zur Warnung. Ein Lauf, der auf einem Plateau (Sattel) stehenbleibt und erst nach der Fortsetzung das Optimum erreicht, zählt als nicht erreicht (1 von 661 bei 10⁻⁸). Gemessen ist das an Zufallsgraphen mit Gewicht 1.
- **Fixpunkt ist nicht Optimum:** ein Sattel ist ein Fixpunkt, den das Zertifikat verneint. Die Demo sagt „Falle oder Sattel“ und prüft keine Hesse-Matrix.
- **Zeiten sind Python-Schleifen** (`for i in range(n)`), keine kompilierte Bibliothek; der Faktor gegen ADMM wäre in C ein anderer. Sie hängen vom Rechner und von der Toleranz ab und werden nicht getestet (Lehre aus Stück 9 bis 12).
- **Das Zertifikat vor dem Lauf** (Rakoto Endor, Waldspurger) gilt hier nur für zusammenhängende bipartite Graphen (Rang-1-Lösung); seine Schärfe (Proposition 2.3, konstruierte Gegenbeispiele) wurde nicht nachgebaut. <!-- lint-ok: dezimalpunkt -->
- **Frank–Wolfe mit Skizze ist nur ein Abschnitt mit Zitaten:** kein Code, keine Messung; Speicher `O(d + r n/ζ)` und „über 20 Millionen Knoten“ sind Aussagen der Arbeit.
- **Der Erfolgsanteil je Rang** hängt an den drei Graphen und den Starts der App; der Lauf mit festem Seed ist reproduzierbar, die Quote ist keine Wahrscheinlichkeit für andere Graphen.

## Verifikation

- **Burer–Monteiro** (`tests/test_bm.py`): der Zielwert an Handbeispielen (Fünfeck 4,5225, `K_n`-Vielecke `n²/4`, `p = 1` der Schnittwert), die Aktualisierung als exakter Minimierer einer Zeile (Gitter von 20 000 Winkeln), Zeilenlänge 1 und monotones Wachsen, `p = 1` als Flip-Suche (ganzzahliger Schnitt, lokales Optimum der Flip-Nachbarschaft), Wert ≤ SDP-Wert für jeden Rang, ausreichender Rang erreicht ihn, `Z(V)` für die Ecken des Quadrats (`Z = J/2`, Schranke 4, `Z V = 0`), eine Falle (`K₅`, Rang 1: kleinster Eigenwert −0,5), Zertifikat ja ⇒ Wert gleich SDP-Wert, Referenzwert gegen den Löser aus Stück 6, Theorie-Rang als kleinstes `p` mit `p(p+1)/2 > n`, Toleranz gegen Sweeps, Reproduzierbarkeit, das Kriterium „Optimum erreicht“ (Rest des Abstiegs als geometrische Reihe von Hand, Randfälle: nicht konvergiert, Lauf über dem Referenzwert, knapp über und unter der Schwelle; der Rang-2-Fixpunkt auf Seed 3 bei Toleranz 10⁻⁸ bis 10⁻¹² nie „erreicht“, Rang 3 immer; ein langsam konvergierender Lauf auf `G(120, 0,1)` ist noch Optimum; der Überschuss ist zweigipflig mit dem Faktor 100 dazwischen).
- **ADMM, Speicher, Zertifikat vor dem Lauf** (`tests/test_admm_rank.py`): ADMM gegen den Löser aus Stück 6; Speicher in Byte gegen `ndarray.nbytes`; `K_{a,a}` hat `κ = 2`, `K_{a,b}` die Eigenwerte `min`, `max`, `a + b`; `L = D + W` hat Kern `x`; Grenzen der Anwendbarkeit (nicht bipartit, nicht zusammenhängend); ausreichender Rang erreicht bei bipartiten Graphen in 10 von 10 Starts das Optimum.
- **Orakel** (`tests/test_oracle_kegel_niedrigrang.py`): ein **eigener** Aufbau des SDP in cvxpy (Clarabel, SCS) gegen den zertifizierten Referenzwert, die Schranke aus `Z(V)`, eine Falle unter dem Optimum, Matrix-ADMM.
- **Graphen, SDP, Rundung, Löser** (aus Stück 12 und 6 übernommen, `tests/test_graphs.py` um zufällige bipartite Graphen, Zweifärbung und Zusammenhang erweitert).
- **Randfälle** (`tests/test_edge_cases.py`): zwei Knoten, Graph ohne Kanten (Wert 0, Zertifikat ja), die Rang-Reihe ohne Kanten, isolierte Knoten (Zeile bleibt unberührt), nicht zusammenhängende Graphen (Summe der Teile), Skalierung der Gewichte (gleiche Zeilen, Wert ×10), Umnummerierung, Rang über `n`, Gewicht 0, alle Reglerecken, **Serialisierbarkeit der zwischengespeicherten Ergebnisse (pickle)**.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt, Art und Text), **Claims** (`tests/test_claims.py`: jede Zahl dieser README und der Hilfetexte), **Oberfläche** (AppTest: jeder Button, jeder Auswahlregler an erstem und letztem Wert, jeder Rang, jeder kleine Graph mit jedem Rang, PDF, Tabellen), **Abbildungen** (aus den gezeichneten Punkten zurückgerechnet: Länge 1, Gram-Matrix, Nachbarwinkel des Fünfecks 144°), Presets, Permalink, Layout, Footer, Laufzeit-Importe, Kern (Prüfsumme gegen das Schwester-Repo).
- **Umfang:** 208 Tests (10 davon, die cvxpy-Orakel gegen Clarabel und SCS, überspringen sich ohne cvxpy; mit cvxpy laufen alle 10 grün), rund 4,5 Minuten (2026-11-03, Windows, Rechner durch andere Läufe ausgelastet; Stand vor dieser Änderung rund 2,5 Minuten); kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy ist Gegenprobe im dev-Satz. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 bis 12 |
| `kbm_bm.py` | Burer–Monteiro: Blockkoordinatenabstieg, `Z(V)`, Zertifikat, Referenzwert, Rang nach der Theorie |
| `kbm_admm.py` | ADMM in Matrixform (Gegenprobe, Stück 9) |
| `kbm_rank.py` | Speicher von `X` und `V`, Zertifikat von Rakoto Endor und Waldspurger |
| `kbm_graphs.py` | Graphen, Familien, zufällige bipartite Graphen, Zweifärbung, Schnittwert |
| `kbm_sdp.py` | Die SDP-Relaxation als Kegelprogramm (aus Stück 12), Lösung, Dual, Zerlegung `X = V Vᵀ` |
| `kbm_rounding.py` | Erwartungswert der Rundung, Zufallshyperebenen, exakter Schnitt |
| `kbm_solver.py`, `kbm_cones.py` | Primal-dualer innerer Punkt (Mehrotra, Nesterov–Todd), aus Stück 6 |
| `kbm_constants.py` | Regler-Reihen, feste Reihen, Farben, Presets |
| `kbm_evaluation.py` | Experimente, Verdikte, Formatierung |
| `kbm_layout.py` | Fester Kopf (Stylesheet) |
| `kbm_pdf_export.py` | PDF-Export (fpdf2) |
| `kbm_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kbm_visualization.py` | Plotly-Abbildungen (Punkte auf Kreis und Kugel, Verlauf, Erfolgsquote, Eigenwerte, Zeitreihe; 2D-Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Burer–Monteiro, ADMM und Rang, Orakel, Graphen, SDP, Rundung, Löser, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe, Kern |

## Bewusst nicht umgesetzt

- **Frank–Wolfe mit Skizze (CGAL)** als Verfahren: nur der Abschnitt mit Zitaten; die Arbeit von Yurtsever u. a. wäre ein eigenes Stück.
- **Riemannsche Verfahren und das Originalverfahren von Burer und Monteiro** (Augmented Lagrangian, L-BFGS) für allgemeine SDPs mit `m` Gleichungen.
- **Eine Hesse-Prüfung** (Falle von Sattel unterscheiden) und eine **matrixfreie Eigenwertsuche** für das Zertifikat.
- **Die konstruierten Gegenbeispiele** (Proposition 2.3 von Rakoto Endor und Waldspurger) und die Fälle mit Fallen oberhalb der Barvinok–Pataki-Schwelle (O'Carroll u. a.). <!-- lint-ok: dezimalpunkt -->
- **Kompilierte Kerne** für die Sweeps; die Zeiten sind Python.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html).
