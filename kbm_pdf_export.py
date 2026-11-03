"""PDF-Export des aktuellen Laufs (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from kbm_evaluation import Settings, de, graph_name, num, sci, tol_text

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "λ": "lambda", "β": "beta", "α": "alpha", "ρ": "rho", "μ": "mu", "τ": "tau", "θ": "theta", "κ": "kappa",
            "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "ℓ": "l", "∞": "inf", "√": "sqrt", "‖": "|",
            "°": "o", "⟨": "<", "⟩": ">", "½": "1/2", "⪰": ">=", "ₙ": "n", "₂": "2"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, m: dict) -> bytes:
    ref = m["ref"]
    n = m["W"].shape[0]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Niedriger Rang: Burer-Monteiro für die MaxCut-SDP"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{graph_name(s)}, Seed {s.seed}, {m['edges']} Kanten, Rang p = {m['rank']}, Toleranz {tol_text(s.tol)}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    rows = [("Wert des Laufs (Relaxation bei X = V V^T)", num(m["value"])), ("zertifizierter Referenzwert", num(ref["value"]) + (f" (Rang {ref['p']})" if ref["certified"] else f" (Rang {ref['p']}, ohne Zertifikat)")),
            ("Abstand zum Referenzwert", sci(max(m["gap"], 0.0))), ("Sweeps", str(m["sweeps"])), ("kleinster Eigenwert von Z(V)", sci(m["lam_min"])), ("Zertifikat (Z(V) PSD)", "ja" if m["certified"] else "nein"),
            ("Rang nach der Theorie p(p+1)/2 > n", str(m["theory_rank"])), ("sqrt(2n)", de(m["sqrt_rank"], 1)), ("Speicher von X / von V", f"{n * n * 8} / {n * m['rank'] * 8} Byte")]
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(100, 6, "Kennzahl", border=1)
    pdf.cell(100, 6, "Wert", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, val in rows:
        pdf.cell(100, 5.5, _s(k), border=1)
        pdf.cell(100, 5.5, _s(val), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kegel-niedrigrang-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
