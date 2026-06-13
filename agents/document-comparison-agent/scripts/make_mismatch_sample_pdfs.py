#!/usr/bin/env python3
"""Generate mismatch-format sample PDFs (reordered sections, renamed headings, tables)."""

from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    raise SystemExit("Install fpdf2: pip install fpdf2")


LEGACY = """PRIVACY POLICY (Legacy Format)
Version 1.0 - January 2020

Section 1. Purpose
This policy defines how the organization collects and uses personal data.

Section 2. Personal Data
Personal data includes names, email addresses, and usage logs.
Data may be retained for up to 5 years unless otherwise required by law.

Section 3. Consent
Users must provide opt-in consent before marketing communications.

Section 4. Security
The organization implements reasonable technical safeguards.
Employees must report suspected breaches within 72 hours.

Section 5. Cross-border Transfers
International transfers are permitted to affiliated entities without additional safeguards.
"""


def write_text_pdf(text: str, path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    width = pdf.w - pdf.l_margin - pdf.r_margin
    for line in text.strip().split("\n"):
        if not line.strip():
            pdf.ln(4)
            continue
        pdf.multi_cell(width, 6, line)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def write_modernized_pdf(path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    width = pdf.w - pdf.l_margin - pdf.r_margin

    # Reordered + renamed headings vs legacy
    blocks = [
        "PRIVACY POLICY (Modern Format)\nVersion 2.0 - January 2024",
        "Art. 3 - Lawful Basis and Consent\n"
        "Explicit opt-in consent is required for marketing and profiling activities.",
        "Art. 1 - Scope and Purpose\n"
        "This policy defines how the organization collects, processes, and protects personal data.",
        "Art. 4 - Security and Breach Notification\n"
        "The organization implements encryption, access controls, and annual security audits.\n"
        "Suspected breaches must be reported to the DPO within 24 hours.",
        "Art. 2 - Categories of Personal Data\n"
        "Personal data includes identifiers, contact details, and behavioral logs.\n"
        "Data retention is limited to 3 years unless legally required.",
        "Art. 5 - International Transfers\n"
        "International transfers require Standard Contractual Clauses or equivalent mechanisms.",
    ]
    for block in blocks:
        for line in block.split("\n"):
            pdf.multi_cell(width, 6, line)
        pdf.ln(4)

    # Table page
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(width, 6, "Appendix - Data Processing Register")
    pdf.ln(6)
    pdf.set_font("Helvetica", size=10)
    headers = ["Processing Activity", "Legal Basis", "Retention"]
    rows = [
        ["Marketing email", "Consent", "2 years"],
        ["Payroll", "Contract", "7 years"],
        ["Web analytics", "Legitimate interest", "13 months"],
    ]
    col_w = width / 3
    for cell in headers:
        pdf.cell(col_w, 8, cell, border=1)
    pdf.ln()
    for row in rows:
        for cell in row:
            pdf.cell(col_w, 8, cell, border=1)
        pdf.ln()

    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "samples"
    write_text_pdf(LEGACY, out / "legacy_mismatch.pdf")
    write_modernized_pdf(out / "modernized_mismatch.pdf")
    print(f"Created {out / 'legacy_mismatch.pdf'}")
    print(f"Created {out / 'modernized_mismatch.pdf'}")


if __name__ == "__main__":
    main()
