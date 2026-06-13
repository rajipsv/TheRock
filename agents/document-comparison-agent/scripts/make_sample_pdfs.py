#!/usr/bin/env python3
"""Generate sample legacy/modernized policy PDFs for API/CLI testing."""

from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    raise SystemExit("Install fpdf2: pip install fpdf2")


LEGACY = """PRIVACY AND DATA PROTECTION POLICY (Legacy)
Version 1.0 - Effective January 2020

Section 1. Purpose
This policy defines how the organization collects and uses personal data.

Section 2. Personal Data
Personal data includes names, email addresses, and usage logs.
Data may be retained for up to 5 years unless otherwise required by law.

Section 3. Consent
Users must provide opt-in consent before marketing communications.
Implied consent is acceptable for essential service operations.

Section 4. Security
The organization implements reasonable technical safeguards.
Employees must report suspected breaches within 72 hours.

Section 5. Cross-border Transfers
International transfers are permitted to affiliated entities without additional safeguards.

Appendix A. Definitions
Controller means the entity determining purposes of processing.
"""


MODERNIZED = """PRIVACY AND DATA PROTECTION POLICY (Modernized)
Version 2.0 - Effective January 2024

Section 1. Purpose
This policy defines how the organization collects, processes, and protects personal data
in compliance with applicable privacy regulations.

Section 2. Personal Data
Personal data includes identifiers, contact details, device identifiers, and behavioral logs.
Data retention is limited to 3 years unless a longer period is legally required or documented.

Section 3. Consent
Explicit opt-in consent is required for marketing and profiling activities.
Essential processing relies on contractual necessity or legitimate interest with documented assessment.

Section 4. Security
The organization implements encryption, access controls, and annual security audits.
Suspected personal data breaches must be reported to the DPO within 24 hours.

Section 5. Cross-border Transfers
International transfers require Standard Contractual Clauses or equivalent approved mechanisms.
Transfer impact assessments are mandatory for high-risk destinations.

Section 6. Automated Decision-Making
Individuals have the right to request human review of solely automated decisions with legal effect.

Appendix A. Definitions
Controller and Processor roles are defined per GDPR Article 4.
Data Protection Officer contact details are published on the corporate website.
"""


def write_pdf(text: str, path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin
    for line in text.strip().split("\n"):
        if not line.strip():
            pdf.ln(4)
            continue
        pdf.multi_cell(effective_width, 6, line)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "samples"
    write_pdf(LEGACY, out / "legacy_policy.pdf")
    write_pdf(MODERNIZED, out / "modernized_policy.pdf")
    print(f"Created {out / 'legacy_policy.pdf'}")
    print(f"Created {out / 'modernized_policy.pdf'}")


if __name__ == "__main__":
    main()
