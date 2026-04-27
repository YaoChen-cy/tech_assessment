# Download WM 10-K and 10-Q SEC filings as plain text via edgartools.
from __future__ import annotations

import time
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "data" / "raw" / "filings"

IDENTITY   = "yao.chen5@mail.mcgill.ca"
TICKER     = "WM"
YEAR_START = 2021
YEAR_END   = 2025


def _quarter_from_period(period_str: str) -> int:
    # Map a period-of-report date string to its fiscal quarter number (1–4).
    import pandas as pd
    return {3: 1, 6: 2, 9: 3, 12: 4}[pd.Timestamp(period_str).month]


def _filing_to_text(filing) -> str | None:
    # Extract full document text from a filing object, trying multiple strategies in order.
    import re

    def _clean(raw: str) -> str:
        # Strip HTML tags and normalize whitespace.
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"\s{3,}", "\n\n", text)
        return text.strip()

    try:
        doc = filing.obj()

        # 1. direct .text attribute (works for 10-K wrappers)
        if hasattr(doc, "text") and isinstance(doc.text, str) and doc.text.strip():
            return _clean(doc.text)

        # 2. edgartools TenQ / TenK expose markdown()
        if hasattr(doc, "markdown"):
            md = doc.markdown()
            if md and md.strip():
                return md.strip()

        # 3. iterate named items (10-Q has item1, item2 …)
        parts = []
        for attr in sorted(dir(doc)):
            if re.match(r"^item\d", attr):
                try:
                    val = getattr(doc, attr)
                    if callable(val):
                        val = val()
                    if isinstance(val, str) and val.strip():
                        parts.append(val.strip())
                except Exception:
                    pass
        if parts:
            return "\n\n".join(parts)

        # 4. fall back to raw primary document HTML
        primary = filing.primary_document
        if primary:
            html = primary.download()
            if html:
                return _clean(html if isinstance(html, str) else html.decode("utf-8", errors="ignore"))

        return None
    except Exception as e:
        print(f"    [WARN] text extraction failed: {e}")
        return None


class FilingCollector:
    def __init__(
        self,
        ticker: str     = TICKER,
        year_start: int = YEAR_START,
        year_end: int   = YEAR_END,
        out_dir: Path   = OUT_DIR,
        sleep: float    = 0.5,
    ):
        # Store collection parameters and output directory.
        self.ticker     = ticker
        self.year_start = year_start
        self.year_end   = year_end
        self.out_dir    = Path(out_dir)
        self.sleep      = sleep

    def _save(self, text: str, form: str, year: int, quarter: int) -> Path:
        # Write filing text to disk at <out_dir>/<form>/<year>Q<quarter>.txt.
        dest = self.out_dir / form / f"{year}Q{quarter}.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return dest

    def _collect_10q(self, company) -> int:
        # Fetch and save all 10-Q filings within the configured year range.
        import pandas as pd
        saved = 0
        print("Fetching 10-Q filings...")
        for f in list(company.get_filings(form="10-Q")):
            if not f.period_of_report:
                continue
            period = pd.Timestamp(f.period_of_report)
            if not (self.year_start <= period.year <= self.year_end):
                continue
            if period.month == 12:   # skip Dec 10-Qs (rare edge case)
                continue
            qtr = _quarter_from_period(f.period_of_report)
            label = f"{period.year} Q{qtr}"
            dest  = self.out_dir / "10-Q" / f"{period.year}Q{qtr}.txt"
            if dest.exists():
                print(f"  10-Q  {label}  [skip — already exists]")
                continue
            print(f"  10-Q  {label}  {f.period_of_report} ...", end=" ", flush=True)
            text = _filing_to_text(f)
            if text:
                path = self._save(text, "10-Q", period.year, qtr)
                print(f"saved ({len(text):,} chars → {path.name})")
                saved += 1
            else:
                print("SKIP (no text)")
            time.sleep(self.sleep)
        return saved

    def _collect_10k(self, company) -> int:
        # Fetch and save all 10-K filings within the configured year range.
        import pandas as pd
        saved = 0
        print("\nFetching 10-K filings...")
        for f in list(company.get_filings(form="10-K", amendments=False)):
            if not f.period_of_report:
                continue
            fy = pd.Timestamp(f.period_of_report).year
            if not (self.year_start <= fy <= self.year_end):
                continue
            dest = self.out_dir / "10-K" / f"{fy}Q4.txt"
            if dest.exists():
                print(f"  10-K  {fy}  [skip — already exists]")
                continue
            print(f"  10-K  {fy}  {f.period_of_report} ...", end=" ", flush=True)
            text = _filing_to_text(f)
            if text:
                path = self._save(text, "10-K", fy, 4)
                print(f"saved ({len(text):,} chars → {path.name})")
                saved += 1
            else:
                print("SKIP (no text)")
            time.sleep(self.sleep)
        return saved

    def run(self) -> dict[str, list[Path]]:
        # Fetch and save 10-Q and 10-K filings; return saved paths grouped by form.
        from edgar import Company, set_identity
        set_identity(IDENTITY)
        company = Company(self.ticker)

        print(f"=== FilingCollector  {self.ticker}  {self.year_start}Q1–{self.year_end}Q4 ===")
        print(f"    Output → {self.out_dir}\n")

        n10q = self._collect_10q(company)
        n10k = self._collect_10k(company)

        saved_10q = sorted((self.out_dir / "10-Q").glob("*.txt")) if (self.out_dir / "10-Q").exists() else []
        saved_10k = sorted((self.out_dir / "10-K").glob("*.txt")) if (self.out_dir / "10-K").exists() else []

        print(f"\n=== Done ===")
        print(f"  10-Q: {n10q} new files  ({len(saved_10q)} total)")
        print(f"  10-K: {n10k} new files  ({len(saved_10k)} total)")
        return {"10-Q": saved_10q, "10-K": saved_10k}


if __name__ == "__main__":
    FilingCollector().run()
