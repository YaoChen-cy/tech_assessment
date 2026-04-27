# Extract quarterly revenue and free cash flow from WM SEC 10-Q and 10-K filings via edgartools.
import time
import warnings
import logging
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)


def _to_millions(value: float | None) -> float | None:
    # Convert a raw dollar value to USD millions, rounded to 1 decimal.
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value) / 1_000_000, 1)


def _period_to_quarter(period_str: str) -> int:
    # Map a period-of-report date string to its fiscal quarter number (1–4).
    return {3: 1, 6: 2, 9: 3, 12: 4}[pd.Timestamp(period_str).month]


class FinancialExtractor:
    def __init__(
        self,
        company,                   # edgar.Company instance, already set_identity'd
        year_start: int = 2021,
        year_end:   int = 2025,
        output_path: str | Path | None = None,
    ):
        # Store company handle, year range, and optional output path.
        self.company     = company
        self.year_start  = year_start
        self.year_end    = year_end
        self.output_path = Path(output_path) if output_path else None

    def _extract_10q(self, filing) -> dict | None:
        # Extract revenue and YTD FCF from a single 10-Q filing.
        period = pd.Timestamp(filing.period_of_report)
        try:
            fin = filing.obj().financials
            return {
                "fiscal_year":    period.year,
                "fiscal_quarter": _period_to_quarter(filing.period_of_report),
                "revenue":        fin.get_revenue(),
                "fcf_ytd":        fin.get_free_cash_flow(),
            }
        except Exception as e:
            print(f"    [WARN] {filing.period_of_report}: {e}")
            return None

    def _extract_10k(self, filing) -> dict | None:
        # Extract full-year revenue and FCF from a single 10-K filing.
        fy = pd.Timestamp(filing.period_of_report).year
        try:
            fin = filing.obj().financials
            return {
                "fiscal_year": fy,
                "revenue_fy":  fin.get_revenue(),
                "fcf_fy":      fin.get_free_cash_flow(),
            }
        except Exception as e:
            print(f"    [WARN] {filing.period_of_report}: {e}")
            return None

    def _collect_10q(self) -> dict[tuple[int, int], dict]:
        # Fetch all 10-Q filings and return a map of (year, quarter) → extracted data.
        print("Fetching 10-Q filings...")
        q_map: dict[tuple[int, int], dict] = {}
        for f in list(self.company.get_filings(form="10-Q")):
            if not f.period_of_report:
                continue
            period = pd.Timestamp(f.period_of_report)
            if not (self.year_start <= period.year <= self.year_end):
                continue
            if period.month == 12:   # Q4 covered by 10-K
                continue
            fq = _period_to_quarter(f.period_of_report)
            print(f"  10-Q  {period.year} Q{fq}  {f.period_of_report} ...", end=" ", flush=True)
            rec = self._extract_10q(f)
            if rec:
                q_map[(period.year, fq)] = rec
                rev = f"{rec['revenue']/1e6:.0f}M" if rec["revenue"] else "None"
                fcf = f"{rec['fcf_ytd']/1e6:.0f}M ytd" if rec["fcf_ytd"] else "None"
                print(f"rev={rev}  fcf={fcf}")
            else:
                print("SKIP")
            time.sleep(0.4)
        return q_map

    def _collect_10k(self) -> dict[int, dict]:
        # Fetch all 10-K filings and return a map of year → extracted data.
        print("\nFetching 10-K filings...")
        k_map: dict[int, dict] = {}
        for f in list(self.company.get_filings(form="10-K", amendments=False)):
            if not f.period_of_report:
                continue
            fy = pd.Timestamp(f.period_of_report).year
            if not (self.year_start <= fy <= self.year_end):
                continue
            print(f"  10-K  {fy}  {f.period_of_report} ...", end=" ", flush=True)
            rec = self._extract_10k(f)
            if rec:
                k_map[fy] = rec
                rev = f"{rec['revenue_fy']/1e6:.0f}M" if rec["revenue_fy"] else "None"
                fcf = f"{rec['fcf_fy']/1e6:.0f}M" if rec["fcf_fy"] else "None"
                print(f"rev={rev}  fcf={fcf}")
            else:
                print("SKIP")
            time.sleep(0.4)
        return k_map

    def _assemble(
        self,
        q_map: dict[tuple[int, int], dict],
        k_map: dict[int, dict],
    ) -> pd.DataFrame:
        # Combine quarterly and annual filing data into a per-quarter YTD revenue and FCF table.
        rows = []
        for year in range(self.year_start, self.year_end + 1):
            fy_rec     = k_map.get(year, {})
            revenue_fy = fy_rec.get("revenue_fy")
            fcf_fy     = fy_rec.get("fcf_fy")
            rev_ytd_acc = 0.0  # accumulate quarterly revenues for YTD

            for qtr in range(1, 5):
                q_rec = q_map.get((year, qtr))

                if qtr in (1, 2, 3):
                    if not q_rec:
                        print(f"  [MISSING] {year} Q{qtr}")
                        rows.append({"fiscal_year": year, "fiscal_quarter": qtr,
                                     "revenue_ytd_M": None, "fcf_ytd_M": None})
                        continue
                    # revenue: quarterly from 10-Q → accumulate for YTD
                    rev_q = q_rec["revenue"] or 0
                    rev_ytd_acc += rev_q
                    revenue_ytd_M = _to_millions(rev_ytd_acc)
                    # fcf: already YTD-cumulative in 10-Q
                    fcf_ytd_M = _to_millions(q_rec["fcf_ytd"])

                else:  # Q4 — use full-year values from 10-K
                    if not fy_rec:
                        print(f"  [MISSING] {year} Q4 (no 10-K)")
                        rows.append({"fiscal_year": year, "fiscal_quarter": 4,
                                     "revenue_ytd_M": None, "fcf_ytd_M": None})
                        continue
                    revenue_ytd_M = _to_millions(revenue_fy)
                    fcf_ytd_M     = _to_millions(fcf_fy)

                rows.append({"fiscal_year": year, "fiscal_quarter": qtr,
                             "revenue_ytd_M": revenue_ytd_M, "fcf_ytd_M": fcf_ytd_M})

        df = pd.DataFrame(rows, columns=["fiscal_year", "fiscal_quarter",
                                         "revenue_ytd_M", "fcf_ytd_M"])
        df["fiscal_year"]    = df["fiscal_year"].astype("Int64")
        df["fiscal_quarter"] = df["fiscal_quarter"].astype("Int64")
        return df

    def run(self) -> pd.DataFrame:
        # Run the full extraction pipeline and optionally save results to CSV.
        print(f"=== FinancialExtractor  {self.year_start}Q1 – {self.year_end}Q4 (USD millions) ===\n")
        q_map = self._collect_10q()
        k_map = self._collect_10k()
        df    = self._assemble(q_map, k_map)

        print("\n=== Results ===")
        print(df.to_string(index=False))

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.output_path, index=False)
            print(f"\nSaved → {self.output_path}")

        return df


if __name__ == "__main__":
    from edgar import Company, set_identity
    set_identity("yao.chen5@mail.mcgill.ca")
    c = Company("WM")
    extractor = FinancialExtractor(
        c,
        year_start=2021,
        year_end=2025,
        output_path=Path(__file__).parent.parent / "data" / "interim" / "wm_financials.csv",
    )
    extractor.run()
