from __future__ import annotations

# Generate synthetic and real fine-tuning data for WM guidance extraction across train/val/test splits.
import argparse
import json
import random
from pathlib import Path
from typing import Callable


def detect_project_root() -> Path:
    # Resolve the project root from this file's location.
    here = Path(__file__).resolve()
    if here.parent.name == "src":
        return here.parent.parent
    return here.parent


ROOT = detect_project_root()
OUT_DIR = ROOT / "data" / "training"
TRAIN_FILE = OUT_DIR / "guidance_train.jsonl"
VAL_FILE = OUT_DIR / "guidance_val.jsonl"
TEST_FILE = OUT_DIR / "guidance_test.jsonl"

DEFAULT_TRAIN_SIZE = 1600
DEFAULT_VAL_SIZE = 250
DEFAULT_SEED = 42

SYSTEM = (
    "Extract only explicitly stated financial guidance values from a WM earnings press release. "
    "Return valid JSON only. Use null for missing fields. Do not guess.\n"
    "Rules:\n"
    "- revenue_min/revenue_max: Total Company revenue guidance only. "
    "Ignore segment, collection, disposal, internal revenue growth, EBITDA, margin, or any non-total-company figure.\n"
    "- If the text shows both current guidance and prior/initial/previous guidance, extract the CURRENT guidance only.\n"
    "- revenue_unit / fcf_unit — determine from context in this order:\n"
    "  1. If the context contains a table header like '(in millions)' or '($ in millions)', bare dollar amounts such as '$25,550' are in millions -> unit='million'.\n"
    "  2. If the context contains '(in billions)' or '($ in billions)', bare dollar amounts are in billions -> unit='billion'.\n"
    "  3. If the number is followed by 'billion' -> unit='billion'.\n"
    "  4. If the number is followed by 'million' -> unit='million'.\n"
    "  5. If the number has a '%' sign -> unit='%'.\n"
    "  6. Otherwise copy digits as-is with unit=null.\n"
    "- Copy digits exactly as they appear; do not convert units.\n"
    "- fcf_min/fcf_max: use free cash flow guidance that INCLUDES sustainability investments. "
    "Ignore any figure labeled 'before sustainability investments' or 'excluding sustainability investments'.\n"
    "- Single-value guidance like 'approximately 10%' should be stored as min=max=10.\n"
    "- Keep min <= max.\n"
    "Examples:\n"
    "  '15.5% to 16.0%' -> revenue_min=15.5, revenue_max=16.0, revenue_unit='%'\n"
    "  '$2.325 billion to $2.425 billion' -> fcf_min=2.325, fcf_max=2.425, fcf_unit='billion'\n"
    "  '(in millions) Revenue: $25,550 - $25,800' -> revenue_min=25550, revenue_max=25800, revenue_unit='million'"
)


def make_example(section: str, fiscal_quarter: str, context: str, output: dict) -> dict:
    # Build a single training conversation dict from section, quarter, context, and expected output.
    user = f"Section title: {section}\nFiscal quarter: {fiscal_quarter}\nContext:\n{context}"
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM},
            {"from": "human", "value": user},
            {"from": "gpt", "value": json.dumps(output, separators=(",", ":"))},
        ]
    }


def j(rev_min, rev_max, rev_unit, fcf_min, fcf_max, fcf_unit):
    # Shorthand to build a guidance output dict.
    return {
        "revenue_min": rev_min,
        "revenue_max": rev_max,
        "revenue_unit": rev_unit,
        "fcf_min": fcf_min,
        "fcf_max": fcf_max,
        "fcf_unit": fcf_unit,
    }


REAL = [
    make_example(
        "OUTLOOK", "Q1",
        "Total Company revenue growth is expected to be 12.5% to 13%. Combined internal revenue growth from yield "
        "and volume in the collection and disposal business is expected to be 4.5% or greater. "
        "Adjusted operating EBITDA is expected to be between $4.875 billion and $4.975 billion in 2021. "
        "Free cash flow is projected to be between $2.325 billion and $2.425 billion in 2021.",
        j(12.5, 13.0, "%", 2.325, 2.425, "billion"),
    ),
    make_example(
        "REVISED 2021 OUTLOOK", "Q2",
        "Total Company revenue growth in 2021 is expected to be 15.5% to 16.0%. Combined internal revenue growth "
        "from yield and volume in the collection and disposal business is expected to be 5.5% or greater. "
        "Adjusted operating EBITDA is expected to be between $5.0 billion and $5.1 billion in 2021. "
        "Free cash flow is projected to be between $2.5 billion and $2.6 billion in 2021.",
        j(15.5, 16.0, "%", 2.5, 2.6, "billion"),
    ),
    make_example(
        "2021 OUTLOOK", "Q3",
        "Total Company revenue growth in 2021 is expected to be between 17% and 17.5%. Combined internal revenue "
        "growth from yield and volume in the collection and disposal business is expected to be about 6.5%. "
        "Adjusted operating EBITDA is expected to be between $5.0 billion and $5.1 billion in 2021. "
        "Free cash flow is projected to be between $2.5 billion and $2.6 billion in 2021.",
        j(17.0, 17.5, "%", 2.5, 2.6, "billion"),
    ),
    make_example(
        "2022 OUTLOOK", "Q4",
        "Revenue Growth Total Company revenue growth is expected to be 5.8% to 6.2%, which includes organic "
        "revenue growth from the collection and disposal business of approximately 6%. "
        "Adjusted operating EBITDA is expected to be in the range of $5.325 to $5.425 billion for the full year. "
        "Free cash flow excluding the acceleration of these targeted sustainability and automation focused capital "
        "investments is projected to be between $2.6 and $2.7 billion. "
        "Free cash flow is projected to be between $2.05 billion and $2.15 billion.",
        j(5.8, 6.2, "%", 2.05, 2.15, "billion"),
    ),
    make_example(
        "REVISED 2022 OUTLOOK", "Q2",
        "Total Company revenue growth in 2022 is expected to be approximately 10%, an increase of 400 basis "
        "points from the midpoint of prior guidance. "
        "Adjusted operating EBITDA is now expected to be between $5.5 and $5.6 billion in 2022, an increase "
        "of $175 million from prior guidance. "
        "Free cash flow is projected to exceed the upper end of the Company's previous guidance range of "
        "$2.6 to $2.7 billion excluding the targeted sustainability growth investments, or $2.05 to $2.15 "
        "billion including sustainability growth investments.",
        j(10.0, 10.0, "%", 2.05, 2.15, "billion"),
    ),
    make_example(
        "2023 OUTLOOK", "Q4",
        "Revenue Growth & Profitability Total Company revenue is expected to grow between 4% and 5.5%, which "
        "includes organic revenue growth from the collection and disposal business, excluding fuel, of "
        "approximately 5.5%. "
        "Free cash flow before these targeted sustainability and automation focused capital investments is "
        "projected to be between $2.6 and $2.7 billion. "
        "Free cash flow is projected to be between $1.5 and $1.6 billion including the sustainability growth investments.",
        j(4.0, 5.5, "%", 1.5, 1.6, "billion"),
    ),
    make_example(
        "REVISED 2023 OUTLOOK", "Q2",
        "Total Company revenue growth is expected to be between 3.25% and 4.25%, which is lower than prior "
        "expectations by 100 basis points at the midpoint. "
        "Free cash flow is projected to be between $1.675 and $1.775 billion, which is above prior expectations "
        "by $175 million at the midpoint. "
        "Free cash flow excluding targeted sustainability growth investments is expected to be between $2.575 "
        "and $2.675 billion, essentially in line with prior guidance.",
        j(3.25, 4.25, "%", 1.675, 1.775, "billion"),
    ),
    make_example(
        "2024 OUTLOOK", "Q4",
        "Revenue Growth & Profitability Total Company revenue is expected to grow between 6% and 7%. "
        "Adjusted operating EBITDA margin is expected to be in the range of 29.0% and 29.4%. "
        "Free cash flow before these targeted sustainability and automation focused capital investments is "
        "projected to be between $2.75 and $2.95 billion. "
        "Free cash flow is projected to be between $1.90 and $2.05 billion including the sustainability growth investments.",
        j(6.0, 7.0, "%", 1.90, 2.05, "billion"),
    ),
    make_example(
        "UPDATED 2024 OUTLOOK", "Q1",
        "Revenue: 5% - 5.75% Growth from Prior Year 6% - 7% Growth from Prior Year "
        "Adjusted Operating EBITDA $6.375 - $6.525 Billion: $6.275 - $6.425 Billion "
        "Free Cash Flow: $2.0 - $2.15 Billion | $1.9 - $2.05 Billion",
        j(5.0, 5.75, "%", 2.0, 2.15, "billion"),
    ),
    make_example(
        "2025 FINANCIAL OUTLOOK", "Q4",
        "WM's 2025 outlook includes a significant step change in revenue and earnings. "
        "Projected Results Projected Year-Over-Year Change at (in millions) Midpoint "
        "Revenue: $25,550 - $25,800 Growth of 16.4%, including 5.7% organic "
        "Free cash flow: $2,675 - $2,775 Growth of 17.6%",
        j(25550, 25800, "million", 2675, 2775, "million"),
    ),
    make_example(
        "2025 Outlook", "Q2",
        "The Company is confident in its ability to deliver upon its full-year outlook for adjusted operating "
        "EBITDA and is positioned to deliver free cash flow in excess of its initial target. "
        "Free cash flow is now projected to be between $2.8 and $2.9 billion, an increase of $125 million "
        "from the Company's initial guidance. "
        "Total Company revenue is now expected to be between $25.275 and $25.475 billion.",
        j(25.275, 25.475, "billion", 2.8, 2.9, "billion"),
    ),
    make_example(
        "2025 Outlook", "Q3",
        "The Company is confident in its ability to deliver upon its full-year outlook ranges for adjusted "
        "operating EBITDA and free cash flow. "
        "Free cash flow guidance remains between $2.8 and $2.9 billion. "
        "Total Company revenue is now expected to be approximately $25.275 billion, which is at the low end "
        "of the Company's prior guidance range.",
        j(25.275, 25.275, "billion", 2.8, 2.9, "billion"),
    ),
    make_example(
        "FINANCIAL OUTLOOK", "Q4",
        "Projected Results Projected Year-Over-Year Change at Midpoint ($ in millions) "
        "Revenue: $26,425 - $26,625 Growth of 5.2% "
        "Free cash flow: $3,750 - $3,850 Growth of 29.4%",
        j(26425, 26625, "million", 3750, 3850, "million"),
    ),
]


SECTION_TITLES = [
    "OUTLOOK",
    "REVISED OUTLOOK",
    "UPDATED OUTLOOK",
    "FINANCIAL OUTLOOK",
    "{year} OUTLOOK",
    "REVISED {year} OUTLOOK",
    "UPDATED {year} OUTLOOK",
    "{year} FINANCIAL OUTLOOK",
    "{year} EXPECTATIONS",
    "FINANCIAL EXPECTATIONS",
]

REV_PCT_PHRASES = [
    "Total Company revenue growth is expected to be {lo}% to {hi}%.",
    "Total Company revenue growth in {year} is expected to be {lo}% to {hi}%.",
    "Total Company revenue is expected to grow between {lo}% and {hi}%.",
    "Revenue growth is expected to be between {lo}% and {hi}% for the full year.",
    "The Company expects Total Company revenue growth of {lo}% to {hi}%.",
    "Full-year Total Company revenue growth is now expected to be {lo}% to {hi}%.",
    "Total Company revenue growth is projected in the range of {lo}% to {hi}%.",
    "Revenue | {lo}% - {hi}% growth from prior year.",
    "Revenue Growth: {lo}% - {hi}%.",
    "Total Company revenue growth is estimated at {lo}% to {hi}%.",
]

REV_SINGLE_PCT_PHRASES = [
    "Total Company revenue growth is expected to be approximately {x}%.",
    "The Company now expects Total Company revenue growth of about {x}%.",
    "Full-year Total Company revenue growth is projected to be roughly {x}%.",
]

REV_ABS_B_PHRASES = [
    "Total Company revenue is expected to be between ${lo} billion and ${hi} billion.",
    "Total Company revenue is now expected to be between ${lo} and ${hi} billion.",
    "Full-year revenue is projected to be ${lo} billion to ${hi} billion.",
    "Revenue: ${lo} - ${hi} billion.",
    "Total Company revenue outlook: ${lo} to ${hi} billion.",
    "Revenue | ${lo} - ${hi} billion.",
]

REV_SINGLE_B_PHRASES = [
    "Total Company revenue is expected to be approximately ${x} billion.",
    "Revenue is now expected to be about ${x} billion.",
    "The Company now expects total revenue of roughly ${x} billion.",
]

REV_ABS_M_PHRASES = [
    "Total Company revenue is expected to be between ${lo} and ${hi}.",
    "Revenue: ${lo} - ${hi}.",
    "Total Company revenue is projected to be ${lo} to ${hi}.",
    "Revenue | ${lo} - ${hi}.",
]

FCF_B_PHRASES = [
    "Free cash flow is projected to be between ${lo} billion and ${hi} billion.",
    "Free cash flow is projected to be between ${lo} and ${hi} billion.",
    "Adjusted free cash flow is expected to be between ${lo} and ${hi} billion.",
    "Free cash flow is expected to be in the range of ${lo} billion to ${hi} billion.",
    "The Company projects free cash flow of ${lo} to ${hi} billion.",
    "Free cash flow guidance remains between ${lo} and ${hi} billion.",
    "Free cash flow is now projected to be between ${lo} and ${hi} billion.",
    "Free Cash Flow: ${lo} - ${hi} billion.",
]

FCF_M_PHRASES = [
    "Free cash flow is projected to be between ${lo} and ${hi}.",
    "Free cash flow: ${lo} - ${hi}.",
    "Free cash flow is expected to be approximately ${lo} to ${hi}.",
    "Free Cash Flow | ${lo} - ${hi}.",
]

NOISE_PHRASES = [
    "Combined internal revenue growth from yield and volume in the collection and disposal business is expected to be {n}% or greater.",
    "Adjusted operating EBITDA margin is expected to be in the range of {n}% and {m}%, expanding {b} basis points at the midpoint.",
    "The Company's disciplined pricing programs and strong execution underpin these expectations.",
    "Core price is expected to be between {n}% and {m}%.",
    "Organic revenue growth from the collection and disposal business is expected to be approximately {n}%.",
    "Free cash flow before sustainability investments is projected to be between ${lo} and ${hi} billion.",
    "Free cash flow excluding the acceleration of these targeted investments is expected to be ${lo} to ${hi} billion.",
    "Rollover from acquisitions is expected to contribute ${n} million of revenue.",
    "Collection and disposal yield is expected to improve by {n}%.",
    "Average recycled commodity prices are assumed to remain near current levels.",
    "Adjusted operating EBITDA is expected to be between ${lo} billion and ${hi} billion.",
    "Net cash provided by operating activities is expected to be between ${lo} billion and ${hi} billion.",
]

TABLE_M_HEADERS = ["(in millions)", "($ in millions)", "(amounts in millions)"]
TABLE_B_HEADERS = ["(in billions)", "($ in billions)"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
YEARS = list(range(2021, 2028))
SEPARATORS = [" ", "\n", "\n\n", " | "]


class SynthBuilder:
    def __init__(self, seed: int):
        # Seed the random generator for reproducible synthetic data.
        self.rng = random.Random(seed)

    def choose(self, items):
        # Pick a random item from a list.
        return self.rng.choice(items)

    def section(self, year: int | None = None, revised_bias: bool = False) -> str:
        # Generate a random section title, optionally biased toward revised/updated variants.
        year = year or self.choose(YEARS)
        section = self.choose(SECTION_TITLES).format(year=year)
        if revised_bias and "REVISED" not in section and "UPDATED" not in section:
            section = self.choose([f"REVISED {year} OUTLOOK", f"UPDATED {year} OUTLOOK", f"{year} EXPECTATIONS"])
        return section

    def quarter(self) -> str:
        # Pick a random fiscal quarter.
        return self.choose(QUARTERS)

    def num(self, lo: float, hi: float, decimals: int = 2) -> float:
        # Generate a random float in [lo, hi] rounded to the given decimal places.
        return round(self.rng.uniform(lo, hi), decimals)

    def noise(self, k_min: int = 1, k_max: int = 3) -> str:
        # Generate 1–3 noise sentences to pad the context with irrelevant financial details.
        phrases = self.rng.sample(NOISE_PHRASES, k=self.rng.randint(k_min, k_max))
        parts = []
        for p in phrases:
            lo = round(self.rng.uniform(1.4, 4.2), 3)
            hi = round(lo + self.rng.uniform(0.05, 0.4), 3)
            n = round(self.rng.uniform(2.5, 10.0), 1)
            m = round(n + self.rng.uniform(0.2, 1.3), 1)
            b = self.choose([20, 30, 40, 50, 60, 75, 100])
            parts.append(p.format(lo=lo, hi=hi, n=n, m=m, b=b))
        return " ".join(parts)

    def join(self, *parts: str) -> str:
        # Join parts with a random separator.
        sep = self.choose(SEPARATORS)
        return sep.join([p for p in parts if p])

    def pct_rev_b_fcf(self) -> dict:
        # Generate a percentage revenue + billion FCF guidance example.
        year = self.choose(YEARS)
        rev_lo = self.num(1.5, 20.0, 2)
        rev_hi = round(rev_lo + self.rng.uniform(0.2, 2.0), 2)
        fcf_lo = self.num(1.1, 4.2, 3)
        fcf_hi = round(fcf_lo + self.rng.uniform(0.05, 0.25), 3)
        context = self.join(
            self.choose(REV_PCT_PHRASES).format(lo=rev_lo, hi=rev_hi, year=year),
            self.noise(),
            self.choose(FCF_B_PHRASES).format(lo=fcf_lo, hi=fcf_hi),
        )
        return make_example(self.section(year, revised_bias=True), self.quarter(), context, j(rev_lo, rev_hi, "%", fcf_lo, fcf_hi, "billion"))

    def abs_b_rev_b_fcf(self) -> dict:
        # Generate an absolute billion revenue + billion FCF guidance example.
        year = self.choose(YEARS)
        rev_lo = self.num(18.0, 29.0, 3)
        rev_hi = round(rev_lo + self.rng.uniform(0.08, 0.6), 3)
        fcf_lo = self.num(1.2, 4.5, 3)
        fcf_hi = round(fcf_lo + self.rng.uniform(0.05, 0.25), 3)
        context = self.join(
            self.choose(REV_ABS_B_PHRASES).format(lo=rev_lo, hi=rev_hi),
            self.noise(),
            self.choose(FCF_B_PHRASES).format(lo=fcf_lo, hi=fcf_hi),
        )
        return make_example(self.section(year), self.quarter(), context, j(rev_lo, rev_hi, "billion", fcf_lo, fcf_hi, "billion"))

    def table_millions(self) -> dict:
        # Generate a table-format guidance example with values in millions.
        year = self.choose(YEARS)
        rev_lo = self.rng.randint(17000, 29000)
        rev_hi = rev_lo + self.rng.randint(100, 600)
        fcf_lo = self.rng.randint(1400, 4300)
        fcf_hi = fcf_lo + self.rng.randint(50, 300)
        rev_growth = self.num(2.0, 20.0, 1)
        fcf_growth = self.num(3.0, 30.0, 1)
        header = self.choose(TABLE_M_HEADERS)
        context = self.join(
            f"Projected Results Projected Year-Over-Year Change at Midpoint {header}",
            f"Revenue: ${rev_lo:,} - ${rev_hi:,} Growth of {rev_growth}%",
            f"Free cash flow: ${fcf_lo:,} - ${fcf_hi:,} Growth of {fcf_growth}%",
            self.noise(1, 2),
        )
        return make_example(self.section(year), self.choose(["Q1", "Q4"]), context, j(rev_lo, rev_hi, "million", fcf_lo, fcf_hi, "million"))

    def table_billions(self) -> dict:
        # Generate a table-format guidance example with values in billions.
        year = self.choose(YEARS)
        rev_lo = self.num(22.0, 31.0, 3)
        rev_hi = round(rev_lo + self.rng.uniform(0.10, 0.60), 3)
        fcf_lo = self.num(1.8, 4.3, 3)
        fcf_hi = round(fcf_lo + self.rng.uniform(0.05, 0.25), 3)
        header = self.choose(TABLE_B_HEADERS)
        context = self.join(
            f"Projected Results at Midpoint {header}",
            f"Revenue: ${rev_lo} - ${rev_hi}",
            f"Free cash flow: ${fcf_lo} - ${fcf_hi}",
            self.noise(1, 2),
        )
        return make_example(self.section(year), self.choose(["Q1", "Q4"]), context, j(rev_lo, rev_hi, "billion", fcf_lo, fcf_hi, "billion"))

    def fcf_only(self) -> dict:
        # Generate an example where only FCF guidance is present and revenue is null.
        year = self.choose(YEARS)
        fcf_lo = self.num(1.3, 4.2, 3)
        fcf_hi = round(fcf_lo + self.rng.uniform(0.05, 0.20), 3)
        context = self.join(
            "The Company is confident in its ability to deliver upon its full-year outlook.",
            self.choose(FCF_B_PHRASES).format(lo=fcf_lo, hi=fcf_hi),
            self.noise(),
        )
        return make_example(f"{year} Outlook", self.choose(["Q2", "Q3"]), context, j(None, None, None, fcf_lo, fcf_hi, "billion"))

    def revenue_only(self) -> dict:
        # Generate an example where only revenue guidance is present and FCF is null.
        year = self.choose(YEARS)
        if self.rng.random() < 0.65:
            rev_lo = self.num(2.0, 18.0, 2)
            rev_hi = round(rev_lo + self.rng.uniform(0.2, 1.6), 2)
            rev_unit = "%"
            rev_phrase = self.choose(REV_PCT_PHRASES).format(lo=rev_lo, hi=rev_hi, year=year)
        else:
            rev_lo = self.num(20.0, 28.0, 3)
            rev_hi = round(rev_lo + self.rng.uniform(0.08, 0.5), 3)
            rev_unit = "billion"
            rev_phrase = self.choose(REV_ABS_B_PHRASES).format(lo=rev_lo, hi=rev_hi)
        context = self.join(
            rev_phrase,
            self.noise(),
            "The Company reiterated its cash generation framework without updating the full-year free cash flow target.",
        )
        return make_example(self.section(year), self.choose(["Q1", "Q2", "Q3"]), context, j(rev_lo, rev_hi, rev_unit, None, None, None))

    def single_value_revenue(self) -> dict:
        # Generate an example with a single-point revenue guidance (min == max).
        year = self.choose(YEARS)
        fcf_lo = self.num(1.4, 4.1, 3)
        fcf_hi = round(fcf_lo + self.rng.uniform(0.05, 0.20), 3)
        if self.rng.random() < 0.6:
            x = self.num(3.0, 12.0, 1)
            rev_phrase = self.choose(REV_SINGLE_PCT_PHRASES).format(x=x)
            out = j(x, x, "%", fcf_lo, fcf_hi, "billion")
        else:
            x = self.num(20.0, 28.0, 3)
            rev_phrase = self.choose(REV_SINGLE_B_PHRASES).format(x=x)
            out = j(x, x, "billion", fcf_lo, fcf_hi, "billion")
        context = self.join(rev_phrase, self.noise(), self.choose(FCF_B_PHRASES).format(lo=fcf_lo, hi=fcf_hi))
        return make_example(f"REVISED {year} OUTLOOK", self.choose(["Q2", "Q3"]), context, out)

    def prior_vs_current(self) -> dict:
        # Generate an example containing both prior and current guidance to test extraction of current only.
        year = self.choose(YEARS)
        use_pct = self.rng.random() < 0.6
        if use_pct:
            prior_lo = self.num(2.0, 10.0, 2)
            prior_hi = round(prior_lo + self.rng.uniform(0.2, 1.2), 2)
            curr_lo = round(prior_lo + self.rng.uniform(-1.0, 2.0), 2)
            curr_hi = round(curr_lo + self.rng.uniform(0.2, 1.5), 2)
            rev_current = self.choose(REV_PCT_PHRASES).format(lo=curr_lo, hi=curr_hi, year=year)
            rev_prior = f"Prior guidance contemplated revenue growth of {prior_lo}% to {prior_hi}%."
            rev_unit = "%"
            rev_lo, rev_hi = curr_lo, curr_hi
        else:
            prior_lo = self.num(20.0, 25.0, 3)
            prior_hi = round(prior_lo + self.rng.uniform(0.08, 0.3), 3)
            curr_lo = round(prior_lo + self.rng.uniform(-0.4, 0.8), 3)
            curr_hi = round(curr_lo + self.rng.uniform(0.08, 0.35), 3)
            rev_current = self.choose(REV_ABS_B_PHRASES).format(lo=curr_lo, hi=curr_hi)
            rev_prior = f"Previous guidance for Total Company revenue was ${prior_lo} to ${prior_hi} billion."
            rev_unit = "billion"
            rev_lo, rev_hi = curr_lo, curr_hi

        fcf_prior_lo = self.num(1.5, 3.4, 3)
        fcf_prior_hi = round(fcf_prior_lo + self.rng.uniform(0.05, 0.15), 3)
        fcf_curr_lo = round(fcf_prior_lo + self.rng.uniform(-0.15, 0.30), 3)
        fcf_curr_hi = round(fcf_curr_lo + self.rng.uniform(0.05, 0.20), 3)

        pieces = [
            f"Updated outlook for {year}:",
            rev_current,
            self.choose(FCF_B_PHRASES).format(lo=fcf_curr_lo, hi=fcf_curr_hi),
            rev_prior,
            f"Previous free cash flow guidance was ${fcf_prior_lo} to ${fcf_prior_hi} billion.",
            self.noise(1, 2),
        ]
        self.rng.shuffle(pieces)
        context = self.join(*pieces)
        return make_example(self.section(year, revised_bias=True), self.choose(["Q1", "Q2", "Q3"]), context, j(rev_lo, rev_hi, rev_unit, fcf_curr_lo, fcf_curr_hi, "billion"))

    def sustainability_pair(self) -> dict:
        # Generate an example with both ex-sustainability and including-sustainability FCF guidance.
        year = self.choose(YEARS)
        rev_lo = self.num(3.0, 12.0, 2)
        rev_hi = round(rev_lo + self.rng.uniform(0.2, 1.2), 2)
        excl_lo = self.num(2.2, 4.0, 3)
        excl_hi = round(excl_lo + self.rng.uniform(0.05, 0.20), 3)
        incl_lo = round(excl_lo - self.rng.uniform(0.4, 1.2), 3)
        incl_hi = round(incl_lo + self.rng.uniform(0.05, 0.20), 3)
        context = self.join(
            self.choose(REV_PCT_PHRASES).format(lo=rev_lo, hi=rev_hi, year=year),
            f"Free cash flow before sustainability investments is projected to be between ${excl_lo} and ${excl_hi} billion.",
            f"Free cash flow is projected to be between ${incl_lo} and ${incl_hi} billion including sustainability investments.",
            self.noise(1, 2),
        )
        return make_example(self.section(year), self.choose(["Q2", "Q4"]), context, j(rev_lo, rev_hi, "%", incl_lo, incl_hi, "billion"))

    def multiline_bullets(self) -> dict:
        # Generate a bullet-list or table formatted guidance example.
        year = self.choose(YEARS)
        use_million = self.rng.random() < 0.4
        if use_million:
            rev_lo = self.rng.randint(19000, 28500)
            rev_hi = rev_lo + self.rng.randint(100, 400)
            fcf_lo = self.rng.randint(1600, 4200)
            fcf_hi = fcf_lo + self.rng.randint(50, 200)
            rev_line = self.choose(REV_ABS_M_PHRASES).format(lo=f"{rev_lo:,}", hi=f"{rev_hi:,}")
            fcf_line = self.choose(FCF_M_PHRASES).format(lo=f"{fcf_lo:,}", hi=f"{fcf_hi:,}")
            unit_rev = unit_fcf = "million"
            header = self.choose(TABLE_M_HEADERS)
        else:
            rev_lo = self.num(20.0, 29.0, 3)
            rev_hi = round(rev_lo + self.rng.uniform(0.08, 0.45), 3)
            fcf_lo = self.num(1.5, 4.3, 3)
            fcf_hi = round(fcf_lo + self.rng.uniform(0.05, 0.20), 3)
            rev_line = self.choose(REV_ABS_B_PHRASES).format(lo=rev_lo, hi=rev_hi)
            fcf_line = self.choose(FCF_B_PHRASES).format(lo=fcf_lo, hi=fcf_hi)
            unit_rev = unit_fcf = "billion"
            header = self.choose(TABLE_B_HEADERS)
        context = (
            f"Financial Outlook\n"
            f"Projected Results {header}\n"
            f"- {rev_line}\n"
            f"- {fcf_line}\n"
            f"- {self.noise(1, 2)}"
        )
        return make_example(self.section(year), self.choose(QUARTERS), context, j(rev_lo, rev_hi, unit_rev, fcf_lo, fcf_hi, unit_fcf))

    def null_case(self) -> dict:
        # Generate an example where no guidance values are present (all nulls expected).
        year = self.choose(YEARS)
        context = self.join(
            "The Company reaffirmed its strategic priorities and capital allocation framework.",
            self.noise(2, 3),
            "Management did not provide an updated full-year revenue or free cash flow target in this section.",
        )
        return make_example(self.section(year), self.choose(["Q1", "Q2", "Q3"]), context, j(None, None, None, None, None, None))


VARIANT_WEIGHTS: list[tuple[str, int]] = [
    ("pct_rev_b_fcf", 24),
    ("abs_b_rev_b_fcf", 16),
    ("table_millions", 12),
    ("table_billions", 8),
    ("fcf_only", 8),
    ("revenue_only", 8),
    ("single_value_revenue", 8),
    ("prior_vs_current", 8),
    ("sustainability_pair", 5),
    ("multiline_bullets", 7),
    ("null_case", 4),
]


def build_synthetic_examples(total: int, seed: int) -> list[dict]:
    # Generate and deduplicate a target number of synthetic training examples.
    builder = SynthBuilder(seed)
    variant_names = [name for name, _ in VARIANT_WEIGHTS]
    weights = [w for _, w in VARIANT_WEIGHTS]
    examples: list[dict] = []
    seen = set()
    max_attempts = total * 20

    attempts = 0
    while len(examples) < total and attempts < max_attempts:
        attempts += 1
        method_name = builder.rng.choices(variant_names, weights=weights, k=1)[0]
        method: Callable[[], dict] = getattr(builder, method_name)
        ex = method()
        key = ex["conversations"][1]["value"]
        if key in seen:
            continue
        seen.add(key)
        examples.append(ex)

    if len(examples) < total:
        raise RuntimeError(f"Could not generate enough unique synthetic examples: wanted {total}, got {len(examples)}")

    builder.rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    # Write a list of dicts as newline-delimited JSON.
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    # Parse CLI args and generate train/val/test splits to JSONL files.
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.train_size <= 0 or args.val_size <= 0:
        raise ValueError("train-size and val-size must both be positive")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    synthetic = build_synthetic_examples(args.train_size + args.val_size, seed=args.seed)
    train_rows = synthetic[: args.train_size]
    val_rows = synthetic[args.train_size :]
    test_rows = REAL

    write_jsonl(TRAIN_FILE, train_rows)
    write_jsonl(VAL_FILE, val_rows)
    write_jsonl(TEST_FILE, test_rows)

    print(f"Train -> {TRAIN_FILE} ({len(train_rows)} synthetic examples)")
    print(f"Val   -> {VAL_FILE} ({len(val_rows)} synthetic examples)")
    print(f"Test  -> {TEST_FILE} ({len(test_rows)} real ground-truth examples)")


if __name__ == "__main__":
    main()
