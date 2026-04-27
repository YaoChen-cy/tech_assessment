# Extract structured revenue and FCF guidance from WM earnings press releases using a local Ollama LLM.
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel, ValidationError, field_validator

ROOT = Path(__file__).parent.parent
INPUT_CSV = ROOT / "data" / "processed" / "extract_rev_fcf.csv"
OUTPUT_CSV = ROOT / "data" / "interim" / "wm_guidance_local.csv"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

DEFAULT_MODEL = "llama3.2"
MAX_RETRIES = 1
REQUEST_TIMEOUT = 300

PREFILTER_CONTEXT = True
MAX_CONTEXT_CHARS = 1800

KEYWORDS = [
    "revenue",
    "free cash flow",
    "cash flow",
    "fcf",
    "guidance",
    "outlook",
    "current expectations",
    "original expectations",
    "prior guidance",
    "lowered",
    "reaffirmed",
    "expect",
    "projected",
    "estimated",
    "range",
    "%",
    "$",
    "billion",
    "million",
]


def ns_to_s(ns: int | float | None) -> float | None:
    # Convert nanoseconds to seconds, rounded to 3 decimal places.
    if ns is None:
        return None
    return round(float(ns) / 1_000_000_000, 3)


def _parse_numeric(v: object) -> float | None:
    # Accept plain floats or strings like '$4.975 billion' / '12.5%' and return a float.
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = v.strip()
        cleaned = cleaned.replace(",", "")
        cleaned = re.sub(r"[$%]", "", cleaned)
        cleaned = re.sub(r"(?i)\b(billion|million)\b", "", cleaned).strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _split_sentences(text: str) -> list[str]:
    # Split press-release prose into sentence/block fragments on punctuation boundaries.
    if not text:
        return []

    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[\.\?!;])\s+(?=[A-Z0-9])", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return parts


def _prefilter_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    # Keep only keyword-matching sentences to reduce prompt tokens; fall back to start of text if no hits.
    text = str(text or "").strip()
    if not text:
        return ""

    sentences = _split_sentences(text)
    if not sentences:
        return text[:max_chars]

    kept: list[str] = []
    lowered_keywords = [k.lower() for k in KEYWORDS]

    for sent in sentences:
        s = sent.lower()
        if any(k in s for k in lowered_keywords):
            kept.append(sent)

    if not kept:
        return text[:max_chars]

    deduped: list[str] = []
    seen = set()
    for sent in kept:
        key = sent.strip().lower()
        if key not in seen:
            deduped.append(sent)
            seen.add(key)

    out = " ".join(deduped)
    return out[:max_chars]


class OutlookGuidance(BaseModel):
    revenue_min: float | None = None
    revenue_max: float | None = None
    revenue_unit: str | None = None   # "%" | "$B" | "$M"
    fcf_min: float | None = None
    fcf_max: float | None = None
    fcf_unit: str | None = None       # "%" | "$B" | "$M"

    @field_validator("revenue_min", "revenue_max", "fcf_min", "fcf_max", mode="before")
    @classmethod
    def coerce_numeric(cls, v: object) -> float | None:
        return _parse_numeric(v)


OLLAMA_FORMAT_SCHEMA: dict[str, Any] = OutlookGuidance.model_json_schema()

SYSTEM_PROMPT = (
    "Extract only explicitly stated financial guidance values from a WM earnings press release. "
    "Return valid JSON only. Use null for missing fields. Do not guess.\n"
    "Rules:\n"
    "- revenue_min/revenue_max: Total Company revenue guidance range only. "
    "Ignore segment, collection, disposal, or internal revenue growth figures.\n"
    "- revenue_unit / fcf_unit — determine from the context in this order:\n"
    "  1. If the context contains a table header like '(in millions)' or '($ in millions)', "
    "bare dollar amounts such as '$25,550' are in millions → unit='million'.\n"
    "  2. If the context contains '(in billions)' or '($ in billions)', bare amounts are in billions → unit='billion'.\n"
    "  3. If the number is followed by 'billion' → unit='billion'.\n"
    "  4. If the number is followed by 'million' → unit='million'.\n"
    "  5. If the number has a '%' sign → unit='%'.\n"
    "  6. Otherwise copy digits as-is with unit=null.\n"
    "- Copy digits exactly as they appear — no conversion.\n"
    "- fcf_min/fcf_max: free cash flow guidance range. "
    "Use the figure that INCLUDES sustainability investments; ignore any figure labeled 'before sustainability investments'.\n"
    "- revenue_min <= revenue_max and fcf_min <= fcf_max (smaller number is always min).\n"
    "Examples:\n"
    "  '15.5% to 16.0%' → revenue_min=15.5, revenue_max=16.0, revenue_unit='%'\n"
    "  '$2.325 billion to $2.425 billion' → fcf_min=2.325, fcf_max=2.425, fcf_unit='billion'\n"
    "  '(in millions) Revenue: $25,550 - $25,800' → revenue_min=25550, revenue_max=25800, revenue_unit='million'"
)

USER_TEMPLATE = """Section title: {section}
Fiscal quarter: {fiscal_quarter}
Context:
{context}
"""


def _call_ollama(user_text: str, model: str) -> tuple[str, dict[str, Any]]:
    # Post a single chat request to the Ollama API and return the response content and metadata.
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "format": OLLAMA_FORMAT_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "options": {
            "temperature": 0.0,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content, data


def _extract_json(raw: str) -> dict[str, Any]:
    # Strip markdown fences and extract the first JSON object from a raw model response string.
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group()

    return json.loads(cleaned)


_UNIT_MAP = {
    "billion": ("$B", 1.0),
    "billions": ("$B", 1.0),
    "$b": ("$B", 1.0),
    "million": ("$B", 1e-3),
    "millions": ("$B", 1e-3),
    "$m": ("$B", 1e-3),
    "%": ("%", 1.0),
    "percent": ("%", 1.0),
}


def _normalize_unit(
    min_val: float | None,
    max_val: float | None,
    unit_raw: str | None,
) -> tuple[float | None, float | None, str | None]:
    # Normalise a raw unit label and scale the associated values to canonical ($B or %) form.
    if unit_raw is None:
        return min_val, max_val, None
    key = unit_raw.strip().lower()
    canonical, factor = _UNIT_MAP.get(key, (unit_raw, 1.0))
    if factor != 1.0:
        min_val = round(min_val * factor, 6) if min_val is not None else None
        max_val = round(max_val * factor, 6) if max_val is not None else None
    return min_val, max_val, canonical


def _postprocess(g: OutlookGuidance) -> OutlookGuidance:
    # Normalise units and enforce min <= max on an extracted guidance record.
    data = g.model_dump()

    for mn, mx, u in [
        ("revenue_min", "revenue_max", "revenue_unit"),
        ("fcf_min", "fcf_max", "fcf_unit"),
    ]:
        data[mn], data[mx], data[u] = _normalize_unit(data[mn], data[mx], data[u])

    for mn, mx in [("revenue_min", "revenue_max"), ("fcf_min", "fcf_max")]:
        x, y = data[mn], data[mx]
        if x is not None and y is not None and x > y:
            data[mn], data[mx] = y, x

    return OutlookGuidance(**data)


class LocalGuidanceExtractor:
    def __init__(
        self,
        input_path: str | Path = INPUT_CSV,
        output_path: str | Path = OUTPUT_CSV,
        model: str = DEFAULT_MODEL,
        prefilter_context: bool = PREFILTER_CONTEXT,
        max_context_chars: int = MAX_CONTEXT_CHARS,
    ):
        # Store paths, model name, and context-filtering settings.
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.model = model
        self.prefilter_context = prefilter_context
        self.max_context_chars = max_context_chars

    def _check_ollama(self) -> None:
        # Verify Ollama is running and the requested model is available, then warm it up.
        try:
            r = requests.get(OLLAMA_TAGS_URL, timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                print(f"[WARN] Model '{self.model}' not found locally. Available: {models}")
                print(f"       Run: ollama pull {self.model}")
        except requests.ConnectionError as e:
            raise RuntimeError("Ollama is not running. Start it with: ollama serve") from e

        print(f"Warming up {self.model} ...", end=" ", flush=True)
        requests.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "think": False,
                "keep_alive": -1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        print("ready")

    def _build_user_text(
        self,
        section: str,
        context: str,
        fiscal_quarter: str = "",
    ) -> tuple[str, str]:
        # Apply context prefilter and format the user prompt for a single row.
        context_in = str(context or "")
        context_used = (
            _prefilter_context(context_in, self.max_context_chars)
            if self.prefilter_context
            else context_in[: self.max_context_chars]
        )

        user_text = USER_TEMPLATE.format(
            section=str(section or ""),
            fiscal_quarter=str(fiscal_quarter or ""),
            context=context_used,
        )
        return user_text, context_used

    def _extract_one(
        self,
        section: str,
        context: str,
        fiscal_quarter: str = "",
    ) -> tuple[OutlookGuidance, bool, dict[str, Any]]:
        # Run extraction with retries for one (section, context) pair; return guidance, success flag, and stats.
        user_text, context_used = self._build_user_text(
            section=section,
            context=context,
            fiscal_quarter=fiscal_quarter,
        )

        last_error: Exception | None = None
        last_meta: dict[str, Any] = {}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw, meta = _call_ollama(user_text, self.model)
                last_meta = meta
                data = _extract_json(raw)
                guidance = OutlookGuidance(**data)
                guidance = _postprocess(guidance)

                stats = {
                    "chars_sent": len(context_used),
                    "prompt_eval_count": meta.get("prompt_eval_count"),
                    "prompt_eval_s": ns_to_s(meta.get("prompt_eval_duration")),
                    "eval_count": meta.get("eval_count"),
                    "eval_s": ns_to_s(meta.get("eval_duration")),
                    "total_s": ns_to_s(meta.get("total_duration")),
                    "load_s": ns_to_s(meta.get("load_duration")),
                }
                return guidance, True, stats

            except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(0.5)

        stats = {
            "chars_sent": len(context_used),
            "prompt_eval_count": last_meta.get("prompt_eval_count"),
            "prompt_eval_s": ns_to_s(last_meta.get("prompt_eval_duration")),
            "eval_count": last_meta.get("eval_count"),
            "eval_s": ns_to_s(last_meta.get("eval_duration")),
            "total_s": ns_to_s(last_meta.get("total_duration")),
            "load_s": ns_to_s(last_meta.get("load_duration")),
            "error": str(last_error) if last_error else None,
        }
        return OutlookGuidance(), False, stats

    def run(self, limit: int | None = None) -> pd.DataFrame:
        # Process all rows in the input CSV and save structured guidance to the output CSV.
        self._check_ollama()

        df = pd.read_csv(self.input_path)
        if limit is not None:
            df = df.head(limit).copy()

        print(f"=== LocalGuidanceExtractor model={self.model} ===")
        print(f"Rows to process: {len(df)}")
        print(f"Prefilter context: {self.prefilter_context}")
        print(f"Max context chars: {self.max_context_chars}\n")

        results: list[dict[str, Any]] = []

        start_all = time.time()

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            fy = row.get("fiscal_year", None)
            fq = str(row.get("fiscal_quarter", ""))
            rd = row.get("release_date", None)
            file_name = row.get("file", None)
            section = str(row.get("section", ""))
            context = str(row.get("context", ""))

            print(f"[{i}/{len(df)}] {fy} {fq} [{section[:28]}] ... ", end="", flush=True)

            t0 = time.time()
            guidance, ok, stats = self._extract_one(
                section=section,
                context=context,
                fiscal_quarter=fq,
            )
            row_s = round(time.time() - t0, 3)

            rec = {
                "fiscal_year": fy,
                "fiscal_quarter": fq,
                "release_date": rd,
                "file": file_name,
                "section": section,
                **guidance.model_dump(),
                "parse_success": ok,
                "chars_sent": stats.get("chars_sent"),
                "prompt_eval_count": stats.get("prompt_eval_count"),
                "prompt_eval_s": stats.get("prompt_eval_s"),
                "eval_count": stats.get("eval_count"),
                "eval_s": stats.get("eval_s"),
                "total_s": stats.get("total_s"),
                "load_s": stats.get("load_s"),
                "row_runtime_s": row_s,
                "error": stats.get("error"),
            }
            results.append(rec)

            status = "ok" if ok else "FAILED"
            rev = (
                f"{guidance.revenue_min}–{guidance.revenue_max}{guidance.revenue_unit or ''}"
                if guidance.revenue_min is not None else "—"
            )
            fcf = (
                f"{guidance.fcf_min}–{guidance.fcf_max}{guidance.fcf_unit or ''}"
                if guidance.fcf_min is not None else "—"
            )

            print(
                f"{status}  "
                f"chars={stats.get('chars_sent')}  "
                f"prompt_toks={stats.get('prompt_eval_count')}  "
                f"prompt_s={stats.get('prompt_eval_s')}  "
                f"gen_toks={stats.get('eval_count')}  "
                f"gen_s={stats.get('eval_s')}  "
                f"row_s={row_s}  "
                f"rev={rev}  fcf={fcf}"
            )

        result_df = pd.DataFrame(results)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(self.output_path, index=False)

        elapsed = round(time.time() - start_all, 2)
        success_n = int(result_df["parse_success"].sum()) if not result_df.empty else 0
        success_rate = (result_df["parse_success"].mean() * 100) if not result_df.empty else 0.0

        print(f"\nSaved → {self.output_path}")
        print(f"Parse success: {success_rate:.1f}% ({success_n}/{len(result_df)})")
        print(f"Total wall time: {elapsed}s")

        if not result_df.empty:
            print("\nAverages:")
            for col in ["chars_sent", "prompt_eval_count", "prompt_eval_s", "eval_count", "eval_s", "row_runtime_s"]:
                if col in result_df.columns:
                    val = pd.to_numeric(result_df[col], errors="coerce").mean()
                    print(f"  {col}: {round(val, 3) if pd.notna(val) else 'n/a'}")

        return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast local Ollama guidance extractor")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test: process only the first 2 rows",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for debugging",
    )
    parser.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable keyword-based context prefilter",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=MAX_CONTEXT_CHARS,
        help=f"Max chars sent to model after filtering (default: {MAX_CONTEXT_CHARS})",
    )
    args = parser.parse_args()

    extractor = LocalGuidanceExtractor(
        model=args.model,
        prefilter_context=not args.no_prefilter,
        max_context_chars=args.max_context_chars,
    )

    if args.test:
        df_raw = pd.read_csv(INPUT_CSV).head(2)
        print("=== Smoke test: first 2 rows ===\n")
        extractor._check_ollama()

        for _, row in df_raw.iterrows():
            fy = row.get("fiscal_year", None)
            fq = str(row.get("fiscal_quarter", ""))
            rd = row.get("release_date", None)
            section = str(row.get("section", ""))
            context = str(row.get("context", ""))

            print(f"Row: {fy} {fq} [{section}]")
            preview = _prefilter_context(context, args.max_context_chars) if not args.no_prefilter else context[:args.max_context_chars]
            print(f"Context preview: {preview[:250]}...\n")

            guidance, ok, stats = extractor._extract_one(
                section=section,
                context=context,
                fiscal_quarter=fq,
            )

            print(json.dumps({
                "fiscal_year": fy,
                "fiscal_quarter": fq,
                "release_date": rd,
                **guidance.model_dump(),
                "parse_success": ok,
                **stats,
            }, indent=2))
            print(f"\n{'─' * 60}\n")
    else:
        extractor.run(limit=args.limit)
