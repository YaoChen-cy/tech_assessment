# Download WM quarterly earnings call transcripts as plain text via defeatbeta-api.
from __future__ import annotations

import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "raw" / "transcripts"

TICKER     = "WM"
YEAR_START = 2021
YEAR_END   = 2025


def _transcript_to_text(df_paragraphs) -> str:
    # Convert a paragraphs DataFrame to plain text with speaker labels.
    parts = []
    for _, row in df_paragraphs.iterrows():
        speaker = row.get("speaker", "")
        content = row.get("content", "")
        if speaker and content:
            parts.append(f"[{speaker}]\n{content}")
        elif content:
            parts.append(content)
    return "\n\n".join(parts)


class TranscriptsCollector:
    def __init__(
        self,
        ticker: str     = TICKER,
        year_start: int = YEAR_START,
        year_end: int   = YEAR_END,
        out_dir: Path   = OUT_DIR,
    ):
        # Store collection parameters and output directory.
        self.ticker     = ticker
        self.year_start = year_start
        self.year_end   = year_end
        self.out_dir    = Path(out_dir)

    def _save(self, text: str, year: int, quarter: int) -> Path:
        # Write transcript text to disk at <out_dir>/<year>Q<quarter>.txt.
        dest = self.out_dir / f"{year}Q{quarter}.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        return dest

    def run(self) -> list[Path]:
        # Fetch all available earnings call transcripts and save as plain text files.
        from defeatbeta_api.data.ticker import Ticker

        print(f"=== TranscriptsCollector  {self.ticker}  {self.year_start}Q1–{self.year_end}Q4 ===")
        print(f"    Output → {self.out_dir}\n")

        tk = Ticker(self.ticker, log_level=logging.ERROR)
        transcripts = tk.earning_call_transcripts()

        available = transcripts.get_transcripts_list()
        print(f"Available transcripts: {len(available)} total\n")

        saved_paths: list[Path] = []
        saved = 0
        skipped = 0
        missing = 0

        for year in range(self.year_start, self.year_end + 1):
            for quarter in range(1, 5):
                label = f"{year} Q{quarter}"
                dest  = self.out_dir / f"{year}Q{quarter}.txt"

                if dest.exists():
                    print(f"  {label}  [skip — already exists]")
                    skipped += 1
                    saved_paths.append(dest)
                    continue

                print(f"  {label} ...", end=" ", flush=True)
                try:
                    df = transcripts.get_transcript(year, quarter)
                    text = _transcript_to_text(df)
                    if text.strip():
                        path = self._save(text, year, quarter)
                        print(f"saved ({len(text):,} chars → {path.name})")
                        saved += 1
                        saved_paths.append(path)
                    else:
                        print("SKIP (empty)")
                        missing += 1
                except ValueError:
                    print("NOT FOUND")
                    missing += 1

        print(f"\n=== Done ===")
        print(f"  Saved: {saved}  |  Already existed: {skipped}  |  Not found: {missing}")
        return sorted(saved_paths)


if __name__ == "__main__":
    TranscriptsCollector().run()
