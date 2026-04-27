# Extract a named section from a WM earnings press release PDF, with date and fiscal period metadata.
import re
import sys
from datetime import datetime
from pathlib import Path

import pdfplumber

_MONTH_DATE_RE = re.compile(
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December'
    r'|Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)'
    r'\s+(\d{1,2}),?\s+(\d{4})\b',
    re.IGNORECASE,
)

_DATE_FORMATS = ('%B %d, %Y', '%B %d %Y', '%b %d, %Y', '%b %d %Y', '%b. %d, %Y')

_QUARTER_END_MONTH = {3: "Q1", 6: "Q2", 9: "Q3", 12: "Q4"}


def _parse_month_date(s: str) -> str | None:
    # Parse a 'Month DD, YYYY' string to YYYYMMDD, tolerating missing comma.
    s = re.sub(r'\s+', ' ', re.sub(r',', ', ', s.strip()))
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime('%Y%m%d')
        except ValueError:
            pass
    return None


def _extract_period_end_date(line: str) -> str | None:
    # Extract period-end date from 'ended Month DD, YYYY' on a single line.
    m = re.search(r'ended\s+(\w+\.?\s+\d{1,2},?\s+\d{4})', line, re.IGNORECASE)
    return _parse_month_date(m.group(1)) if m else None


def _extract_release_date(line: str) -> str | None:
    # Extract release date from a 'HOUSTON — Month DD, YYYY —' announcement line.
    has_houston = 'houston' in line.lower()
    has_dash_date = bool(re.search(r'[—–]\s*' + _MONTH_DATE_RE.pattern + r'\s*[—–]', line, re.IGNORECASE))
    if not has_houston and not has_dash_date:
        return None
    m = _MONTH_DATE_RE.search(line)
    return _parse_month_date(f"{m.group(1)} {m.group(2)}, {m.group(3)}") if m else None


def _date_to_fiscal(yyyymmdd: str | None) -> tuple[int | None, str | None]:
    # Convert a YYYYMMDD period-end date to (fiscal_year, fiscal_quarter) per WM calendar.
    if not yyyymmdd:
        return None, None
    try:
        dt = datetime.strptime(yyyymmdd, '%Y%m%d')
    except ValueError:
        return None, None
    quarter = _QUARTER_END_MONTH.get(dt.month)
    return (dt.year, quarter) if quarter else (None, None)


def _is_numeric_token(t: str) -> bool:
    # Return True for dollar amounts, percentages, plain numbers, and common placeholders.
    return bool(re.match(r"^[\$\(\-]?\d[\d,\.]*%?\)?$", t) or t in ("-", "—", "N/A"))


def _is_bullet_line(t: str) -> bool:
    # Return True if the line starts with a bullet character.
    return t.lstrip().startswith(("•", "–", "-", "*", "", "•"))


def _is_separator_line(t: str) -> bool:
    # Return True for horizontal rules or footnote openers that signal the end of section content.
    ln = t.strip()
    if re.match(r'^[-─═_·.]{5,}$', re.sub(r"\s+", "", ln)):
        return True
    return bool(re.match(r'^\([a-z]\)\s', ln))


def _merge_currency_tokens(words: list) -> list:
    # Merge split '$' + digit tokens and consecutive digit fragments from the PDF renderer.
    if not words:
        return words
    out, skip_next = [], False
    for i, w in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        nxt = words[i + 1] if i + 1 < len(words) else None
        if w["text"] == "$" and nxt and re.match(r"^\d", nxt["text"]) and (nxt["x0"] - w["x1"]) < 12:
            merged = {**w, "text": "$" + nxt["text"], "x1": nxt["x1"]}
            out.append(merged)
            skip_next = True
        elif (re.match(r"^\d+$", w["text"]) and nxt and re.match(r"^\d+$", nxt["text"])
              and (nxt["x0"] - w["x1"]) < 8):
            merged = {**w, "text": w["text"] + nxt["text"], "x1": nxt["x1"]}
            out.append(merged)
            skip_next = True
        else:
            out.append(w)
    return out


def _page_word_rows(page) -> list[tuple[float, list]]:
    # Group page words into rows by y-coordinate (3 pt tolerance), sorted top-to-bottom.
    rows: dict[float, list] = {}
    for w in page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3):
        y = round(w["top"])
        key = next((k for k in rows if abs(k - y) <= 3), y)
        rows.setdefault(key, []).append(w)
    return [(y, _merge_currency_tokens(sorted(ws, key=lambda w: w["x0"])))
            for y, ws in sorted(rows.items())]


def _row_text(words: list) -> str:
    # Join word tokens left-to-right into a single string.
    return " ".join(w["text"] for w in sorted(words, key=lambda w: w["x0"]))


def _heading_matches_target(line: str, target: str) -> bool:
    # Return True if line contains any keyword from the pipe/comma-separated target string.
    keywords = [t.strip().upper() for t in re.split(r"\||,", target) if t.strip()]
    return any(kw in line.upper() for kw in keywords)


def _is_section_heading(page, y: float, ln: str) -> bool:
    # Return True if the row at y is a bold standalone section heading based on font and shape checks.
    ln = ln.strip()
    if not ln or len(ln) > 120 or ln[0] in ("•", "-", "*", "–", "(", "["):
        return False
    chars_at_y = [c for c in page.chars if abs(round(c["top"]) - y) <= 3]
    if not chars_at_y:
        return False
    if sum(1 for c in chars_at_y if "bold" in c.get("fontname", "").lower()) / len(chars_at_y) <= 0.5:
        return False
    clean = re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)', '', ln)).strip()
    if not clean or len(clean) > 100:
        return False
    if re.fullmatch(r'[A-Z0-9][A-Z0-9 &/\-]*', clean) and clean.isupper():
        return True
    if re.search(r'\b\d{4}\b', clean) and len(clean) <= 50 and not re.search(r'[\$%]|\d{1,3},\d{3}', clean):
        return True
    return False


def _find_heading_y(page, target: str) -> float | None:
    # Return the y-position of the first bold section heading on the page that matches target.
    for y, words in _page_word_rows(page):
        ln = _row_text(words).strip()
        if _is_section_heading(page, y, ln) and _heading_matches_target(ln, target):
            return y
    return None


def _find_next_section_y(page, after_y: float, current_heading: str) -> float | None:
    # Return the y-position of the next bold section heading after after_y, or None if not found.
    current_clean = re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)', '', current_heading)).strip().upper()
    for y, words in _page_word_rows(page):
        if y <= after_y or y - after_y < 40:
            continue
        lt = _row_text(words)
        if _is_separator_line(lt):
            return y
        if not _is_section_heading(page, y, lt.strip()):
            continue
        clean = re.sub(r'\s+', ' ', re.sub(r'\([^)]*\)', '', lt)).strip().upper()
        if clean != current_clean:
            return y
    return None


_TABLE_GAP_PT = 50  # minimum x-gap (pt) between label and value columns


def _table_row_to_text(words: list) -> str | None:
    # Convert a table-layout row to 'Label: col1 | col2 | ...' text, or None if not a table row.
    sorted_words = sorted(words, key=lambda w: w["x0"])
    segments: list[list[str]] = [[]]
    for i, w in enumerate(sorted_words):
        if i > 0 and (w["x0"] - sorted_words[i - 1]["x1"]) > _TABLE_GAP_PT:
            segments.append([])
        segments[-1].append(w["text"])
    if len(segments) < 2:
        return None
    if not any(_is_numeric_token(t) for seg in segments[1:] for t in seg):
        return None
    label = " ".join(segments[0])
    values = " | ".join(" ".join(seg) for seg in segments[1:] if seg)
    return f"{label}: {values}" if label else values


def SectionExtractor(
    pdf_path: str | Path,
    heading: str,
    *,
    max_pages_after: int = 3,
) -> dict:
    # Extract text belonging to the named heading from the PDF, returning section text and fiscal metadata.
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    target        = heading.strip()
    section_rows: list[tuple[float, list]] = []
    document_date: str | None = None
    release_date:  str | None = None
    in_section    = False
    heading_found = ""
    pages_after   = 0

    with pdfplumber.open(pdf_path) as pdf:
        for early_page in pdf.pages[:3]:
            rows  = _page_word_rows(early_page)
            lines = [_row_text(ws) for _, ws in rows]
            for i, line in enumerate(lines):
                if not release_date:
                    release_date = _extract_release_date(line)
                    if not release_date and line.strip().upper() == 'HOUSTON':
                        for neighbour in lines[max(0, i - 2): i] + lines[i + 1: i + 3]:
                            release_date = _extract_release_date(neighbour)
                            if release_date:
                                break
                if not document_date and 'ended' in line.lower():
                    document_date = _extract_period_end_date(line)
                    if not document_date:
                        for next_line in lines[i + 1: i + 3]:
                            m = _MONTH_DATE_RE.search(next_line)
                            if m:
                                document_date = _parse_month_date(m.group(0))
                                break
            if document_date and release_date:
                break

        for page in pdf.pages:
            if not in_section:
                heading_y = _find_heading_y(page, target)
                if heading_y is None:
                    continue
                in_section  = True
                all_rows    = _page_word_rows(page)
                closest_y   = min((y for y, _ in all_rows), key=lambda y: abs(y - (heading_y or 0)))
                heading_found = _row_text(dict(all_rows)[closest_y])
                next_y      = _find_next_section_y(page, heading_y + 2, heading_found)
                for y, ws in all_rows:
                    if y <= heading_y:
                        continue
                    if next_y and y >= next_y:
                        break
                    section_rows.append((y, ws))
                if next_y:
                    in_section = False
            else:
                pages_after += 1
                if pages_after > max_pages_after:
                    break
                all_rows = _page_word_rows(page)
                next_y   = _find_next_section_y(page, 0, heading_found)
                for y, ws in all_rows:
                    if next_y and y >= next_y:
                        break
                    section_rows.append((y, ws))
                if next_y:
                    in_section = False
                    break

    paragraphs: list[str] = []
    current_paragraph: str | None = None

    for _, ws in section_rows:
        lt = _row_text(ws)
        if _is_separator_line(lt):
            break
        table_text = _table_row_to_text(ws)
        if table_text is not None:
            if current_paragraph is not None:
                paragraphs.append(current_paragraph)
                current_paragraph = None
            paragraphs.append(table_text)
        elif _is_bullet_line(lt):
            if current_paragraph is not None:
                paragraphs.append(current_paragraph)
            current_paragraph = re.sub(r'^[\s••\-\*–]+', '', lt).strip()
        else:
            stripped = lt.strip()
            if stripped:
                if current_paragraph is not None:
                    current_paragraph += " " + stripped
                else:
                    current_paragraph = stripped

    if current_paragraph is not None:
        paragraphs.append(current_paragraph)

    fiscal_year, fiscal_quarter = _date_to_fiscal(document_date)

    return {
        "section":        heading_found,
        "text":           "\n".join(paragraphs).strip(),
        "date":           document_date,
        "fiscal_year":    fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "release_date":   release_date,
    }


def _print_result(result: dict) -> None:
    # Print a section extraction result to stdout.
    print(f"\n{'='*60}")
    print(f"Section  : {result['section']}")
    print(f"Period   : {result['date']}  FY{result['fiscal_year']} {result['fiscal_quarter']}")
    print(f"Released : {result['release_date']}")
    print(f"{'='*60}\n")
    if result["text"]:
        print(result["text"])


if __name__ == "__main__":
    result = SectionExtractor("/Users/chenyao/Downloads/otpp_tech_assessment/data/raw/news_releases/4Q_2022_Press_Release_01_31_23_Final_Consolidated.pdf", "OUTLOOK|EXPECTATIONS|GUIDANCE")
    _print_result(result)
