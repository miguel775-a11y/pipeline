from pathlib import Path
import argparse
import os
import json
import logging
import sys
import re
from datetime import datetime

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib import colors

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas
from config import MASTER_MERGED_PATH_DEFAULT, OUTPUT_DIR_DEFAULT
from loader import load_master_merged, validate_required_columns, coerce_numeric_columns
from features import compute_features, add_advanced_filters
from scoring import (
    apply_normalized_scores,
    build_factor_library,
    score_engines,
    validate_scored_output,
)
from deepseek_reports import generate_reports, safe_filename
import selection_policy
import reference_ranking


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stock_pipeline")


# =====================================================================
# Drive-agnostic path discovery
# ---------------------------------------------------------------------
# The pipeline no longer depends on a fixed drive letter such as H:, D:,
# or C:. By default it discovers Master_merged.xlsx relative to this
# pipeline.py file and creates scoring_output next to the discovered
# master file. Command-line arguments still override discovery.
# =====================================================================

MASTER_MERGED_CANDIDATE_FILENAMES = ("Master_merged.xlsx", "Master_merged.csv")


def _path_from_env(name: str) -> Path | None:
    """Return a Path from an environment variable if it is set."""
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _normalize_cli_path(path: Path) -> Path:
    """Normalize a CLI path, preserving explicit relative paths against CWD."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def _candidate_master_paths(script_dir: Path) -> list[Path]:
    """Candidate master-file locations, in safest search order.

    Supported layouts:
      A) Master_merged.xlsx in the same folder as pipeline.py
      B) pipeline.py inside Stock_scoring_pipeline, Master_merged.xlsx in parent
      C) Running from a project folder that contains Master_merged.xlsx
      D) Running from a scripts subfolder, Master_merged.xlsx in CWD parent
    """
    bases = [script_dir, script_dir.parent, Path.cwd(), Path.cwd().parent]
    candidates: list[Path] = []
    seen: set[str] = set()

    for base in bases:
        for filename in MASTER_MERGED_CANDIDATE_FILENAMES:
            candidate = (base / filename).expanduser()
            key = str(candidate.resolve(strict=False)).lower()
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    return candidates


def resolve_master_merged_path(master_arg: Path | None, script_dir: Path) -> Path:
    """Resolve the master file path without relying on a drive letter."""
    if master_arg is not None:
        master_path = _normalize_cli_path(master_arg)
        if not master_path.exists():
            raise FileNotFoundError(
                f"Master file supplied through --master-merged was not found: {master_path}"
            )
        return master_path

    env_path = _path_from_env("STOCK_PIPELINE_MASTER_MERGED")
    if env_path is not None:
        env_path = _normalize_cli_path(env_path)
        if not env_path.exists():
            raise FileNotFoundError(
                "Environment variable STOCK_PIPELINE_MASTER_MERGED points to a missing file: "
                f"{env_path}"
            )
        return env_path

    candidates = _candidate_master_paths(script_dir)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    candidate_list = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "Could not find Master_merged.xlsx or Master_merged.csv automatically.\n"
        "Place Master_merged.xlsx in the same folder as pipeline.py, or in the parent folder, "
        "or pass it explicitly using --master-merged.\n"
        f"Checked:\n{candidate_list}"
    )


def resolve_output_dir(output_arg: Path | None, master_path: Path, script_dir: Path) -> Path:
    """Resolve output directory without relying on a drive letter.

    Default: create/use scoring_output next to the discovered master file.
    This preserves the existing professional structure when pipeline.py is
    inside Stock_scoring_pipeline and Master_merged.xlsx is one level above it.
    """
    if output_arg is not None:
        return _normalize_cli_path(output_arg)

    env_path = _path_from_env("STOCK_PIPELINE_OUTPUT_DIR")
    if env_path is not None:
        return _normalize_cli_path(env_path)

    # Preferred default: keep all dated outputs beside Master_merged.xlsx.
    return (master_path.parent / "scoring_output").resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def versioned_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 2
    while True:
        candidate = parent / f"{stem}_V{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def build_run_directories(base_output_dir: Path) -> dict:
    date_str = datetime.now().strftime("%Y-%m-%d")
    day_dir = ensure_dir(base_output_dir / date_str)

    return {
        "day_dir": day_dir,
        "deepseek_root": ensure_dir(day_dir / "deepseek_reports"),
        "deepseek_json": ensure_dir(day_dir / "deepseek_reports" / "json"),
        "deepseek_pdf": ensure_dir(day_dir / "deepseek_reports" / "pdf"),
        "deepseek_summary": ensure_dir(day_dir / "deepseek_reports" / "summary"),
    }




# --- Patch RR-H.QA1: rupee symbol PDF rendering helper ---
# ReportLab's default Helvetica font does not include U+20B9 (₹).
# Substitute "Rs." in PDF text only. The underlying JSON / CSV
# report_text retains "₹" (UTF-8 handled natively by those formats).
# Future patch may register a Unicode font and remove this helper.
def _rupee_for_pdf(value: object) -> str:
    """Return the input as a string with ₹ replaced by Rs. for PDF rendering.

    Defensive: tolerates non-string input by stringifying first; tolerates
    None by returning empty string.
    """
    if value is None:
        return ""
    return str(value).replace("₹", "Rs.")


def _extract_essential_metrics(report_item: dict) -> list[tuple[str, str]]:
    payload = report_item.get("python_report_payload") or {}
    rows = payload.get("essential_metrics") or []
    clean_rows: list[tuple[str, str]] = []
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            clean_rows.append((str(row[0]), str(row[1])))
    return clean_rows


# ---------------------------------------------------------------------
# 2. Colour palette + verdict mapping. Place these as module-level
#    constants just above _rupee_for_pdf (around the current line 185).
# ---------------------------------------------------------------------

# Hex strings are kept alongside HexColor() objects so they can be used
# both in TableStyle commands (which want a colour object) and inside
# Paragraph inline markup like <font color="#1F4E79">...</font> (which
# wants a string).
_HEX_SNAPSHOT_BG       = "#E8F1FB"   # light blue
_HEX_SNAPSHOT_BORDER   = "#1F4E79"   # dark blue
_HEX_SECTION_BG        = "#EEEEEE"   # light grey
_HEX_SECTION_RULE      = "#BDBDBD"   # mid grey
_HEX_LABEL_BLUE        = "#1F4E79"   # inline **Label:** prefix
_HEX_STRENGTH_BORDER   = "#2E7D32"   # green left rule on strength bullets
_HEX_WEAKNESS_BORDER   = "#C62828"   # red left rule on weakness bullets
_HEX_KEY_RISK          = "#E65100"   # orange — Key Risk:
_HEX_WHAT_MUST_IMPROVE = "#1565C0"   # blue   — What Must Improve:
_HEX_POSITIVE          = "#2E7D32"   # green  — Positive flags:
_HEX_RED_FLAG          = "#C62828"   # red    — Red flags:
_HEX_FOOTER            = "#757575"   # mid grey for page footer
_HEX_GROUP_HEADER_BG   = "#F4F6F8"   # very light grey, metric group rows
_HEX_GROUP_HEADER_FG   = "#1F4E79"   # dark blue text in group header

_COL_SNAPSHOT_BG       = HexColor(_HEX_SNAPSHOT_BG)
_COL_SNAPSHOT_BORDER   = HexColor(_HEX_SNAPSHOT_BORDER)
_COL_SECTION_BG        = HexColor(_HEX_SECTION_BG)
_COL_SECTION_RULE      = HexColor(_HEX_SECTION_RULE)
_COL_STRENGTH_BORDER   = HexColor(_HEX_STRENGTH_BORDER)
_COL_WEAKNESS_BORDER   = HexColor(_HEX_WEAKNESS_BORDER)
_COL_FOOTER            = HexColor(_HEX_FOOTER)
_COL_GROUP_HEADER_BG   = HexColor(_HEX_GROUP_HEADER_BG)
_COL_GROUP_HEADER_FG   = HexColor(_HEX_GROUP_HEADER_FG)

# Verdict label -> (background colour, text colour).
# Matched case-insensitively as a substring of the verdict string, so
# minor wording differences ("Strong Buy", "Buy ", "Tactical Buy.")
# still classify correctly. Order matters: longer phrases first so
# "tactical buy" is not swallowed by "buy".
_VERDICT_PALETTE: list[tuple[str, tuple]] = [
    ("avoid for now",  (HexColor("#C62828"), colors.white)),  # red
    ("avoid",          (HexColor("#C62828"), colors.white)),  # red (fallback)
    ("tactical buy",   (HexColor("#EF6C00"), colors.white)),  # amber/orange
    ("strong buy",     (HexColor("#2E7D32"), colors.white)),  # green
    ("buy",            (HexColor("#2E7D32"), colors.white)),  # green
]
_VERDICT_DEFAULT = (HexColor("#616161"), colors.white)         # grey


def _verdict_colours(label: str) -> tuple:
    """Map a verdict label to (bg, fg) colours. Grey fallback."""
    key = (label or "").strip().lower()
    for needle, palette in _VERDICT_PALETTE:
        if needle in key:
            return palette
    return _VERDICT_DEFAULT


# ---------------------------------------------------------------------
# 3. Replacement for _strip_text_metrics_section.
#    The old marker `"\n\nEssential metrics table\n"` did not match the
#    real text, where the heading sometimes appears glued mid-sentence
#    (e.g. "...align with profit growth. Essential metrics table").
#    We now cut at the first case-insensitive occurrence of the phrase
#    regardless of surrounding whitespace.
# ---------------------------------------------------------------------

_METRICS_HEADING_RE = re.compile(r"\bEssential metrics table\b", re.IGNORECASE)


def _strip_text_metrics_section(report_text: str) -> str:
    if not report_text:
        return report_text
    m = _METRICS_HEADING_RE.search(report_text)
    if not m:
        return report_text
    return report_text[: m.start()].rstrip(" \n\t")


# ---------------------------------------------------------------------
# 4. Text / section parsing helpers.
# ---------------------------------------------------------------------

# Inline `**bold**` becomes a dark-blue bold span (the "Label:" prefix
# style used heavily in sub-bullets).
_INLINE_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

# Matches `**N. Title**` headings (N = 1..9). The non-greedy capture
# stops at the closing `**`, so glued first bullets like
# `**1. Core strengths**- This is...` parse correctly.
_NUMBERED_HEADING_RE = re.compile(r"\*\*(\d+)\.\s+([^*]+?)\*\*")

# Inline label colouring rules applied to lines in section 6 and 7
# (and anywhere else they happen to appear). Each tuple is
# (compiled_regex, hex_colour). The regex must have one capture group
# wrapping the substring that should be coloured.
_INLINE_LABEL_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(Positive flags:\s*\d+)", re.IGNORECASE), _HEX_POSITIVE),
    (re.compile(r"(Red flags:\s*\d+)",      re.IGNORECASE), _HEX_RED_FLAG),
]


def _escape_html(text: str) -> str:
    """Minimal HTML escape for ReportLab Paragraph input."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _inline_markup(text: str) -> str:
    """Escape HTML and convert **bold** spans to dark-blue bold.

    Also colours inline 'Positive flags: N' / 'Red flags: N' spans.
    """
    safe = _escape_html(text)
    safe = _INLINE_BOLD_RE.sub(
        rf'<b><font color="{_HEX_LABEL_BLUE}">\1</font></b>',
        safe,
    )
    for pattern, hex_colour in _INLINE_LABEL_RULES:
        safe = pattern.sub(
            rf'<b><font color="{hex_colour}">\1</font></b>',
            safe,
        )
    return safe


def _parse_report_sections(report_text: str) -> list[dict]:
    """Split report_text into ordered sections.

    Returns a list of dicts:
        [{"number": "0", "title": "Report Snapshot", "body": "..."},
         {"number": "1", "title": "Core strengths",  "body": "..."},
         ...]

    Section 0 is detected by the plain heading "0. Report Snapshot".
    Sections 1+ are detected by the `**N. Title**` regex; the body of
    each runs from just after the closing `**` to just before the next
    heading (or end of text).
    """
    sections: list[dict] = []
    text = report_text or ""

    snap_match = re.search(r"^0\.\s+Report Snapshot\s*$", text, re.MULTILINE)
    heading_matches = list(_NUMBERED_HEADING_RE.finditer(text))

    if snap_match:
        snap_start = snap_match.end()
        snap_end = heading_matches[0].start() if heading_matches else len(text)
        sections.append({
            "number": "0",
            "title": "Report Snapshot",
            "body": text[snap_start:snap_end].strip(),
        })

    for i, m in enumerate(heading_matches):
        body_start = m.end()
        body_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
        sections.append({
            "number": m.group(1),
            "title": m.group(2).strip(),
            "body": text[body_start:body_end].strip(),
        })

    return sections


def _parse_section_blocks(body: str) -> list[dict]:
    """Classify each non-empty line of a section body.

    Each block is one of:
        {"kind": "top_bullet", "text": "..."}   # lines starting with `- `
        {"kind": "sub_bullet", "text": "..."}   # lines starting with `* `
        {"kind": "para",       "text": "..."}   # everything else
    """
    blocks: list[dict] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            blocks.append({"kind": "top_bullet", "text": line[2:].strip()})
        elif line.startswith("* "):
            blocks.append({"kind": "sub_bullet", "text": line[2:].strip()})
        else:
            blocks.append({"kind": "para", "text": line})
    return blocks


def _extract_verdict_fields(section_body: str) -> dict:
    """Pick out 'Verdict Label', 'Best Horizon', 'Key Risk',
    'What Must Improve' from section 7's body."""
    out = {"verdict": "", "horizon": "", "key_risk": "", "what_must_improve": ""}
    label_map = {
        "Verdict Label":     "verdict",
        "Best Horizon":      "horizon",
        "Key Risk":          "key_risk",
        "What Must Improve": "what_must_improve",
    }
    for raw_line in section_body.splitlines():
        line = raw_line.strip()
        for prefix, key in label_map.items():
            if line.lower().startswith(prefix.lower() + ":"):
                out[key] = line[len(prefix) + 1:].strip()
                break
    return out


def _extract_confidence_fields(report_text: str) -> dict:
    """Pull 'analysis confidence score', 'Positive flags', 'Red flags'
    out of section 6 (they sit inside running prose, not on own lines)."""
    out = {"confidence": "", "positive": "", "red": ""}
    if not report_text:
        return out
    m = re.search(r"analysis confidence score is\s+(\d+)", report_text, re.IGNORECASE)
    if m:
        out["confidence"] = m.group(1)
    m = re.search(r"Positive flags:\s*(\d+)", report_text, re.IGNORECASE)
    if m:
        out["positive"] = m.group(1)
    m = re.search(r"Red flags:\s*(\d+)", report_text, re.IGNORECASE)
    if m:
        out["red"] = m.group(1)
    return out


def _extract_trade_map(section5_body: str) -> list[tuple[str, str]]:
    """Pull entry-trigger / invalidation numbers out of section 5 prose.

    Returns ordered (label, value) rows ready for a small ReportLab table.
    Only entries actually found in the prose are returned.
    """
    if not section5_body:
        return []

    # Each tuple: (display label, list of regex patterns to try).
    # Patterns capture the bare value (e.g. "Rs.76.18" or "22.39").
    rules: list[tuple[str, list[str]]] = [
        ("Current Price",     [r"[Cc]urrent price is\s+(Rs\.[\d,]+(?:\.\d+)?)"]),
        ("Pivot Point",       [r"pivot point at\s+(Rs\.[\d,]+(?:\.\d+)?)"]),
        ("VWAP",              [r"VWAP at\s+(Rs\.[\d,]+(?:\.\d+)?)"]),
        ("ADX",               [r"ADX at\s+([\d.]+)"]),
        ("Resistance R1",     [r"pivot resistance R1 \((Rs\.[\d,]+(?:\.\d+)?)\)",
                               r"pivot resistance R1 at\s+(Rs\.[\d,]+(?:\.\d+)?)"]),
        ("Resistance R2",     [r"pivot resistance R2 \((Rs\.[\d,]+(?:\.\d+)?)\)",
                               r"pivot resistance R2 at\s+(Rs\.[\d,]+(?:\.\d+)?)"]),
        ("Resistance R3",     [r"pivot resistance R3 \((Rs\.[\d,]+(?:\.\d+)?)\)",
                               r"pivot resistance R3 at\s+(Rs\.[\d,]+(?:\.\d+)?)"]),
        ("Support S1",        [r"pivot support S1 at\s+(Rs\.[\d,]+(?:\.\d+)?)",
                               r"pivot support S1 \((Rs\.[\d,]+(?:\.\d+)?)\)"]),
        ("Support S2",        [r"pivot support S2 at\s+(Rs\.[\d,]+(?:\.\d+)?)",
                               r"pivot support S2 \((Rs\.[\d,]+(?:\.\d+)?)\)"]),
        ("Support S3",        [r"pivot support S3 at\s+(Rs\.[\d,]+(?:\.\d+)?)",
                               r"pivot support S3 \((Rs\.[\d,]+(?:\.\d+)?)\)"]),
    ]

    rows: list[tuple[str, str]] = []
    for label, patterns in rules:
        for pattern in patterns:
            m = re.search(pattern, section5_body)
            if m:
                rows.append((label, m.group(1)))
                break
    return rows


# Mapping from metric label substring -> group name. Order matters: the
# first matching keyword wins, so put more specific keywords earlier.
_METRIC_GROUPS: list[tuple[str, list[str]]] = [
    ("Snapshot",      ["current price", "market cap"]),
    ("Valuation",     ["pe (", "sector pe", "peg", "p/b", "price-to", "earnings yield",
                       "ev/ebitda", "price/", "p/s"]),
    ("Growth",        ["growth", "qoq", "yoy"]),
    ("Profitability", ["roe", "roce", "roa", "opm", "margin"]),
    ("Balance Sheet", ["debt", "current ratio", "interest coverage", "altman"]),
    ("Technicals",    ["sma", "rsi", "pivot", "vwap", "macd", "atr", "adx",
                       "52-week", "day range", "month low"]),
    ("Peer / Sector", ["sector", "industry", "rs vs", "relative strength"]),
]
_METRIC_GROUP_ORDER = [name for name, _ in _METRIC_GROUPS] + ["Other"]


def _classify_metric(label: str) -> str:
    key = (label or "").strip().lower()
    for group, needles in _METRIC_GROUPS:
        for needle in needles:
            if needle in key:
                return group
    return "Other"


# ---------------------------------------------------------------------
# 5. ReportLab block builders.
# ---------------------------------------------------------------------

def _build_snapshot_box(snapshot_body: str, available_width: float,
                        snapshot_style: ParagraphStyle) -> Table:
    """Wrap the entire snapshot section in a 1-cell table with a light
    blue background and a 2pt dark-blue border."""
    flowables: list = []
    for block in _parse_section_blocks(snapshot_body):
        if block["kind"] == "top_bullet":
            html = "• " + _inline_markup(block["text"])
        elif block["kind"] == "sub_bullet":
            html = "&nbsp;&nbsp;· " + _inline_markup(block["text"])
        else:
            html = _inline_markup(block["text"])
        flowables.append(Paragraph(html, snapshot_style))

    if not flowables:
        flowables.append(Paragraph("&nbsp;", snapshot_style))

    t = Table([[flowables]], colWidths=[available_width])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), _COL_SNAPSHOT_BG),
        ("BOX",          (0, 0), (-1, -1), 2, _COL_SNAPSHOT_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _build_section_title(number: str, title: str, available_width: float,
                         title_style: ParagraphStyle) -> Table:
    """Section heading bar — light grey background, dark rule below."""
    p = Paragraph(f"<b>{_escape_html(number)}. {_escape_html(title)}</b>", title_style)
    t = Table([[p]], colWidths=[available_width])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), _COL_SECTION_BG),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW",     (0, 0), (-1, -1), 1.2, _COL_SECTION_RULE),
    ]))
    return t


def _build_top_bullet(text: str, available_width: float,
                      body_style: ParagraphStyle,
                      border_colour=None) -> Table:
    """Top-level bullet with optional coloured left rule (green for
    strengths, red for weaknesses)."""
    html = "• " + _inline_markup(text)
    p = Paragraph(html, body_style)
    t = Table([[p]], colWidths=[available_width])
    style_cmds = [
        ("LEFTPADDING",   (0, 0), (-1, -1), 10 if border_colour else 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]
    if border_colour is not None:
        style_cmds.append(("LINEBEFORE", (0, 0), (0, 0), 3, border_colour))
    t.setStyle(TableStyle(style_cmds))
    return t


def _build_sub_bullet(text: str, sub_style: ParagraphStyle) -> Paragraph:
    """Indented secondary bullet."""
    return Paragraph("· " + _inline_markup(text), sub_style)


def _build_verdict_card(verdict: str, horizon: str, key_risk: str,
                        what_must_improve: str, confidence: str,
                        positive: str, red: str,
                        available_width: float,
                        body_style: ParagraphStyle) -> Table:
    """At-a-glance card placed right after the snapshot box.

    Two rows tall:
        Row 1: VERDICT (large, coloured) | Horizon | Confidence pills
        Row 2: Key Risk + What Must Improve (full width, two columns)
    """
    bg, fg = _verdict_colours(verdict)
    fg_hex = "#FFFFFF" if fg == colors.white else "#000000"

    verdict_html = (
        f'<para align="center"><b><font size="13" color="{fg_hex}">'
        f'{_escape_html(verdict or "—")}</font></b></para>'
    )
    horizon_html = (
        f'<para align="center"><font size="9" color="#555555">HORIZON</font><br/>'
        f'<b><font size="11">{_escape_html(horizon or "—")}</font></b></para>'
    )

    pill_parts = []
    if confidence:
        pill_parts.append(
            f'<font size="9" color="#555555">CONFIDENCE</font><br/>'
            f'<b><font size="11">{_escape_html(confidence)}</font></b>'
        )
    if positive:
        pill_parts.append(
            f'<font size="9" color="{_HEX_POSITIVE}">POSITIVE</font><br/>'
            f'<b><font size="11" color="{_HEX_POSITIVE}">{_escape_html(positive)}</font></b>'
        )
    if red:
        pill_parts.append(
            f'<font size="9" color="{_HEX_RED_FLAG}">RED FLAGS</font><br/>'
            f'<b><font size="11" color="{_HEX_RED_FLAG}">{_escape_html(red)}</font></b>'
        )
    pills_html = (
        '<para align="center">' + '&nbsp;&nbsp;&nbsp;'.join(pill_parts) + '</para>'
        if pill_parts else '<para align="center">&nbsp;</para>'
    )

    risk_html = (
        f'<b><font color="{_HEX_KEY_RISK}">Key Risk:</font></b> '
        f'{_escape_html(key_risk) if key_risk else "—"}'
    )
    improve_html = (
        f'<b><font color="{_HEX_WHAT_MUST_IMPROVE}">What Must Improve:</font></b> '
        f'{_escape_html(what_must_improve) if what_must_improve else "—"}'
    )

    verdict_p = Paragraph(verdict_html, body_style)
    horizon_p = Paragraph(horizon_html, body_style)
    pills_p   = Paragraph(pills_html, body_style)
    risk_p    = Paragraph(risk_html, body_style)
    impr_p    = Paragraph(improve_html, body_style)

    col_a = available_width * 0.40
    col_b = available_width * 0.25
    col_c = available_width - col_a - col_b

    data = [
        [verdict_p, horizon_p, pills_p],
        [risk_p, impr_p, ""],
    ]
    t = Table(data, colWidths=[col_a, col_b, col_c])
    t.setStyle(TableStyle([
        # Row 0: verdict strip — coloured background
        ("BACKGROUND",    (0, 0), (0, 0), bg),
        ("BACKGROUND",    (1, 0), (-1, 0), HexColor("#FAFAFA")),
        # Row 1: risk + improve, neutral background, span across cols
        ("SPAN",          (1, 1), (2, 1)),
        ("BACKGROUND",    (0, 1), (-1, 1), HexColor("#FFFFFF")),
        # Borders
        ("BOX",           (0, 0), (-1, -1), 0.8, _COL_SECTION_RULE),
        ("LINEBELOW",     (0, 0), (-1, 0), 0.6, _COL_SECTION_RULE),
        # Padding
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _build_trade_map_table(rows: list[tuple[str, str]],
                           available_width: float) -> Table:
    """Compact two-column Entry / Invalidation table for section 5."""
    if not rows:
        return None

    # Split into resistance (and current/pivot) vs support rows so the
    # left column reads "upside" and the right reads "downside".
    upside_labels = {"Current Price", "Pivot Point", "VWAP", "ADX",
                     "Resistance R1", "Resistance R2", "Resistance R3"}
    upside = [(lbl, val) for lbl, val in rows if lbl in upside_labels]
    downside = [(lbl, val) for lbl, val in rows if lbl not in upside_labels]

    max_len = max(len(upside), len(downside))
    while len(upside) < max_len:
        upside.append(("", ""))
    while len(downside) < max_len:
        downside.append(("", ""))

    header = [Paragraph("<b>Upside / Trigger</b>", _trade_map_header_style()),
              "",
              Paragraph("<b>Downside / Invalidation</b>", _trade_map_header_style()),
              ""]
    body_rows = []
    for (l1, v1), (l2, v2) in zip(upside, downside):
        body_rows.append([l1, v1, l2, v2])

    data = [header] + body_rows
    col = available_width / 4.0
    t = Table(data, colWidths=[col, col, col, col])
    t.setStyle(TableStyle([
        ("SPAN",          (0, 0), (1, 0)),
        ("SPAN",          (2, 0), (3, 0)),
        ("BACKGROUND",    (0, 0), (1, 0), HexColor("#E8F5E9")),  # soft green
        ("BACKGROUND",    (2, 0), (3, 0), HexColor("#FFEBEE")),  # soft red
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("LEADING",       (0, 0), (-1, -1), 11),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID",          (0, 1), (-1, -1), 0.3, colors.lightgrey),
        ("BOX",           (0, 0), (-1, -1), 0.6, _COL_SECTION_RULE),
        ("FONTNAME",      (1, 1), (1, -1), "Helvetica-Bold"),
        ("FONTNAME",      (3, 1), (3, -1), "Helvetica-Bold"),
    ]))
    return t


def _trade_map_header_style() -> ParagraphStyle:
    return ParagraphStyle(
        "TradeMapHeader",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=11,
        alignment=0,
        textColor=HexColor("#212121"),
    )


def _build_essential_metrics_table(metric_rows: list[tuple[str, str]]) -> Table:
    """Replacement for the old single-column attribute/value table.

    Groups rows into Snapshot / Valuation / Growth / Profitability /
    Balance Sheet / Technicals / Peer & Sector, then lays out each
    group as two label/value columns. Group-header rows span the full
    width and use a tinted background.
    """
    if not metric_rows:
        return None

    # Bucket rows by group, preserving original within-group order.
    buckets: dict[str, list[tuple[str, str]]] = {name: [] for name in _METRIC_GROUP_ORDER}
    for label, value in metric_rows:
        buckets[_classify_metric(label)].append((label, value))

    data: list[list] = []
    # 4 columns: label_left, value_left, label_right, value_right.
    # Group-header rows span all 4.
    style_cmds: list = []
    row_idx = 0
    for group_name in _METRIC_GROUP_ORDER:
        rows = buckets.get(group_name) or []
        if not rows:
            continue

        # Group header row
        data.append([group_name, "", "", ""])
        style_cmds.append(("SPAN",        (0, row_idx), (-1, row_idx)))
        style_cmds.append(("BACKGROUND",  (0, row_idx), (-1, row_idx), _COL_GROUP_HEADER_BG))
        style_cmds.append(("TEXTCOLOR",   (0, row_idx), (-1, row_idx), _COL_GROUP_HEADER_FG))
        style_cmds.append(("FONTNAME",    (0, row_idx), (-1, row_idx), "Helvetica-Bold"))
        style_cmds.append(("FONTSIZE",    (0, row_idx), (-1, row_idx), 9))
        style_cmds.append(("LEFTPADDING", (0, row_idx), (-1, row_idx), 6))
        style_cmds.append(("TOPPADDING",  (0, row_idx), (-1, row_idx), 5))
        style_cmds.append(("BOTTOMPADDING", (0, row_idx), (-1, row_idx), 4))
        row_idx += 1

        # Pair rows two at a time -> one table row each.
        for i in range(0, len(rows), 2):
            left  = rows[i]
            right = rows[i + 1] if i + 1 < len(rows) else ("", "")
            data.append([left[0], left[1], right[0], right[1]])
            row_idx += 1

    # Column widths: labels wider than values.
    label_w = 42 * mm
    value_w = 28 * mm
    col_widths = [label_w, value_w, label_w, value_w]

    t = Table(data, colWidths=col_widths)
    base_style = [
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("LEADING",       (0, 0), (-1, -1), 11),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID",          (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BOX",           (0, 0), (-1, -1), 0.6, colors.grey),
        # Make value cells bold and right-aligned for readability.
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME",      (3, 0), (3, -1), "Helvetica-Bold"),
        ("ALIGN",         (1, 0), (1, -1), "RIGHT"),
        ("ALIGN",         (3, 0), (3, -1), "RIGHT"),
    ]
    t.setStyle(TableStyle(base_style + style_cmds))
    return t


# ---------------------------------------------------------------------
# 6. NumberedCanvas — adds "<stock> — <strategy>" on the left and
#    "Page X of Y" on the right of every page's footer.
# ---------------------------------------------------------------------

def _make_numbered_canvas(footer_left: str):
    """Return a Canvas subclass with the given left-footer text baked in."""

    class _NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states: list[dict] = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            page_count = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_footer(page_count)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

        def _draw_footer(self, page_count: int):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(_COL_FOOTER)
            page_w, _page_h = A4
            margin = 18 * mm
            if footer_left:
                self.drawString(margin, 10 * mm, footer_left[:120])
            self.drawRightString(
                page_w - margin,
                10 * mm,
                f"Page {self._pageNumber} of {page_count}",
            )
            self.restoreState()

    return _NumberedCanvas


# ---------------------------------------------------------------------
# 7. Replacement save_report_pdf.
# ---------------------------------------------------------------------

def save_report_pdf(output_file: Path, report_item: dict) -> None:
    styles = getSampleStyleSheet()

    # ----- Paragraph styles -----
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=2,
        textColor=HexColor("#1A1A1A"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12,
        textColor=HexColor("#555555"),
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceAfter=4,
        textColor=HexColor("#1A1A1A"),
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["BodyText"],
        fontSize=11.5,
        leading=14,
        textColor=HexColor("#1A1A1A"),
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=4,
        textColor=HexColor("#1A1A1A"),
    )
    snapshot_style = ParagraphStyle(
        "SnapshotBody",
        parent=body_style,
        fontSize=11,
        leading=16,
        spaceAfter=4,
        textColor=HexColor("#0F2540"),
    )
    sub_style = ParagraphStyle(
        "SubBullet",
        parent=body_style,
        fontSize=9.5,
        leading=13,
        leftIndent=18,
        textColor=HexColor("#333333"),
        spaceAfter=2,
    )
    verdict_card_style = ParagraphStyle(
        "VerdictCard",
        parent=body_style,
        fontSize=10,
        leading=14,
        textColor=HexColor("#1A1A1A"),
        spaceAfter=0,
    )

    # ----- Document -----
    margin = 18 * mm
    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=margin,
        bottomMargin=margin + 6 * mm,  # extra space for footer
    )
    available_width = doc.width

    # ----- Pull data out of report_item -----
    stock_name        = report_item.get("stock_name", "Unknown Stock")
    strategy          = report_item.get("strategy", "")
    report_text       = report_item.get("report_text")
    error             = report_item.get("error")
    report_source     = report_item.get("report_source", "")
    essential_metric_rows = _extract_essential_metrics(report_item)

    report_text_for_pdf = _strip_text_metrics_section(str(report_text)) if report_text else report_text
    if report_text_for_pdf:
        report_text_for_pdf = _rupee_for_pdf(report_text_for_pdf)
    essential_metric_rows = [
        (_rupee_for_pdf(label), _rupee_for_pdf(value))
        for label, value in essential_metric_rows
    ]

    # Try to surface a current price under the title.
    current_price = ""
    for label, value in essential_metric_rows:
        if label.strip().lower() == "current price":
            current_price = value
            break

    generated_on = datetime.now().strftime("%d %b %Y")

    # ----- Story -----
    story: list = []

    # Title block + subtitle strip
    story.append(Paragraph(_escape_html(stock_name), title_style))
    subtitle_bits = []
    if strategy:
        subtitle_bits.append(f"Strategy: <b>{_escape_html(strategy)}</b>")
    if current_price:
        subtitle_bits.append(f"Current Price: <b>{_escape_html(current_price)}</b>")
    subtitle_bits.append(f"Generated: {generated_on}")
    story.append(Paragraph(" &nbsp;&nbsp;|&nbsp;&nbsp; ".join(subtitle_bits), subtitle_style))

    if report_text_for_pdf:
        heading = (
            "DeepSeek Report" if report_source == "deepseek"
            else "Python Engine Stock Detailed Report"
        )
        story.append(Paragraph(heading, heading_style))
        story.append(Spacer(1, 4))

        sections = _parse_report_sections(report_text_for_pdf)
        section_lookup = {s["number"]: s for s in sections}

        # Extract verdict + confidence for the top callout card.
        verdict_fields = _extract_verdict_fields(section_lookup.get("7", {}).get("body", ""))
        conf_fields    = _extract_confidence_fields(report_text_for_pdf)

        for section in sections:
            number = section["number"]
            title  = section["title"]
            body   = section["body"]

            if number == "0":
                # Snapshot in a boxed callout.
                story.append(_build_snapshot_box(body, available_width, snapshot_style))
                story.append(Spacer(1, 8))

                # Verdict callout card right after the snapshot.
                if any(verdict_fields.values()) or any(conf_fields.values()):
                    story.append(_build_verdict_card(
                        verdict           = verdict_fields.get("verdict", ""),
                        horizon           = verdict_fields.get("horizon", ""),
                        key_risk          = verdict_fields.get("key_risk", ""),
                        what_must_improve = verdict_fields.get("what_must_improve", ""),
                        confidence        = conf_fields.get("confidence", ""),
                        positive          = conf_fields.get("positive", ""),
                        red               = conf_fields.get("red", ""),
                        available_width   = available_width,
                        body_style        = verdict_card_style,
                    ))
                    story.append(Spacer(1, 10))
                continue

            # Heading bar for every other section. Section 5 needs its
            # heading bar + Trade Map kept together so the heading
            # doesn't strand at the foot of a page when KeepTogether
            # pushes the table forward.
            trade_table = None
            if number == "5":
                trade_rows = _extract_trade_map(body)
                trade_table = _build_trade_map_table(trade_rows, available_width)

            heading_bar = _build_section_title(number, title, available_width, section_title_style)
            if trade_table is not None:
                story.append(KeepTogether([
                    heading_bar,
                    Spacer(1, 4),
                    trade_table,
                    Spacer(1, 6),
                ]))
            else:
                story.append(heading_bar)
                story.append(Spacer(1, 4))

            # Colour the left border on strengths (section 1) and
            # weaknesses (section 3) for fast scanning.
            border_for_top_bullet = None
            if number == "1":
                border_for_top_bullet = _COL_STRENGTH_BORDER
            elif number == "3":
                border_for_top_bullet = _COL_WEAKNESS_BORDER

            for block in _parse_section_blocks(body):
                if block["kind"] == "top_bullet":
                    story.append(_build_top_bullet(
                        block["text"], available_width, body_style,
                        border_colour=border_for_top_bullet,
                    ))
                elif block["kind"] == "sub_bullet":
                    story.append(_build_sub_bullet(block["text"], sub_style))
                else:
                    # Plain paragraph. In section 7, label-prefixed lines
                    # (Verdict Label / Key Risk / What Must Improve /
                    # Best Horizon) get a coloured label treatment.
                    text = block["text"]
                    rendered = _render_labeled_paragraph(text, body_style, available_width)
                    story.append(rendered)

            story.append(Spacer(1, 8))

        # Essential Metrics table at the end.
        if essential_metric_rows:
            story.append(Spacer(1, 4))
            story.append(_build_section_title(
                "8", "Essential Metrics", available_width, section_title_style,
            ))
            story.append(Spacer(1, 4))
            table = _build_essential_metrics_table(essential_metric_rows)
            if table is not None:
                story.append(table)
    else:
        story.append(Paragraph("Report Status", heading_style))
        if error:
            safe_error = _escape_html(f"Error: {error}")
            story.append(Paragraph(safe_error, body_style))
        else:
            story.append(Paragraph("No report text generated.", body_style))

    footer_left = f"{stock_name}"
    if strategy:
        footer_left += f" — {strategy}"
    doc.build(story, canvasmaker=_make_numbered_canvas(footer_left))


def _render_labeled_paragraph(text: str, body_style: ParagraphStyle,
                              available_width: float = None):
    """Render a plain-paragraph line, special-casing the four labeled
    lines that appear in section 7.

    "Verdict Label: <X>"     -> full-width pill, background tied to verdict
                                colour, white text (red for "Avoid for Now",
                                amber for "Tactical Buy", green for "Buy",
                                grey otherwise).
    "Best Horizon: <X>"      -> bold dark-blue label, normal value.
    "Key Risk: <X>"          -> bold orange label, normal value.
    "What Must Improve: <X>" -> bold blue label, normal value.

    Returns a Paragraph or a Table (both are Flowables, so callers can
    append the result to the story uniformly).
    """
    label_rules = [
        ("Verdict Label",     None),                       # special: full-line pill
        ("Best Horizon",      _HEX_LABEL_BLUE),
        ("Key Risk",          _HEX_KEY_RISK),
        ("What Must Improve", _HEX_WHAT_MUST_IMPROVE),
    ]
    stripped = text.strip()
    for prefix, hex_colour in label_rules:
        if stripped.lower().startswith(prefix.lower() + ":"):
            rest = stripped[len(prefix) + 1:].strip()
            if prefix == "Verdict Label":
                bg, fg = _verdict_colours(rest)
                fg_hex = "#FFFFFF" if fg == colors.white else "#000000"
                pill_html = (
                    f'<b><font color="{fg_hex}" size="11">Verdict Label: '
                    f'{_escape_html(rest)}</font></b>'
                )
                pill_p = Paragraph(pill_html, body_style)
                width = available_width if available_width else 170 * mm
                t = Table([[pill_p]], colWidths=[width])
                t.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), bg),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
                    ("TOPPADDING",    (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ]))
                return t
            else:
                html = (
                    f'<b><font color="{hex_colour}">{prefix}:</font></b> '
                    f'{_escape_html(rest)}'
                )
                return Paragraph(html, body_style)

    return Paragraph(_inline_markup(text), body_style)

def save_daily_summary_pdf(output_file: Path, shortlist_map: dict[str, pd.DataFrame]) -> None:
    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = ParagraphStyle(
        "SummaryBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        spaceAfter=4,
    )

    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    story = []
    story.append(Paragraph("Daily Shortlist Summary", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        body_style
    ))
    story.append(Spacer(1, 10))

    section_order = [
        "Swing_Shortlist",
        "ShortTerm_Shortlist",
        "LongTerm_Core_Shortlist",
        "LongTerm_Opp_Shortlist",
    ]

    for idx, section_name in enumerate(section_order):
        df = shortlist_map.get(section_name)
        if df is None or df.empty:
            continue

        story.append(Paragraph(section_name.replace("_", " "), heading_style))
        story.append(Spacer(1, 6))

        table_data = [[
            "Stock Name",
            "NSE Code",
            "Category",
            "Entry Quality",
            "MCap Bucket",
        ]]

        for _, row in df.iterrows():
            # Patch RR-H.QA1: rupee → Rs. defensively for PDF rendering.
            # None of these columns currently carry ₹, but applying the
            # helper protects against future column changes.
            table_data.append([
                _rupee_for_pdf(row.get("stock_name", "")),
                _rupee_for_pdf(row.get("nse_code", "")),
                section_name.replace("_Shortlist", "").replace("_", " "),
                _rupee_for_pdf(row.get("entry_quality_tag", "")),
                _rupee_for_pdf(row.get("market_cap_bucket", "")),
            ])

        table = Table(
            table_data,
            repeatRows=1,
            colWidths=[65 * mm, 28 * mm, 42 * mm, 32 * mm, 25 * mm],
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEADING", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.beige]),
        ]))

        story.append(table)

        if idx < len(section_order) - 1:
            story.append(PageBreak())

    doc.build(story)


def export_outputs(
    df: pd.DataFrame,
    required_checks: pd.DataFrame,
    scoring_checks: pd.DataFrame,
    output_dir: Path
) -> dict:
    dirs = build_run_directories(output_dir)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    ranked_xlsx = versioned_path(dirs["day_dir"] / f"Scored_master_{timestamp}.xlsx")
    ranked_csv = versioned_path(dirs["day_dir"] / f"Scored_master_{timestamp}.csv")

    df.to_csv(ranked_csv, index=False)

    summary_cols = [
        "stock_name",
        "nse_code",
        "sector_name",
        "industry_name",
        "current_price",
        "market_capitalization",
        "market_cap_bucket",
        "quality_safety_score",
        "tradability_score",
        "crowded_trend_flag",
        "entry_quality_tag",
        "swing_score",
        "short_term_score",
        "long_term_score",
        "swing_score_v2",
        "short_term_score_v2",
        "long_term_score_v2",
        "primary_strategy_tag_v2",
        "primary_strategy_tag",
        "red_flag_count",
        "positive_flag_count",
        "analysis_confidence_score",
        "setup_quality_tag",
        "setup_confirmation_tag",
        "setup_risk_tag",
        "red_flags",
        "positive_flags",
        "swing_rank",
        "short_term_rank",
        "long_term_rank",
        "swing_rank_v2",
        "short_term_rank_v2",
        "long_term_rank_v2",
        "swing_rank_within_bucket",
        "short_term_rank_within_bucket",
        "long_term_rank_within_bucket",
        "swing_rank_within_bucket_v2",
        "short_term_rank_within_bucket_v2",
        "long_term_rank_within_bucket_v2",
        "growth_factor",
        "business_quality_factor",
        "cashflow_quality_factor",
        "risk_factor",
        "valuation_factor",
        "catalyst_proxy_factor",
    ]

    existing = [c for c in summary_cols if c in df.columns]

    # --- Patch C: selection logic fully delegated to selection_policy ---
    # All shortlist selection (and the strategy-tag-active eligible frames)
    # come from policy_result. The local variable names below are preserved
    # so the Excel writer block and the return dict need no changes.
    #
    # DeepSeek queue is intentionally not consumed here — the report-
    # generation block in main() still owns that decision (queue migration
    # is deferred to a later patch).
    policy_result = selection_policy.run_selection_policy(
        df, deepseek_enabled=False
    )

    # --- Patch RR-F: live reference grading ---
    # reference_ranking consumes policy_result and produces the six
    # reference DataFrames. This replaces the RR-E shadow call: the
    # outputs are now real workbook sheets, not just log lines. A
    # missing or broken reference_ranking module fails loudly here
    # (no defensive try/except) — reference grading is now a required
    # pipeline output, the same way selection is.
    reference_outputs = reference_ranking.run_reference_ranking(policy_result)

    # Active column names (same string values as the previous inline logic).
    swing_score_active = policy_result["active_score_cols"]["swing"]
    short_term_score_active = policy_result["active_score_cols"]["short_term"]
    long_term_score_active = policy_result["active_score_cols"]["long_term"]
    strategy_tag_active = policy_result["strategy_tag_active"]

    # Raw eligibility universes (composite-score gates only, no tag filter).
    swing_eligible = policy_result["swing_eligible"]
    short_term_eligible = policy_result["short_term_eligible"]
    long_term_eligible = policy_result["long_term_eligible"]

    # Strategy-tag-active eligible frames — now sourced directly from
    # selection_policy (Patch C). Previously recomputed locally in Patch B.
    swing_active_eligible = policy_result["swing_active_eligible"]
    short_term_active_eligible = policy_result["short_term_active_eligible"]
    long_term_active_eligible = policy_result["long_term_active_eligible"]

    # Long Term Core / Opportunity factor split — sourced from policy.
    long_term_core_eligible = policy_result["long_term_core_eligible"]
    long_term_opportunity_eligible = policy_result["long_term_opportunity_eligible"]

    # Final shortlists — sourced from policy. Identity verified by the
    # row-count and exact-match equivalence tests during migration.
    swing_shortlist = policy_result["swing_shortlist"]
    short_term_shortlist = policy_result["short_term_shortlist"]
    long_term_shortlist = policy_result["long_term_shortlist"]
    long_term_core_shortlist = policy_result["long_term_core_shortlist"]
    long_term_opportunity_shortlist = policy_result["long_term_opportunity_shortlist"]

    # Audit-only universe — eligible somewhere but not in any shortlist.
    # New in Patch C; surfaces 458 rows that were previously hidden.
    audit_only_universe = policy_result["audit_only_universe"]

    with pd.ExcelWriter(ranked_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Scored_Master", index=False)
        required_checks.to_excel(writer, sheet_name="Required_Column_Checks", index=False)
        scoring_checks.to_excel(writer, sheet_name="Scoring_Validation", index=False)

        df[existing].sort_values(swing_score_active, ascending=False).head(200).to_excel(
            writer, sheet_name="Top_200_Swing_Global", index=False
        )
        df[existing].sort_values(short_term_score_active, ascending=False).head(200).to_excel(
            writer, sheet_name="Top_200_Short_Global", index=False
        )
        df[existing].sort_values(long_term_score_active, ascending=False).head(200).to_excel(
            writer, sheet_name="Top_200_Long_Global", index=False
        )

        # Optional audit sheets to preserve old v1 ranking views for comparison.
        if {"swing_score", "short_term_score", "long_term_score"}.issubset(df.columns):
            df[existing].sort_values("swing_score", ascending=False).head(200).to_excel(
                writer, sheet_name="Top_200_Swing_Global_v1", index=False
            )
            df[existing].sort_values("short_term_score", ascending=False).head(200).to_excel(
                writer, sheet_name="Top_200_Short_Global_v1", index=False
            )
            df[existing].sort_values("long_term_score", ascending=False).head(200).to_excel(
                writer, sheet_name="Top_200_Long_Global_v1", index=False
            )

        swing_active_eligible[existing].to_excel(writer, sheet_name="Swing_Eligible_Universe", index=False)
        short_term_active_eligible[existing].to_excel(writer, sheet_name="ShortTerm_Eligible_Universe", index=False)
        long_term_active_eligible[existing].to_excel(writer, sheet_name="LongTerm_Eligible_Universe", index=False)
        long_term_core_eligible[existing].to_excel(writer, sheet_name="LongTerm_Core_Eligible", index=False)
        long_term_opportunity_eligible[existing].to_excel(writer, sheet_name="LongTerm_Opp_Eligible", index=False)

        swing_shortlist[existing].to_excel(writer, sheet_name="Swing_Shortlist", index=False)
        short_term_shortlist[existing].to_excel(writer, sheet_name="ShortTerm_Shortlist", index=False)
        long_term_shortlist[existing].to_excel(writer, sheet_name="LongTerm_Shortlist", index=False)
        long_term_core_shortlist[existing].to_excel(writer, sheet_name="LongTerm_Core_Shortlist", index=False)
        long_term_opportunity_shortlist[existing].to_excel(writer, sheet_name="LongTerm_Opp_Shortlist", index=False)

        # Patch C: Audit_Only sheet — stocks that passed at least one
        # eligibility gate but did not appear in any shortlist. Surfaces
        # ~458 rows that were previously invisible in the workbook.
        # Sort by quality_safety_score desc, then tradability_score desc
        # when those columns exist; otherwise write rows in natural order.
        audit_sort_cols = [
            c for c in ("quality_safety_score", "tradability_score")
            if c in audit_only_universe.columns
        ]
        if audit_sort_cols:
            audit_only_sorted = audit_only_universe.sort_values(
                audit_sort_cols, ascending=[False] * len(audit_sort_cols)
            )
        else:
            audit_only_sorted = audit_only_universe
        audit_only_sorted[existing].to_excel(
            writer, sheet_name="Audit_Only", index=False
        )

        # --- Patch RR-F: write the six reference-ranking sheets ---
        # Appended after the existing 20 sheets in the documented order.
        # Each reference DataFrame is written as-is: reference_ranking
        # already produces the canonical 23-column schema, so no column
        # selector is needed. Empty 23-column frames (e.g. when a source
        # shortlist was empty) are written with header row only — a
        # legitimate state, surfaced by the rows=0 log line below.
        _REFERENCE_SHEET_ORDER = (
            "Swing_Reference",
            "ShortTerm_Reference",
            "LongTerm_Core_Reference",
            "LongTerm_Opp_Reference",
            "LongTerm_Reference",
            "Audit_Only_Reference",
        )
        for ref_sheet_name in _REFERENCE_SHEET_ORDER:
            ref_df = reference_outputs[ref_sheet_name]
            ref_df.to_excel(writer, sheet_name=ref_sheet_name, index=False)

    # --- Patch RR-F: log row counts and grade summaries ---
    # Same format as RR-E's shadow logs so trainers can grep
    # "reference_ranking:" exactly as before. The lines now appear after
    # the workbook write rather than after a no-op shadow.
    for ref_sheet_name in (
        "Swing_Reference",
        "ShortTerm_Reference",
        "LongTerm_Core_Reference",
        "LongTerm_Opp_Reference",
        "LongTerm_Reference",
        "Audit_Only_Reference",
    ):
        ref_df = reference_outputs[ref_sheet_name]
        n_rows = len(ref_df)
        if n_rows == 0 or "reference_grade" not in ref_df.columns:
            grades = {}
        else:
            grades = ref_df["reference_grade"].value_counts().to_dict()
        logger.info(
            "reference_ranking: %s rows=%d grades=%s",
            ref_sheet_name, n_rows, grades,
        )

    # --- Patch RR-G: simulated reference-grade report queue (read-only) ---
    # Builds a hypothetical report queue from reference_outputs and writes
    # it as a separate CSV (no workbook sheet). Twelve INFO lines compare
    # it against the live queue. Pure simulation: actual report
    # generation in main() is not affected.
    reference_report_queue = _build_reference_report_queue(reference_outputs)

    # Per-source contribution counts (pre-dedup) for the by_source_sheet log.
    by_source: dict = {}
    for sheet_name in _REFERENCE_QUEUE_SOURCE_SHEETS:
        ref_df = reference_outputs.get(sheet_name)
        if not isinstance(ref_df, pd.DataFrame) or len(ref_df) == 0 \
                or "must_generate_report" not in ref_df.columns:
            by_source[sheet_name] = 0
        else:
            yes_mask = ref_df["must_generate_report"].astype(str).str.upper() == "YES"
            by_source[sheet_name] = int(yes_mask.sum())
    pre_dedup_total = sum(by_source.values())
    dedup_collisions = pre_dedup_total - len(reference_report_queue)

    # Audit Only YES count — excluded from the queue but tracked.
    audit_ref = reference_outputs.get("Audit_Only_Reference")
    if isinstance(audit_ref, pd.DataFrame) and len(audit_ref) > 0 \
            and "must_generate_report" in audit_ref.columns:
        audit_yes_count = int(
            (audit_ref["must_generate_report"].astype(str).str.upper() == "YES").sum()
        )
    else:
        audit_yes_count = 0

    # Priority distribution (1..5).
    if len(reference_report_queue) > 0 and "report_priority" in reference_report_queue.columns:
        priority_dist = reference_report_queue["report_priority"].value_counts().sort_index().to_dict()
        priority_dist = {int(k): int(v) for k, v in priority_dist.items()}
    else:
        priority_dist = {}

    # Build "current queue" view (today's actual report selection logic).
    # We pass a synthetic outputs dict with the keys this helper consumes —
    # the real outputs dict is not yet built at this point in the function.
    synthetic_outputs = {
        "active_score_cols": {
            "swing":      swing_score_active,
            "short_term": short_term_score_active,
            "long_term":  long_term_score_active,
        },
        "swing_shortlist":                 swing_shortlist,
        "short_term_shortlist":            short_term_shortlist,
        "long_term_core_shortlist":        long_term_core_shortlist,
        "long_term_opportunity_shortlist": long_term_opportunity_shortlist,
    }
    current_queue_keys = _compute_current_report_queue_keys(synthetic_outputs)

    # Compare the two queues.
    comparison = _compare_report_queues(
        reference_report_queue, current_queue_keys, reference_outputs
    )

    # Save the simulated queue as a CSV in day_dir (no workbook sheet).
    reference_queue_csv = versioned_path(
        dirs["day_dir"] / f"Reference_Report_Queue_{timestamp}.csv"
    )
    reference_report_queue.to_csv(reference_queue_csv, index=False)

    # Twelve INFO log lines under a single grep prefix.
    logger.info("reference_report_queue: simulated rows=%d", len(reference_report_queue))
    logger.info("reference_report_queue: priority_distribution=%s", priority_dist)
    logger.info("reference_report_queue: by_source_sheet=%s", by_source)
    logger.info("reference_report_queue: dedup_collisions=%d", dedup_collisions)
    logger.info(
        "reference_report_queue: audit_only YES count=%d excluded from simulated queue",
        audit_yes_count,
    )
    logger.info("reference_report_queue: vs current report queue")
    logger.info("reference_report_queue: in_both=%d", comparison["in_both"])
    logger.info("reference_report_queue: only_in_reference=%d", comparison["only_in_reference"])
    logger.info("reference_report_queue: only_in_current=%d", comparison["only_in_current"])
    logger.info(
        "reference_report_queue: sample only_in_reference: %s",
        comparison["sample_only_ref"],
    )
    logger.info(
        "reference_report_queue: sample only_in_current: %s",
        comparison["sample_only_cur"],
    )
    logger.info("reference_report_queue: csv saved to %s", reference_queue_csv)

    return {
        "ranked_xlsx": ranked_xlsx,
        "ranked_csv": ranked_csv,
        "timestamp": timestamp,
        "dirs": dirs,
        "active_score_cols": {
            "swing": swing_score_active,
            "short_term": short_term_score_active,
            "long_term": long_term_score_active,
        },
        "strategy_tag_active": strategy_tag_active,
        "swing_shortlist": swing_shortlist,
        "short_term_shortlist": short_term_shortlist,
        "long_term_shortlist": long_term_shortlist,
        "long_term_core_shortlist": long_term_core_shortlist,
        "long_term_opportunity_shortlist": long_term_opportunity_shortlist,
        # Patch RR-F: reference grading outputs are now first-class.
        "swing_reference": reference_outputs["Swing_Reference"],
        "short_term_reference": reference_outputs["ShortTerm_Reference"],
        "long_term_core_reference": reference_outputs["LongTerm_Core_Reference"],
        "long_term_opportunity_reference": reference_outputs["LongTerm_Opp_Reference"],
        "long_term_combined_reference": reference_outputs["LongTerm_Reference"],
        "audit_only_reference": reference_outputs["Audit_Only_Reference"],
        # Patch RR-G: simulated report queue (CSV-only, no workbook sheet).
        "reference_report_queue": reference_report_queue,
        "reference_report_queue_csv": reference_queue_csv,
    }


# =====================================================================
# Patch RR-G: simulated reference-grade report queue (read-only)
# ---------------------------------------------------------------------
# Builds a hypothetical report queue from the reference-grading outputs
# so trainers can compare it side-by-side with today's actual report
# queue (head(30) by score across the four shortlists). Pure simulation:
#   * not consumed by main()'s report-generation block
#   * does not change which reports get generated
#   * does not change DeepSeek behaviour
#   * does not add a workbook sheet
#
# Output: a separate CSV file in day_dir + 12 INFO log lines.
# =====================================================================

# 15-column schema for the simulated queue (CSV columns, in order).
_REFERENCE_REPORT_QUEUE_COLUMNS = [
    "report_queue_rank",
    "source_reference_sheet",
    "reference_rank",
    "reference_grade",
    "stable_stock_key",
    "stock_name",
    "nse_code",
    "strategy_category",
    "strategy_score",
    "must_generate_report",
    "report_priority",
    "suggested_action_view",
    "why_ranked_here",
    "top_strengths",
    "main_weakness",
]

# Reference sheets that contribute to the simulated queue.
# Audit_Only_Reference is intentionally excluded — it is a review list
# (no A+, "review" framing in suggested_action_view), not a candidate
# list. Its YES count is logged separately for transparency.
_REFERENCE_QUEUE_SOURCE_SHEETS = (
    "Swing_Reference",
    "ShortTerm_Reference",
    "LongTerm_Core_Reference",
    "LongTerm_Opp_Reference",
    "LongTerm_Reference",
)


def _build_reference_report_queue(reference_outputs: dict) -> pd.DataFrame:
    """Build the simulated report queue from reference outputs.

    Walks the five trading/long-term reference DataFrames, keeps rows
    where must_generate_report == "YES", dedups by stable_stock_key,
    sorts deterministically, and returns a 15-column DataFrame.

    Rules:
      * Source iteration order: Swing_Reference, ShortTerm_Reference,
        LongTerm_Core_Reference, LongTerm_Opp_Reference, LongTerm_Reference.
      * Dedup tie-break: lowest report_priority -> lowest reference_rank ->
        lexicographic source_reference_sheet name.
      * Final sort: report_priority asc, reference_rank asc,
        stable_stock_key asc.
      * Empty inputs are tolerated; result is an empty 15-column frame.

    Audit_Only_Reference is NOT consumed (handled separately for the log).
    """
    # Required columns from each reference DataFrame.
    required = [
        "stable_stock_key", "stock_name", "nse_code", "strategy_category",
        "strategy_score", "reference_rank", "reference_grade",
        "must_generate_report", "report_priority", "suggested_action_view",
        "why_ranked_here", "top_strengths", "main_weakness",
    ]

    rows = []
    for sheet_name in _REFERENCE_QUEUE_SOURCE_SHEETS:
        df = reference_outputs.get(sheet_name)
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            continue
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(
                "reference_report_queue: %s missing columns %s — skipped.",
                sheet_name, missing,
            )
            continue
        yes_mask = df["must_generate_report"].astype(str).str.upper() == "YES"
        if not yes_mask.any():
            continue
        sub = df.loc[yes_mask, required].copy()
        sub["source_reference_sheet"] = sheet_name
        rows.append(sub)

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in _REFERENCE_REPORT_QUEUE_COLUMNS})

    combined = pd.concat(rows, ignore_index=True)

    # Dedup: tie-break by [report_priority asc, reference_rank asc,
    # source_reference_sheet asc]. mergesort is stable, preserving order
    # within ties.
    combined = combined.sort_values(
        by=["report_priority", "reference_rank", "source_reference_sheet"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    deduped = combined.drop_duplicates(
        subset=["stable_stock_key"], keep="first"
    ).reset_index(drop=True)

    # Final sort + rank.
    deduped = deduped.sort_values(
        by=["report_priority", "reference_rank", "stable_stock_key"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    deduped["report_queue_rank"] = range(1, len(deduped) + 1)

    return deduped[_REFERENCE_REPORT_QUEUE_COLUMNS].reset_index(drop=True)


def _compute_current_report_queue_keys(outputs: dict) -> set:
    """Compute stable_stock_keys representing today's actual report queue.

    Mirrors main()'s live report-generation logic exactly:
        Swing_Shortlist sorted by active swing score, head(30)
        ShortTerm_Shortlist sorted by active short_term score, head(30)
        LongTerm_Core_Shortlist sorted by active long_term score, head(30)
        LongTerm_Opp_Shortlist sorted by active long_term score, head(30)

    If main()'s logic ever changes, this helper must change in lockstep.
    Returns a set[str] of stable_stock_keys.
    """
    score_cols = outputs.get("active_score_cols") or {}
    pairs = [
        (outputs.get("swing_shortlist"),                 score_cols.get("swing", "swing_score_v2")),
        (outputs.get("short_term_shortlist"),            score_cols.get("short_term", "short_term_score_v2")),
        (outputs.get("long_term_core_shortlist"),        score_cols.get("long_term", "long_term_score_v2")),
        (outputs.get("long_term_opportunity_shortlist"), score_cols.get("long_term", "long_term_score_v2")),
    ]

    keys: set = set()
    for sl, score_col in pairs:
        if not isinstance(sl, pd.DataFrame) or len(sl) == 0 or score_col not in sl.columns:
            continue
        top = sl.sort_values(score_col, ascending=False).head(30)
        for _, row in top.iterrows():
            keys.add(reference_ranking.build_stable_stock_key(row))
    return keys


def _compare_report_queues(
    reference_queue: pd.DataFrame,
    current_keys: set,
    reference_outputs: dict,
) -> dict:
    """Compute intersection + the two 'only-in' sets between the two queues.

    Returns a dict with counts and small samples for logging only:
      {
        'in_both':           int,
        'only_in_reference': int,
        'only_in_current':   int,
        'sample_only_ref':   list[str],
        'sample_only_cur':   list[str],
      }
    """
    if isinstance(reference_queue, pd.DataFrame) and "stable_stock_key" in reference_queue.columns:
        ref_keys = set(reference_queue["stable_stock_key"].astype(str).tolist())
    else:
        ref_keys = set()

    in_both = ref_keys & current_keys
    only_in_ref = ref_keys - current_keys
    only_in_cur = current_keys - ref_keys

    # Build a key -> (stock_name, reference_grade) lookup from reference_queue
    ref_lookup = {}
    if isinstance(reference_queue, pd.DataFrame) and len(reference_queue) > 0:
        for _, r in reference_queue.iterrows():
            ref_lookup[str(r.get("stable_stock_key", ""))] = (
                str(r.get("stock_name", "")),
                str(r.get("reference_grade", "")),
            )

    # For only_in_current samples, attempt to find each key in any reference
    # output (might exist with grade < B+ so excluded from the queue).
    cur_grade_lookup = {}
    for sheet_name, df in reference_outputs.items():
        if not isinstance(df, pd.DataFrame) or "stable_stock_key" not in df.columns:
            continue
        for _, r in df.iterrows():
            k = str(r.get("stable_stock_key", ""))
            if k and k not in cur_grade_lookup:
                cur_grade_lookup[k] = (
                    str(r.get("stock_name", "")),
                    str(r.get("reference_grade", "")),
                )

    sample_only_ref = []
    for k in sorted(only_in_ref)[:5]:
        name, grade = ref_lookup.get(k, (k, "?"))
        sample_only_ref.append(f"{name} (grade {grade})")

    sample_only_cur = []
    for k in sorted(only_in_cur)[:5]:
        if k in cur_grade_lookup:
            name, grade = cur_grade_lookup[k]
            sample_only_cur.append(f"{name} (would be grade {grade})")
        else:
            sample_only_cur.append(f"{k} (not graded by reference policy)")

    return {
        "in_both":           len(in_both),
        "only_in_reference": len(only_in_ref),
        "only_in_current":   len(only_in_cur),
        "sample_only_ref":   sample_only_ref,
        "sample_only_cur":   sample_only_cur,
    }


# =====================================================================
# Patch RR-H: live report queue migration (default-mode report logic)
# ---------------------------------------------------------------------
# Replaces the old head(30)-by-score logic in main()'s default branch
# with a controlled live queue derived from outputs["reference_report_queue"].
# Test-filter mode (--deepseek-test-codes / --deepseek-test-names) is
# untouched.
#
# Policy: A+ and A only for first live phase. B+ stays in the
# Reference_Report_Queue CSV for trainer review but is not reported
# until RR-H.1 promotes it.
# =====================================================================

# First-phase live grade policy. RR-H.1 may extend to ("A+", "A", "B+")
# after trainer review of RR-H output quality.
_LIVE_REPORT_GRADE_POLICY = ("A+", "A")

# --- Patch RR-H.1: Controlled Short Term B+ Booster ---
# Hard cap on the number of ShortTerm B+ rows that can be promoted into
# the live report queue per run. Setting this to 0 disables the booster
# without removing the helper or its log lines (soft rollback).
_LIVE_REPORT_BPLUS_SHORTTERM_BOOSTER_MAX = 3

# The exact phrase produced by reference_ranking._format_main_weakness
# when no weakness rule fires. The booster matches this case-insensitively
# with whitespace stripped, so trivial upstream wording changes don't
# silently break eligibility detection.
_LIVE_REPORT_BPLUS_NO_MAJOR_WEAKNESS_PHRASE = "no major weakness flagged"

# Source-sheet -> (lookup_outputs_key, report_bucket_name) routing.
# LongTerm_Reference rows that survive dedup (Long Term — Other) are
# looked up in the combined long_term_shortlist (which contains them
# by definition) and routed to the LongTerm_Opp_Shortlist report bucket.
_LIVE_QUEUE_ROUTING = {
    "Swing_Reference":         ("swing_shortlist",                 "Swing_Shortlist"),
    "ShortTerm_Reference":     ("short_term_shortlist",            "ShortTerm_Shortlist"),
    "LongTerm_Core_Reference": ("long_term_core_shortlist",        "LongTerm_Core_Shortlist"),
    "LongTerm_Opp_Reference":  ("long_term_opportunity_shortlist", "LongTerm_Opp_Shortlist"),
    "LongTerm_Reference":      ("long_term_shortlist",             "LongTerm_Opp_Shortlist"),
}


def _build_live_report_queue(reference_report_queue: pd.DataFrame) -> tuple:
    """Filter the full reference_report_queue down to the live policy.

    Steps:
      1. Keep only rows where reference_grade is in _LIVE_REPORT_GRADE_POLICY.
      2. Drop LongTerm_Reference rows whose stable_stock_key is already
         present (after step 1) in LongTerm_Core_Reference or
         LongTerm_Opp_Reference. The surviving LongTerm_Reference rows
         are "Long Term — Other" candidates worth reporting.

    Returns a tuple:
      (live_queue_df, dedup_dropped, combined_only_kept)

    Defensive: returns an empty DataFrame + zero counts if the input is
    None, empty, or missing required columns.
    """
    required_cols = ("reference_grade", "source_reference_sheet", "stable_stock_key")
    if (
        not isinstance(reference_report_queue, pd.DataFrame)
        or len(reference_report_queue) == 0
        or any(c not in reference_report_queue.columns for c in required_cols)
    ):
        return reference_report_queue.iloc[0:0].copy() if isinstance(reference_report_queue, pd.DataFrame) else pd.DataFrame(), 0, 0

    # Step 1: grade filter.
    graded = reference_report_queue[
        reference_report_queue["reference_grade"].isin(_LIVE_REPORT_GRADE_POLICY)
    ].copy()

    if len(graded) == 0:
        return graded, 0, 0

    # Step 2: LongTerm_Reference dedup.
    core_opp_mask = graded["source_reference_sheet"].isin(
        ["LongTerm_Core_Reference", "LongTerm_Opp_Reference"]
    )
    core_opp_keys = set(graded.loc[core_opp_mask, "stable_stock_key"].astype(str))

    combined_mask = graded["source_reference_sheet"] == "LongTerm_Reference"
    combined_dup_mask = combined_mask & graded["stable_stock_key"].astype(str).isin(core_opp_keys)
    combined_only_mask = combined_mask & ~combined_dup_mask

    dedup_dropped = int(combined_dup_mask.sum())
    combined_only_kept = int(combined_only_mask.sum())

    live_queue = graded[~combined_dup_mask].reset_index(drop=True)
    return live_queue, dedup_dropped, combined_only_kept


def _map_queue_to_full_rows(live_queue_df: pd.DataFrame, outputs: dict) -> tuple:
    """Map live-queue rows back to full shortlist rows by stable_stock_key.

    For each row in live_queue_df, look up the source shortlist via
    _LIVE_QUEUE_ROUTING[source_reference_sheet], find the matching row
    by stable_stock_key (built fresh on the shortlist via
    reference_ranking.build_stable_stock_key), and append that full row
    to the appropriate report bucket.

    Returns a tuple:
      (shortlist_map, unmapped_count, per_bucket_counts)

    shortlist_map keys are the four canonical bucket names that
    generate_reports() expects: Swing_Shortlist, ShortTerm_Shortlist,
    LongTerm_Core_Shortlist, LongTerm_Opp_Shortlist. Empty buckets are
    omitted from the dict (downstream loop logs "Skipping ... no stocks
    available" for empties anyway, but cleanliness is nicer).

    Unmapped rows (queue key not found in the source shortlist) are
    skipped with a WARNING log line. The pipeline does not crash unless
    a structural error occurs (which itself is caught in main()).
    """
    bucket_lists: dict = {
        "Swing_Shortlist":         [],
        "ShortTerm_Shortlist":     [],
        "LongTerm_Core_Shortlist": [],
        "LongTerm_Opp_Shortlist":  [],
    }
    per_bucket_counts: dict = {k: 0 for k in bucket_lists}
    unmapped_count = 0

    if not isinstance(live_queue_df, pd.DataFrame) or len(live_queue_df) == 0:
        return {}, 0, per_bucket_counts

    # Cache: build {stable_stock_key -> full_row Series} for each shortlist
    # we may need. Built lazily on first use.
    shortlist_key_index: dict = {}

    def _ensure_index(outputs_key: str):
        if outputs_key in shortlist_key_index:
            return shortlist_key_index[outputs_key]
        sl_df = outputs.get(outputs_key)
        if not isinstance(sl_df, pd.DataFrame) or len(sl_df) == 0:
            shortlist_key_index[outputs_key] = {}
            return {}
        idx: dict = {}
        for _, full_row in sl_df.iterrows():
            key = reference_ranking.build_stable_stock_key(full_row)
            # First occurrence wins; duplicates within a shortlist are
            # not expected (selection_policy guarantees uniqueness).
            idx.setdefault(key, full_row)
        shortlist_key_index[outputs_key] = idx
        return idx

    # Walk queue rows in queue order (already sorted by report_priority,
    # reference_rank in _build_reference_report_queue).
    for _, qrow in live_queue_df.iterrows():
        source_sheet = str(qrow.get("source_reference_sheet", ""))
        routing = _LIVE_QUEUE_ROUTING.get(source_sheet)
        if routing is None:
            unmapped_count += 1
            logger.warning(
                "live_report_queue: queue row source_reference_sheet=%s "
                "has no routing entry; skipping stock=%s",
                source_sheet, qrow.get("stock_name", "?"),
            )
            continue
        outputs_key, bucket_name = routing
        idx = _ensure_index(outputs_key)
        qkey = str(qrow.get("stable_stock_key", ""))
        full_row = idx.get(qkey)
        if full_row is None:
            unmapped_count += 1
            logger.warning(
                "live_report_queue: stable_stock_key=%s (stock=%s) not "
                "found in %s; skipping.",
                qkey, qrow.get("stock_name", "?"), outputs_key,
            )
            continue
        bucket_lists[bucket_name].append(full_row)
        per_bucket_counts[bucket_name] += 1

    # Convert lists of Series back to DataFrames; drop empty buckets.
    shortlist_map: dict = {}
    for bucket_name, rows in bucket_lists.items():
        if not rows:
            continue
        # pd.DataFrame from a list of Series preserves column order from
        # the source shortlist (each Series carries its original index).
        shortlist_map[bucket_name] = pd.DataFrame(rows).reset_index(drop=True)

    return shortlist_map, unmapped_count, per_bucket_counts


def _apply_shortterm_bplus_booster(
    reference_report_queue: pd.DataFrame,
    live_queue_df: pd.DataFrame,
) -> tuple:
    """Patch RR-H.1: append eligible ShortTerm B+ rows to the live queue.

    Eligibility (all five must hold):
      - reference_grade           == "B+"
      - source_reference_sheet    == "ShortTerm_Reference"
      - must_generate_report      == "YES"
      - report_priority           == 3
      - main_weakness             ≈  _LIVE_REPORT_BPLUS_NO_MAJOR_WEAKNESS_PHRASE
                                     (case-insensitive, whitespace-tolerant)

    Anti-dup: rows whose stable_stock_key already appears in the A+/A
    live queue are dropped (defensive — not currently expected, since a
    stock can't be both A+/A and B+ in the same run).

    Sort: ascending [report_priority, reference_rank, stable_stock_key].
    Cap : _LIVE_REPORT_BPLUS_SHORTTERM_BOOSTER_MAX.

    Defensive: if any required column is missing from the queue, returns
    the live queue unchanged with zero counts and an empty names list.
    The caller's outer try/except provides a second safety net.

    Returns:
        (combined_queue, eligible_count, added_count, skipped_count, added_names)
    """
    required_cols = (
        "reference_grade",
        "source_reference_sheet",
        "must_generate_report",
        "report_priority",
        "stable_stock_key",
        "main_weakness",
        "reference_rank",
        "stock_name",
    )

    # Defensive guard 1: empty / non-DataFrame input.
    if (
        not isinstance(reference_report_queue, pd.DataFrame)
        or len(reference_report_queue) == 0
    ):
        return live_queue_df, 0, 0, 0, []

    # Defensive guard 2: missing required columns.
    missing = [c for c in required_cols if c not in reference_report_queue.columns]
    if missing:
        logger.warning(
            "live_report_queue: bplus_booster skipped — missing columns: %s",
            missing,
        )
        return live_queue_df, 0, 0, 0, []

    # Defensive guard 3: cap = 0 means booster disabled (soft rollback).
    if _LIVE_REPORT_BPLUS_SHORTTERM_BOOSTER_MAX <= 0:
        return live_queue_df, 0, 0, 0, []

    q = reference_report_queue

    # Build the eligibility mask. All five filters AND'd together.
    grade_mask    = q["reference_grade"] == "B+"
    source_mask   = q["source_reference_sheet"] == "ShortTerm_Reference"
    yes_mask      = q["must_generate_report"].astype(str).str.strip().str.upper() == "YES"
    priority_mask = pd.to_numeric(q["report_priority"], errors="coerce") == 3
    weakness_mask = (
        q["main_weakness"]
        .astype(str)
        .str.strip()
        .str.lower()
        == _LIVE_REPORT_BPLUS_NO_MAJOR_WEAKNESS_PHRASE
    )
    eligible_mask = grade_mask & source_mask & yes_mask & priority_mask & weakness_mask

    eligible_df = q[eligible_mask].copy()
    eligible_count = int(len(eligible_df))

    if eligible_count == 0:
        return live_queue_df, 0, 0, 0, []

    # Anti-dup: drop any eligible row whose stable_stock_key is already
    # present in the A+/A live queue.
    if isinstance(live_queue_df, pd.DataFrame) and "stable_stock_key" in live_queue_df.columns:
        existing_keys = set(live_queue_df["stable_stock_key"].astype(str))
        eligible_df = eligible_df[
            ~eligible_df["stable_stock_key"].astype(str).isin(existing_keys)
        ]

    # Deterministic sort, then apply the hard cap.
    sort_cols = ["report_priority", "reference_rank", "stable_stock_key"]
    eligible_df = (
        eligible_df.sort_values(sort_cols, ascending=True, kind="mergesort")
        .reset_index(drop=True)
    )

    booster_rows = eligible_df.head(_LIVE_REPORT_BPLUS_SHORTTERM_BOOSTER_MAX).copy()
    added_count = int(len(booster_rows))
    skipped_count = int(eligible_count - added_count)
    added_names = [str(n) for n in booster_rows["stock_name"].tolist()]

    # Append to the live queue. The 15-column queue schema matches
    # exactly, so concat is straightforward; reset_index for cleanliness.
    if added_count > 0 and isinstance(live_queue_df, pd.DataFrame):
        combined_queue = pd.concat(
            [live_queue_df, booster_rows], ignore_index=True, sort=False
        )
    else:
        combined_queue = live_queue_df

    return combined_queue, eligible_count, added_count, skipped_count, added_names


def save_deepseek_summary_csv(
    summary_items: list[dict],
    output_file: Path
) -> Path | None:
    if not summary_items:
        return None

    df = pd.DataFrame(summary_items)

    preferred_cols = [
        "stock_name",
        "nse_code",
        "strategy",
        "report_source",
        "fallback_used",
        "fallback_reason",
        "verdict_label",
        "best_horizon",
        "key_risk",
        "what_must_improve",
        "red_flag_count",
        "positive_flag_count",
        "analysis_confidence_score",
        "primary_strategy_tag",
        "entry_quality_tag",
        "setup_quality_tag",
        "setup_confirmation_tag",
        "setup_risk_tag",
        "swing_score",
        "short_term_score",
        "long_term_score",
        "swing_score_v2",
        "short_term_score_v2",
        "long_term_score_v2",
        "primary_strategy_tag_v2",
    ]
    existing = [c for c in preferred_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    df = df[existing + remaining]

    output_file = versioned_path(output_file)
    df.to_csv(output_file, index=False)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="3-engine stock scoring pipeline")

    parser.add_argument(
        "--master-merged",
        type=Path,
        default=None,
        help=(
            "Optional path to Master_merged.xlsx/CSV. If omitted, the pipeline "
            "looks in the script folder, the parent folder, the current working "
            "directory, and the current working directory parent."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. If omitted, scoring_output is created "
            "next to the discovered Master_merged file."
        ),
    )
    parser.add_argument("--generate-deepseek-prompts", action="store_true")
    parser.add_argument("--call-deepseek", action="store_true")
    parser.add_argument("--deepseek-api-key", type=str, default="")
    parser.add_argument("--deepseek-test-codes", type=str, default="")
    parser.add_argument("--deepseek-test-names", type=str, default="")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    args.master_merged = resolve_master_merged_path(args.master_merged, script_dir)
    args.output_dir = resolve_output_dir(args.output_dir, args.master_merged, script_dir)

    logger.info("Starting stock scoring pipeline...")
    logger.info(f"Pipeline script folder: {script_dir}")
    logger.info(f"Master input: {args.master_merged}")
    logger.info(f"Output directory: {args.output_dir}")

    test_codes = {
        x.strip().upper()
        for x in args.deepseek_test_codes.split(",")
        if x.strip()
    }

    test_names = {
        x.strip()
        for x in args.deepseek_test_names.split(",")
        if x.strip()
    }

    if args.call_deepseek:
        logger.info("Report mode: DeepSeek API")
    elif args.generate_deepseek_prompts:
        logger.info("Report mode: Python fallback / prompt generation path")
    else:
        logger.info("Report mode: scoring only (no report generation)")

    if test_codes or test_names:
        logger.info(
            f"Test filter active | codes={len(test_codes)} | names={len(test_names)}"
        )

    logger.info("Loading master input file...")
    df = load_master_merged(args.master_merged)
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.")

    logger.info("Validating required columns...")
    required_checks, required_errors = validate_required_columns(df)
    if required_errors:
        raise ValueError("Required columns missing/unpopulated:\n" + "\n".join(required_errors))
    logger.info("Required column validation passed.")

    logger.info("Coercing numeric columns...")
    df = coerce_numeric_columns(df)
    logger.info("Computing features...")
    df = compute_features(df)
    logger.info("Applying advanced filters...")
    df = add_advanced_filters(df)
    logger.info("Applying normalized scores...")
    df = apply_normalized_scores(df)
    logger.info("Building factor library...")
    df = build_factor_library(df)
    logger.info("Running scoring engines (v1 + v2)...")
    df = score_engines(df)

    logger.info("Validating scored output...")
    scoring_checks = validate_scored_output(df)
    failed_scoring = scoring_checks[scoring_checks["status"] == "FAIL"]
    if not failed_scoring.empty:
        raise ValueError("Scoring validation failed:\n" + failed_scoring.to_string(index=False))
    logger.info("Scored output validation passed.")

    logger.info("Exporting ranked master and shortlist sheets...")
    outputs = export_outputs(df, required_checks, scoring_checks, args.output_dir)

    logger.info("Creating daily shortlist summary PDF...")
    summary_pdf = versioned_path(outputs["dirs"]["day_dir"] / "Daily_Shortlist_Summary.pdf")
    save_daily_summary_pdf(
        summary_pdf,
        {
            "Swing_Shortlist": outputs["swing_shortlist"],
            "ShortTerm_Shortlist": outputs["short_term_shortlist"],
            "LongTerm_Core_Shortlist": outputs["long_term_core_shortlist"],
            "LongTerm_Opp_Shortlist": outputs["long_term_opportunity_shortlist"],
        },
    )
    logger.info(f"Saved summary PDF: {summary_pdf}")

    deepseek_summary_csv = None

    if args.generate_deepseek_prompts or args.call_deepseek:
        logger.info("Preparing report generation stage...")
        json_root = outputs["dirs"]["deepseek_json"]
        pdf_root = outputs["dirs"]["deepseek_pdf"]
        summary_root = outputs["dirs"]["deepseek_summary"]

        active_score_cols = outputs.get("active_score_cols", {})
        swing_score_active = active_score_cols.get("swing", "swing_score_v2")
        short_term_score_active = active_score_cols.get("short_term", "short_term_score_v2")
        long_term_score_active = active_score_cols.get("long_term", "long_term_score_v2")
        strategy_tag_active = outputs.get("strategy_tag_active", "primary_strategy_tag_v2")

        def resolve_active_strategy_label(row: pd.Series) -> str:
            tag = str(row.get(strategy_tag_active, "") or "").strip()
            if tag == "Swing":
                return "Swing_Shortlist"
            if tag == "Short Term":
                return "ShortTerm_Shortlist"
            if tag == "Long Term":
                is_core = (
                    pd.notna(row.get("business_quality_factor")) and float(row.get("business_quality_factor", 0)) >= 60 and
                    pd.notna(row.get("cashflow_quality_factor")) and float(row.get("cashflow_quality_factor", 0)) >= 50 and
                    pd.notna(row.get("risk_factor")) and float(row.get("risk_factor", 0)) >= 55
                )
                return "LongTerm_Core_Shortlist" if is_core else "LongTerm_Opp_Shortlist"
            return ""

        if test_codes or test_names:
            test_df = df.copy()
            if test_codes:
                test_df = test_df[
                    test_df["nse_code"].astype(str).str.upper().isin(test_codes)
                ].copy()
            if test_names:
                test_df = test_df[
                    test_df["stock_name"].astype(str).isin(test_names)
                ].copy()

            if not test_df.empty:
                test_df["__active_strategy_label__"] = test_df.apply(resolve_active_strategy_label, axis=1)
                test_df = test_df[test_df["__active_strategy_label__"] != ""].copy()

            shortlist_map = {
                label: grp.copy()
                for label, grp in test_df.groupby("__active_strategy_label__", sort=False)
            }

            if not shortlist_map:
                logger.warning("Skipping report generation: no stocks matched the test filter.")
            else:
                logger.info(f"Matched {sum(len(v) for v in shortlist_map.values())} stocks across {len(shortlist_map)} strategy bucket(s) for report generation.")
        else:
            # --- Patch RR-H: live report queue migration ---
            # Replaces the old head(30)-by-score logic with a controlled
            # live queue derived from outputs["reference_report_queue"].
            # Policy: A+ and A only (B+ stays in CSV for review).
            # Mapping: stable_stock_key only (never nse_code).
            # Failure modes log WARNING and skip; no silent fallback to
            # head(30).
            reference_queue = outputs.get("reference_report_queue")
            ref_queue_size = len(reference_queue) if isinstance(reference_queue, pd.DataFrame) else 0

            # Old head(30) count for comparison logging.
            old_head30_count = (
                min(30, len(outputs.get("swing_shortlist", pd.DataFrame())))
                + min(30, len(outputs.get("short_term_shortlist", pd.DataFrame())))
                + min(30, len(outputs.get("long_term_core_shortlist", pd.DataFrame())))
                + min(30, len(outputs.get("long_term_opportunity_shortlist", pd.DataFrame())))
            )

            if ref_queue_size == 0:
                logger.warning(
                    "live_report_queue: reference_report_queue is empty/missing — "
                    "zero reports will be generated."
                )
                shortlist_map = {}
                unmapped_count = 0
                per_bucket = {"Swing_Shortlist": 0, "ShortTerm_Shortlist": 0,
                              "LongTerm_Core_Shortlist": 0, "LongTerm_Opp_Shortlist": 0}
                live_queue_size = 0
                by_grade: dict = {}
                by_source_sheet: dict = {}
                bplus_excluded = 0
                dedup_dropped = 0
                combined_only_kept = 0
                # Patch RR-H.1: booster zero-defaults so the
                # unconditional log block below doesn't NameError.
                booster_eligible = 0
                booster_added = 0
                booster_skipped = 0
                booster_added_names: list = []
            else:
                # Counts from the FULL queue, computed before the live filter.
                bplus_excluded = int(
                    (reference_queue["reference_grade"] == "B+").sum()
                ) if "reference_grade" in reference_queue.columns else 0

                # Filter to live policy + apply LongTerm_Reference dedup.
                live_queue, dedup_dropped, combined_only_kept = _build_live_report_queue(reference_queue)
                live_queue_size = len(live_queue)

                if live_queue_size == 0:
                    logger.warning(
                        "live_report_queue: no rows match policy %s after dedup — "
                        "zero reports will be generated.",
                        _LIVE_REPORT_GRADE_POLICY,
                    )
                    by_grade = {}
                    by_source_sheet = {}
                else:
                    by_grade = {
                        g: int((live_queue["reference_grade"] == g).sum())
                        for g in _LIVE_REPORT_GRADE_POLICY
                    }
                    by_source_sheet = (
                        live_queue["source_reference_sheet"].value_counts().to_dict()
                    )
                    by_source_sheet = {str(k): int(v) for k, v in by_source_sheet.items()}

                # --- Patch RR-H.1: ShortTerm B+ booster ---
                # Apply the booster on top of the A+/A live queue. Wrapped
                # in try/except: any failure falls back to the unmodified
                # A+/A queue so the booster cannot break A+/A reporting.
                try:
                    (
                        extended_queue,
                        booster_eligible,
                        booster_added,
                        booster_skipped,
                        booster_added_names,
                    ) = _apply_shortterm_bplus_booster(reference_queue, live_queue)
                except Exception as exc:
                    logger.warning(
                        "live_report_queue: bplus_booster failed (%s) — "
                        "falling back to A+/A live queue only.",
                        exc,
                    )
                    extended_queue = live_queue
                    booster_eligible = 0
                    booster_added = 0
                    booster_skipped = 0
                    booster_added_names = []

                # Map the EXTENDED queue (A+/A plus booster rows) back to
                # full shortlist rows. The mapper uses stable_stock_key
                # routing and is unchanged from RR-H.
                shortlist_map, unmapped_count, per_bucket = _map_queue_to_full_rows(
                    extended_queue, outputs
                )

            # --- live_report_queue log block (12 lines) ---
            new_total = sum(per_bucket.values())
            logger.info("live_report_queue: policy=%s", _LIVE_REPORT_GRADE_POLICY)
            logger.info("live_report_queue: reference_queue_size=%d", ref_queue_size)
            logger.info("live_report_queue: live_queue_size=%d", live_queue_size)
            logger.info("live_report_queue: by_grade=%s", by_grade)
            logger.info("live_report_queue: by_source_sheet=%s", by_source_sheet)
            logger.info("live_report_queue: B+ excluded count=%d", bplus_excluded)
            logger.info(
                "live_report_queue: LongTerm_Reference duplicates dropped=%d",
                dedup_dropped,
            )
            logger.info(
                "live_report_queue: LongTerm_Reference combined-only kept=%d",
                combined_only_kept,
            )
            logger.info(
                "live_report_queue: comparison vs old head(30) — old=%d, new=%d",
                old_head30_count, new_total,
            )
            logger.info("live_report_queue: unmapped_keys=%d", unmapped_count)
            logger.info(
                "live_report_queue: final reports queued: Swing=%d, ShortTerm=%d, "
                "LongTerm_Core=%d, LongTerm_Opp=%d",
                per_bucket.get("Swing_Shortlist", 0),
                per_bucket.get("ShortTerm_Shortlist", 0),
                per_bucket.get("LongTerm_Core_Shortlist", 0),
                per_bucket.get("LongTerm_Opp_Shortlist", 0),
            )

            # --- Patch RR-H.1: ShortTerm B+ booster log block (6 lines) ---
            logger.info(
                "live_report_queue: bplus_booster policy=ShortTerm only, max=%d",
                _LIVE_REPORT_BPLUS_SHORTTERM_BOOSTER_MAX,
            )
            logger.info(
                "live_report_queue: bplus_booster eligible=%d", booster_eligible
            )
            logger.info(
                "live_report_queue: bplus_booster added=%d", booster_added
            )
            logger.info(
                "live_report_queue: bplus_booster skipped=%d", booster_skipped
            )
            logger.info(
                "live_report_queue: bplus_booster names=%s", booster_added_names
            )
            logger.info(
                "live_report_queue: total_with_booster=%d",
                live_queue_size + booster_added,
            )

        all_summary_rows: list[dict] = []

        for label, shortlist_df in shortlist_map.items():
            working_df = shortlist_df.copy()

            if working_df.empty:
                logger.info(f"Skipping {label}: no stocks available for this bucket.")
                continue

            logger.info(f"Generating reports for {label} | stocks={len(working_df)}")
            reports = generate_reports(
                working_df,
                strategy=label,
                call_api=args.call_deepseek,
                api_key=args.deepseek_api_key if args.deepseek_api_key else None,
            )

            json_file = versioned_path(json_root / f"{label.lower()}.json")
            json_file.write_text(
                json.dumps(reports, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Saved report JSON: {json_file}")

            strategy_pdf_dir = ensure_dir(pdf_root / label)
            total_reports = len(reports)
            for idx, item in enumerate(reports, start=1):
                stock_name = safe_filename(item.get("stock_name"))
                logger.info(f"[{label}] Saving PDF {idx}/{total_reports}: {stock_name}")
                pdf_file = versioned_path(strategy_pdf_dir / f"{stock_name}.pdf")
                save_report_pdf(pdf_file, item)

            merge_cols = [
                "stock_name",
                "nse_code",
                "primary_strategy_tag",
                "primary_strategy_tag_v2",
                "entry_quality_tag",
                "setup_quality_tag",
                "setup_confirmation_tag",
                "setup_risk_tag",
                "red_flag_count",
                "positive_flag_count",
                "analysis_confidence_score",
                "swing_score",
                "short_term_score",
                "long_term_score",
                "swing_score_v2",
                "short_term_score_v2",
                "long_term_score_v2",
            ]
            available_merge_cols = [c for c in merge_cols if c in working_df.columns]

            if reports:
                report_df = pd.DataFrame(reports)
                context_df = working_df[available_merge_cols].drop_duplicates().copy()

                if "stock_name" in report_df.columns and "stock_name" in context_df.columns:
                    merged = report_df.merge(
                        context_df,
                        on="stock_name",
                        how="left",
                        suffixes=("", "_ctx"),
                    )
                else:
                    merged = report_df.copy()

                for _, row in merged.iterrows():
                    all_summary_rows.append({
                        "stock_name": row.get("stock_name"),
                        "nse_code": row.get("nse_code"),
                        "strategy": row.get("strategy"),
                        "report_source": row.get("report_source"),
                        "fallback_used": row.get("fallback_used"),
                        "fallback_reason": row.get("fallback_reason"),
                        "verdict_label": row.get("verdict_label"),
                        "best_horizon": row.get("best_horizon"),
                        "key_risk": row.get("key_risk"),
                        "what_must_improve": row.get("what_must_improve"),
                        "red_flag_count": row.get("red_flag_count"),
                        "positive_flag_count": row.get("positive_flag_count"),
                        "analysis_confidence_score": row.get("analysis_confidence_score"),
                        "primary_strategy_tag": row.get("primary_strategy_tag"),
                        "primary_strategy_tag_v2": row.get("primary_strategy_tag_v2"),
                        "entry_quality_tag": row.get("entry_quality_tag"),
                        "setup_quality_tag": row.get("setup_quality_tag"),
                        "setup_confirmation_tag": row.get("setup_confirmation_tag"),
                        "setup_risk_tag": row.get("setup_risk_tag"),
                        "swing_score": row.get("swing_score"),
                        "short_term_score": row.get("short_term_score"),
                        "long_term_score": row.get("long_term_score"),
                        "swing_score_v2": row.get("swing_score_v2"),
                        "short_term_score_v2": row.get("short_term_score_v2"),
                        "long_term_score_v2": row.get("long_term_score_v2"),
                    })

        deepseek_summary_csv = save_deepseek_summary_csv(
            all_summary_rows,
            summary_root / f"deepseek_report_summary_{outputs['timestamp']}.csv"
        )
        if deepseek_summary_csv:
            logger.info(f"Saved report summary CSV: {deepseek_summary_csv}")

    logger.info("Pipeline completed successfully.")
    logger.info(f"ranked_xlsx: {outputs['ranked_xlsx']}")
    logger.info(f"ranked_csv: {outputs['ranked_csv']}")
    logger.info(f"summary_pdf: {summary_pdf}")
    if deepseek_summary_csv:
        logger.info(f"deepseek_summary_csv: {deepseek_summary_csv}")


if __name__ == "__main__":
    main()