"""
selection_policy.py
====================

Single source of truth for *selection* in the Stock Market Intelligence
Pipeline. This module is the deterministic Python layer that decides:

    1. Which stocks are eligible for each strategy
    2. Which stocks make the shortlist for each strategy
    3. Which stocks go into the Python fallback report queue
    4. Which stocks go into the optional DeepSeek report queue
    5. Which stocks are kept only as audit-only context

Architecture role:

        Data  ->  Scores  ->  Tags  ->  [SELECT]  ->  Reports
                                          ^^^^^^
                                       (this file)

Design rules:

  * Python is the source of truth. DeepSeek is optional and never imported here.
  * No I/O. Pure functions: DataFrame in, DataFrame (or dict) out.
  * No new metrics, no freshness gate yet (those are later steps).
  * Thresholds and bucket mixes live as module-level constants for now and
    will eventually move into config.py without changing behaviour.

Status:
    SKELETON. Not yet imported by pipeline.py. The thresholds, bucket mixes
    and eligibility rules below mirror the live logic currently embedded in
    pipeline.py.export_outputs(...) so that, when pipeline.py adopts this
    module, behaviour does not change.

Multibagger:
    Stub only. Real Multibagger logic is a later stage decision and must
    not be invented here.

Patch history:
    * Added build_stable_stock_key / add_stable_stock_key helpers and
      switched build_audit_only_universe to use the stable key. This
      protects identity matching when nse_code is blank on some rows.
      Shortlist selection is unaffected.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ------------------------------------------------------------
# These are copied from pipeline.py.export_outputs(...) verbatim so that
# the behaviour is preserved when this module is adopted. They are kept
# at the top of the file so students can read the rules in one place.
# Eventually they will move to config.py.
# ============================================================

# --- Eligibility gates (composite-score thresholds) -------------------
SWING_ELIGIBILITY = {
    "min_tradability_score": 45,
    "min_quality_safety_score": 45,
}

SHORT_TERM_ELIGIBILITY = {
    "min_tradability_score": 40,
    "min_quality_safety_score": 40,
}

LONG_TERM_ELIGIBILITY = {
    "min_quality_safety_score": 50,
    "min_tradability_score": 25,
}

# --- Long Term Core sub-gate (factor-level thresholds) ----------------
LONG_TERM_CORE_GATE = {
    "min_business_quality_factor": 60,
    "min_cashflow_quality_factor": 50,
    "min_risk_factor": 55,
}

# --- Multibagger (PLACEHOLDER, not active) ----------------------------
# Real thresholds will be set in a later stage. Kept here so the shape is
# visible to students.
MULTIBAGGER_ELIGIBILITY = {
    "min_quality_safety_score": None,
    "min_tradability_score": None,
    "min_business_quality_factor": None,
    "min_cashflow_quality_factor": None,
}

# --- Bucket mixes (market_cap_bucket -> max stocks in shortlist) ------
SWING_BUCKET_MIX = {"Large Cap": 40, "Mid Cap": 80, "Small Cap": 25, "Micro Cap": 5}
SHORT_TERM_BUCKET_MIX = {"Large Cap": 20, "Mid Cap": 80, "Small Cap": 60, "Micro Cap": 10}
LONG_TERM_CORE_BUCKET_MIX = {"Large Cap": 40, "Mid Cap": 80, "Small Cap": 20, "Micro Cap": 0}
LONG_TERM_OPP_BUCKET_MIX = {"Large Cap": 10, "Mid Cap": 50, "Small Cap": 35, "Micro Cap": 5}
LONG_TERM_BUCKET_MIX = {"Large Cap": 40, "Mid Cap": 80, "Small Cap": 30, "Micro Cap": 5}

# --- Report queue depth -----------------------------------------------
# How many stocks per strategy go into the Python fallback report queue
# (and, when enabled, into the optional DeepSeek queue).
REPORT_QUEUE_TOP_N = 30

# --- Stable stock key -------------------------------------------------
# Identifier columns in fallback order. The first non-blank value wins.
# A final synthetic "row:<idx>" fallback ensures two truly-blank rows
# are never collapsed into the same identity.
STABLE_KEY_COL = "_stable_stock_key"
STABLE_KEY_FALLBACK_ORDER = [
    "nse_code",
    "isin",
    "bse_code",
    "stock_code",
    "best_stock_key",
    "stock_name",
]

# Strings (lower-cased) that count as "blank" when building the stable key.
_INVALID_KEY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--"}


# ============================================================
# DEFENSIVE HELPERS
# ============================================================

def _has_columns(df: pd.DataFrame, cols: list[str], context: str) -> bool:
    """Return True if every column exists. Otherwise log a warning
    and return False. Lets the caller decide what to do."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(
            "selection_policy: %s missing columns %s. Returning empty result.",
            context, missing,
        )
        return False
    return True


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    """Return a zero-row DataFrame with the same columns, used as a safe
    fallback when required columns are missing."""
    return df.head(0).copy()


def _is_blank(value) -> bool:
    """True if value should be treated as missing for identity purposes."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    s = str(value).strip().lower()
    return s in _INVALID_KEY_TOKENS


# ============================================================
# STABLE STOCK KEY
# ------------------------------------------------------------
# When nse_code is missing on a row (which happens for hundreds of rows
# in current Master_merged data), set-based identity comparisons collapse
# all such rows to one bucket. The stable key fixes this by walking a
# fallback order of identifier columns and falling back to a synthetic
# row-index key only as a last resort.
#
# Format examples:
#       "nse:INFY"           -> nse_code present
#       "isin:INE009A01021"  -> nse_code blank, isin present
#       "name:Infosys Ltd"   -> only the company name was available
#       "row:1842"           -> nothing usable; synthetic per-row key
#
# The prefix keeps keys from different identifier types from colliding,
# so a hypothetical bse_code "INFY" can never match an nse_code "INFY".
# ============================================================

# Short prefixes per identifier source. Keeps keys from different
# columns from colliding (e.g. nse_code "FOO" vs bse_code "FOO").
_STABLE_KEY_PREFIXES = {
    "nse_code": "nse",
    "isin": "isin",
    "bse_code": "bse",
    "stock_code": "stk",
    "best_stock_key": "bsk",
    "stock_name": "name",
}


def build_stable_stock_key(row: pd.Series) -> str:
    """Return a stable, never-blank identity string for a single row.

    Walks STABLE_KEY_FALLBACK_ORDER and returns the first non-blank value
    found, prefixed by the column it came from. If nothing is usable,
    returns a synthetic per-row fallback so blank rows never collide.
    """
    for col in STABLE_KEY_FALLBACK_ORDER:
        if col in row.index:
            value = row[col]
            if not _is_blank(value):
                cleaned = str(value).strip()
                # Code-like fields go uppercase; preserve stock_name casing.
                if col != "stock_name":
                    cleaned = cleaned.upper()
                prefix = _STABLE_KEY_PREFIXES.get(col, col)
                return f"{prefix}:{cleaned}"

    # Last-resort synthetic key — uses the row's index label.
    # Two blank rows with different DataFrame indices stay distinct.
    return f"row:{row.name}"


def add_stable_stock_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a `_stable_stock_key` column added.

    Pure helper. Does not change any other column. Safe to call on any
    DataFrame, including empty ones and ones missing all identifier
    columns (in which case every row gets a synthetic 'row:<idx>' key).
    """
    if df is None or len(df) == 0:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if STABLE_KEY_COL not in out.columns:
            out[STABLE_KEY_COL] = pd.Series([], dtype=object)
        return out

    out = df.copy()
    out[STABLE_KEY_COL] = out.apply(build_stable_stock_key, axis=1)
    return out


# ============================================================
# ACTIVE COLUMN RESOLUTION
# ------------------------------------------------------------
# The pipeline currently produces both v1 and v2 score columns. The v2
# columns are preferred when present. This logic lives here so every
# selection function uses the same rule.
# ============================================================

def resolve_active_score_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return the active score column name for each strategy.
    Prefers *_v2 columns when available."""
    return {
        "swing": "swing_score_v2" if "swing_score_v2" in df.columns else "swing_score",
        "short_term": "short_term_score_v2" if "short_term_score_v2" in df.columns else "short_term_score",
        "long_term": "long_term_score_v2" if "long_term_score_v2" in df.columns else "long_term_score",
    }


def resolve_active_strategy_tag_column(df: pd.DataFrame) -> str:
    """Return the active strategy-tag column name. Prefers v2."""
    return "primary_strategy_tag_v2" if "primary_strategy_tag_v2" in df.columns else "primary_strategy_tag"


def filter_by_strategy_tag(df: pd.DataFrame, tag_value: str, tag_col: str) -> pd.DataFrame:
    """Keep only rows whose active strategy tag equals tag_value."""
    if tag_col not in df.columns:
        logger.warning("selection_policy: strategy tag column '%s' missing.", tag_col)
        return _empty_like(df)
    return df[df[tag_col].astype(str).eq(tag_value)].copy()


# ============================================================
# ELIGIBILITY FUNCTIONS
# ------------------------------------------------------------
# Each function applies a strategy's composite-score thresholds.
# It does NOT apply the strategy tag — the tag is applied separately
# so we can clearly see eligibility universe vs strategy-active universe.
# ============================================================

def eligible_for_swing(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_columns(df, ["tradability_score", "quality_safety_score"], "eligible_for_swing"):
        return _empty_like(df)
    g = SWING_ELIGIBILITY
    return df[
        (df["tradability_score"] >= g["min_tradability_score"]) &
        (df["quality_safety_score"] >= g["min_quality_safety_score"])
    ].copy()


def eligible_for_short_term(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_columns(df, ["tradability_score", "quality_safety_score"], "eligible_for_short_term"):
        return _empty_like(df)
    g = SHORT_TERM_ELIGIBILITY
    return df[
        (df["tradability_score"] >= g["min_tradability_score"]) &
        (df["quality_safety_score"] >= g["min_quality_safety_score"])
    ].copy()


def eligible_for_long_term(df: pd.DataFrame) -> pd.DataFrame:
    if not _has_columns(df, ["tradability_score", "quality_safety_score"], "eligible_for_long_term"):
        return _empty_like(df)
    g = LONG_TERM_ELIGIBILITY
    return df[
        (df["quality_safety_score"] >= g["min_quality_safety_score"]) &
        (df["tradability_score"] >= g["min_tradability_score"])
    ].copy()


def eligible_for_long_term_core(long_term_active_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Long Term Core factor gate on top of the Long-Term-active
    eligible universe."""
    needed = ["business_quality_factor", "cashflow_quality_factor", "risk_factor"]
    if not _has_columns(long_term_active_df, needed, "eligible_for_long_term_core"):
        return _empty_like(long_term_active_df)
    g = LONG_TERM_CORE_GATE
    return long_term_active_df[
        (long_term_active_df["business_quality_factor"] >= g["min_business_quality_factor"]) &
        (long_term_active_df["cashflow_quality_factor"] >= g["min_cashflow_quality_factor"]) &
        (long_term_active_df["risk_factor"] >= g["min_risk_factor"])
    ].copy()


def eligible_for_long_term_opportunity(
    long_term_active_df: pd.DataFrame,
    core_df: pd.DataFrame,
) -> pd.DataFrame:
    """Long-Term-active stocks that did NOT make Core."""
    if long_term_active_df.empty:
        return _empty_like(long_term_active_df)
    return long_term_active_df[
        ~long_term_active_df.index.isin(core_df.index)
    ].copy()


def eligible_for_multibagger(df: pd.DataFrame) -> pd.DataFrame:
    """STUB. Returns an empty frame and logs a warning. Real Multibagger
    selection rules will be designed in a later stage."""
    logger.info(
        "selection_policy: Multibagger eligibility is not implemented yet. "
        "Returning empty frame."
    )
    return _empty_like(df)


# ============================================================
# BUCKET SHORTLIST BUILDER
# ------------------------------------------------------------
# Same algorithm currently inlined inside pipeline.py.export_outputs.
# Sorts each market_cap_bucket by score and takes the top N per bucket.
# ============================================================

def build_bucket_shortlist(
    data: pd.DataFrame,
    score_col: str,
    bucket_mix: dict[str, int],
) -> pd.DataFrame:
    """Pick the top-N stocks per market_cap_bucket and return them sorted
    by score."""
    if data.empty:
        return _empty_like(data)
    if "market_cap_bucket" not in data.columns:
        logger.warning("selection_policy: 'market_cap_bucket' missing — returning empty shortlist.")
        return _empty_like(data)
    if score_col not in data.columns:
        logger.warning("selection_policy: score column '%s' missing — returning empty shortlist.", score_col)
        return _empty_like(data)

    parts = []
    for bucket, count in bucket_mix.items():
        if count <= 0:
            continue
        subset = (
            data[data["market_cap_bucket"] == bucket]
            .sort_values(score_col, ascending=False)
            .head(count)
        )
        parts.append(subset)

    if not parts:
        return _empty_like(data)

    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(score_col, ascending=False).reset_index(drop=True)


# ============================================================
# PER-STRATEGY SHORTLIST BUILDERS
# ------------------------------------------------------------
# Each composes: eligibility -> strategy-tag filter -> bucket shortlist.
# Long Term has two child shortlists (Core and Opportunity).
# ============================================================

def build_swing_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    score_col = resolve_active_score_columns(df)["swing"]
    tag_col = resolve_active_strategy_tag_column(df)
    eligible = eligible_for_swing(df)
    active = filter_by_strategy_tag(eligible, "Swing", tag_col)
    return build_bucket_shortlist(active, score_col, SWING_BUCKET_MIX)


def build_short_term_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    score_col = resolve_active_score_columns(df)["short_term"]
    tag_col = resolve_active_strategy_tag_column(df)
    eligible = eligible_for_short_term(df)
    active = filter_by_strategy_tag(eligible, "Short Term", tag_col)
    return build_bucket_shortlist(active, score_col, SHORT_TERM_BUCKET_MIX)


def build_long_term_active(df: pd.DataFrame) -> pd.DataFrame:
    """Helper: Long-Term-eligible AND tag = 'Long Term'."""
    tag_col = resolve_active_strategy_tag_column(df)
    eligible = eligible_for_long_term(df)
    return filter_by_strategy_tag(eligible, "Long Term", tag_col)


def build_long_term_core_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    score_col = resolve_active_score_columns(df)["long_term"]
    lt_active = build_long_term_active(df)
    core = eligible_for_long_term_core(lt_active)
    return build_bucket_shortlist(core, score_col, LONG_TERM_CORE_BUCKET_MIX)


def build_long_term_opportunity_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    score_col = resolve_active_score_columns(df)["long_term"]
    lt_active = build_long_term_active(df)
    core = eligible_for_long_term_core(lt_active)
    opp = eligible_for_long_term_opportunity(lt_active, core)
    return build_bucket_shortlist(opp, score_col, LONG_TERM_OPP_BUCKET_MIX)


def build_long_term_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """Combined Long Term shortlist (Core + Opportunity universe pooled)."""
    score_col = resolve_active_score_columns(df)["long_term"]
    lt_active = build_long_term_active(df)
    return build_bucket_shortlist(lt_active, score_col, LONG_TERM_BUCKET_MIX)


def build_multibagger_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """STUB. Returns an empty frame. Multibagger selection logic is
    intentionally deferred."""
    return eligible_for_multibagger(df)


# ============================================================
# AUDIT-ONLY UNIVERSE
# ------------------------------------------------------------
# Stocks that passed at least one eligibility gate but did NOT appear in
# any shortlist. Useful for transparency and Trainer-side review.
#
# PATCH: previously used nse_code as the identity key, which collapsed
# hundreds of blank-nse_code rows into one false-duplicate group. Now
# uses the stable stock key built from a fallback chain of identifiers.
# ============================================================

def build_audit_only_universe(
    df: pd.DataFrame,
    shortlists: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return rows that are eligible somewhere but not in any shortlist.

    Identity is matched via the stable stock key (nse_code -> isin ->
    bse_code -> stock_code -> best_stock_key -> stock_name -> synthetic
    row index). This is robust to blank nse_code values upstream.
    """
    # Add stable key to the master frame (internal, not exposed to caller)
    df_keyed = add_stable_stock_key(df)

    # Build the union of "eligible somewhere" stable keys
    swing_e = eligible_for_swing(df_keyed)
    short_e = eligible_for_short_term(df_keyed)
    long_e = eligible_for_long_term(df_keyed)

    eligible_keys: set[str] = set()
    for frame in (swing_e, short_e, long_e):
        if STABLE_KEY_COL in frame.columns:
            eligible_keys.update(frame[STABLE_KEY_COL].astype(str).tolist())

    if not eligible_keys:
        return _empty_like(df)

    # Build the union of shortlisted stable keys
    shortlisted_keys: set[str] = set()
    for sl in shortlists.values():
        if not isinstance(sl, pd.DataFrame) or sl.empty:
            continue
        sl_keyed = add_stable_stock_key(sl)
        if STABLE_KEY_COL in sl_keyed.columns:
            shortlisted_keys.update(sl_keyed[STABLE_KEY_COL].astype(str).tolist())

    audit_keys = eligible_keys - shortlisted_keys
    if not audit_keys:
        return _empty_like(df)

    # Return original rows from df (without exposing the stable key column).
    audit_mask = df_keyed[STABLE_KEY_COL].astype(str).isin(audit_keys)
    return df.loc[df_keyed.index[audit_mask]].copy()


# ============================================================
# REPORT QUEUE BUILDERS
# ------------------------------------------------------------
# The report queue says: "for each strategy, here are the top-N stocks
# the report layer should generate a report for."
#
# Two builders:
#   - Python fallback queue: ALWAYS available. Source of truth.
#   - DeepSeek queue: same shape, but only populated when enabled=True.
#     If enabled=False, returns an empty queue and logs an info line.
# ============================================================

def _top_n_per_shortlist(
    shortlists: dict[str, pd.DataFrame],
    score_cols: dict[str, str],
    top_n: int,
) -> dict[str, pd.DataFrame]:
    """Sort each shortlist by its strategy's score column, head(top_n)."""
    queue: dict[str, pd.DataFrame] = {}
    label_to_score_key = {
        "Swing_Shortlist": "swing",
        "ShortTerm_Shortlist": "short_term",
        "LongTerm_Core_Shortlist": "long_term",
        "LongTerm_Opp_Shortlist": "long_term",
        "LongTerm_Shortlist": "long_term",
        "Multibagger_Shortlist": "long_term",  # placeholder; revisit later
    }

    for label, sl in shortlists.items():
        if not isinstance(sl, pd.DataFrame) or sl.empty:
            queue[label] = sl if isinstance(sl, pd.DataFrame) else pd.DataFrame()
            continue
        score_key = label_to_score_key.get(label, "swing")
        score_col = score_cols.get(score_key)
        if score_col and score_col in sl.columns:
            queue[label] = sl.sort_values(score_col, ascending=False).head(top_n).copy()
        else:
            queue[label] = sl.head(top_n).copy()

    return queue


def build_python_fallback_queue(
    shortlists: dict[str, pd.DataFrame],
    score_cols: dict[str, str],
    top_n: int = REPORT_QUEUE_TOP_N,
) -> dict[str, pd.DataFrame]:
    """Always-available report queue for Python fallback reports.
    This is the deterministic source of truth used by every student run."""
    return _top_n_per_shortlist(shortlists, score_cols, top_n)


def build_deepseek_queue(
    shortlists: dict[str, pd.DataFrame],
    score_cols: dict[str, str],
    top_n: int = REPORT_QUEUE_TOP_N,
    enabled: bool = False,
) -> dict[str, pd.DataFrame]:
    """Optional DeepSeek queue. Same shape as the Python queue but only
    populated when the Trainer explicitly enables DeepSeek. When
    enabled=False, returns an empty dict and the pipeline silently
    skips DeepSeek calls — students never need a DeepSeek API key."""
    if not enabled:
        logger.info(
            "selection_policy: DeepSeek queue disabled. "
            "Python fallback queue is the active report queue."
        )
        return {}
    return _top_n_per_shortlist(shortlists, score_cols, top_n)


# ============================================================
# MASTER ORCHESTRATOR
# ------------------------------------------------------------
# Single entry point for pipeline.py to call (later). Returns a dict
# whose keys are deliberately compatible with what export_outputs
# already produces, so the migration is a swap, not a redesign.
# ============================================================

def run_selection_policy(
    df: pd.DataFrame,
    deepseek_enabled: bool = False,
    report_queue_top_n: int = REPORT_QUEUE_TOP_N,
) -> dict:
    """Run the full Python selection layer end to end.

    Inputs:
        df: the scored & tagged master DataFrame produced upstream
            (must contain quality_safety_score, tradability_score,
            market_cap_bucket, primary_strategy_tag[_v2], score columns).
        deepseek_enabled: Trainer flag. False = students path. Default False.
        report_queue_top_n: how deep the report queue goes per strategy.

    Returns a dict with:
        active_score_cols           dict[str, str]
        strategy_tag_active         str
        swing_eligible              DataFrame
        short_term_eligible         DataFrame
        long_term_eligible          DataFrame
        swing_active_eligible       DataFrame   (eligible AND tag == "Swing")
        short_term_active_eligible  DataFrame   (eligible AND tag == "Short Term")
        long_term_active_eligible   DataFrame   (eligible AND tag == "Long Term")
        long_term_core_eligible     DataFrame
        long_term_opportunity_eligible  DataFrame
        swing_shortlist             DataFrame
        short_term_shortlist        DataFrame
        long_term_shortlist         DataFrame
        long_term_core_shortlist    DataFrame
        long_term_opportunity_shortlist DataFrame
        multibagger_shortlist       DataFrame  (empty stub for now)
        audit_only_universe         DataFrame
        python_fallback_queue       dict[str, DataFrame]
        deepseek_queue              dict[str, DataFrame]  (empty unless enabled)
    """
    score_cols = resolve_active_score_columns(df)
    tag_col = resolve_active_strategy_tag_column(df)

    # Eligibility universes (raw gates, no strategy tag yet)
    swing_e = eligible_for_swing(df)
    short_e = eligible_for_short_term(df)
    long_e = eligible_for_long_term(df)

    # Strategy-tag-active eligible universes (eligible AND primary tag matches).
    # Patch C: exposed as top-level return keys so pipeline.py does not need
    # to recompute them.
    swing_active_e = filter_by_strategy_tag(swing_e, "Swing", tag_col)
    short_term_active_e = filter_by_strategy_tag(short_e, "Short Term", tag_col)
    lt_active = filter_by_strategy_tag(long_e, "Long Term", tag_col)

    # Long Term Core / Opportunity universes (after tag filter)
    lt_core_e = eligible_for_long_term_core(lt_active)
    lt_opp_e = eligible_for_long_term_opportunity(lt_active, lt_core_e)

    # Shortlists
    swing_sl = build_swing_shortlist(df)
    short_sl = build_short_term_shortlist(df)
    long_sl = build_long_term_shortlist(df)
    long_core_sl = build_long_term_core_shortlist(df)
    long_opp_sl = build_long_term_opportunity_shortlist(df)
    multibagger_sl = build_multibagger_shortlist(df)

    shortlists_for_reports = {
        "Swing_Shortlist": swing_sl,
        "ShortTerm_Shortlist": short_sl,
        "LongTerm_Core_Shortlist": long_core_sl,
        "LongTerm_Opp_Shortlist": long_opp_sl,
    }

    # Patch C.1: the audit-only universe must subtract ALL shortlists,
    # including the combined Long Term shortlist. The combined LongTerm
    # bucket mix can include Mid Cap stocks that fall outside the Core/Opp
    # sub-mix (Core has 80 Mid slots, Opp has 50, combined LongTerm has 80;
    # stocks failing the Core factor gate that also miss Opp's tighter Mid
    # cut land only in the combined view). Those stocks were leaking into
    # Audit_Only.
    #
    # Report queues continue to use the 4-key shortlists_for_reports dict
    # so report-generation behaviour is unchanged.
    all_shortlists = {
        **shortlists_for_reports,
        "LongTerm_Shortlist": long_sl,
    }

    audit_only = build_audit_only_universe(df, all_shortlists)

    python_queue = build_python_fallback_queue(
        shortlists_for_reports, score_cols, top_n=report_queue_top_n,
    )
    deepseek_queue = build_deepseek_queue(
        shortlists_for_reports, score_cols, top_n=report_queue_top_n,
        enabled=deepseek_enabled,
    )

    return {
        "active_score_cols": score_cols,
        "strategy_tag_active": tag_col,
        "swing_eligible": swing_e,
        "short_term_eligible": short_e,
        "long_term_eligible": long_e,
        "swing_active_eligible": swing_active_e,
        "short_term_active_eligible": short_term_active_e,
        "long_term_active_eligible": lt_active,
        "long_term_core_eligible": lt_core_e,
        "long_term_opportunity_eligible": lt_opp_e,
        "swing_shortlist": swing_sl,
        "short_term_shortlist": short_sl,
        "long_term_shortlist": long_sl,
        "long_term_core_shortlist": long_core_sl,
        "long_term_opportunity_shortlist": long_opp_sl,
        "multibagger_shortlist": multibagger_sl,
        "audit_only_universe": audit_only,
        "python_fallback_queue": python_queue,
        "deepseek_queue": deepseek_queue,
    }


# ============================================================
# MODULE GUARD
# ============================================================

if __name__ == "__main__":
    # This module is not meant to be run directly. Import it from
    # pipeline.py (in a later stage) and call run_selection_policy(df).
    print(
        "selection_policy.py is a library module.\n"
        "Import it and call run_selection_policy(df). "
        "Not yet wired into pipeline.py."
    )
