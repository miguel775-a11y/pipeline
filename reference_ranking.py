"""
reference_ranking.py
=====================

Reference Ranking Layer (Patch RR-A skeleton).

Purpose
-------
For each stock category produced by selection_policy.run_selection_policy(...),
build a *reference list* that ranks the rows from best to worst and adds
human-friendly grades (A+ / A / B+ / B / C / Avoid), top strengths, main
weakness, and a recommended report priority.

Reference grades and phrases are a *presentation and decision-support*
layer. They never replace the underlying numeric scores. Numeric scoring
in scoring.py and selection logic in selection_policy.py remain the
source of truth.

Architecture role
-----------------

        Data -> Scores -> Tags -> Select (selection_policy.py)
                                    |
                                    v
                        ----------------------------
                        | reference_ranking.py     |   <- THIS FILE
                        ----------------------------
                                    |
                                    v
                              Report (pipeline.py / deepseek_reports.py)

Design rules
------------
* Python is the source of truth. This module never imports DeepSeek.
* Pure functions: DataFrame in, DataFrame (or dict) out. No I/O.
* DeepSeek is optional and never required. Students run without it.
* `stable_stock_key` is the required identity. `nse_code` is optional and
  display-only. Rows must NOT be dropped because nse_code is missing.

Patch status (RR-A)
-------------------
This patch ships only the skeleton:

  * module-level constants for the actual tag vocabulary used by the workbook
  * the 23-column output schema
  * defensive helpers (safe getters, stable-key builder, score resolver)
  * report-priority + must-generate-report logic (rules are simple and
    grade-driven so they land safely now)
  * phrase formatters as safe stubs
  * six public `build_*_reference(...)` functions that return empty
    23-column frames
  * `run_reference_ranking(policy_result)` orchestrator returning the
    six expected keys

Real grading rules (quintiles, modifiers, A+/A/B+/B/C/Avoid assignment),
real phrase content, and the Long Term Combined sub-typing land in later
patches (RR-B, RR-C, RR-D).

This module is NOT yet imported by pipeline.py. Wiring is RR-E.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# CONSTANTS — actual tag values (audited from real workbook)
# =====================================================================

# entry_quality_tag (best -> worst)
ENTRY_QUALITY_CLEAN = "Clean Entry"
ENTRY_QUALITY_PULLBACK = "Watch on Pullback"
ENTRY_QUALITY_NEUTRAL = "Neutral"
ENTRY_QUALITY_INSUFFICIENT = "Insufficient Data"
ENTRY_QUALITY_CROWDED = "Crowded Trend"

ENTRY_QUALITY_VALUES = (
    ENTRY_QUALITY_CLEAN,
    ENTRY_QUALITY_PULLBACK,
    ENTRY_QUALITY_NEUTRAL,
    ENTRY_QUALITY_INSUFFICIENT,
    ENTRY_QUALITY_CROWDED,
)

# setup_quality_tag (best -> worst)
SETUP_QUALITY_HIGH = "High Quality"
SETUP_QUALITY_ACCEPTABLE = "Acceptable Quality"
SETUP_QUALITY_FRAGILE = "Fragile Quality"

SETUP_QUALITY_VALUES = (
    SETUP_QUALITY_HIGH,
    SETUP_QUALITY_ACCEPTABLE,
    SETUP_QUALITY_FRAGILE,
)

# setup_confirmation_tag (best -> worst)
SETUP_CONFIRMATION_STRONG = "Strong Confirmation"
SETUP_CONFIRMATION_PARTIAL = "Partial Confirmation"
SETUP_CONFIRMATION_WEAK = "Weak Confirmation"

SETUP_CONFIRMATION_VALUES = (
    SETUP_CONFIRMATION_STRONG,
    SETUP_CONFIRMATION_PARTIAL,
    SETUP_CONFIRMATION_WEAK,
)

# setup_risk_tag (best -> worst)
SETUP_RISK_CONTAINED = "Contained Risk"
SETUP_RISK_MODERATE = "Moderate Risk"
SETUP_RISK_ELEVATED = "Elevated Risk"

SETUP_RISK_VALUES = (
    SETUP_RISK_CONTAINED,
    SETUP_RISK_MODERATE,
    SETUP_RISK_ELEVATED,
)

# market_cap_bucket (matches selection_policy.py keys)
MARKET_CAP_LARGE = "Large Cap"
MARKET_CAP_MID = "Mid Cap"
MARKET_CAP_SMALL = "Small Cap"
MARKET_CAP_MICRO = "Micro Cap"

MARKET_CAP_VALUES = (
    MARKET_CAP_LARGE,
    MARKET_CAP_MID,
    MARKET_CAP_SMALL,
    MARKET_CAP_MICRO,
)


# =====================================================================
# CONSTANTS — grades and strategy categories
# =====================================================================

GRADE_A_PLUS = "A+"
GRADE_A = "A"
GRADE_B_PLUS = "B+"
GRADE_B = "B"
GRADE_C = "C"
GRADE_AVOID = "Avoid"

GRADE_VALUES = (
    GRADE_A_PLUS,
    GRADE_A,
    GRADE_B_PLUS,
    GRADE_B,
    GRADE_C,
    GRADE_AVOID,
)

# Strategy categories used in the `strategy_category` output column.
STRATEGY_SWING = "Swing"
STRATEGY_SHORT_TERM = "Short Term"
STRATEGY_LONG_TERM_CORE = "Long Term Core"
STRATEGY_LONG_TERM_OPP = "Long Term Opportunity"
STRATEGY_LONG_TERM_COMBINED_CORE = "Long Term — Core"
STRATEGY_LONG_TERM_COMBINED_OPP = "Long Term — Opportunity"
STRATEGY_LONG_TERM_COMBINED_OTHER = "Long Term — Other"
STRATEGY_AUDIT_ONLY = "Audit Only"


# =====================================================================
# CONSTANTS — output schema, report priority, must-generate-report
# =====================================================================

# Column order for every reference DataFrame (23 columns).
REFERENCE_COLUMNS = [
    "reference_rank",
    "reference_grade",
    "stable_stock_key",
    "stock_name",
    "nse_code",
    "market_cap_bucket",
    "strategy_category",
    "strategy_score",
    "quality_safety_score",
    "tradability_score",
    "entry_quality_tag",
    "setup_quality_tag",
    "setup_confirmation_tag",
    "setup_risk_tag",
    "red_flag_count",
    "positive_flag_count",
    "analysis_confidence_score",
    "top_strengths",
    "main_weakness",
    "why_ranked_here",
    "suggested_action_view",
    "must_generate_report",
    "report_priority",
]

# Output keys for run_reference_ranking(...) — match planned sheet names.
REFERENCE_OUTPUT_KEYS = (
    "Swing_Reference",
    "ShortTerm_Reference",
    "LongTerm_Core_Reference",
    "LongTerm_Opp_Reference",
    "LongTerm_Reference",
    "Audit_Only_Reference",
)

# Grade -> report priority (1 = highest, 5 = lowest).
REPORT_PRIORITY_BY_GRADE = {
    GRADE_A_PLUS: 1,
    GRADE_A: 2,
    GRADE_B_PLUS: 3,
    GRADE_B: 4,
    GRADE_C: 5,
    GRADE_AVOID: 5,
}

# Per-strategy must_generate_report rule, by grade.
# Audit-only "A" grade -> review report (YES); A+ is disabled for audit-only.
MUST_GENERATE_REPORT_RULE = {
    STRATEGY_SWING: {
        GRADE_A_PLUS: "YES", GRADE_A: "YES", GRADE_B_PLUS: "YES",
        GRADE_B: "NO", GRADE_C: "NO", GRADE_AVOID: "NO",
    },
    STRATEGY_SHORT_TERM: {
        GRADE_A_PLUS: "YES", GRADE_A: "YES", GRADE_B_PLUS: "YES",
        GRADE_B: "NO", GRADE_C: "NO", GRADE_AVOID: "NO",
    },
    STRATEGY_LONG_TERM_CORE: {
        GRADE_A_PLUS: "YES", GRADE_A: "YES", GRADE_B_PLUS: "YES",
        GRADE_B: "NO", GRADE_C: "NO", GRADE_AVOID: "NO",
    },
    STRATEGY_LONG_TERM_OPP: {
        GRADE_A_PLUS: "YES", GRADE_A: "YES", GRADE_B_PLUS: "YES",
        GRADE_B: "NO", GRADE_C: "NO", GRADE_AVOID: "NO",
    },
    STRATEGY_AUDIT_ONLY: {
        # A+ is intentionally absent for audit-only; A is the top.
        GRADE_A: "YES", GRADE_B_PLUS: "NO",
        GRADE_B: "NO", GRADE_C: "NO", GRADE_AVOID: "NO",
    },
}


# =====================================================================
# CONSTANTS — Swing phrase rules (RR-B)
# =====================================================================
# Rules are stored as data, not code. Each entry is a (predicate, label)
# tuple. The predicates are small lambdas reading via _safe_get* so they
# never raise on missing values. The labels are the exact phrases that
# appear in the Excel output.
#
# To add or change a rule, edit this list — no logic changes needed.
# =====================================================================

# Up to three labels picked in this priority order.
_SWING_STRENGTH_RULES = [
    # (predicate(row) -> bool, label)
    (lambda r: (_safe_get_float(r, "tradability_score") or 0) >= 70,
     "high tradability"),
    (lambda r: _safe_get(r, "setup_confirmation_tag") == SETUP_CONFIRMATION_STRONG,
     "strong confirmation"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_CLEAN,
     "clean entry"),
    (lambda r: _safe_get(r, "setup_quality_tag") == SETUP_QUALITY_HIGH,
     "high quality setup"),
    (lambda r: _safe_get(r, "setup_risk_tag") == SETUP_RISK_CONTAINED,
     "contained risk"),
    (lambda r: _safe_get_int(r, "red_flag_count") == 0,
     "few red flags"),
    (lambda r: (_safe_get_int(r, "positive_flag_count") or 0) >= 4,
     "many positive flags"),
]
_SWING_STRENGTHS_DEFAULT = "limited strengths"

# First-applicable rule wins. Order = severity (worst first).
_SWING_WEAKNESS_RULES = [
    (lambda r: (_safe_get_int(r, "red_flag_count") or 0) >= 3,
     lambda r: f"multiple red flags ({_safe_get_int(r, 'red_flag_count')})"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_CROWDED,
     lambda r: "crowded trend setup"),
    (lambda r: _safe_get(r, "setup_confirmation_tag") == SETUP_CONFIRMATION_WEAK,
     lambda r: "weak confirmation"),
    (lambda r: _safe_get(r, "setup_risk_tag") == SETUP_RISK_ELEVATED,
     lambda r: "elevated setup risk"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_INSUFFICIENT,
     lambda r: "insufficient data for clear setup"),
    (lambda r: _safe_get(r, "setup_quality_tag") == SETUP_QUALITY_FRAGILE,
     lambda r: "fragile setup quality"),
    (lambda r: (_safe_get_float(r, "tradability_score") is not None
                and _safe_get_float(r, "tradability_score") < 45),
     lambda r: "low tradability"),
    (lambda r: (_safe_get_float(r, "analysis_confidence_score") is not None
                and _safe_get_float(r, "analysis_confidence_score") < 0.6),
     lambda r: "lower analysis confidence"),
    (lambda r: ((_safe_get_int(r, "positive_flag_count") or 0) <= 1
                and (_safe_get_int(r, "red_flag_count") or 0) >= 1),
     lambda r: "few positive offsets"),
]
_SWING_WEAKNESS_DEFAULT = "no major weakness flagged"

# Grade -> action phrase. Phrases avoid "buy", "sell", "guaranteed", "target".
SUGGESTED_ACTION_VIEW_SWING = {
    GRADE_A_PLUS: "Strong candidate — review entry levels carefully",
    GRADE_A:      "Good candidate — review entry plan",
    GRADE_B_PLUS: "Watchlist — wait for confirmation",
    GRADE_B:      "Watchlist only",
    GRADE_C:      "Skip for now",
    GRADE_AVOID:  "Avoid in current setup",
}

# why_ranked_here templates per grade. {score} is filled with a 1-decimal
# float; {detail} with a strength or weakness phrase.
WHY_RANKED_TEMPLATE_SWING = {
    GRADE_A_PLUS: "Top quintile swing score of {score:.1f}, with {detail}.",
    GRADE_A:      "Strong swing score of {score:.1f} and {detail}.",
    GRADE_B_PLUS: "Solid swing score of {score:.1f} but {detail}.",
    GRADE_B:      "Acceptable score of {score:.1f}; held back by {detail}.",
    GRADE_C:      "Below-average score of {score:.1f}; {detail}.",
    GRADE_AVOID:  "Currently unsuitable: {detail}.",
}

# Track which optional columns we've already logged as missing — log once.
_RR_LOGGED_MISSING_OPTIONAL: set = set()


# =====================================================================
# CONSTANTS — Short Term phrase rules (RR-C)
# =====================================================================

_SHORT_TERM_STRENGTH_RULES = [
    (lambda r: (_safe_get_float(r, "tradability_score") or 0) >= 70,
     "high tradability"),
    (lambda r: _safe_get(r, "setup_confirmation_tag") == SETUP_CONFIRMATION_STRONG,
     "strong confirmation"),
    (lambda r: _safe_get(r, "setup_quality_tag") == SETUP_QUALITY_HIGH,
     "high quality setup"),
    (lambda r: _safe_get(r, "setup_risk_tag") == SETUP_RISK_CONTAINED,
     "contained risk"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_CLEAN,
     "clean entry"),
    (lambda r: _safe_get_int(r, "red_flag_count") == 0,
     "few red flags"),
    (lambda r: (_safe_get_int(r, "positive_flag_count") or 0) >= 4,
     "many positive flags"),
]

_SHORT_TERM_WEAKNESS_RULES = [
    (lambda r: (_safe_get_int(r, "red_flag_count") or 0) >= 3,
     lambda r: f"multiple red flags ({_safe_get_int(r, 'red_flag_count')})"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_CROWDED,
     lambda r: "crowded trend setup"),
    (lambda r: _safe_get(r, "setup_confirmation_tag") == SETUP_CONFIRMATION_WEAK,
     lambda r: "weak confirmation"),
    (lambda r: _safe_get(r, "setup_risk_tag") == SETUP_RISK_ELEVATED,
     lambda r: "elevated setup risk"),
    (lambda r: _safe_get(r, "setup_quality_tag") == SETUP_QUALITY_FRAGILE,
     lambda r: "fragile setup quality"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_INSUFFICIENT,
     lambda r: "insufficient data for clear setup"),
    (lambda r: (_safe_get_float(r, "tradability_score") is not None
                and _safe_get_float(r, "tradability_score") < 45),
     lambda r: "low tradability"),
    (lambda r: (_safe_get_float(r, "analysis_confidence_score") is not None
                and _safe_get_float(r, "analysis_confidence_score") < 0.6),
     lambda r: "lower analysis confidence"),
    (lambda r: ((_safe_get_int(r, "positive_flag_count") or 0) <= 1
                and (_safe_get_int(r, "red_flag_count") or 0) >= 1),
     lambda r: "few positive offsets"),
]

SUGGESTED_ACTION_VIEW_SHORT_TERM = {
    GRADE_A_PLUS: "Strong short-term candidate — review entry levels carefully",
    GRADE_A:      "Good short-term candidate — review entry plan",
    GRADE_B_PLUS: "Watchlist — wait for confirmation",
    GRADE_B:      "Watchlist only",
    GRADE_C:      "Skip for now",
    GRADE_AVOID:  "Avoid in current setup",
}

WHY_RANKED_TEMPLATE_SHORT_TERM = {
    GRADE_A_PLUS: "Top quintile short-term score of {score:.1f}, with {detail}.",
    GRADE_A:      "Strong short-term score of {score:.1f} and {detail}.",
    GRADE_B_PLUS: "Solid short-term score of {score:.1f} but {detail}.",
    GRADE_B:      "Acceptable score of {score:.1f}; held back by {detail}.",
    GRADE_C:      "Below-average score of {score:.1f}; {detail}.",
    GRADE_AVOID:  "Currently unsuitable: {detail}.",
}


# =====================================================================
# CONSTANTS — Long Term Core phrase rules (RR-C)
# =====================================================================

_LONG_TERM_CORE_STRENGTH_RULES = [
    (lambda r: (_safe_get_float(r, "business_quality_factor") or 0) >= 70,
     "strong business quality"),
    (lambda r: (_safe_get_float(r, "cashflow_quality_factor") or 0) >= 60,
     "healthy cash flows"),
    (lambda r: (_safe_get_float(r, "risk_factor") or 0) >= 60,
     "low financial risk"),
    (lambda r: (_safe_get_float(r, "valuation_factor") or 0) >= 50,
     "reasonable valuation"),
    (lambda r: (_safe_get_float(r, "quality_safety_score") or 0) >= 75,
     "high quality+safety"),
    (lambda r: (_safe_get_int(r, "red_flag_count") is not None
                and _safe_get_int(r, "red_flag_count") <= 1),
     "few red flags"),
]

_LONG_TERM_CORE_WEAKNESS_RULES = [
    (lambda r: (_safe_get_int(r, "red_flag_count") or 0) >= 3,
     lambda r: f"multiple red flags ({_safe_get_int(r, 'red_flag_count')})"),
    (lambda r: (_safe_get_float(r, "risk_factor") is not None
                and _safe_get_float(r, "risk_factor") < 50),
     lambda r: "elevated business risk"),
    (lambda r: (_safe_get_float(r, "valuation_factor") is not None
                and _safe_get_float(r, "valuation_factor") < 30),
     lambda r: "stretched valuation"),
    (lambda r: (_safe_get_float(r, "quality_safety_score") is not None
                and _safe_get_float(r, "quality_safety_score") < 60),
     lambda r: "modest quality+safety"),
    (lambda r: _safe_get(r, "setup_quality_tag") == SETUP_QUALITY_FRAGILE,
     lambda r: "fragile setup quality"),
    (lambda r: (_safe_get_float(r, "analysis_confidence_score") is not None
                and _safe_get_float(r, "analysis_confidence_score") < 0.6),
     lambda r: "lower analysis confidence"),
]

SUGGESTED_ACTION_VIEW_LONG_TERM_CORE = {
    GRADE_A_PLUS: "Strong long-term candidate — review thesis and valuation",
    GRADE_A:      "Good long-term candidate — review thesis",
    GRADE_B_PLUS: "Long-term watchlist — track quarterly results",
    GRADE_B:      "Long-term watchlist only",
    GRADE_C:      "Low priority for long-term",
    GRADE_AVOID:  "Not a long-term fit currently",
}

WHY_RANKED_TEMPLATE_LONG_TERM_CORE = {
    GRADE_A_PLUS: "Top quintile long-term score of {score:.1f}, with {detail}.",
    GRADE_A:      "Strong long-term score of {score:.1f} and {detail}.",
    GRADE_B_PLUS: "Solid long-term score of {score:.1f} but {detail}.",
    GRADE_B:      "Acceptable score of {score:.1f}; held back by {detail}.",
    GRADE_C:      "Below-average score of {score:.1f}; {detail}.",
    GRADE_AVOID:  "Currently unsuitable: {detail}.",
}


# =====================================================================
# CONSTANTS — Long Term Opportunity phrase rules (RR-C)
# =====================================================================

_LONG_TERM_OPP_STRENGTH_RULES = [
    (lambda r: (_safe_get_float(r, "growth_factor") or 0) >= 70,
     "high growth potential"),
    (lambda r: (_safe_get_float(r, "catalyst_proxy_factor") or 0) >= 60,
     "near-term catalyst"),
    (lambda r: (_safe_get_float(r, "valuation_factor") or 0) >= 60,
     "attractive valuation"),
    (lambda r: (_safe_get_float(r, "risk_factor") or 0) >= 50,
     "decent risk profile"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_CLEAN,
     "clean entry"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_PULLBACK,
     "watchable pullback"),
    (lambda r: (_safe_get_int(r, "red_flag_count") is not None
                and _safe_get_int(r, "red_flag_count") <= 1),
     "few red flags"),
]

_LONG_TERM_OPP_WEAKNESS_RULES = [
    (lambda r: (_safe_get_int(r, "red_flag_count") or 0) >= 3,
     lambda r: f"multiple red flags ({_safe_get_int(r, 'red_flag_count')})"),
    (lambda r: (_safe_get_float(r, "risk_factor") is not None
                and _safe_get_float(r, "risk_factor") < 40),
     lambda r: "elevated risk profile"),
    (lambda r: (_safe_get_float(r, "valuation_factor") is not None
                and _safe_get_float(r, "valuation_factor") < 30),
     lambda r: "stretched valuation"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_CROWDED,
     lambda r: "crowded trend setup"),
    (lambda r: (_safe_get_float(r, "analysis_confidence_score") is not None
                and _safe_get_float(r, "analysis_confidence_score") < 0.6),
     lambda r: "lower analysis confidence"),
]

SUGGESTED_ACTION_VIEW_LONG_TERM_OPP = {
    GRADE_A_PLUS: "Opportunity candidate — verify catalyst and timing",
    GRADE_A:      "Opportunity candidate — track for setup",
    GRADE_B_PLUS: "Opportunity watchlist — wait for clearer entry",
    GRADE_B:      "Opportunity watchlist only",
    GRADE_C:      "Low priority opportunity",
    GRADE_AVOID:  "Not an opportunity fit currently",
}

WHY_RANKED_TEMPLATE_LONG_TERM_OPP = {
    GRADE_A_PLUS: "Top quintile long-term score of {score:.1f}, with {detail}.",
    GRADE_A:      "Strong long-term score of {score:.1f} and {detail}.",
    GRADE_B_PLUS: "Solid long-term score of {score:.1f} but {detail}.",
    GRADE_B:      "Acceptable score of {score:.1f}; held back by {detail}.",
    GRADE_C:      "Below-average score of {score:.1f}; {detail}.",
    GRADE_AVOID:  "Currently unsuitable: {detail}.",
}


# =====================================================================
# CONSTANTS — Audit Only phrase rules (RR-D)
# ---------------------------------------------------------------------
# Audit Only is a REVIEW list, not a recommendation list. The grade
# scale deliberately excludes A+. Phrase wording uses "review" framing
# rather than "candidate" framing — no "buy", "sell", "guaranteed",
# "target", or "candidate".
# =====================================================================

_AUDIT_ONLY_STRENGTH_RULES = [
    (lambda r: (_safe_get_float(r, "quality_safety_score") or 0) >= 70,
     "high quality+safety"),
    (lambda r: (_safe_get_float(r, "tradability_score") or 0) >= 50,
     "decent tradability"),
    (lambda r: (_safe_get_int(r, "red_flag_count") is not None
                and _safe_get_int(r, "red_flag_count") <= 1),
     "few red flags"),
    (lambda r: (_safe_get_int(r, "positive_flag_count") or 0) >= 4,
     "many positive flags"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_CLEAN,
     "clean entry"),
    (lambda r: _safe_get(r, "setup_risk_tag") == SETUP_RISK_CONTAINED,
     "contained risk"),
]

_AUDIT_ONLY_WEAKNESS_RULES = [
    (lambda r: (_safe_get_int(r, "red_flag_count") or 0) >= 3,
     lambda r: f"multiple red flags ({_safe_get_int(r, 'red_flag_count')})"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_CROWDED,
     lambda r: "crowded trend setup"),
    (lambda r: _safe_get(r, "entry_quality_tag") == ENTRY_QUALITY_INSUFFICIENT,
     lambda r: "insufficient data for clear setup"),
    (lambda r: _safe_get(r, "setup_risk_tag") == SETUP_RISK_ELEVATED,
     lambda r: "elevated setup risk"),
    (lambda r: (_safe_get_float(r, "tradability_score") is not None
                and _safe_get_float(r, "tradability_score") < 35),
     lambda r: "low tradability"),
    (lambda r: (_safe_get_float(r, "analysis_confidence_score") is not None
                and _safe_get_float(r, "analysis_confidence_score") < 0.6),
     lambda r: "lower analysis confidence"),
]

# Note: NO GRADE_A_PLUS key — A+ is structurally disabled for Audit Only.
SUGGESTED_ACTION_VIEW_AUDIT_ONLY = {
    GRADE_A:      "Worth reviewing in next session",
    GRADE_B_PLUS: "Mark for later review",
    GRADE_B:      "Note only — no action",
    GRADE_C:      "Low-priority audit row",
    GRADE_AVOID:  "Skip — too many flags",
}

# Note: NO GRADE_A_PLUS key — review framing, not candidate framing.
WHY_RANKED_TEMPLATE_AUDIT_ONLY = {
    GRADE_A:      "High audit score of {score:.1f}: {detail}. Worth reviewing further.",
    GRADE_B_PLUS: "Audit score of {score:.1f}; {detail}. Mark for later review.",
    GRADE_B:      "Audit score of {score:.1f}; {detail}. Note only.",
    GRADE_C:      "Low audit score of {score:.1f}; {detail}.",
    GRADE_AVOID:  "Currently inappropriate for review queue: {detail}.",
}

# Synthetic ordering score for Audit Only:
#   audit_reference_score = 0.6 * quality_safety_score + 0.4 * tradability_score
# Documented as data so a trainer can adjust without touching logic.
_AUDIT_QSS_WEIGHT = 0.6
_AUDIT_TRADABILITY_WEIGHT = 0.4


# =====================================================================
# CONSTANTS — strategy dispatch (RR-C)
# =====================================================================
# Maps each strategy to its rule lists / phrase tables. Keeps the four
# phrase formatters small: each does a single dict lookup, then walks the
# selected rule list with the same loop body RR-B already ships.
# =====================================================================

_STRENGTH_RULES_BY_STRATEGY = {
    STRATEGY_SWING:           _SWING_STRENGTH_RULES,
    STRATEGY_SHORT_TERM:      _SHORT_TERM_STRENGTH_RULES,
    STRATEGY_LONG_TERM_CORE:  _LONG_TERM_CORE_STRENGTH_RULES,
    STRATEGY_LONG_TERM_OPP:   _LONG_TERM_OPP_STRENGTH_RULES,
    STRATEGY_AUDIT_ONLY:      _AUDIT_ONLY_STRENGTH_RULES,
}

_WEAKNESS_RULES_BY_STRATEGY = {
    STRATEGY_SWING:           _SWING_WEAKNESS_RULES,
    STRATEGY_SHORT_TERM:      _SHORT_TERM_WEAKNESS_RULES,
    STRATEGY_LONG_TERM_CORE:  _LONG_TERM_CORE_WEAKNESS_RULES,
    STRATEGY_LONG_TERM_OPP:   _LONG_TERM_OPP_WEAKNESS_RULES,
    STRATEGY_AUDIT_ONLY:      _AUDIT_ONLY_WEAKNESS_RULES,
}

_SUGGESTED_ACTION_VIEW_BY_STRATEGY = {
    STRATEGY_SWING:           SUGGESTED_ACTION_VIEW_SWING,
    STRATEGY_SHORT_TERM:      SUGGESTED_ACTION_VIEW_SHORT_TERM,
    STRATEGY_LONG_TERM_CORE:  SUGGESTED_ACTION_VIEW_LONG_TERM_CORE,
    STRATEGY_LONG_TERM_OPP:   SUGGESTED_ACTION_VIEW_LONG_TERM_OPP,
    STRATEGY_AUDIT_ONLY:      SUGGESTED_ACTION_VIEW_AUDIT_ONLY,
}

_WHY_RANKED_TEMPLATE_BY_STRATEGY = {
    STRATEGY_SWING:           WHY_RANKED_TEMPLATE_SWING,
    STRATEGY_SHORT_TERM:      WHY_RANKED_TEMPLATE_SHORT_TERM,
    STRATEGY_LONG_TERM_CORE:  WHY_RANKED_TEMPLATE_LONG_TERM_CORE,
    STRATEGY_LONG_TERM_OPP:   WHY_RANKED_TEMPLATE_LONG_TERM_OPP,
    STRATEGY_AUDIT_ONLY:      WHY_RANKED_TEMPLATE_AUDIT_ONLY,
}


# =====================================================================
# CONSTANTS — stable stock key
# =====================================================================

# Identifier columns in fallback order (matches selection_policy.py).
STABLE_KEY_COL = "_stable_stock_key"
STABLE_KEY_FALLBACK_ORDER = [
    "nse_code",
    "isin",
    "bse_code",
    "stock_code",
    "best_stock_key",
    "stock_name",
]

# Short prefix per identifier source so different sources can't collide.
_STABLE_KEY_PREFIXES = {
    "nse_code": "nse",
    "isin": "isin",
    "bse_code": "bse",
    "stock_code": "stk",
    "best_stock_key": "bsk",
    "stock_name": "name",
}

# Strings (lower-cased) that count as "blank" everywhere in this module.
_INVALID_KEY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "--"}


# =====================================================================
# DEFENSIVE HELPERS
# =====================================================================

def _is_blank(value) -> bool:
    """True if value should be treated as missing for identity or text use."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    s = str(value).strip().lower()
    return s in _INVALID_KEY_TOKENS


def _safe_get(row, col, default=None):
    """Read a cell from a Series-like row; return default if column missing
    or value is blank."""
    try:
        if col not in row.index:
            return default
    except Exception:
        return default
    value = row[col]
    if _is_blank(value):
        return default
    return value


def _safe_get_float(row, col) -> Optional[float]:
    """Read a cell as float; return None on missing/blank/non-numeric."""
    value = _safe_get(row, col, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_get_int(row, col) -> Optional[int]:
    """Read a cell as int; return None on missing/blank/non-numeric."""
    value = _safe_get(row, col, None)
    if value is None:
        return None
    try:
        # tolerate "3", 3.0, etc.
        return int(float(value))
    except (TypeError, ValueError):
        return None


# =====================================================================
# STABLE STOCK KEY (mirrors selection_policy.py for identity consistency)
# =====================================================================

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
    return f"row:{row.name}"


def add_stable_stock_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a `_stable_stock_key` column added.

    Pure helper. Does not change any other column. Safe on empty frames
    or frames missing all identifier columns (every row gets a synthetic
    'row:<idx>' key in that case).
    """
    if df is None or len(df) == 0:
        out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        if STABLE_KEY_COL not in out.columns:
            out[STABLE_KEY_COL] = pd.Series([], dtype=object)
        return out
    out = df.copy()
    out[STABLE_KEY_COL] = out.apply(build_stable_stock_key, axis=1)
    return out


# =====================================================================
# SCORE COLUMN RESOLUTION
# =====================================================================

def _resolve_score_column(
    df: pd.DataFrame,
    primary: str,
    fallback: str,
) -> Optional[str]:
    """Return the active score column name, preferring the v2 primary.

    Returns None if neither column exists in df. Callers should treat
    None as "required column missing — return empty reference frame".
    """
    if primary in df.columns:
        return primary
    if fallback in df.columns:
        return fallback
    return None


# =====================================================================
# EMPTY-FRAME BUILDER
# =====================================================================

def _empty_reference_frame() -> pd.DataFrame:
    """Return a zero-row DataFrame with the canonical 23-column schema."""
    return pd.DataFrame({col: pd.Series(dtype=object) for col in REFERENCE_COLUMNS})


# =====================================================================
# REPORT PRIORITY + MUST-GENERATE-REPORT
# =====================================================================

def _assign_report_priority(grade: str) -> int:
    """Map a reference grade to an integer report priority (1 = highest)."""
    return REPORT_PRIORITY_BY_GRADE.get(grade, 5)


def _compute_must_generate_report(
    grade: str,
    row: pd.Series,
    strategy: str,
) -> str:
    """Decide whether the report layer should generate a report for this row.

    Rules:
      1. Look up the grade in MUST_GENERATE_REPORT_RULE[strategy].
      2. Override to NO if analysis_confidence_score < 0.4 (when present).

    Patch RR-G.1: the previous override "Override to NO if
    entry_quality_tag is 'Insufficient Data' and grade <= B+" has been
    REMOVED. Insufficient Data is now treated as a caution — it appears
    in main_weakness and (for A+/A grades) in the why_ranked_here
    suffix, but it does not force NO. The trainer makes the manual
    judgment when the report shows the caution.

    The crowded_trend_flag = 1 demotion still affects the *grade*, not
    this function — it is applied during grading.
    """
    rule = MUST_GENERATE_REPORT_RULE.get(strategy, {})
    decision = rule.get(grade, "NO")

    confidence = _safe_get_float(row, "analysis_confidence_score")
    if confidence is not None and confidence < 0.4:
        return "NO"

    return decision


# =====================================================================
# PHRASE FORMATTERS (skeleton — real content lands in RR-B onward)
# =====================================================================

def _format_top_strengths(row: pd.Series, strategy: str) -> str:
    """Return a short comma-separated list of strengths (max 3).

    RR-C: walks the strength rule list registered for `strategy` in
    _STRENGTH_RULES_BY_STRATEGY. Strategies not in the dispatch table
    (e.g. Long Term Combined and Audit Only until RR-D) return "n/a".
    """
    rules = _STRENGTH_RULES_BY_STRATEGY.get(strategy)
    if rules is None:
        return "n/a"
    found = []
    for predicate, label in rules:
        try:
            if predicate(row):
                found.append(label)
                if len(found) >= 3:
                    break
        except Exception:
            # A predicate must never bring down the pipeline. Skip on error.
            continue
    if not found:
        return _SWING_STRENGTHS_DEFAULT  # universal default phrase
    return ", ".join(found)


def _format_main_weakness(row: pd.Series, strategy: str) -> str:
    """Return a single short phrase describing the main weakness.

    RR-C: walks the weakness rule list registered for `strategy` in
    _WEAKNESS_RULES_BY_STRATEGY (first applicable rule wins).
    Strategies not in the dispatch table return "n/a".
    """
    rules = _WEAKNESS_RULES_BY_STRATEGY.get(strategy)
    if rules is None:
        return "n/a"
    for predicate, label_fn in rules:
        try:
            if predicate(row):
                return label_fn(row)
        except Exception:
            continue
    return _SWING_WEAKNESS_DEFAULT  # universal default phrase


def _format_why_ranked_here(
    row: pd.Series,
    grade: str,
    strategy: str,
    score_value: Optional[float],
) -> str:
    """Return a single 12-25 word sentence explaining the ranking.

    RR-C: walks the template dict registered for `strategy` in
    _WHY_RANKED_TEMPLATE_BY_STRATEGY. Template-driven, deterministic,
    never AI-generated. Strategies not in the dispatch table return "n/a".

    Patch RR-G.1: when entry_quality_tag is "Insufficient Data" AND the
    grade is A+ or A on a trading/long-term strategy, append a short
    caution suffix so high-grade rows surface the data gap explicitly.
    For B+/B/C/Avoid grades, the existing weakness-driven template
    already shows "insufficient data for clear setup" via the {detail}
    slot, so no suffix is needed there. Audit Only is unaffected (it
    uses its own templates and doesn't carry the same A+ surface).
    """
    template_dict = _WHY_RANKED_TEMPLATE_BY_STRATEGY.get(strategy)
    if template_dict is None:
        return "n/a"
    if score_value is None:
        # Defensive: should not happen for the four implemented strategies
        # (score is the sort key) but keep a safe fallback so a freak NaN
        # row never crashes the function.
        score_value = 0.0
    template = template_dict.get(grade)
    if template is None:
        return "n/a"
    # Pick a detail: a positive strength for A+/A, a weakness otherwise.
    if grade in (GRADE_A_PLUS, GRADE_A):
        detail = _format_top_strengths(row, strategy)
        if detail in ("n/a", _SWING_STRENGTHS_DEFAULT):
            detail = "decent overall profile"
    else:
        detail = _format_main_weakness(row, strategy)
        if detail in ("n/a", _SWING_WEAKNESS_DEFAULT):
            detail = "average overall profile"
    sentence = template.format(score=float(score_value), detail=detail)

    # Patch RR-G.1: caution suffix for A+/A Insufficient Data rows on
    # the four trading/long-term strategies. Audit Only opts out (it
    # already uses review framing).
    if (
        grade in (GRADE_A_PLUS, GRADE_A)
        and strategy in (
            STRATEGY_SWING,
            STRATEGY_SHORT_TERM,
            STRATEGY_LONG_TERM_CORE,
            STRATEGY_LONG_TERM_OPP,
        )
        and _safe_get(row, "entry_quality_tag") == ENTRY_QUALITY_INSUFFICIENT
    ):
        sentence = sentence + " Caution: entry timing data incomplete — confirm entry manually."

    return sentence


def _format_suggested_action_view(grade: str, strategy: str) -> str:
    """Return a learning-oriented action view phrase.

    RR-C: looks up the action dict registered for `strategy` in
    _SUGGESTED_ACTION_VIEW_BY_STRATEGY. Strategies not in the dispatch
    table return "n/a". Phrases deliberately avoid 'buy', 'sell',
    'guaranteed', 'target'.
    """
    action_dict = _SUGGESTED_ACTION_VIEW_BY_STRATEGY.get(strategy)
    if action_dict is None:
        return "n/a"
    return action_dict.get(grade, "n/a")


# =====================================================================
# QUINTILE BASE GRADES (RR-B)
# =====================================================================

# Order of grades from best (rank 1) to worst, used by quintile assignment
# and the one-step confidence downgrade.
_GRADE_ORDER = [
    GRADE_A_PLUS,
    GRADE_A,
    GRADE_B_PLUS,
    GRADE_B,
    GRADE_C,
    GRADE_AVOID,
]


def _compute_quintile_grades(n_rows: int) -> list:
    """Return n_rows base grades for a sorted-best-to-worst input.

    For n >= 5: percentile-based assignment.
        position 0..n-1, percentile = position / n
        percentile <  0.2 -> A+
        percentile <  0.4 -> A
        percentile <  0.6 -> B+
        percentile <  0.8 -> B
        otherwise         -> C
    For n < 5: rank-1 -> A, rest -> B+ (per design, small lists fall back
    to absolute rules so quintile arithmetic is never applied to <5 rows).
    """
    if n_rows <= 0:
        return []
    if n_rows < 5:
        return [GRADE_A] + [GRADE_B_PLUS] * (n_rows - 1)
    grades = []
    for position in range(n_rows):
        pct = position / n_rows
        if pct < 0.2:
            grades.append(GRADE_A_PLUS)
        elif pct < 0.4:
            grades.append(GRADE_A)
        elif pct < 0.6:
            grades.append(GRADE_B_PLUS)
        elif pct < 0.8:
            grades.append(GRADE_B)
        else:
            grades.append(GRADE_C)
    return grades


def _downgrade_one_step(grade: str) -> str:
    """Return the next-worse grade (A+ -> A -> B+ -> B -> C -> Avoid).
    Avoid stays Avoid (cannot be downgraded further)."""
    if grade not in _GRADE_ORDER:
        return grade
    idx = _GRADE_ORDER.index(grade)
    return _GRADE_ORDER[min(idx + 1, len(_GRADE_ORDER) - 1)]


# =====================================================================
# AUDIT-ONLY QUINTILE BASE GRADES (RR-D)
# =====================================================================
# A+ is intentionally disabled. The bottom two quintiles both map to C
# because Audit Only is a review list — there is no need to differentiate
# "low priority" from "very low priority" with a fifth grade band.
# =====================================================================

def _compute_audit_quintile_grades(n_rows: int) -> list:
    """Return n_rows base grades for an Audit-Only sorted-best-to-worst input.

    A+ is structurally disabled. Distribution:
        position 0..n-1, percentile = position / n
        percentile <  0.2 -> A
        percentile <  0.4 -> B+
        percentile <  0.6 -> B
        percentile >= 0.6 -> C
    For n < 5: rank-1 -> A, rest -> B+ (same fallback shape as the
    standard quintile helper, capped at A).
    """
    if n_rows <= 0:
        return []
    if n_rows < 5:
        return [GRADE_A] + [GRADE_B_PLUS] * (n_rows - 1)
    grades = []
    for position in range(n_rows):
        pct = position / n_rows
        if pct < 0.2:
            grades.append(GRADE_A)
        elif pct < 0.4:
            grades.append(GRADE_B_PLUS)
        elif pct < 0.6:
            grades.append(GRADE_B)
        else:
            grades.append(GRADE_C)
    return grades


# =====================================================================
# SWING GRADE ASSIGNMENT (RR-B)
# =====================================================================

def _assign_swing_grade(row: pd.Series, base_grade: str) -> str:
    """Apply Swing modifier rules to a base quintile grade.

    Strict order, no stacking:
        1. Avoid          (terminal — no further checks)
        2. Drop to B      (sets grade to B)
        3. Drop to B+     (caps grade at B+; only effective if better)
        4. A+ Lift        (only effective if base was A+ or A)
        5. Confidence     (one-step downgrade if score < 0.6)
    """
    # ---- 1. Avoid conditions (any one => Avoid; terminal except for confidence)
    red_flags = _safe_get_int(row, "red_flag_count")
    tradability = _safe_get_float(row, "tradability_score")
    entry_tag = _safe_get(row, "entry_quality_tag")
    risk_tag = _safe_get(row, "setup_risk_tag")
    confirm_tag = _safe_get(row, "setup_confirmation_tag")
    crowded_flag = _safe_get_int(row, "crowded_trend_flag")

    avoid_hit = (
        (red_flags is not None and red_flags >= 5)
        or (tradability is not None and tradability < 35)
        or (entry_tag == ENTRY_QUALITY_CROWDED and risk_tag == SETUP_RISK_ELEVATED)
    )
    if avoid_hit:
        grade = GRADE_AVOID
    else:
        grade = base_grade

        # ---- 2. Drop to B (any one => B, only if grade is currently better than B)
        drop_to_b = (
            (red_flags is not None and red_flags >= 3)
            or (tradability is not None and tradability < 50)
            or (risk_tag == SETUP_RISK_ELEVATED)
        )
        if drop_to_b and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_B):
            grade = GRADE_B

        # ---- 3. Drop to B+ (caps at B+; only effective if grade is better than B+)
        # Patch RR-G.1: ENTRY_QUALITY_INSUFFICIENT removed from the
        # demoter set. Insufficient Data is now caution-only — see
        # _format_why_ranked_here for the A+/A caution suffix and the
        # weakness-rule list for the main_weakness phrase.
        drop_to_bplus = (
            confirm_tag == SETUP_CONFIRMATION_WEAK
            or entry_tag == ENTRY_QUALITY_CROWDED
            or crowded_flag == 1
        )
        if drop_to_bplus and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_B_PLUS):
            grade = GRADE_B_PLUS

        # ---- 4. A+ Lift (only effective if base was already A+ or A — cannot
        # lift a B+/B/C/Avoid stock to A+; stops sub-quintile-A stocks from
        # being promoted by tag values alone)
        lift_to_aplus = (
            entry_tag == ENTRY_QUALITY_CLEAN
            and confirm_tag == SETUP_CONFIRMATION_STRONG
            and red_flags is not None and red_flags <= 1
        )
        if lift_to_aplus and base_grade in (GRADE_A_PLUS, GRADE_A) and grade in (GRADE_A_PLUS, GRADE_A):
            grade = GRADE_A_PLUS

    # ---- 5. Confidence downgrade (applied last, applies even to Avoid which
    # stays Avoid via _downgrade_one_step's terminal handling)
    confidence = _safe_get_float(row, "analysis_confidence_score")
    if confidence is not None and confidence < 0.6:
        grade = _downgrade_one_step(grade)

    return grade


# =====================================================================
# SHORT TERM GRADE ASSIGNMENT (RR-C)
# =====================================================================

def _assign_short_term_grade(row: pd.Series, base_grade: str) -> str:
    """Apply Short Term modifier rules to a base quintile grade.

    Strict order, no stacking:
        1. Avoid          (terminal except for confidence)
        2. Drop to B
        3. Drop to B+
        4. A+ Lift        (only if base was already A+ or A)
        5. Confidence     (one-step downgrade if score < 0.6)
    """
    red_flags = _safe_get_int(row, "red_flag_count")
    tradability = _safe_get_float(row, "tradability_score")
    entry_tag = _safe_get(row, "entry_quality_tag")
    risk_tag = _safe_get(row, "setup_risk_tag")
    confirm_tag = _safe_get(row, "setup_confirmation_tag")
    quality_tag = _safe_get(row, "setup_quality_tag")

    # ---- 1. Avoid
    avoid_hit = (
        (red_flags is not None and red_flags >= 5)
        or (tradability is not None and tradability < 30)
        or (entry_tag == ENTRY_QUALITY_CROWDED)
    )
    if avoid_hit:
        grade = GRADE_AVOID
    else:
        grade = base_grade

        # ---- 2. Drop to B
        # Patch RR-G.1: entry_tag == ENTRY_QUALITY_INSUFFICIENT removed
        # from the demoter set. Insufficient Data is now caution-only —
        # see _format_why_ranked_here for the A+/A caution suffix and
        # the weakness-rule list for the main_weakness phrase.
        drop_to_b = (
            (red_flags is not None and red_flags >= 3)
            or (tradability is not None and tradability < 45)
        )
        if drop_to_b and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_B):
            grade = GRADE_B

        # ---- 3. Drop to B+
        drop_to_bplus = (
            confirm_tag == SETUP_CONFIRMATION_WEAK
            or risk_tag == SETUP_RISK_ELEVATED
            or quality_tag == SETUP_QUALITY_FRAGILE
        )
        if drop_to_bplus and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_B_PLUS):
            grade = GRADE_B_PLUS

        # ---- 4. A+ Lift (only effective if base was already A+ or A)
        lift_to_aplus = (
            confirm_tag == SETUP_CONFIRMATION_STRONG
            and quality_tag == SETUP_QUALITY_HIGH
            and red_flags is not None and red_flags <= 1
        )
        if lift_to_aplus and base_grade in (GRADE_A_PLUS, GRADE_A) and grade in (GRADE_A_PLUS, GRADE_A):
            grade = GRADE_A_PLUS

    # ---- 5. Confidence downgrade
    confidence = _safe_get_float(row, "analysis_confidence_score")
    if confidence is not None and confidence < 0.6:
        grade = _downgrade_one_step(grade)

    return grade


# =====================================================================
# LONG TERM CORE GRADE ASSIGNMENT (RR-C)
# =====================================================================

def _assign_long_term_core_grade(row: pd.Series, base_grade: str) -> str:
    """Apply Long Term Core modifier rules to a base quintile grade.

    Strict order, no stacking:
        1. Avoid (red_flag_count >= 5)
        2. Drop to B
        3. Drop to B+
        4. A+ Lift
        5. Confidence downgrade
    """
    red_flags = _safe_get_int(row, "red_flag_count")
    valuation = _safe_get_float(row, "valuation_factor")
    risk_factor = _safe_get_float(row, "risk_factor")
    quality_safety = _safe_get_float(row, "quality_safety_score")
    business_quality = _safe_get_float(row, "business_quality_factor")
    cashflow_quality = _safe_get_float(row, "cashflow_quality_factor")
    quality_tag = _safe_get(row, "setup_quality_tag")

    # ---- 1. Avoid
    avoid_hit = (red_flags is not None and red_flags >= 5)
    if avoid_hit:
        grade = GRADE_AVOID
    else:
        grade = base_grade

        # ---- 2. Drop to B
        drop_to_b = (
            (red_flags is not None and red_flags >= 3)
            or (valuation is not None and valuation < 30)
        )
        if drop_to_b and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_B):
            grade = GRADE_B

        # ---- 3. Drop to B+
        drop_to_bplus = (
            (quality_safety is not None and quality_safety < 60)
            or (risk_factor is not None and risk_factor < 55)
            or (quality_tag == SETUP_QUALITY_FRAGILE)
        )
        if drop_to_bplus and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_B_PLUS):
            grade = GRADE_B_PLUS

        # ---- 4. A+ Lift (only effective if base was already A+ or A)
        lift_to_aplus = (
            business_quality is not None and business_quality >= 70
            and cashflow_quality is not None and cashflow_quality >= 60
            and risk_factor is not None and risk_factor >= 60
            and red_flags is not None and red_flags <= 1
        )
        if lift_to_aplus and base_grade in (GRADE_A_PLUS, GRADE_A) and grade in (GRADE_A_PLUS, GRADE_A):
            grade = GRADE_A_PLUS

    # ---- 5. Confidence downgrade
    confidence = _safe_get_float(row, "analysis_confidence_score")
    if confidence is not None and confidence < 0.6:
        grade = _downgrade_one_step(grade)

    return grade


# =====================================================================
# LONG TERM OPPORTUNITY GRADE ASSIGNMENT (RR-C)
# =====================================================================

def _assign_long_term_opp_grade(row: pd.Series, base_grade: str) -> str:
    """Apply Long Term Opportunity modifier rules to a base quintile grade.

    Strict order, no stacking:
        1. Avoid
        2. Drop to B
        3. Drop to B+
        4. A+ Lift
        5. Confidence downgrade
    """
    red_flags = _safe_get_int(row, "red_flag_count")
    risk_factor = _safe_get_float(row, "risk_factor")
    valuation = _safe_get_float(row, "valuation_factor")
    growth = _safe_get_float(row, "growth_factor")
    catalyst = _safe_get_float(row, "catalyst_proxy_factor")
    entry_tag = _safe_get(row, "entry_quality_tag")

    # ---- 1. Avoid
    avoid_hit = (
        (red_flags is not None and red_flags >= 5)
        or (risk_factor is not None and risk_factor < 30)
    )
    if avoid_hit:
        grade = GRADE_AVOID
    else:
        grade = base_grade

        # ---- 2. Drop to B
        drop_to_b = (
            (red_flags is not None and red_flags >= 3)
            or (risk_factor is not None and risk_factor < 40)
        )
        if drop_to_b and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_B):
            grade = GRADE_B

        # ---- 3. Drop to B+
        drop_to_bplus = (
            (valuation is not None and valuation < 30)
            or (entry_tag == ENTRY_QUALITY_CROWDED)
        )
        if drop_to_bplus and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_B_PLUS):
            grade = GRADE_B_PLUS

        # ---- 4. A+ Lift (only effective if base was already A+ or A)
        lift_to_aplus = (
            growth is not None and growth >= 70
            and catalyst is not None and catalyst >= 60
            and entry_tag in (ENTRY_QUALITY_CLEAN, ENTRY_QUALITY_PULLBACK)
        )
        if lift_to_aplus and base_grade in (GRADE_A_PLUS, GRADE_A) and grade in (GRADE_A_PLUS, GRADE_A):
            grade = GRADE_A_PLUS

    # ---- 5. Confidence downgrade
    confidence = _safe_get_float(row, "analysis_confidence_score")
    if confidence is not None and confidence < 0.6:
        grade = _downgrade_one_step(grade)

    return grade


# =====================================================================
# AUDIT ONLY GRADE ASSIGNMENT (RR-D)
# ---------------------------------------------------------------------
# Differs from Swing/Short Term/Core/Opp grade assigners in three ways:
#   1. The "lift" step lifts to A (not A+). A+ is disabled.
#   2. The intermediate downgrade is "Drop to C" (not Drop to B).
#   3. There is no Drop-to-B+ step — Audit Only's only middle modifier
#      is the lift; everything else lands at the quintile grade.
# =====================================================================

def _assign_audit_only_grade(row: pd.Series, base_grade: str) -> str:
    """Apply Audit Only modifier rules to a base quintile grade.

    Strict order, no stacking:
        1. Avoid           (terminal except for confidence)
        2. Drop to C       (sets grade to C)
        3. Lift to A       (only effective if base was already A)
        4. Confidence      (one-step downgrade if score < 0.6)

    A+ is never assigned — neither the base helper nor this function
    produces it.
    """
    red_flags = _safe_get_int(row, "red_flag_count")
    entry_tag = _safe_get(row, "entry_quality_tag")
    risk_tag = _safe_get(row, "setup_risk_tag")
    qss = _safe_get_float(row, "quality_safety_score")

    # ---- 1. Avoid
    avoid_hit = (
        (red_flags is not None and red_flags >= 5)
        or (entry_tag == ENTRY_QUALITY_CROWDED and risk_tag == SETUP_RISK_ELEVATED)
    )
    if avoid_hit:
        grade = GRADE_AVOID
    else:
        grade = base_grade

        # ---- 2. Drop to C (any one => C, only if grade is currently better than C)
        # Patch RR-G.1: entry_tag == ENTRY_QUALITY_INSUFFICIENT removed
        # from the demoter set. Insufficient Data is now caution-only.
        # red_flag_count >= 3 remains the sole Drop-to-C trigger.
        drop_to_c = (red_flags is not None and red_flags >= 3)
        if drop_to_c and _GRADE_ORDER.index(grade) < _GRADE_ORDER.index(GRADE_C):
            grade = GRADE_C

        # ---- 3. Lift to A (only effective if base was already A — cannot
        # lift a B+/B/C/Avoid stock to A; lift never produces A+)
        lift_to_a = (
            qss is not None and qss >= 70
            and red_flags is not None and red_flags <= 1
            and entry_tag != ENTRY_QUALITY_CROWDED
        )
        if lift_to_a and base_grade == GRADE_A and grade == GRADE_A:
            # Lift confirms A; no actual change needed since base was A and
            # no demoter fired (grade still equals A). The branch is here
            # to make the rule explicit and symmetric with the other
            # categories' lift logic. If a future rule pushes the lift
            # threshold higher, this is the place to enforce it.
            grade = GRADE_A

    # ---- 4. Confidence downgrade (applied last; respects Avoid terminal)
    confidence = _safe_get_float(row, "analysis_confidence_score")
    if confidence is not None and confidence < 0.6:
        grade = _downgrade_one_step(grade)

    return grade


# =====================================================================
# OPTIONAL-COLUMN PRESENCE LOG (RR-B)
# =====================================================================

def _log_missing_optional(df: pd.DataFrame, columns: list) -> None:
    """Log one INFO line the first time each optional column is missing.
    Tracking is module-global so the log is calm in a long-running session.
    """
    for col in columns:
        if col in df.columns:
            continue
        if col in _RR_LOGGED_MISSING_OPTIONAL:
            continue
        _RR_LOGGED_MISSING_OPTIONAL.add(col)
        logger.info(
            "reference_ranking: optional column '%s' missing — "
            "related grading modifiers and phrase rules will not fire.", col,
        )


# =====================================================================
# PUBLIC SKELETON FUNCTIONS — one per category
# =====================================================================
# In RR-A, every public build_* function returns an empty 23-column frame.
# Signatures are stable so RR-B onward fills bodies without changing call
# sites. Each docstring records exactly what its later body will do.
# =====================================================================

def build_swing_reference(swing_shortlist_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Swing reference list (RR-B implementation).

    Inputs:
        swing_shortlist_df: DataFrame produced by
            selection_policy.run_selection_policy(...)["swing_shortlist"].
            stable_stock_key is the required identity; nse_code is optional
            and display-only.

    Returns:
        DataFrame with the canonical 23-column schema. Empty 23-column
        frame if the input is None/empty or required columns are missing.
        Rows are NEVER dropped because nse_code is missing.

    Required columns: stock_name, market_cap_bucket, swing_score_v2 OR
        swing_score. Identity is built from the fallback chain via
        add_stable_stock_key, so even rows with all primary IDs blank
        receive a synthetic 'row:<idx>' key and remain in the output.

    Optional columns: nse_code, isin, bse_code, stock_code, best_stock_key,
        entry_quality_tag, setup_quality_tag, setup_confirmation_tag,
        setup_risk_tag, crowded_trend_flag, red_flag_count,
        positive_flag_count, analysis_confidence_score,
        quality_safety_score, tradability_score. Missing optional values
        leave the corresponding output cell as None / 'n/a'; modifiers
        tied to that column simply do not fire.
    """
    # ---- 1. Empty/None guard
    if swing_shortlist_df is None or len(swing_shortlist_df) == 0:
        return _empty_reference_frame()

    # ---- 2. Required-column check
    score_col = _resolve_score_column(
        swing_shortlist_df, "swing_score_v2", "swing_score"
    )
    if score_col is None:
        logger.warning(
            "reference_ranking.build_swing_reference: neither "
            "'swing_score_v2' nor 'swing_score' present — returning empty frame."
        )
        return _empty_reference_frame()
    for required_col in ("stock_name", "market_cap_bucket"):
        if required_col not in swing_shortlist_df.columns:
            logger.warning(
                "reference_ranking.build_swing_reference: required column "
                "'%s' missing — returning empty frame.", required_col,
            )
            return _empty_reference_frame()

    # Inform (once) about any missing optional columns we'd normally use.
    _log_missing_optional(swing_shortlist_df, [
        "entry_quality_tag", "setup_quality_tag", "setup_confirmation_tag",
        "setup_risk_tag", "crowded_trend_flag", "red_flag_count",
        "positive_flag_count", "analysis_confidence_score",
        "quality_safety_score", "tradability_score", "nse_code",
    ])

    # ---- 3. Add stable identity key (working copy)
    keyed = add_stable_stock_key(swing_shortlist_df)

    # ---- 4. Sort: score desc, stable key asc (deterministic tie-break)
    keyed = keyed.sort_values(
        by=[score_col, STABLE_KEY_COL],
        ascending=[False, True],
        kind="mergesort",  # stable sort, preserves earlier order on full ties
    ).reset_index(drop=True)

    # ---- 5. Compute base quintile grades by sorted position
    base_grades = _compute_quintile_grades(len(keyed))

    # ---- 6+7. Apply modifiers and build output rows
    out_rows = []
    for position, (_, row) in enumerate(keyed.iterrows()):
        score_value = _safe_get_float(row, score_col)
        # If the score itself is missing (rare — would mean None made it
        # through despite the column being present), use 0.0 for templates
        # but still grade by base quintile only.
        base = base_grades[position]
        final_grade = _assign_swing_grade(row, base)

        out_rows.append({
            "reference_rank": position + 1,
            "reference_grade": final_grade,
            "stable_stock_key": row[STABLE_KEY_COL],
            "stock_name": _safe_get(row, "stock_name", "n/a"),
            "nse_code": _safe_get(row, "nse_code", None),
            "market_cap_bucket": _safe_get(row, "market_cap_bucket", "n/a"),
            "strategy_category": STRATEGY_SWING,
            "strategy_score": score_value,
            "quality_safety_score": _safe_get_float(row, "quality_safety_score"),
            "tradability_score": _safe_get_float(row, "tradability_score"),
            "entry_quality_tag": _safe_get(row, "entry_quality_tag", "n/a"),
            "setup_quality_tag": _safe_get(row, "setup_quality_tag", "n/a"),
            "setup_confirmation_tag": _safe_get(row, "setup_confirmation_tag", "n/a"),
            "setup_risk_tag": _safe_get(row, "setup_risk_tag", "n/a"),
            "red_flag_count": _safe_get_int(row, "red_flag_count"),
            "positive_flag_count": _safe_get_int(row, "positive_flag_count"),
            "analysis_confidence_score": _safe_get_float(row, "analysis_confidence_score"),
            "top_strengths": _format_top_strengths(row, STRATEGY_SWING),
            "main_weakness": _format_main_weakness(row, STRATEGY_SWING),
            "why_ranked_here": _format_why_ranked_here(
                row, final_grade, STRATEGY_SWING, score_value
            ),
            "suggested_action_view": _format_suggested_action_view(final_grade, STRATEGY_SWING),
            "must_generate_report": _compute_must_generate_report(
                final_grade, row, STRATEGY_SWING
            ),
            "report_priority": _assign_report_priority(final_grade),
        })

    # ---- 8. Construct DataFrame in canonical column order
    if not out_rows:
        return _empty_reference_frame()
    out_df = pd.DataFrame(out_rows)
    return out_df[REFERENCE_COLUMNS].reset_index(drop=True)


def build_short_term_reference(short_term_shortlist_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Short Term reference list (RR-C implementation).

    Inputs:
        short_term_shortlist_df: DataFrame produced by
            selection_policy.run_selection_policy(...)["short_term_shortlist"].
            stable_stock_key is the required identity; nse_code is optional
            and display-only.

    Returns:
        DataFrame with the canonical 23-column schema. Empty 23-column
        frame if the input is None/empty or required columns are missing.
        Rows are NEVER dropped because nse_code is missing.

    Required columns: stock_name, market_cap_bucket, short_term_score_v2 OR
        short_term_score.

    Optional columns: nse_code, isin, bse_code, stock_code, best_stock_key,
        entry_quality_tag, setup_quality_tag, setup_confirmation_tag,
        setup_risk_tag, red_flag_count, positive_flag_count,
        analysis_confidence_score, quality_safety_score, tradability_score.
        Missing values leave the corresponding output cell as None / 'n/a';
        modifiers tied to that column simply do not fire.
    """
    # ---- 1. Empty/None guard
    if short_term_shortlist_df is None or len(short_term_shortlist_df) == 0:
        return _empty_reference_frame()

    # ---- 2. Required-column check
    score_col = _resolve_score_column(
        short_term_shortlist_df, "short_term_score_v2", "short_term_score"
    )
    if score_col is None:
        logger.warning(
            "reference_ranking.build_short_term_reference: neither "
            "'short_term_score_v2' nor 'short_term_score' present — "
            "returning empty frame."
        )
        return _empty_reference_frame()
    for required_col in ("stock_name", "market_cap_bucket"):
        if required_col not in short_term_shortlist_df.columns:
            logger.warning(
                "reference_ranking.build_short_term_reference: required "
                "column '%s' missing — returning empty frame.", required_col,
            )
            return _empty_reference_frame()

    _log_missing_optional(short_term_shortlist_df, [
        "entry_quality_tag", "setup_quality_tag", "setup_confirmation_tag",
        "setup_risk_tag", "red_flag_count", "positive_flag_count",
        "analysis_confidence_score", "quality_safety_score",
        "tradability_score", "nse_code",
    ])

    # ---- 3. Add stable identity key
    keyed = add_stable_stock_key(short_term_shortlist_df)

    # ---- 4. Sort: score desc, stable key asc (deterministic tie-break)
    keyed = keyed.sort_values(
        by=[score_col, STABLE_KEY_COL],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---- 5. Compute base quintile grades
    base_grades = _compute_quintile_grades(len(keyed))

    # ---- 6+7. Apply modifiers and build output rows
    out_rows = []
    for position, (_, row) in enumerate(keyed.iterrows()):
        score_value = _safe_get_float(row, score_col)
        base = base_grades[position]
        final_grade = _assign_short_term_grade(row, base)

        out_rows.append({
            "reference_rank": position + 1,
            "reference_grade": final_grade,
            "stable_stock_key": row[STABLE_KEY_COL],
            "stock_name": _safe_get(row, "stock_name", "n/a"),
            "nse_code": _safe_get(row, "nse_code", None),
            "market_cap_bucket": _safe_get(row, "market_cap_bucket", "n/a"),
            "strategy_category": STRATEGY_SHORT_TERM,
            "strategy_score": score_value,
            "quality_safety_score": _safe_get_float(row, "quality_safety_score"),
            "tradability_score": _safe_get_float(row, "tradability_score"),
            "entry_quality_tag": _safe_get(row, "entry_quality_tag", "n/a"),
            "setup_quality_tag": _safe_get(row, "setup_quality_tag", "n/a"),
            "setup_confirmation_tag": _safe_get(row, "setup_confirmation_tag", "n/a"),
            "setup_risk_tag": _safe_get(row, "setup_risk_tag", "n/a"),
            "red_flag_count": _safe_get_int(row, "red_flag_count"),
            "positive_flag_count": _safe_get_int(row, "positive_flag_count"),
            "analysis_confidence_score": _safe_get_float(row, "analysis_confidence_score"),
            "top_strengths": _format_top_strengths(row, STRATEGY_SHORT_TERM),
            "main_weakness": _format_main_weakness(row, STRATEGY_SHORT_TERM),
            "why_ranked_here": _format_why_ranked_here(
                row, final_grade, STRATEGY_SHORT_TERM, score_value
            ),
            "suggested_action_view": _format_suggested_action_view(
                final_grade, STRATEGY_SHORT_TERM
            ),
            "must_generate_report": _compute_must_generate_report(
                final_grade, row, STRATEGY_SHORT_TERM
            ),
            "report_priority": _assign_report_priority(final_grade),
        })

    if not out_rows:
        return _empty_reference_frame()
    out_df = pd.DataFrame(out_rows)
    return out_df[REFERENCE_COLUMNS].reset_index(drop=True)


def build_long_term_core_reference(long_term_core_shortlist_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Long Term Core reference list (RR-C implementation).

    Inputs:
        long_term_core_shortlist_df: DataFrame produced by
            selection_policy.run_selection_policy(...)["long_term_core_shortlist"].

    Returns:
        DataFrame with the canonical 23-column schema. Empty if input is
        None/empty or required columns are missing.

    Required columns: stock_name, market_cap_bucket, long_term_score_v2
        OR long_term_score.

    Optional columns: business_quality_factor, cashflow_quality_factor,
        risk_factor, valuation_factor, quality_safety_score,
        setup_quality_tag, red_flag_count, positive_flag_count,
        analysis_confidence_score, entry_quality_tag, tradability_score,
        nse_code, isin, bse_code, stock_code, best_stock_key.
    """
    # ---- 1. Empty/None guard
    if long_term_core_shortlist_df is None or len(long_term_core_shortlist_df) == 0:
        return _empty_reference_frame()

    # ---- 2. Required-column check
    score_col = _resolve_score_column(
        long_term_core_shortlist_df, "long_term_score_v2", "long_term_score"
    )
    if score_col is None:
        logger.warning(
            "reference_ranking.build_long_term_core_reference: neither "
            "'long_term_score_v2' nor 'long_term_score' present — "
            "returning empty frame."
        )
        return _empty_reference_frame()
    for required_col in ("stock_name", "market_cap_bucket"):
        if required_col not in long_term_core_shortlist_df.columns:
            logger.warning(
                "reference_ranking.build_long_term_core_reference: required "
                "column '%s' missing — returning empty frame.", required_col,
            )
            return _empty_reference_frame()

    _log_missing_optional(long_term_core_shortlist_df, [
        "business_quality_factor", "cashflow_quality_factor", "risk_factor",
        "valuation_factor", "quality_safety_score", "setup_quality_tag",
        "red_flag_count", "positive_flag_count",
        "analysis_confidence_score", "entry_quality_tag",
        "tradability_score", "nse_code",
    ])

    # ---- 3. Add stable identity key
    keyed = add_stable_stock_key(long_term_core_shortlist_df)

    # ---- 4. Sort: score desc, stable key asc
    keyed = keyed.sort_values(
        by=[score_col, STABLE_KEY_COL],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---- 5. Compute base quintile grades
    base_grades = _compute_quintile_grades(len(keyed))

    # ---- 6+7. Apply modifiers and build output rows
    out_rows = []
    for position, (_, row) in enumerate(keyed.iterrows()):
        score_value = _safe_get_float(row, score_col)
        base = base_grades[position]
        final_grade = _assign_long_term_core_grade(row, base)

        out_rows.append({
            "reference_rank": position + 1,
            "reference_grade": final_grade,
            "stable_stock_key": row[STABLE_KEY_COL],
            "stock_name": _safe_get(row, "stock_name", "n/a"),
            "nse_code": _safe_get(row, "nse_code", None),
            "market_cap_bucket": _safe_get(row, "market_cap_bucket", "n/a"),
            "strategy_category": STRATEGY_LONG_TERM_CORE,
            "strategy_score": score_value,
            "quality_safety_score": _safe_get_float(row, "quality_safety_score"),
            "tradability_score": _safe_get_float(row, "tradability_score"),
            "entry_quality_tag": _safe_get(row, "entry_quality_tag", "n/a"),
            "setup_quality_tag": _safe_get(row, "setup_quality_tag", "n/a"),
            "setup_confirmation_tag": _safe_get(row, "setup_confirmation_tag", "n/a"),
            "setup_risk_tag": _safe_get(row, "setup_risk_tag", "n/a"),
            "red_flag_count": _safe_get_int(row, "red_flag_count"),
            "positive_flag_count": _safe_get_int(row, "positive_flag_count"),
            "analysis_confidence_score": _safe_get_float(row, "analysis_confidence_score"),
            "top_strengths": _format_top_strengths(row, STRATEGY_LONG_TERM_CORE),
            "main_weakness": _format_main_weakness(row, STRATEGY_LONG_TERM_CORE),
            "why_ranked_here": _format_why_ranked_here(
                row, final_grade, STRATEGY_LONG_TERM_CORE, score_value
            ),
            "suggested_action_view": _format_suggested_action_view(
                final_grade, STRATEGY_LONG_TERM_CORE
            ),
            "must_generate_report": _compute_must_generate_report(
                final_grade, row, STRATEGY_LONG_TERM_CORE
            ),
            "report_priority": _assign_report_priority(final_grade),
        })

    if not out_rows:
        return _empty_reference_frame()
    out_df = pd.DataFrame(out_rows)
    return out_df[REFERENCE_COLUMNS].reset_index(drop=True)


def build_long_term_opportunity_reference(long_term_opportunity_shortlist_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Long Term Opportunity reference list (RR-C implementation).

    Inputs:
        long_term_opportunity_shortlist_df: DataFrame produced by
            selection_policy.run_selection_policy(...)["long_term_opportunity_shortlist"].

    Returns:
        DataFrame with the canonical 23-column schema. Empty if input is
        None/empty or required columns are missing.

    Required columns: stock_name, market_cap_bucket, long_term_score_v2
        OR long_term_score.

    Optional columns: growth_factor, catalyst_proxy_factor,
        valuation_factor, risk_factor, entry_quality_tag, red_flag_count,
        positive_flag_count, analysis_confidence_score,
        quality_safety_score, tradability_score, setup_quality_tag,
        nse_code, isin, bse_code, stock_code, best_stock_key.
    """
    # ---- 1. Empty/None guard
    if long_term_opportunity_shortlist_df is None or len(long_term_opportunity_shortlist_df) == 0:
        return _empty_reference_frame()

    # ---- 2. Required-column check
    score_col = _resolve_score_column(
        long_term_opportunity_shortlist_df, "long_term_score_v2", "long_term_score"
    )
    if score_col is None:
        logger.warning(
            "reference_ranking.build_long_term_opportunity_reference: neither "
            "'long_term_score_v2' nor 'long_term_score' present — "
            "returning empty frame."
        )
        return _empty_reference_frame()
    for required_col in ("stock_name", "market_cap_bucket"):
        if required_col not in long_term_opportunity_shortlist_df.columns:
            logger.warning(
                "reference_ranking.build_long_term_opportunity_reference: "
                "required column '%s' missing — returning empty frame.",
                required_col,
            )
            return _empty_reference_frame()

    _log_missing_optional(long_term_opportunity_shortlist_df, [
        "growth_factor", "catalyst_proxy_factor", "valuation_factor",
        "risk_factor", "entry_quality_tag", "red_flag_count",
        "positive_flag_count", "analysis_confidence_score",
        "quality_safety_score", "tradability_score", "nse_code",
    ])

    # ---- 3. Add stable identity key
    keyed = add_stable_stock_key(long_term_opportunity_shortlist_df)

    # ---- 4. Sort: score desc, stable key asc
    keyed = keyed.sort_values(
        by=[score_col, STABLE_KEY_COL],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---- 5. Compute base quintile grades
    base_grades = _compute_quintile_grades(len(keyed))

    # ---- 6+7. Apply modifiers and build output rows
    out_rows = []
    for position, (_, row) in enumerate(keyed.iterrows()):
        score_value = _safe_get_float(row, score_col)
        base = base_grades[position]
        final_grade = _assign_long_term_opp_grade(row, base)

        out_rows.append({
            "reference_rank": position + 1,
            "reference_grade": final_grade,
            "stable_stock_key": row[STABLE_KEY_COL],
            "stock_name": _safe_get(row, "stock_name", "n/a"),
            "nse_code": _safe_get(row, "nse_code", None),
            "market_cap_bucket": _safe_get(row, "market_cap_bucket", "n/a"),
            "strategy_category": STRATEGY_LONG_TERM_OPP,
            "strategy_score": score_value,
            "quality_safety_score": _safe_get_float(row, "quality_safety_score"),
            "tradability_score": _safe_get_float(row, "tradability_score"),
            "entry_quality_tag": _safe_get(row, "entry_quality_tag", "n/a"),
            "setup_quality_tag": _safe_get(row, "setup_quality_tag", "n/a"),
            "setup_confirmation_tag": _safe_get(row, "setup_confirmation_tag", "n/a"),
            "setup_risk_tag": _safe_get(row, "setup_risk_tag", "n/a"),
            "red_flag_count": _safe_get_int(row, "red_flag_count"),
            "positive_flag_count": _safe_get_int(row, "positive_flag_count"),
            "analysis_confidence_score": _safe_get_float(row, "analysis_confidence_score"),
            "top_strengths": _format_top_strengths(row, STRATEGY_LONG_TERM_OPP),
            "main_weakness": _format_main_weakness(row, STRATEGY_LONG_TERM_OPP),
            "why_ranked_here": _format_why_ranked_here(
                row, final_grade, STRATEGY_LONG_TERM_OPP, score_value
            ),
            "suggested_action_view": _format_suggested_action_view(
                final_grade, STRATEGY_LONG_TERM_OPP
            ),
            "must_generate_report": _compute_must_generate_report(
                final_grade, row, STRATEGY_LONG_TERM_OPP
            ),
            "report_priority": _assign_report_priority(final_grade),
        })

    if not out_rows:
        return _empty_reference_frame()
    out_df = pd.DataFrame(out_rows)
    return out_df[REFERENCE_COLUMNS].reset_index(drop=True)


def build_long_term_combined_reference(
    long_term_shortlist_df: pd.DataFrame,
    core_keys: Optional[set] = None,
    opportunity_keys: Optional[set] = None,
) -> pd.DataFrame:
    """Build the Long Term Combined reference list (RR-D implementation).

    Each row's sub-type is determined by stable_stock_key membership:
        * "Long Term — Core" if key in core_keys
        * "Long Term — Opportunity" if key in opportunity_keys
        * "Long Term — Other" otherwise (the bucket-mix-gap rows that
          Patch C.1 surfaced — stocks that made the combined view but
          neither Core nor Opp sub-list)

    Grading uses the appropriate parent-strategy helper:
        * Core rows: _assign_long_term_core_grade
        * Opportunity and Other rows: _assign_long_term_opp_grade

    Other rows have a clarifying suffix appended to why_ranked_here.

    Inputs:
        long_term_shortlist_df: policy_result["long_term_shortlist"]
        core_keys / opportunity_keys: stable_stock_key sets from the
            Core and Opp shortlists. Accepts None, list, tuple, set —
            normalised to set inside the function.

    Returns:
        DataFrame with the canonical 23-column schema. Empty if input is
        None/empty or required columns are missing.
    """
    # ---- 1. Empty/None guard
    if long_term_shortlist_df is None or len(long_term_shortlist_df) == 0:
        return _empty_reference_frame()

    # ---- 2. Required-column check
    score_col = _resolve_score_column(
        long_term_shortlist_df, "long_term_score_v2", "long_term_score"
    )
    if score_col is None:
        logger.warning(
            "reference_ranking.build_long_term_combined_reference: neither "
            "'long_term_score_v2' nor 'long_term_score' present — "
            "returning empty frame."
        )
        return _empty_reference_frame()
    for required_col in ("stock_name", "market_cap_bucket"):
        if required_col not in long_term_shortlist_df.columns:
            logger.warning(
                "reference_ranking.build_long_term_combined_reference: "
                "required column '%s' missing — returning empty frame.",
                required_col,
            )
            return _empty_reference_frame()

    # Normalise the sub-type lookup sets — accept None, list, tuple, set.
    core_keys = set(core_keys) if core_keys else set()
    opportunity_keys = set(opportunity_keys) if opportunity_keys else set()
    if not core_keys and not opportunity_keys:
        logger.info(
            "reference_ranking.build_long_term_combined_reference: "
            "core_keys and opportunity_keys are both empty — all rows "
            "graded as 'Long Term — Other'."
        )

    _log_missing_optional(long_term_shortlist_df, [
        "business_quality_factor", "cashflow_quality_factor", "risk_factor",
        "valuation_factor", "growth_factor", "catalyst_proxy_factor",
        "quality_safety_score", "setup_quality_tag", "red_flag_count",
        "positive_flag_count", "analysis_confidence_score",
        "entry_quality_tag", "tradability_score", "nse_code",
    ])

    # ---- 3. Add stable identity key
    keyed = add_stable_stock_key(long_term_shortlist_df)

    # ---- 4. Sort: score desc, stable key asc
    keyed = keyed.sort_values(
        by=[score_col, STABLE_KEY_COL],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---- 5. Compute base quintile grades
    base_grades = _compute_quintile_grades(len(keyed))

    # ---- 6+7. Sub-type each row, dispatch to appropriate grader, build output
    OTHER_SUFFIX = " (Combined-only: did not make Core or Opp sub-list.)"
    out_rows = []
    for position, (_, row) in enumerate(keyed.iterrows()):
        score_value = _safe_get_float(row, score_col)
        base = base_grades[position]
        stable_key = str(row[STABLE_KEY_COL])

        # Determine sub-type and dispatch.
        if stable_key in core_keys:
            sub_category = STRATEGY_LONG_TERM_COMBINED_CORE
            parent_strategy = STRATEGY_LONG_TERM_CORE
            final_grade = _assign_long_term_core_grade(row, base)
            is_other = False
        elif stable_key in opportunity_keys:
            sub_category = STRATEGY_LONG_TERM_COMBINED_OPP
            parent_strategy = STRATEGY_LONG_TERM_OPP
            final_grade = _assign_long_term_opp_grade(row, base)
            is_other = False
        else:
            sub_category = STRATEGY_LONG_TERM_COMBINED_OTHER
            parent_strategy = STRATEGY_LONG_TERM_OPP
            final_grade = _assign_long_term_opp_grade(row, base)
            is_other = True

        why = _format_why_ranked_here(row, final_grade, parent_strategy, score_value)
        if is_other and why and why != "n/a":
            why = why + OTHER_SUFFIX

        out_rows.append({
            "reference_rank": position + 1,
            "reference_grade": final_grade,
            "stable_stock_key": row[STABLE_KEY_COL],
            "stock_name": _safe_get(row, "stock_name", "n/a"),
            "nse_code": _safe_get(row, "nse_code", None),
            "market_cap_bucket": _safe_get(row, "market_cap_bucket", "n/a"),
            "strategy_category": sub_category,
            "strategy_score": score_value,
            "quality_safety_score": _safe_get_float(row, "quality_safety_score"),
            "tradability_score": _safe_get_float(row, "tradability_score"),
            "entry_quality_tag": _safe_get(row, "entry_quality_tag", "n/a"),
            "setup_quality_tag": _safe_get(row, "setup_quality_tag", "n/a"),
            "setup_confirmation_tag": _safe_get(row, "setup_confirmation_tag", "n/a"),
            "setup_risk_tag": _safe_get(row, "setup_risk_tag", "n/a"),
            "red_flag_count": _safe_get_int(row, "red_flag_count"),
            "positive_flag_count": _safe_get_int(row, "positive_flag_count"),
            "analysis_confidence_score": _safe_get_float(row, "analysis_confidence_score"),
            "top_strengths": _format_top_strengths(row, parent_strategy),
            "main_weakness": _format_main_weakness(row, parent_strategy),
            "why_ranked_here": why,
            "suggested_action_view": _format_suggested_action_view(
                final_grade, parent_strategy
            ),
            "must_generate_report": _compute_must_generate_report(
                final_grade, row, parent_strategy
            ),
            "report_priority": _assign_report_priority(final_grade),
        })

    if not out_rows:
        return _empty_reference_frame()
    out_df = pd.DataFrame(out_rows)
    return out_df[REFERENCE_COLUMNS].reset_index(drop=True)


def build_audit_only_reference(audit_only_universe_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Audit Only reference list (RR-D implementation).

    Audit Only is a REVIEW list, not a recommendation list. Five
    structural safeguards make this clear in the output:
        1. A+ is structurally disabled (never produced by the quintile
           helper or the grade assigner; not present in the action /
           why-ranked dicts).
        2. strategy_category = "Audit Only" on every row.
        3. suggested_action_view phrases use "review" framing only.
        4. must_generate_report is conservative: only A grade => YES.
        5. why_ranked_here templates explain rather than recommend.

    Ranking score:
        audit_reference_score = 0.6 * quality_safety_score
                              + 0.4 * tradability_score
        (computed locally; placed inside the existing strategy_score
        column — no new output column added.)

    Inputs:
        audit_only_universe_df: policy_result["audit_only_universe"].
            stable_stock_key is the required identity; nse_code is
            optional and display-only.

    Returns:
        DataFrame with the canonical 23-column schema. Empty if input is
        None/empty or if BOTH quality_safety_score and tradability_score
        columns are missing (without either, the synthetic ordering
        score is meaningless).
    """
    # ---- 1. Empty/None guard
    if audit_only_universe_df is None or len(audit_only_universe_df) == 0:
        return _empty_reference_frame()

    # ---- 2. Required-column check
    if "stock_name" not in audit_only_universe_df.columns:
        logger.warning(
            "reference_ranking.build_audit_only_reference: required "
            "column 'stock_name' missing — returning empty frame."
        )
        return _empty_reference_frame()
    has_qss = "quality_safety_score" in audit_only_universe_df.columns
    has_trad = "tradability_score" in audit_only_universe_df.columns
    if not (has_qss or has_trad):
        logger.warning(
            "reference_ranking.build_audit_only_reference: both "
            "'quality_safety_score' and 'tradability_score' missing — "
            "synthetic audit score cannot be computed; returning empty frame."
        )
        return _empty_reference_frame()

    _log_missing_optional(audit_only_universe_df, [
        "entry_quality_tag", "setup_quality_tag", "setup_confirmation_tag",
        "setup_risk_tag", "red_flag_count", "positive_flag_count",
        "analysis_confidence_score", "market_cap_bucket", "nse_code",
    ])

    # ---- 3. Add stable identity key
    keyed = add_stable_stock_key(audit_only_universe_df)

    # ---- 4. Compute synthetic audit_reference_score per row.
    # Lives only as a temporary working column; not added to the
    # output schema. Missing-value safe: a row with both fields missing
    # gets 0.0 (sorts to bottom but is preserved in the output).
    def _audit_score(row: pd.Series) -> float:
        qss = _safe_get_float(row, "quality_safety_score")
        trad = _safe_get_float(row, "tradability_score")
        score = 0.0
        if qss is not None:
            score += _AUDIT_QSS_WEIGHT * qss
        if trad is not None:
            score += _AUDIT_TRADABILITY_WEIGHT * trad
        return score

    AUDIT_SCORE_COL = "_audit_reference_score"
    keyed[AUDIT_SCORE_COL] = keyed.apply(_audit_score, axis=1)

    # ---- 5. Sort: synthetic score desc, stable key asc
    keyed = keyed.sort_values(
        by=[AUDIT_SCORE_COL, STABLE_KEY_COL],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # ---- 6. Compute base quintile grades (A+ disabled)
    base_grades = _compute_audit_quintile_grades(len(keyed))

    # ---- 7. Apply Audit-Only modifiers and build output rows
    out_rows = []
    for position, (_, row) in enumerate(keyed.iterrows()):
        audit_score_value = float(row[AUDIT_SCORE_COL])
        base = base_grades[position]
        final_grade = _assign_audit_only_grade(row, base)

        out_rows.append({
            "reference_rank": position + 1,
            "reference_grade": final_grade,
            "stable_stock_key": row[STABLE_KEY_COL],
            "stock_name": _safe_get(row, "stock_name", "n/a"),
            "nse_code": _safe_get(row, "nse_code", None),
            "market_cap_bucket": _safe_get(row, "market_cap_bucket", "n/a"),
            "strategy_category": STRATEGY_AUDIT_ONLY,
            # Per RR-D spec: synthetic audit score occupies the existing
            # strategy_score column rather than adding a new column.
            "strategy_score": audit_score_value,
            "quality_safety_score": _safe_get_float(row, "quality_safety_score"),
            "tradability_score": _safe_get_float(row, "tradability_score"),
            "entry_quality_tag": _safe_get(row, "entry_quality_tag", "n/a"),
            "setup_quality_tag": _safe_get(row, "setup_quality_tag", "n/a"),
            "setup_confirmation_tag": _safe_get(row, "setup_confirmation_tag", "n/a"),
            "setup_risk_tag": _safe_get(row, "setup_risk_tag", "n/a"),
            "red_flag_count": _safe_get_int(row, "red_flag_count"),
            "positive_flag_count": _safe_get_int(row, "positive_flag_count"),
            "analysis_confidence_score": _safe_get_float(row, "analysis_confidence_score"),
            "top_strengths": _format_top_strengths(row, STRATEGY_AUDIT_ONLY),
            "main_weakness": _format_main_weakness(row, STRATEGY_AUDIT_ONLY),
            "why_ranked_here": _format_why_ranked_here(
                row, final_grade, STRATEGY_AUDIT_ONLY, audit_score_value
            ),
            "suggested_action_view": _format_suggested_action_view(
                final_grade, STRATEGY_AUDIT_ONLY
            ),
            "must_generate_report": _compute_must_generate_report(
                final_grade, row, STRATEGY_AUDIT_ONLY
            ),
            "report_priority": _assign_report_priority(final_grade),
        })

    if not out_rows:
        return _empty_reference_frame()
    out_df = pd.DataFrame(out_rows)
    return out_df[REFERENCE_COLUMNS].reset_index(drop=True)


# =====================================================================
# MASTER ORCHESTRATOR
# =====================================================================

def run_reference_ranking(policy_result: dict) -> dict:
    """Build all six reference lists from a selection_policy result dict.

    Inputs:
        policy_result: the dict returned by
            selection_policy.run_selection_policy(df). Must contain the
            keys: swing_shortlist, short_term_shortlist,
            long_term_core_shortlist, long_term_opportunity_shortlist,
            long_term_shortlist, audit_only_universe.

    Returns a dict with the six output keys (REFERENCE_OUTPUT_KEYS):
        Swing_Reference, ShortTerm_Reference, LongTerm_Core_Reference,
        LongTerm_Opp_Reference, LongTerm_Reference, Audit_Only_Reference.

    Each value is a DataFrame with the canonical 23-column schema. Empty
    in RR-A; populated in RR-B onward. A missing input key yields an
    empty reference frame for that category and an INFO log line — never
    an exception.
    """
    if not isinstance(policy_result, dict):
        logger.warning(
            "reference_ranking.run_reference_ranking: expected dict, "
            "got %s — returning all empty frames.",
            type(policy_result).__name__,
        )
        return {key: _empty_reference_frame() for key in REFERENCE_OUTPUT_KEYS}

    def _frame(key: str) -> pd.DataFrame:
        value = policy_result.get(key)
        if value is None:
            logger.info(
                "reference_ranking: policy_result missing key '%s' — "
                "returning empty reference frame.", key,
            )
            return None
        return value

    # Build core sub-keys for the Combined builder. RR-A passes these
    # through unused; RR-D will consume them. Computing them here keeps
    # the orchestrator's signature stable.
    core_sl = _frame("long_term_core_shortlist")
    opp_sl = _frame("long_term_opportunity_shortlist")
    core_keys: set = set()
    opp_keys: set = set()
    if isinstance(core_sl, pd.DataFrame) and len(core_sl) > 0:
        core_keys = set(add_stable_stock_key(core_sl)[STABLE_KEY_COL].astype(str).tolist())
    if isinstance(opp_sl, pd.DataFrame) and len(opp_sl) > 0:
        opp_keys = set(add_stable_stock_key(opp_sl)[STABLE_KEY_COL].astype(str).tolist())

    return {
        "Swing_Reference":          build_swing_reference(_frame("swing_shortlist")),
        "ShortTerm_Reference":      build_short_term_reference(_frame("short_term_shortlist")),
        "LongTerm_Core_Reference":  build_long_term_core_reference(_frame("long_term_core_shortlist")),
        "LongTerm_Opp_Reference":   build_long_term_opportunity_reference(_frame("long_term_opportunity_shortlist")),
        "LongTerm_Reference":       build_long_term_combined_reference(
                                        _frame("long_term_shortlist"),
                                        core_keys=core_keys,
                                        opportunity_keys=opp_keys,
                                    ),
        "Audit_Only_Reference":     build_audit_only_reference(_frame("audit_only_universe")),
    }


# =====================================================================
# MODULE GUARD
# =====================================================================

if __name__ == "__main__":
    # This module is not meant to be run directly. Import it from
    # pipeline.py (in a later patch) and call run_reference_ranking(...).
    print(
        "reference_ranking.py is a library module.\n"
        "Import it and call run_reference_ranking(policy_result).\n"
        "Status: RR-A skeleton — returns empty 23-column frames."
    )
