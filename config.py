from pathlib import Path

# ==============================
# FILE PATHS — DRIVE-AGNOSTIC DEFAULTS
# ==============================
# These defaults are now relative to the folder where this config.py file
# is kept. pipeline.py also has a stronger discovery layer that can find
# Master_merged.xlsx in either the script folder or its parent folder.
# This keeps the pipeline independent of H:, D:, C:, Google Drive drive
# letters, or external-drive letters.
SCRIPT_DIR_DEFAULT = Path(__file__).resolve().parent
MASTER_MERGED_FILENAME = "Master_merged.xlsx"
OUTPUT_DIR_NAME = "scoring_output"

MASTER_MERGED_PATH_DEFAULT = SCRIPT_DIR_DEFAULT / MASTER_MERGED_FILENAME
OUTPUT_DIR_DEFAULT = SCRIPT_DIR_DEFAULT / OUTPUT_DIR_NAME

# ==============================
# REQUIRED COLUMNS
# ==============================
REQUIRED_COLUMNS = [
    "stock_name","nse_code","isin","sector_name","industry_name",
    "current_price","market_capitalization",
    "trendlyne_momentum_score","day_rsi","day_adx",
    "day_macd","day_macd_signal_line",
    "day_sma50","day_sma200",
    "day_volume","week_volume_avg","month_volume_avg",
    "year_1_high","rr_nifty50_year_pct",

    "sales_growth_3y_pct","sales_growth_5y_pct",
    "profit_growth_3y_pct","profit_growth_5y_pct",
    "eps_growth_3y_pct","eps_growth_5y_pct",

    "sales_q_latest","sales_q_prev","sales_q_yoy_base",
    "profit_q_latest","profit_q_prev","profit_q_yoy_base",
    "eps_q_latest","eps_q_prev","eps_q_yoy_base",

    "roce","roce_3y_avg","roce_5y_avg",
    "roe","roe_3y_avg",

    "opm_current","opm_last_year","opm_5y_avg",

    "cfo_latest","cfo_prev",
    "fcf_latest","fcf_prev","fcf_3y",

    "debt_to_equity","current_ratio","interest_coverage",
    "altman_z_score",

    "peg_ratio","price_to_fcf","price_to_cfo",
    "price_to_sales","pe_ttm",

    "promoter_holding_latest_pct",
    "promoter_holding_change_qoq_pct",
    "fii_holding_change_qoq_pct",
    "mf_holding_change_qoq_pct",
]

# ==============================
# SCORING WEIGHTS
# ==============================

SWING_WEIGHTS = {
    "trend_health_factor": 0.25,
    "entry_timing_factor": 0.25,
    "volume_factor": 0.20,
    "catalyst_proxy_factor": 0.15,
    "risk_factor": 0.10,
    "valuation_factor": 0.05,
}

SHORT_TERM_WEIGHTS = {
    "earnings_accel_factor": 0.25,
    "trend_health_factor": 0.20,
    "momentum_factor": 0.15,
    "ownership_factor": 0.15,
    "growth_factor": 0.10,
    "risk_factor": 0.10,
    "valuation_factor": 0.05,
}

LONG_TERM_WEIGHTS = {
    "growth_factor": 0.25,
    "business_quality_factor": 0.25,
    "cashflow_quality_factor": 0.20,
    "risk_factor": 0.15,
    "valuation_factor": 0.10,
    "ownership_factor": 0.03,
    "technical_sanity_factor": 0.02,
}


# ==============================
# PHASE 1 V2 WEIGHTS
# ==============================

SWING_WEIGHTS_V2 = {
    "swing_volume_v2": 0.32,
    "swing_relative_strength_v2": 0.23,
    "swing_trend_v2": 0.19,
    "swing_execution_v2": 0.16,
    "swing_range_v2": 0.10,
}

SHORT_TERM_WEIGHTS_V2 = {
    "short_term_volume_v2": 0.28,
    "short_term_relative_strength_v2": 0.20,
    "short_term_tactical_momentum_v2": 0.22,
    "short_term_earnings_accel_v2": 0.15,
    "short_term_execution_v2": 0.15,
}

LONG_TERM_WEIGHTS_V2 = {
    "long_term_growth_v2": 0.22,
    "long_term_quality_v2": 0.20,
    "long_term_resilience_v2": 0.15,
    "long_term_sponsorship_v2": 0.125,
    "long_term_cashflow_v2": 0.105,
    "long_term_working_capital_v2": 0.08,
    "long_term_leadership_v2": 0.08,
    "long_term_valuation_v2": 0.04,
}

# ==============================
# PENALTIES
# ==============================

SWING_PENALTIES = {
    "overextended_sma200": 10,
    "rsi_overheat": 6,
    "weak_volume": 6,
    "negative_profit_yoy": 8,
    "high_debt": 10,
}

SHORT_TERM_PENALTIES = {
    "negative_profit_and_eps_yoy": 10,
    "very_overextended_sma200": 8,
    "negative_fii_and_mf": 5,
    "weak_altman": 6,
}

LONG_TERM_PENALTIES = {
    "high_debt": 15,
    "critical_altman": 12,
    "double_negative_fcf": 10,
    "weak_roce_5y": 8,
    "negative_profit_growth_5y": 10,
}