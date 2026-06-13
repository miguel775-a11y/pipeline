
import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)

from config import (
    SWING_WEIGHTS,
    SHORT_TERM_WEIGHTS,
    LONG_TERM_WEIGHTS,
    SWING_WEIGHTS_V2,
    SHORT_TERM_WEIGHTS_V2,
    LONG_TERM_WEIGHTS_V2,
    SWING_PENALTIES,
    SHORT_TERM_PENALTIES,
    LONG_TERM_PENALTIES,
)


# ======================================
# NORMALIZATION HELPERS
# ======================================

def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = series.astype(float)
    ranked = s.rank(pct=True, method="average")
    score = ranked * 100 if higher_is_better else (1 - ranked) * 100
    return score.fillna(0).clip(0, 100)


def piecewise_score(value: pd.Series, points: list[tuple[float, float]]) -> pd.Series:
    x = value.astype(float)
    xp = [p[0] for p in points]
    fp = [p[1] for p in points]
    out = np.interp(x.fillna(np.nan), xp, fp)
    s = pd.Series(out, index=value.index)
    s[value.isna()] = 0
    return s.clip(0, 100)


def piecewise_score_neutral_missing(value: pd.Series, points: list[tuple[float, float]], missing_score: float = 50) -> pd.Series:
    x = value.astype(float)
    xp = [p[0] for p in points]
    fp = [p[1] for p in points]
    out = np.interp(x.fillna(np.nan), xp, fp)
    s = pd.Series(out, index=value.index)
    s[value.isna()] = missing_score
    return s.clip(0, 100)


def rsi_score(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    result = pd.Series(0.0, index=s.index)
    result = np.where(s < 40, 10, result)
    result = np.where((s >= 40) & (s < 50), 10 + (s - 40) * 4, result)
    result = np.where((s >= 50) & (s <= 68), 50 + (s - 50) * (50 / 18), result)
    result = np.where((s > 68) & (s <= 75), 100 - (s - 68) * (40 / 7), result)
    result = np.where(s > 75, 30, result)
    return pd.Series(result, index=s.index).fillna(0).clip(0, 100)


def adx_score(series: pd.Series) -> pd.Series:
    return piecewise_score(series, [(0, 20), (20, 40), (25, 65), (35, 85), (50, 100)])


def debt_to_equity_score(series: pd.Series) -> pd.Series:
    return piecewise_score(series, [(0, 100), (0.5, 85), (1.0, 60), (2.0, 20), (3.0, 0)])


def current_ratio_score(series: pd.Series) -> pd.Series:
    return piecewise_score(series, [(0.8, 0), (1.0, 25), (1.5, 60), (2.0, 85), (3.0, 100)])


def altman_z_score_rule(series: pd.Series) -> pd.Series:
    return piecewise_score(series, [(1.8, 0), (2.2, 35), (3.0, 65), (4.0, 85), (6.0, 100)])


def peg_score(series: pd.Series) -> pd.Series:
    return piecewise_score(series, [(0.7, 100), (1.0, 85), (1.5, 55), (2.0, 25), (3.0, 0)])


def distance_52w_high_score(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return pd.Series(
        np.where((s <= 0) & (s >= -0.08), 100,
                 np.where((s < -0.08) & (s >= -0.15), 75,
                          np.where((s < -0.15) & (s >= -0.25), 45,
                                   np.where(s < -0.25, 15, 90)))),
        index=s.index,
    ).clip(0, 100)


def piotroski_score_rule(series: pd.Series) -> pd.Series:
    return piecewise_score_neutral_missing(series, [(2, 10), (4, 35), (6, 65), (7, 80), (8, 90), (9, 100)], 50)


def earnings_yield_score_rule(series: pd.Series) -> pd.Series:
    return piecewise_score_neutral_missing(series, [(0, 0), (2, 20), (4, 40), (6, 60), (8, 80), (12, 100)], 50)


def ev_ebitda_score_rule(series: pd.Series) -> pd.Series:
    return piecewise_score_neutral_missing(series, [(2, 100), (6, 85), (10, 65), (15, 40), (20, 20), (30, 0)], 50)


def cash_conversion_cycle_score_rule(series: pd.Series) -> pd.Series:
    return piecewise_score_neutral_missing(series, [(-20, 100), (0, 85), (30, 65), (60, 40), (120, 10), (200, 0)], 50)


def debtor_days_score_rule(series: pd.Series) -> pd.Series:
    return piecewise_score_neutral_missing(series, [(0, 100), (30, 85), (60, 65), (90, 40), (150, 10), (250, 0)], 50)


def inventory_turnover_score_rule(series: pd.Series) -> pd.Series:
    return piecewise_score_neutral_missing(series, [(0, 0), (1, 20), (2, 40), (4, 65), (6, 80), (10, 100)], 50)


def _safe_score(df: pd.DataFrame, col: str, default: float = 50.0) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _binary_score(df: pd.DataFrame, col: str, default: float = 50.0) -> pd.Series:
    if col in df.columns:
        return (df[col].fillna(0).astype(float) * 100).clip(0, 100)
    return pd.Series(default, index=df.index, dtype=float)



# ======================================
# STEP 1: NORMALIZED SCORES
# ======================================

def apply_normalized_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    base_features = [
        "sales_growth_long", "profit_growth_long", "eps_growth_long",
        "sales_qoq", "sales_yoy", "profit_qoq", "profit_yoy",
        "eps_qoq", "eps_yoy", "operating_leverage",
        "roce_quality_raw", "roe_quality_raw", "margin_quality_raw",
        "cfo_growth", "fcf_growth", "ownership_flow_raw",
        "trendlyne_momentum_score", "rr_nifty50_year_pct",
        "volume_ratio_week", "volume_ratio_month", "volume_ratio_blended",
        "price_vs_sma5_pct", "price_vs_sma30_pct", "price_vs_sma50_pct", "price_vs_sma100_pct", "price_vs_sma200_pct",
        "macd_spread", "promoter_holding_latest_pct", "promoter_holding_change_qoq_pct",
        "fii_holding_change_qoq_pct", "mf_holding_change_qoq_pct",
        "momentum_delta_day", "momentum_delta_week", "momentum_delta_month",
        "sales_vs_sector_yoy_gap", "profit_vs_sector_yoy_gap",
        "sma50_to_sma200_spread_pct", "room_to_month_high_pct",
        "ownership_trend_score", "debtor_days_delta_vs_3y",
        "inventory_turnover_ratio", "inventory_turnover_trend",
    ]
    for f in base_features:
        if f in out.columns:
            out[f"{f}_score"] = percentile_score(out[f], True)

    for f in ["pe_ttm", "cashflow_multiple", "price_to_sales"]:
        if f in out.columns:
            out[f"{f}_score"] = percentile_score(out[f], False)

    out["day_rsi_score"] = rsi_score(out["day_rsi"])
    out["day_adx_score"] = adx_score(out["day_adx"])
    out["debt_to_equity_score"] = debt_to_equity_score(out["debt_to_equity"])
    out["current_ratio_score"] = current_ratio_score(out["current_ratio"])
    out["altman_z_score_score"] = altman_z_score_rule(out["altman_z_score"])
    out["peg_ratio_score"] = peg_score(out["peg_ratio"])
    out["distance_from_52w_high_pct_score"] = distance_52w_high_score(out["distance_from_52w_high_pct"])

    out["fcf_positive_score"] = out["fcf_positive"].fillna(0) * 100
    out["fcf_consistency_score"] = out["fcf_consistency"].fillna(0) * 100
    out["not_overextended_score"] = out["not_overextended_raw"].fillna(0) * 100

    # Screener-field normalization
    for f in ["earnings_yield", "piotroski_score", "inventory_turnover_ratio", "inventory_turnover_trend"]:
        if f in out.columns:
            out[f"{f}_score"] = percentile_score(out[f], True)

    for f in ["ev_ebitda", "debtor_days", "cash_conversion_cycle_days", "debtor_days_delta_vs_3y"]:
        if f in out.columns:
            out[f"{f}_score"] = percentile_score(out[f], False)

    out["piotroski_rule_score"] = piotroski_score_rule(out["piotroski_score"])
    out["earnings_yield_rule_score"] = earnings_yield_score_rule(out["earnings_yield"])
    out["ev_ebitda_rule_score"] = ev_ebitda_score_rule(out["ev_ebitda"])
    out["cash_conversion_cycle_days_rule_score"] = cash_conversion_cycle_score_rule(out["cash_conversion_cycle_days"])
    out["debtor_days_rule_score"] = debtor_days_score_rule(out["debtor_days"])
    out["inventory_turnover_rule_score"] = inventory_turnover_score_rule(out["inventory_turnover_ratio"])

    if "is_financial_like" in out.columns:
        fin_mask = out["is_financial_like"] == 1
        for col in ["ev_ebitda_rule_score", "cash_conversion_cycle_days_rule_score", "debtor_days_rule_score", "inventory_turnover_rule_score"]:
            out.loc[fin_mask, col] = 50

    out["inventory_turnover_trend_score"] = percentile_score(out["inventory_turnover_trend"].fillna(0), True)
    out.loc[out["inventory_turnover_trend"].isna(), "inventory_turnover_trend_score"] = 50
    if "is_financial_like" in out.columns:
        out.loc[out["is_financial_like"] == 1, "inventory_turnover_trend_score"] = 50

    # Phase 1 migration score columns
    phase1_high = [
        "sales_growth_3y_pct", "sales_growth_5y_pct", "profit_growth_3y_pct", "eps_growth_3y_pct",
        "roce_3y_avg", "roe_3y_avg", "current_ratio", "interest_coverage",
        "institutional_holding_current_qtr_pct", "institutional_holding_change_8qtr_pct",
        "cash_from_operating_activity_annual", "rr_sector_3year_pct",
        "rr_sector_week_pct", "rr_sector_month_pct", "rr_industry_week_pct", "rr_industry_month_pct",
        "rr_nifty50_week_pct", "rr_nifty50_quarter_pct",
        "day_roc21", "revenue_qoq_growth_pct", "net_profit_qoq_growth_pct", "qtr_change_pct",
        "volume_surge_strength", "room_to_month_high_pct", "standard_r2_to_price_diff_pct",
        "standard_r3_to_price_diff_pct", "standard_s3_to_price_diff_pct",
        "qtr_range_position", "day_close_strength", "month_change_pct",
        "roe_vs_sector", "roe_vs_industry", "roa_vs_sector", "roa_vs_industry",
        "days_traded_below_current_pe_pct", "avg_debtor_days_3y", "inventory_turnover_ratio_5y_back",
        "long_term_debt_to_equity_annual", "institutional_holding_change_8qtr_pct"
    ]
    for f in phase1_high:
        if f in out.columns:
            out[f"{f}_score"] = percentile_score(out[f], True)
        elif f + "_score" not in out.columns:
            out[f"{f}_score"] = 50.0

    phase1_low = [
        "pe_vs_3y_avg_pct", "standard_r1_to_price_diff_pct", "standard_s1_to_price_diff_pct",
        "standard_s2_to_price_diff_pct", "peg_gap_vs_sector", "peg_gap_vs_industry"
    ]
    for f in phase1_low:
        if f in out.columns:
            out[f"{f}_score"] = percentile_score(out[f], False)
        elif f + "_score" not in out.columns:
            out[f"{f}_score"] = 50.0

    out["volume_confirmation_flag_score"] = _binary_score(out, "volume_confirmation_flag")
    out["breakout_volume_confirmation_flag_score"] = _binary_score(out, "breakout_volume_confirmation_flag")
    out["cashflow_alignment_flag_score"] = _binary_score(out, "cashflow_alignment_flag")

    return out


# ======================================
# STEP 2: FACTORS
# ======================================

def build_factor_library(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["growth_factor"] = (
        0.35 * out["sales_growth_long_score"] +
        0.35 * out["profit_growth_long_score"] +
        0.30 * out["eps_growth_long_score"]
    )
    out["earnings_accel_factor"] = (
        0.15 * out["sales_qoq_score"] +
        0.20 * out["sales_yoy_score"] +
        0.15 * out["profit_qoq_score"] +
        0.20 * out["profit_yoy_score"] +
        0.10 * out["eps_qoq_score"] +
        0.10 * out["eps_yoy_score"] +
        0.10 * out["operating_leverage_score"]
    )
    out["business_quality_factor"] = (
        0.45 * out["roce_quality_raw_score"] +
        0.25 * out["roe_quality_raw_score"] +
        0.30 * out["margin_quality_raw_score"]
    )
    out["cashflow_quality_factor"] = (
        0.35 * out["cfo_growth_score"] +
        0.35 * out["fcf_growth_score"] +
        0.15 * out["fcf_positive_score"] +
        0.15 * out["fcf_consistency_score"]
    )
    out["risk_factor"] = (
        0.40 * out["debt_to_equity_score"] +
        0.20 * out["current_ratio_score"] +
        0.20 * percentile_score(out["interest_coverage"], True) +
        0.20 * out["altman_z_score_score"]
    )
    out["valuation_factor"] = (
        0.40 * out["peg_ratio_score"] +
        0.25 * out["cashflow_multiple_score"] +
        0.20 * out["price_to_sales_score"] +
        0.15 * out["pe_ttm_score"]
    )
    out["ownership_factor"] = (
        0.40 * out["promoter_holding_latest_pct_score"] +
        0.20 * out["promoter_holding_change_qoq_pct_score"] +
        0.20 * out["fii_holding_change_qoq_pct_score"] +
        0.20 * out["mf_holding_change_qoq_pct_score"]
    )
    out["trend_health_factor"] = (
        0.30 * out["price_vs_sma50_pct_score"] +
        0.35 * out["price_vs_sma200_pct_score"] +
        0.20 * out["day_adx_score"] +
        0.15 * out["macd_spread_score"]
    )
    out["entry_timing_factor"] = (
        0.35 * out["day_rsi_score"] +
        0.25 * out["distance_from_52w_high_pct_score"] +
        0.20 * out["volume_ratio_week_score"] +
        0.20 * out["volume_ratio_month_score"]
    )
    out["volume_factor"] = (
        0.60 * out["volume_ratio_week_score"] +
        0.40 * out["volume_ratio_month_score"]
    )
    out["momentum_factor"] = (
        0.50 * out["trendlyne_momentum_score_score"] +
        0.30 * out["rr_nifty50_year_pct_score"] +
        0.20 * out["distance_from_52w_high_pct_score"]
    )
    out["technical_sanity_factor"] = (
        0.70 * out["price_vs_sma200_pct_score"] +
        0.30 * out["not_overextended_score"]
    )
    out["catalyst_proxy_factor"] = (
        0.35 * out["earnings_accel_factor"] +
        0.20 * out["operating_leverage_score"] +
        0.20 * out["ownership_flow_raw_score"] +
        0.25 * out["trend_health_factor"]
    )

    out["forensic_quality_factor"] = (
        0.40 * out["piotroski_rule_score"] +
        0.25 * out["debtor_days_rule_score"] +
        0.20 * out["cash_conversion_cycle_days_rule_score"] +
        0.15 * out["altman_z_score_score"]
    )
    out["working_capital_factor"] = (
        0.35 * out["debtor_days_rule_score"] +
        0.30 * out["cash_conversion_cycle_days_rule_score"] +
        0.20 * out["inventory_turnover_rule_score"] +
        0.15 * out["inventory_turnover_trend_score"]
    )
    out["value_yield_factor"] = (
        0.45 * out["earnings_yield_rule_score"] +
        0.35 * out["ev_ebitda_rule_score"] +
        0.20 * out["valuation_factor"]
    )

    if "is_financial_like" in out.columns:
        fin_mask = out["is_financial_like"] == 1
        out.loc[fin_mask, "working_capital_factor"] = 50
        out.loc[fin_mask, "forensic_quality_factor"] = (
            0.70 * out.loc[fin_mask, "piotroski_rule_score"] +
            0.30 * out.loc[fin_mask, "altman_z_score_score"]
        )
        out.loc[fin_mask, "value_yield_factor"] = (
            0.70 * out.loc[fin_mask, "earnings_yield_rule_score"] +
            0.30 * out.loc[fin_mask, "valuation_factor"]
        )

    # ======================================
    # PHASE 1 V2 FACTORS
    # ======================================
    out["swing_volume_v2"] = (
        0.40 * _safe_score(out, "volume_ratio_blended_score") +
        0.35 * _safe_score(out, "volume_surge_strength_score") +
        0.15 * _safe_score(out, "volume_confirmation_flag_score") +
        0.10 * _safe_score(out, "breakout_volume_confirmation_flag_score")
    )
    out["swing_relative_strength_v2"] = (
        0.20 * _safe_score(out, "rr_sector_week_pct_score") +
        0.25 * _safe_score(out, "rr_sector_month_pct_score") +
        0.20 * _safe_score(out, "rr_industry_week_pct_score") +
        0.25 * _safe_score(out, "rr_industry_month_pct_score") +
        0.10 * _safe_score(out, "rr_nifty50_week_pct_score")
    )
    out["swing_trend_v2"] = (
        0.35 * _safe_score(out, "day_roc21_score") +
        0.25 * _safe_score(out, "macd_spread_score") +
        0.20 * _safe_score(out, "day_adx_score") +
        0.20 * _safe_score(out, "price_vs_sma100_pct_score")
    )
    out["swing_execution_v2"] = (
        0.35 * _safe_score(out, "room_to_month_high_pct_score") +
        0.25 * _safe_score(out, "standard_r2_to_price_diff_pct_score") +
        0.20 * _safe_score(out, "standard_s3_to_price_diff_pct_score") +
        0.20 * _safe_score(out, "standard_r1_to_price_diff_pct_score")
    )
    out["swing_range_v2"] = (
        0.30 * _safe_score(out, "month_change_pct_score") +
        0.25 * _safe_score(out, "qtr_change_pct_score") +
        0.25 * _safe_score(out, "qtr_range_position_score") +
        0.20 * _safe_score(out, "day_close_strength_score")
    )

    out["short_term_volume_v2"] = (
        0.45 * _safe_score(out, "volume_ratio_blended_score") +
        0.20 * _safe_score(out, "day_mfi_score") +
        0.20 * _safe_score(out, "volume_confirmation_flag_score") +
        0.15 * _safe_score(out, "breakout_volume_confirmation_flag_score")
    )
    out["short_term_relative_strength_v2"] = (
        0.25 * _safe_score(out, "rr_nifty50_week_pct_score") +
        0.25 * _safe_score(out, "rr_sector_month_pct_score") +
        0.25 * _safe_score(out, "rr_industry_month_pct_score") +
        0.25 * _safe_score(out, "rr_nifty50_quarter_pct_score")
    )
    out["short_term_tactical_momentum_v2"] = (
        0.35 * _safe_score(out, "day_roc21_score") +
        0.25 * _safe_score(out, "macd_spread_score") +
        0.20 * _safe_score(out, "day_adx_score") +
        0.20 * _safe_score(out, "normalized_momentum_score_score")
    )
    out["short_term_earnings_accel_v2"] = (
        0.45 * _safe_score(out, "revenue_qoq_growth_pct_score") +
        0.45 * _safe_score(out, "net_profit_qoq_growth_pct_score") +
        0.10 * _safe_score(out, "qtr_change_pct_score")
    )
    out["short_term_execution_v2"] = (
        0.35 * _safe_score(out, "standard_r1_to_price_diff_pct_score") +
        0.35 * _safe_score(out, "room_to_month_high_pct_score") +
        0.15 * _safe_score(out, "qtr_range_position_score") +
        0.15 * _safe_score(out, "day_close_strength_score")
    )

    out["long_term_growth_v2"] = (
        0.30 * _safe_score(out, "sales_growth_3y_pct_score") +
        0.20 * _safe_score(out, "sales_growth_5y_pct_score") +
        0.25 * _safe_score(out, "profit_growth_3y_pct_score") +
        0.25 * _safe_score(out, "eps_growth_3y_pct_score")
    )
    out["long_term_quality_v2"] = (
        0.35 * _safe_score(out, "roce_3y_avg_score") +
        0.25 * _safe_score(out, "roe_3y_avg_score") +
        0.20 * _safe_score(out, "roe_vs_sector_score") +
        0.20 * _safe_score(out, "roe_vs_industry_score")
    )
    out["long_term_resilience_v2"] = (
        0.40 * _safe_score(out, "current_ratio_score") +
        0.40 * _safe_score(out, "interest_coverage_score") +
        0.20 * _safe_score(out, "long_term_debt_to_equity_annual_score")
    )
    out["long_term_sponsorship_v2"] = (
        0.50 * _safe_score(out, "institutional_holding_current_qtr_pct_score") +
        0.50 * _safe_score(out, "institutional_holding_change_8qtr_pct_score")
    )
    out["long_term_cashflow_v2"] = (
        0.60 * _safe_score(out, "cash_from_operating_activity_annual_score") +
        0.40 * _safe_score(out, "cashflow_alignment_flag_score")
    )
    out["long_term_working_capital_v2"] = (
        0.40 * _safe_score(out, "avg_debtor_days_3y_score") +
        0.20 * _safe_score(out, "debtor_days_delta_vs_3y_score") +
        0.20 * _safe_score(out, "inventory_turnover_ratio_5y_back_score") +
        0.20 * _safe_score(out, "inventory_turnover_trend_score")
    )
    out["long_term_leadership_v2"] = 1.0 * _safe_score(out, "rr_sector_3year_pct_score")
    out["long_term_valuation_v2"] = (
        0.50 * _safe_score(out, "pe_vs_3y_avg_pct_score") +
        0.50 * _safe_score(out, "days_traded_below_current_pe_pct_score")
    )

    return out


# ======================================
# STEP 3: PENALTIES
# ======================================

def apply_penalties(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Base config penalties using explicit flags
    out["swing_penalty"] = (
        out["overextended_sma200_flag"] * SWING_PENALTIES["overextended_sma200"] +
        out["rsi_overheat_flag"] * SWING_PENALTIES["rsi_overheat"] +
        out["weak_volume_flag"] * SWING_PENALTIES["weak_volume"] +
        out["negative_profit_yoy_flag"] * SWING_PENALTIES["negative_profit_yoy"] +
        out["high_debt_flag"] * SWING_PENALTIES["high_debt"]
    ).astype(float)

    out["short_term_penalty"] = (
        out["negative_profit_and_eps_yoy_flag"] * SHORT_TERM_PENALTIES["negative_profit_and_eps_yoy"] +
        out["very_overextended_sma200_flag"] * SHORT_TERM_PENALTIES["very_overextended_sma200"] +
        out["negative_fii_and_mf_flag"] * SHORT_TERM_PENALTIES["negative_fii_and_mf"] +
        out["weak_altman_flag"] * SHORT_TERM_PENALTIES["weak_altman"]
    ).astype(float)

    out["long_term_penalty"] = (
        out["high_debt_flag"] * LONG_TERM_PENALTIES["high_debt"] +
        out["weak_altman_flag"] * LONG_TERM_PENALTIES["critical_altman"] +
        out["double_negative_fcf_flag"] * LONG_TERM_PENALTIES["double_negative_fcf"] +
        out["weak_roce_5y_flag"] * LONG_TERM_PENALTIES["weak_roce_5y"] +
        out["negative_profit_growth_5y_flag"] * LONG_TERM_PENALTIES["negative_profit_growth_5y"]
    ).astype(float)

    # Additional nuanced penalties from newer diagnostic layer
    out["swing_penalty"] += (
        2.0 * out["valuation_and_cashflow_contradiction_flag"] +
        1.0 * out["working_capital_stress_flag"] +
        1.0 * out["weak_piotroski_flag"]
    )

    out["short_term_penalty"] += (
        2.5 * out["working_capital_stress_flag"] +
        1.5 * out["debtor_deterioration_flag"] +
        1.5 * out["weak_piotroski_flag"] +
        1.0 * out["margin_deterioration_flag_final"]
    )

    out["long_term_penalty"] += (
        3.0 * out["working_capital_stress_flag"] +
        2.5 * out["debtor_deterioration_flag"] +
        2.0 * out["weak_inventory_trend_flag"] +
        4.0 * out["weak_piotroski_flag"] +
        2.0 * out["valuation_and_cashflow_contradiction_flag"] +
        2.0 * out["falling_institutional_support_flag"]
    )

    return out


def apply_bucket_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["swing_bucket_penalty"] = 0.0
    out["short_term_bucket_penalty"] = 0.0
    out["long_term_bucket_penalty"] = 0.0

    out.loc[out["market_cap_bucket"] == "Micro Cap", "swing_bucket_penalty"] += 12
    out.loc[out["market_cap_bucket"] == "Small Cap", "swing_bucket_penalty"] += 4
    out.loc[out["tradability_score"] < 35, "swing_bucket_penalty"] += 10

    out.loc[out["market_cap_bucket"] == "Micro Cap", "short_term_bucket_penalty"] += 10
    out.loc[out["tradability_score"] < 35, "short_term_bucket_penalty"] += 8

    out.loc[out["market_cap_bucket"] == "Micro Cap", "long_term_bucket_penalty"] += 15
    out.loc[(out["market_cap_bucket"] == "Micro Cap") & (out["quality_safety_score"] < 60), "long_term_bucket_penalty"] += 10

    return out


# ======================================
# FINAL SCORING
# ======================================

def _weighted_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for col, wt in weights.items():
        score += df[col] * wt
    return score


def score_engines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["swing_score_raw"] = _weighted_score(out, SWING_WEIGHTS)
    out["short_term_score_raw"] = _weighted_score(out, SHORT_TERM_WEIGHTS)
    out["long_term_score_raw"] = _weighted_score(out, LONG_TERM_WEIGHTS)

    out["swing_score_raw_v2"] = _weighted_score(out, SWING_WEIGHTS_V2)
    out["short_term_score_raw_v2"] = _weighted_score(out, SHORT_TERM_WEIGHTS_V2)
    out["long_term_score_raw_v2"] = _weighted_score(out, LONG_TERM_WEIGHTS_V2)

    # New Screener-field influence by horizon
    out["swing_score_raw"] += (
        0.02 * out["working_capital_factor"] +
        0.01 * out["forensic_quality_factor"] +
        0.005 * out["value_yield_factor"] +
        0.25 * out["value_support_flag"] +
        0.20 * out["strong_piotroski_flag"] -
        1.5 * out["working_capital_stress_flag"] -
        1.0 * out["debtor_deterioration_flag"]
    )

    out["short_term_score_raw"] += (
        0.03 * out["working_capital_factor"] +
        0.015 * out["forensic_quality_factor"] +
        0.01 * out["value_yield_factor"] +
        0.50 * out["strong_piotroski_flag"] -
        2.5 * out["working_capital_stress_flag"] -
        1.5 * out["debtor_deterioration_flag"] -
        1.5 * out["weak_piotroski_flag"]
    )

    out["long_term_score_raw"] += (
        0.02 * out["working_capital_factor"] +
        0.035 * out["forensic_quality_factor"] +
        0.04 * out["value_yield_factor"] +
        1.0 * out["strong_piotroski_flag"] +
        1.0 * out["deep_value_support_flag"] +
        0.75 * out["compounder_setup_flag"] -
        3.0 * out["working_capital_stress_flag"] -
        2.5 * out["debtor_deterioration_flag"] -
        2.0 * out["weak_inventory_trend_flag"] -
        4.0 * out["weak_piotroski_flag"]
    )

    out = apply_penalties(out)
    out = apply_bucket_adjustments(out)

    out["swing_score"] = (out["swing_score_raw"] - out["swing_penalty"] - out["swing_bucket_penalty"]).clip(0, 100)
    out["short_term_score"] = (out["short_term_score_raw"] - out["short_term_penalty"] - out["short_term_bucket_penalty"]).clip(0, 100)
    out["long_term_score"] = (out["long_term_score_raw"] - out["long_term_penalty"] - out["long_term_bucket_penalty"]).clip(0, 100)

    out["swing_score_v2"] = (out["swing_score_raw_v2"] - out["swing_penalty"] - out["swing_bucket_penalty"]).clip(0, 100)
    out["short_term_score_v2"] = (out["short_term_score_raw_v2"] - out["short_term_penalty"] - out["short_term_bucket_penalty"]).clip(0, 100)
    out["long_term_score_v2"] = (out["long_term_score_raw_v2"] - out["long_term_penalty"] - out["long_term_bucket_penalty"]).clip(0, 100)

    out["swing_score_delta_v2"] = out["swing_score_v2"] - out["swing_score"]
    out["short_term_score_delta_v2"] = out["short_term_score_v2"] - out["short_term_score"]
    out["long_term_score_delta_v2"] = out["long_term_score_v2"] - out["long_term_score"]

    out["swing_rank"] = out["swing_score"].rank(ascending=False, method="dense")
    out["short_term_rank"] = out["short_term_score"].rank(ascending=False, method="dense")
    out["long_term_rank"] = out["long_term_score"].rank(ascending=False, method="dense")

    out["swing_rank_v2"] = out["swing_score_v2"].rank(ascending=False, method="dense")
    out["short_term_rank_v2"] = out["short_term_score_v2"].rank(ascending=False, method="dense")
    out["long_term_rank_v2"] = out["long_term_score_v2"].rank(ascending=False, method="dense")

    out["swing_rank_within_bucket"] = out.groupby("market_cap_bucket")["swing_score"].rank(ascending=False, method="dense")
    out["short_term_rank_within_bucket"] = out.groupby("market_cap_bucket")["short_term_score"].rank(ascending=False, method="dense")
    out["long_term_rank_within_bucket"] = out.groupby("market_cap_bucket")["long_term_score"].rank(ascending=False, method="dense")

    out["swing_rank_within_bucket_v2"] = out.groupby("market_cap_bucket")["swing_score_v2"].rank(ascending=False, method="dense")
    out["short_term_rank_within_bucket_v2"] = out.groupby("market_cap_bucket")["short_term_score_v2"].rank(ascending=False, method="dense")
    out["long_term_rank_within_bucket_v2"] = out.groupby("market_cap_bucket")["long_term_score_v2"].rank(ascending=False, method="dense")

    # Primary strategy tag and timing risk now that final scores exist
    score_mat = out[["swing_score", "short_term_score", "long_term_score"]]
    out["primary_strategy_tag"] = score_mat.idxmax(axis=1).map({
        "swing_score": "Swing",
        "short_term_score": "Short Term",
        "long_term_score": "Long Term",
    }).fillna("Swing")

    score_mat_v2 = out[["swing_score_v2", "short_term_score_v2", "long_term_score_v2"]]
    out["primary_strategy_tag_v2"] = score_mat_v2.idxmax(axis=1).map({
        "swing_score_v2": "Swing",
        "short_term_score_v2": "Short Term",
        "long_term_score_v2": "Long Term",
    }).fillna("Swing")
    out["long_term_timing_risk_flag"] = (
        (out["primary_strategy_tag"] == "Long Term") &
        (
            out["entry_quality_tag"].isin(["Watch on Pullback", "Crowded Trend", "Neutral"]) |
            (out["price_vs_sma200_pct"].fillna(0) < 0)
        )
    ).astype(int)

    return out


# ======================================
# VALIDATION
# ======================================

def validate_scored_output(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    for col in ["swing_score", "short_term_score", "long_term_score", "swing_score_v2", "short_term_score_v2", "long_term_score_v2"]:
        mn = float(df[col].min())
        mx = float(df[col].max())
        checks.append({
            "check": col,
            "status": "PASS" if (mn >= 0 and mx <= 100) else "FAIL",
            "detail": f"min={mn:.2f}, max={mx:.2f}",
        })
    return pd.DataFrame(checks)
