import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import requests

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL_DEFAULT = "deepseek-chat"
TOP_N_DEEPSEEK_DEFAULT = 30

# --- Patch RR-H.QA1: relative-strength wording threshold ---
# A value's absolute magnitude must exceed this band before the wording
# can call it "supportive" (positive) or "weak" (negative). Values in
# [-band, +band] are described as "mixed". Tuneable in one place.
_RS_NEUTRAL_BAND_PCT = 2.0


FIELD_NAME_LABELS: Dict[str, str] = {
    "rr_nifty50_month_pct": "1-month relative strength vs Nifty 50",
    "rr_nifty50_week_pct": "1-week relative strength vs Nifty 50",
    "rr_nifty50_quarter_pct": "quarterly relative strength vs Nifty 50",
    "rr_nifty50_year_pct": "1-year relative strength vs Nifty 50",
    "rr_sensex_week_pct": "1-week relative strength vs Sensex",
    "rr_sensex_month_pct": "1-month relative strength vs Sensex",
    "rr_sensex_quarter_pct": "quarterly relative strength vs Sensex",
    "rr_sensex_year_pct": "1-year relative strength vs Sensex",
    "rr_sensex_3year_pct": "3-year relative strength vs Sensex",
    "rr_sensex_5year_pct": "5-year relative strength vs Sensex",
    "rr_sensex_10year_pct": "10-year relative strength vs Sensex",
    "rr_sector_week_pct": "1-week relative strength vs sector",
    "rr_sector_month_pct": "1-month relative strength vs sector",
    "rr_sector_quarter_pct": "quarterly relative strength vs sector",
    "rr_sector_5year_pct": "5-year relative strength vs sector",
    "rr_industry_week_pct": "1-week relative strength vs industry",
    "rr_industry_month_pct": "1-month relative strength vs industry",
    "rr_industry_quarter_pct": "quarterly relative strength vs industry",
    "rr_industry_5year_pct": "5-year relative strength vs industry",
    "rr_nifty50_10year_pct": "10-year relative strength vs Nifty 50",
    "rr_sector_10year_pct": "10-year relative strength vs sector",
    "rr_industry_10year_pct": "10-year relative strength vs industry",
    "day_high": "day high",
    "day_low": "day low",
    "month_low": "month low",
    "volume_ratio_month": "current volume vs 1-month average",
    "volume_ratio_week": "current volume vs 1-week average",
    "price_vs_sma200_pct": "distance from SMA200",
    "price_vs_sma50_pct": "distance from SMA50",
    "profit_yoy": "profit YoY growth",
    "sales_yoy": "sales YoY growth",
    "eps_yoy": "EPS YoY growth",
    "profit_qoq": "profit QoQ growth",
    "sales_qoq": "sales QoQ growth",
    "eps_qoq": "EPS QoQ growth",
    "day_adx": "ADX",
    "day_rsi": "daily RSI",
    "pe_ttm": "PE (TTM)",
    "pe_5yr_average": "5-year average PE",
    "pe_3yr_average": "3-year average PE",
    "sector_pe_ttm": "sector PE",
    "industry_pe_ttm": "industry PE",
    "peg_ratio": "PEG ratio",
    "ev_ebitda": "EV/EBITDA",
    "piotroski_score": "Piotroski score",
    "debtor_days_delta_vs_3y": "debtor days versus 3-year average",
    "trend_structure_stage": "trend structure stage",
    "compounder_setup_flag": "compounder setup flag",
    "earnings_reacceleration_flag": "earnings reacceleration flag",
    "debt_trend_flag": "debt trend",
    "is_financial_like": "financial-stock context",
    "standard_pivot_point": "pivot point",
    "standard_resistance_r1": "pivot resistance R1",
    "standard_resistance_r2": "pivot resistance R2",
    "standard_resistance_r3": "pivot resistance R3",
    "standard_support_s1": "pivot support S1",
    "standard_support_s2": "pivot support S2",
    "standard_support_s3": "pivot support S3",
    "earnings_yield": "earnings yield",
    "cash_conversion_cycle_days": "cash conversion cycle",
    "inventory_turnover_ratio": "inventory turnover",
    "inventory_turnover_trend": "inventory turnover trend",
    "cashflow_alignment_flag": "cash-flow alignment flag",
    "trend_extension_bucket": "trend extension bucket",
    "volume_conviction_bucket": "volume conviction bucket",
    "volume_surge_strength": "volume surge strength",
    "room_to_month_high_pct": "room to month high",
    "promoter_holding_change_qoq_pct": "promoter holding change QoQ",
    "promoter_holding_latest_pct": "promoter holding",
    "fii_holding_change_qoq_pct": "FII holding change QoQ",
    "mf_holding_change_qoq_pct": "MF holding change QoQ",
    "promoter_holding_pledge_percentage_qtr_pct": "promoter pledge percentage",
    "year_10_high": "10-year high",
    "year_10_low": "10-year low",
    "year_5_high": "5-year high",
    "year_5_low": "5-year low",
    "year_1_change_pct": "1-year price change",
    "year_2_price_change_pct": "2-year price change",
    "year_3_price_change_pct": "3-year price change",
    "year_5_price_change_pct": "5-year price change",
    "year_10_price_change_pct": "10-year price change",
    "roe_annual_pct": "annual ROE",
    "roa_annual_pct": "annual ROA",
    "basic_eps_ttm": "basic EPS (TTM)",
    "beta_1month": "1-month beta",
    "beta_3month": "3-month beta",
    "beta_1year": "1-year beta",
    "beta_3year": "3-year beta",
    "long_term_debt_to_equity_annual": "annual long-term debt-to-equity",
    "eps_ttm_growth_pct": "EPS TTM growth",
    "operating_revenue_ttm": "TTM operating revenue",
    "net_profit_ttm": "TTM net profit",
    "operating_profit_margin_qtr_pct": "quarterly operating margin",
    "operating_profit_margin_qtr_4qtr_ago_pct": "quarterly operating margin four quarters ago",
    "qtr_change_pct": "quarter price change",
    "sector_return_on_equity_roe": "sector ROE",
    "industry_return_on_equity_roe": "industry ROE",
    "sector_return_on_assets": "sector ROA",
    "industry_return_on_assets": "industry ROA",
    "sector_revenue_growth_qtr_yoy_pct": "sector revenue growth YoY",
    "sector_net_profit_growth_qtr_yoy_pct": "sector profit growth YoY",
    "sector_revenue_growth_qtr_qoq_pct": "sector revenue growth QoQ",
    "sector_net_profit_growth_qtr_qoq_pct": "sector profit growth QoQ",
    "rr_sector_year_pct": "1-year relative strength vs sector",
    "rr_industry_year_pct": "1-year relative strength vs industry",
    "rr_sector_3year_pct": "3-year relative strength vs sector",
    "rr_industry_3year_pct": "3-year relative strength vs industry",
    "promoter_holding_change_4qtr_pct": "4-quarter promoter holding change",
    "fii_holding_change_4qtr_pct": "4-quarter FII holding change",
    "mf_holding_change_4qtr_pct": "4-quarter MF holding change",
    "institutional_holding_change_4qtr_pct": "4-quarter institutional holding change",
    "days_traded_below_current_pe_pct": "historical days traded below current PE",
    "price_to_sales": "price to sales",
    "debtor_days": "debtor days",
    "day_sma100": "day SMA100",
    "standard_pivot_point": "pivot point",
    "standard_resistance_r2": "pivot resistance R2",
    "standard_support_s3": "pivot support S3",
    "current_ratio": "current ratio",
    "interest_coverage": "interest coverage",
    "sales_growth_3y_pct": "3-year sales growth",
    "sales_growth_5y_pct": "5-year sales growth",
    "profit_growth_3y_pct": "3-year profit growth",
    "eps_growth_3y_pct": "3-year EPS growth",
    "eps_growth_5y_pct": "5-year EPS growth",
    "year_1_high": "52-week high",
    "year_1_low": "52-week low",
    "revenue_qoq_growth_pct": "revenue QoQ growth",
    "net_profit_qoq_growth_pct": "net profit QoQ growth",
    "price_to_book_value_adjusted": "price to book",
    "days_traded_below_current_price_to_book_value_pct": "historical days traded below current price-to-book",
    "roce_3y_avg": "3-year average ROCE",
    "roe_3y_avg": "3-year average ROE",
    "opm_5y_avg": "5-year average operating margin",
    "standard_resistance_r3": "pivot resistance R3",
    "vwap_day": "daily VWAP",
}


SHORTLIST_HORIZON_MAP = {
    "Swing": "Swing",
    "ShortTerm": "Short Term",
    "LongTerm": "Long Term",
}


def _humanize_field_name(token: str) -> str:
    token = token.strip().strip("`")
    if token in FIELD_NAME_LABELS:
        return FIELD_NAME_LABELS[token]
    text = token.replace("_pct", " pct").replace("_qoq", " QoQ").replace("_yoy", " YoY")
    text = text.replace("_ttm", " TTM").replace("_5yr", " 5Y").replace("_3yr", " 3Y")
    text = text.replace("_", " ").strip()
    return text


def _inject_net_annual_cash_flow(text: str, row: pd.Series) -> str:
    if not text:
        return text
    try:
        net_annual_cf = _get_numeric(row, "net_cash_flow_annual", np.nan)
    except Exception:
        return text
    if pd.isna(net_annual_cf) or abs(float(net_annual_cf)) < 1:
        return text
    if re.search(r"\bnet annual cash flow\b", text, flags=re.IGNORECASE):
        return text

    sentence = f" Net annual cash flow is {'positive' if net_annual_cf >= 0 else 'negative'} at ₹{_fmt_num(net_annual_cf)} Cr."

    if net_annual_cf >= 0:
        heading_match = re.search(r"(?:\*\*2\. What confirms the setup\*\*|\n2\. What confirms the setup)", text)
        if heading_match:
            return text[:heading_match.start()].rstrip() + sentence + "\n\n" + text[heading_match.start():].lstrip()
        return text.rstrip() + sentence

    weakness_match = re.search(r"(?:\*\*4\. Strategy fit\*\*|\n4\. Strategy fit)", text)
    if weakness_match:
        return text[:weakness_match.start()].rstrip() + sentence + "\n\n" + text[weakness_match.start():].lstrip()
    return text.rstrip() + sentence




def _inject_trendlyne_durability(text: str, row: pd.Series) -> str:
    if not text:
        return text
    try:
        durability = _get_numeric(row, "trendlyne_durability_score", np.nan)
        valuation = _get_numeric(row, "trendlyne_valuation_score", np.nan)
    except Exception:
        return text
    if pd.isna(durability):
        return text
    if re.search(r"\bTrendlyne durability\b", text, flags=re.IGNORECASE):
        return text

    durability = float(durability)
    valuation_text = ""
    if not pd.isna(valuation):
        valuation_text = f" The accompanying Trendlyne valuation score is {float(valuation):.0f}."

    if durability >= 55:
        sentence = f" Trendlyne durability score is strong at {durability:.0f}, reinforcing long-cycle quality and business consistency." + valuation_text
        heading_match = re.search(r"(?:\*\*2\. What confirms the setup\*\*|\n2\. What confirms the setup)", text)
        if heading_match:
            return text[:heading_match.start()].rstrip() + sentence + "\n\n" + text[heading_match.start():].lstrip()
        return text.rstrip() + sentence

    if durability <= 35:
        sentence = f" Trendlyne durability score is weak at {durability:.0f}, which reduces conviction on long-cycle quality and consistency." + valuation_text
        weakness_match = re.search(r"(?:\*\*4\. Strategy fit\*\*|\n4\. Strategy fit)", text)
        if weakness_match:
            return text[:weakness_match.start()].rstrip() + sentence + "\n\n" + text[weakness_match.start():].lstrip()
        return text.rstrip() + sentence

    return text


def _inject_long_price_history(text: str, row: pd.Series) -> str:
    if not text:
        return text
    current_price = _get_numeric(row, "current_price", np.nan)
    day_high = _get_numeric(row, "day_high", np.nan)
    day_low = _get_numeric(row, "day_low", np.nan)
    month_low = _get_numeric(row, "month_low", np.nan)
    year_5_high = _get_numeric(row, "year_5_high", np.nan)
    year_5_low = _get_numeric(row, "year_5_low", np.nan)
    year_10_high = _get_numeric(row, "year_10_high", np.nan)
    year_10_low = _get_numeric(row, "year_10_low", np.nan)
    year_1_change = _get_numeric(row, "year_1_change_pct", np.nan)
    year_2_change = _get_numeric(row, "year_2_price_change_pct", np.nan)
    year_3_change = _get_numeric(row, "year_3_price_change_pct", np.nan)
    year_5_change = _get_numeric(row, "year_5_price_change_pct", np.nan)
    year_10_change = _get_numeric(row, "year_10_price_change_pct", np.nan)

    strengths: List[str] = []
    confirmations: List[str] = []
    weaknesses: List[str] = []

    if not re.search(r"5-year high|10-year high|5-year low|10-year low", text, flags=re.IGNORECASE):
        range_parts: List[str] = []
        if not pd.isna(year_5_low) and not pd.isna(year_5_high) and year_5_high > 0:
            range_parts.append(f"the 5-year range spans ₹{_fmt_num(year_5_low)} to ₹{_fmt_num(year_5_high)}")
        if not pd.isna(year_10_low) and not pd.isna(year_10_high) and year_10_high > 0:
            range_parts.append(f"the 10-year range spans ₹{_fmt_num(year_10_low)} to ₹{_fmt_num(year_10_high)}")
        if range_parts:
            strengths.append("* **Long Price History:** " + " and ".join(range_parts) + ".")

    if not re.search(r"1-year return|2-year return|3-year return|5-year return|10-year return", text, flags=re.IGNORECASE):
        change_parts: List[str] = []
        for label, value in [("1-year", year_1_change), ("2-year", year_2_change), ("3-year", year_3_change), ("5-year", year_5_change), ("10-year", year_10_change)]:
            if not pd.isna(value):
                change_parts.append(f"{label} return {_fmt_signed_pct(value)}")
        if change_parts:
            strengths.append("* **Long-Cycle Price Performance:** " + ", ".join(change_parts) + ".")

    if not re.search(r"below the 5-year high|below the 10-year high|near the 5-year high|near the 10-year high", text, flags=re.IGNORECASE) and not pd.isna(current_price):
        confirm_parts: List[str] = []
        weak_parts: List[str] = []
        if not pd.isna(year_5_high) and year_5_high > 0:
            dist5 = (current_price / year_5_high) - 1
            if dist5 >= -0.15:
                confirm_parts.append(f"only {_fmt_frac_pct(abs(dist5))} below the 5-year high of ₹{_fmt_num(year_5_high)}")
            elif dist5 <= -0.35:
                weak_parts.append(f"still {_fmt_frac_pct(abs(dist5))} below the 5-year high of ₹{_fmt_num(year_5_high)}")
        if not pd.isna(year_10_high) and year_10_high > 0:
            dist10 = (current_price / year_10_high) - 1
            if dist10 >= -0.15:
                confirm_parts.append(f"only {_fmt_frac_pct(abs(dist10))} below the 10-year high of ₹{_fmt_num(year_10_high)}")
            elif dist10 <= -0.35:
                weak_parts.append(f"still {_fmt_frac_pct(abs(dist10))} below the 10-year high of ₹{_fmt_num(year_10_high)}")
        if confirm_parts:
            confirmations.append("* **Long-Cycle Breakout Context:** The current price is " + " and ".join(confirm_parts) + ".")
        if weak_parts:
            weaknesses.append("* **Long-Cycle Repair Context:** The stock is " + " and ".join(weak_parts) + ", so major historical breakout territory has not yet been reclaimed.")

    if strengths:
        text = _insert_before_heading(text, [r"(?:\*\*2\. What confirms the setup\*\*|2\. What confirms the setup)"], "\n".join(strengths))
    if confirmations:
        text = _insert_before_heading(text, [r"(?:\*\*3\. What weakens the setup\*\*|3\. What weakens the setup)"], "\n".join(confirmations))
    if weaknesses:
        text = _insert_before_heading(text, [r"(?:\*\*4\. Strategy fit\*\*|4\. Strategy fit)"], "\n".join(weaknesses))
    return text


def _insert_before_heading(text: str, heading_patterns: List[str], sentence: str) -> str:
    if not sentence.strip():
        return text
    for pattern in heading_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[:match.start()].rstrip() + "\n" + sentence.rstrip() + "\n\n" + text[match.start():].lstrip()
    return text.rstrip() + "\n" + sentence.rstrip()


def _inject_tier3_visibility(text: str, row: pd.Series) -> str:
    try:
        current_price = _get_numeric(row, "current_price", np.nan)
        day_ema12 = row.get("day_ema12")
        day_ema20 = row.get("day_ema20")
        day_ema50 = row.get("day_ema50")
        day_ema100 = row.get("day_ema100")
        day_atr = row.get("day_atr")
        normalized_momentum_score = row.get("normalized_momentum_score")
        dvm_classification_text = _get_text(row, "dvm_classification_text", "")
    except Exception:
        return text

    confirmations: List[str] = []
    weaknesses: List[str] = []

    if not re.search(r"\bEMA\b", text, flags=re.IGNORECASE):
        ema12 = _get_numeric(row, "day_ema12", np.nan)
        ema20 = _get_numeric(row, "day_ema20", np.nan)
        ema50 = _get_numeric(row, "day_ema50", np.nan)
        ema100 = _get_numeric(row, "day_ema100", np.nan)
        above12 = (not pd.isna(ema12)) and current_price > ema12
        above20 = (not pd.isna(ema20)) and current_price > ema20
        above50 = (not pd.isna(ema50)) and current_price > ema50
        above100 = (not pd.isna(ema100)) and current_price > ema100
        if above12 and above20 and above50 and above100:
            confirmations.append("* **EMA Structure:** The price is above the 12-day, 20-day, 50-day, and 100-day EMA stack, which confirms a constructive trend alignment.")
        elif above12 and above20 and above50 and not pd.isna(ema100):
            confirmations.append(f"* **EMA Structure:** The price is above the 12-day, 20-day, and 50-day EMAs, though the 100-day EMA at ₹{_fmt_num(day_ema100)} remains the next repair hurdle.")
        elif above12 and above20 and not pd.isna(ema50):
            confirmations.append(f"* **EMA Structure:** The price has reclaimed the 12-day and 20-day EMAs, but the 50-day EMA at ₹{_fmt_num(day_ema50)} remains an important trend-repair checkpoint.")
        elif (not pd.isna(ema20)) and (not pd.isna(ema50)) and current_price < ema20 and ema20 < ema50:
            weaknesses.append("* **EMA Structure:** The price remains below the 20-day EMA and the 20-day EMA sits below the 50-day EMA, which shows the trend repair is not yet complete.")

    if not re.search(r"\bATR\b|Average True Range", text, flags=re.IGNORECASE) and not _is_missing(day_atr) and current_price and not pd.isna(current_price):
        atr_pct = (_get_numeric(row, "day_atr", 0) / current_price) * 100.0 if current_price else np.nan
        if not np.isnan(atr_pct):
            if atr_pct <= 4.5:
                confirmations.append(f"* **ATR / Volatility:** ATR stands at ₹{_fmt_num(day_atr)} or about {_fmt_num(atr_pct,1)}% of price, which suggests manageable volatility.")
            elif atr_pct > 6:
                weaknesses.append(f"* **ATR / Volatility:** ATR stands at ₹{_fmt_num(day_atr)} or about {_fmt_num(atr_pct,1)}% of price, which implies higher execution risk and wider stop requirements.")
            else:
                confirmations.append(f"* **ATR / Volatility:** ATR stands at ₹{_fmt_num(day_atr)} or about {_fmt_num(atr_pct,1)}% of price, which points to moderate but still workable volatility.")

    if not re.search(r"normalized momentum", text, flags=re.IGNORECASE) and not _is_missing(normalized_momentum_score):
        nm = _get_numeric(row, "normalized_momentum_score", np.nan)
        if not pd.isna(nm):
            if nm >= 60:
                confirmations.append(f"* **Normalized Momentum:** The normalized momentum score is supportive at {_fmt_num(nm,0)}, reinforcing trend quality.")
            elif nm < 40:
                weaknesses.append(f"* **Normalized Momentum:** The normalized momentum score is weak at {_fmt_num(nm,0)}, which lowers conviction in trend durability.")
            else:
                confirmations.append(f"* **Normalized Momentum:** The normalized momentum score is {_fmt_num(nm,0)}, which indicates a broadly neutral-to-improving momentum regime.")

    if dvm_classification_text and not re.search(r"\bDVM\b|Expensive Performer|Strong Performer|Under Radar|Value Stock|Turnaround Potential|Weak Stock|Momentum Trap|Slowing Down|Value Trap|Expensive Underperformer|Falling Comet|Mid-range Performer|Expensive Star", text, flags=re.IGNORECASE):
        dvm_text_lower = dvm_classification_text.strip().lower()
        dvm_supportive = any(token in dvm_text_lower for token in ["strong performer", "under radar", "value stock", "turnaround potential"])
        dvm_weak = any(token in dvm_text_lower for token in ["weak stock", "momentum trap", "slowing down", "value trap", "expensive underperformer", "falling comet"])
        dvm_mixed = bool(dvm_text_lower) and not dvm_supportive and not dvm_weak
        if dvm_supportive:
            confirmations.append(f"* **DVM Regime:** The DVM classification is '{dvm_classification_text}', which supports a constructive momentum regime.")
        elif dvm_weak:
            weaknesses.append(f"* **DVM Regime:** The DVM classification is '{dvm_classification_text}', which reflects a weak regime label and lowers confidence in the move.")
        elif dvm_mixed:
            if "expensive" in dvm_text_lower:
                weaknesses.append(f"* **DVM Regime:** The DVM classification is '{dvm_classification_text}', which suggests the stock is performing but already in a richer regime.")
            else:
                confirmations.append(f"* **DVM Regime:** The DVM classification is '{dvm_classification_text}', which is constructive but not yet a top-tier momentum regime.")

    if confirmations:
        text = _insert_before_heading(text, [r"(?:\*\*3\. What weakens the setup\*\*|3\. What weakens the setup)"], "\n".join(confirmations))
    if weaknesses:
        text = _insert_before_heading(text, [r"(?:\*\*4\. Strategy fit\*\*|4\. Strategy fit)"], "\n".join(weaknesses))
    return text




def _inject_sensex_relative_strength(text: str, row: pd.Series) -> str:
    if not text:
        return text

    # Skip if Sensex context is already present.
    if re.search(r"Sensex", text, flags=re.IGNORECASE):
        return text

    short_pairs = []
    for label, value in [
        ("1-week", _get_numeric(row, "rr_sensex_week_pct", np.nan)),
        ("1-month", _get_numeric(row, "rr_sensex_month_pct", np.nan)),
        ("quarter", _get_numeric(row, "rr_sensex_quarter_pct", np.nan)),
        ("1-year", _get_numeric(row, "rr_sensex_year_pct", np.nan)),
    ]:
        if not _is_missing(value):
            short_pairs.append((label, float(value)))
    long_pairs = _period_label_pairs({
        "3-year": _get_numeric(row, "rr_sensex_3year_pct", np.nan),
        "5-year": _get_numeric(row, "rr_sensex_5year_pct", np.nan),
        "10-year": _get_numeric(row, "rr_sensex_10year_pct", np.nan),
    })

    strengths: List[str] = []
    confirmations: List[str] = []
    weaknesses: List[str] = []

    if long_pairs:
        pos_count = sum(1 for _, value in long_pairs if value > 0)
        neg_count = sum(1 for _, value in long_pairs if value < 0)
        parts = [f"{label.lower()} {_fmt_signed_pct(value)}" for label, value in long_pairs]
        if pos_count >= 2:
            strengths.append("* **Sensex Long-Horizon Relative Strength:** " +
                             "Outperformance versus Sensex is visible across longer cycles, with " +
                             ", ".join(parts[:-1]) + " and " + parts[-1] + ".")
        elif neg_count >= 2:
            weaknesses.append("* **Sensex Long-Horizon Relative Strength:** " +
                              "Relative performance versus Sensex is weak across longer cycles, with " +
                              ", ".join(parts[:-1]) + " and " + parts[-1] + ".")

    if short_pairs:
        # Patch RR-H.QA1: count only periods that genuinely exceed the
        # ±2% neutral band. Previously any positive value counted as
        # "positive", so 1-week +0.4% and 1-month +0.5% etc. could
        # produce a "supportive" headline despite weak readings.
        pos_count = sum(1 for _, value in short_pairs if value >  _RS_NEUTRAL_BAND_PCT)
        neg_count = sum(1 for _, value in short_pairs if value < -_RS_NEUTRAL_BAND_PCT)
        parts = [f"{label.lower()} {_fmt_signed_pct(value)}" for label, value in short_pairs]
        if pos_count >= 3:
            confirmations.append("* **Sensex Relative Strength:** " +
                                 "Broad-market performance versus Sensex is supportive across near-term horizons, with " +
                                 ", ".join(parts[:-1]) + " and " + parts[-1] + ".")
        elif neg_count >= 3:
            weaknesses.append("* **Sensex Relative Strength:** " +
                              "Broad-market performance versus Sensex is weak across near-term horizons, with " +
                              ", ".join(parts[:-1]) + " and " + parts[-1] + ".")
        elif len(parts) >= 3:
            confirmations.append("* **Sensex Relative Strength:** " +
                                 "Performance versus Sensex is mixed but informative, with " +
                                 ", ".join(parts[:-1]) + " and " + parts[-1] + ".")

    if strengths:
        text = _insert_before_heading(text, [r"(?:\*\*2\. What confirms the setup\*\*|2\. What confirms the setup)"], "\n".join(strengths))
    if confirmations:
        text = _insert_before_heading(text, [r"(?:\*\*3\. What weakens the setup\*\*|3\. What weakens the setup)"], "\n".join(confirmations))
    if weaknesses:
        text = _insert_before_heading(text, [r"(?:\*\*4\. Strategy fit\*\*|4\. Strategy fit)"], "\n".join(weaknesses))
    return text


def _inject_phase1_risk_ttm_context(text: str, row: pd.Series) -> str:
    if not text:
        return text
    beta_1y = _get_numeric(row, "beta_1year", np.nan)
    beta_3y = _get_numeric(row, "beta_3year", np.nan)
    ltdte_annual = _get_numeric(row, "long_term_debt_to_equity_annual", np.nan)
    eps_ttm_growth = _get_numeric(row, "eps_ttm_growth_pct", np.nan)
    operating_revenue_ttm = _get_numeric(row, "operating_revenue_ttm", np.nan)
    net_profit_ttm = _get_numeric(row, "net_profit_ttm", np.nan)
    is_financial_like = bool(_get_numeric(row, "is_financial_like", 0) == 1)

    strengths: List[str] = []
    weaknesses: List[str] = []
    confirmations: List[str] = []

    if not re.search(r"TTM operating revenue|TTM net profit|EPS TTM growth", text, flags=re.IGNORECASE):
        ttm_parts: List[str] = []
        if not pd.isna(operating_revenue_ttm):
            ttm_parts.append(f"TTM operating revenue at ₹{_fmt_num(operating_revenue_ttm)} Cr")
        if not pd.isna(net_profit_ttm):
            ttm_parts.append(f"TTM net profit at ₹{_fmt_num(net_profit_ttm)} Cr")
        if not pd.isna(eps_ttm_growth):
            ttm_parts.append(f"EPS TTM growth of {_fmt_pct(eps_ttm_growth)}")
        if ttm_parts:
            sentence = "Fresh trailing-twelve-month context remains supportive, with " + ", ".join(ttm_parts) + "."
            if not pd.isna(eps_ttm_growth) and float(eps_ttm_growth) < 0:
                weaknesses.append(sentence)
            else:
                strengths.append(sentence)

    if not re.search(r"\bbeta\b|1-year beta|3-year beta", text, flags=re.IGNORECASE):
        beta_parts: List[str] = []
        if not pd.isna(beta_1y):
            beta_parts.append(f"1-year beta of {_fmt_num(beta_1y)}")
        if not pd.isna(beta_3y):
            beta_parts.append(f"3-year beta of {_fmt_num(beta_3y)}")
        if beta_parts:
            max_beta = max([float(x) for x in [beta_1y, beta_3y] if not pd.isna(x)])
            min_beta = min([float(x) for x in [beta_1y, beta_3y] if not pd.isna(x)])
            if max_beta >= 1.25:
                weaknesses.append("Market sensitivity is elevated, with " + " and ".join(beta_parts) + ", which can amplify benchmark swings.")
            elif max_beta <= 1.05 and min_beta <= 0.95:
                strengths.append("The beta profile is relatively controlled, with " + " and ".join(beta_parts) + ", which supports a steadier risk profile.")
            else:
                confirmations.append("The beta profile is close to market-like risk, with " + " and ".join(beta_parts) + ".")

    if (not is_financial_like) and (not re.search(r"long-term debt-to-equity", text, flags=re.IGNORECASE)):
        if not pd.isna(ltdte_annual):
            ltdte_text = f"Annual long-term debt-to-equity stands at {_fmt_num(ltdte_annual)}."
            if float(ltdte_annual) <= 0.5:
                strengths.append(ltdte_text + " This supports balance-sheet resilience.")
            elif float(ltdte_annual) <= 1.0:
                confirmations.append(ltdte_text + " Leverage looks manageable on an annual basis.")
            else:
                weaknesses.append(ltdte_text + " This keeps structural leverage on the higher side.")

    if strengths:
        insertion = " " + " ".join(strengths)
        heading_match = re.search(r"(?:\*\*2\. What confirms the setup\*\*|\n2\. What confirms the setup)", text)
        if heading_match:
            text = text[:heading_match.start()].rstrip() + insertion + "\n\n" + text[heading_match.start():].lstrip()
        else:
            text = text.rstrip() + insertion

    if confirmations:
        insertion = " " + " ".join(confirmations)
        heading_match = re.search(r"(?:\*\*3\. What weakens the setup\*\*|\n3\. What weakens the setup)", text)
        if heading_match:
            text = text[:heading_match.start()].rstrip() + insertion + "\n\n" + text[heading_match.start():].lstrip()
        else:
            text = text.rstrip() + insertion

    if weaknesses:
        insertion = " " + " ".join(weaknesses)
        heading_match = re.search(r"(?:\*\*4\. Strategy fit\*\*|\n4\. Strategy fit)", text)
        if heading_match:
            text = text[:heading_match.start()].rstrip() + insertion + "\n\n" + text[heading_match.start():].lstrip()
        else:
            text = text.rstrip() + insertion

    return text


def _inject_annual_profitability_context(text: str, row: pd.Series) -> str:
    if not text:
        return text
    roe_annual = _get_numeric(row, "roe_annual_pct", np.nan)
    roa_annual = _get_numeric(row, "roa_annual_pct", np.nan)
    basic_eps = _get_numeric(row, "basic_eps_ttm", np.nan)

    if pd.isna(roe_annual) and pd.isna(roa_annual) and pd.isna(basic_eps):
        return text
    if re.search(r"\bannual ROE\b|\bannual ROA\b|\bbasic EPS\b", text, flags=re.IGNORECASE):
        return text

    is_financial_like = bool(_get_numeric(row, "is_financial_like", 0) == 1)
    strong = False
    weak = False
    if not pd.isna(roe_annual) and float(roe_annual) >= 15:
        strong = True
    if not pd.isna(roa_annual):
        if (is_financial_like and float(roa_annual) >= 1.0) or ((not is_financial_like) and float(roa_annual) >= 5.0):
            strong = True
    if not pd.isna(roe_annual) and float(roe_annual) < 10:
        weak = True
    if not pd.isna(roa_annual):
        if (is_financial_like and float(roa_annual) < 0.8) or ((not is_financial_like) and float(roa_annual) < 3.0):
            weak = True
    if not pd.isna(basic_eps) and float(basic_eps) <= 0:
        weak = True

    parts = []
    if not pd.isna(roe_annual):
        parts.append(f"annual ROE at ₹{_fmt_num(roe_annual)}%" if False else f"annual ROE at {_fmt_pct(roe_annual)}")
    if not pd.isna(roa_annual):
        parts.append(f"annual ROA at {_fmt_pct(roa_annual)}")
    if not pd.isna(basic_eps):
        parts.append(f"basic EPS (TTM) at ₹{_fmt_num(basic_eps)}")
    sentence = " Annual profitability context is supportive, with " + ", ".join(parts) + "." if strong and not weak else " Annual profitability context needs monitoring, with " + ", ".join(parts) + "."

    if strong and not weak:
        heading_match = re.search(r"(?:\*\*2\. What confirms the setup\*\*|\n2\. What confirms the setup)", text)
        if heading_match:
            return text[:heading_match.start()].rstrip() + sentence + "\n\n" + text[heading_match.start():].lstrip()
        return text.rstrip() + sentence

    weakness_match = re.search(r"(?:\*\*4\. Strategy fit\*\*|\n4\. Strategy fit)", text)
    if weakness_match:
        return text[:weakness_match.start()].rstrip() + sentence + "\n\n" + text[weakness_match.start():].lstrip()
    return text.rstrip() + sentence



def _inject_good_to_have_visibility(text: str, row: pd.Series) -> str:
    if not text:
        return text

    body_text = re.split(r"\n*Essential metrics table\n", text, maxsplit=1)[0]

    beta_1m = _get_numeric(row, "beta_1month", np.nan)
    beta_3m = _get_numeric(row, "beta_3month", np.nan)
    opm_qtr = _get_numeric(row, "operating_profit_margin_qtr_pct", np.nan)
    opm_qtr_4q = _get_numeric(row, "operating_profit_margin_qtr_4qtr_ago_pct", np.nan)
    qtr_change = _get_numeric(row, "qtr_change_pct", np.nan)
    revenue_qoq_growth = _get_numeric(row, "revenue_qoq_growth_pct", np.nan)
    net_profit_qoq_growth = _get_numeric(row, "net_profit_qoq_growth_pct", np.nan)
    roce_current = _get_numeric(row, "roce", np.nan)
    roce_3y_avg = _get_numeric(row, "roce_3y_avg", np.nan)
    roe_current = _get_numeric(row, "roe", np.nan)
    roe_3y_avg = _get_numeric(row, "roe_3y_avg", np.nan)
    opm_current = _get_numeric(row, "opm_current", np.nan)
    opm_5y_avg = _get_numeric(row, "opm_5y_avg", np.nan)

    pe_ttm = _get_numeric(row, "pe_ttm", np.nan)
    pe_3y = _get_numeric(row, "pe_3yr_average", np.nan)
    days_below_pe = _get_numeric(row, "days_traded_below_current_pe_pct", np.nan)
    price_to_sales = _get_numeric(row, "price_to_sales", np.nan)
    price_to_book_value_adjusted = _get_numeric(row, "price_to_book_value_adjusted", np.nan)
    days_below_pbv = _get_numeric(row, "days_traded_below_current_price_to_book_value_pct", np.nan)
    sector_price_to_book = _get_numeric(row, "sector_price_to_book_ttm", np.nan)
    industry_price_to_book = _get_numeric(row, "industry_price_to_book_ttm", np.nan)
    price_to_cfo = _get_numeric(row, "price_to_cfo", np.nan)
    price_to_fcf = _get_numeric(row, "price_to_fcf", np.nan)
    peg_ratio = _get_numeric(row, "peg_ratio", np.nan)
    rr_nifty50_week = _get_numeric(row, "rr_nifty50_week_pct", np.nan)
    rr_nifty50_quarter = _get_numeric(row, "rr_nifty50_quarter_pct", np.nan)
    rr_nifty50_year = _get_numeric(row, "rr_nifty50_year_pct", np.nan)
    rr_nifty50_10year = _get_numeric(row, "rr_nifty50_10year_pct", np.nan)
    rr_sector_year = _get_numeric(row, "rr_sector_year_pct", np.nan)
    rr_industry_year = _get_numeric(row, "rr_industry_year_pct", np.nan)
    rr_sector_week = _get_numeric(row, "rr_sector_week_pct", np.nan)
    rr_sector_month = _get_numeric(row, "rr_sector_month_pct", np.nan)
    rr_sector_quarter = _get_numeric(row, "rr_sector_quarter_pct", np.nan)
    rr_sector_3y = _get_numeric(row, "rr_sector_3year_pct", np.nan)
    rr_sector_5y = _get_numeric(row, "rr_sector_5year_pct", np.nan)
    rr_sector_10year = _get_numeric(row, "rr_sector_10year_pct", np.nan)
    rr_industry_week = _get_numeric(row, "rr_industry_week_pct", np.nan)
    rr_industry_month = _get_numeric(row, "rr_industry_month_pct", np.nan)
    rr_industry_quarter = _get_numeric(row, "rr_industry_quarter_pct", np.nan)
    rr_industry_3y = _get_numeric(row, "rr_industry_3year_pct", np.nan)
    rr_industry_5y = _get_numeric(row, "rr_industry_5year_pct", np.nan)
    rr_industry_10year = _get_numeric(row, "rr_industry_10year_pct", np.nan)
    promoter_qoq = _get_numeric(row, "promoter_holding_change_qoq_pct", np.nan)
    promoter_change_4q = _get_numeric(row, "promoter_holding_change_4qtr_pct", np.nan)
    promoter_change_8q = _get_numeric(row, "promoter_holding_change_8qtr_pct", np.nan)
    fii_change_4q = _get_numeric(row, "fii_holding_change_4qtr_pct", np.nan)
    fii_change_8q = _get_numeric(row, "fii_holding_change_8qtr_pct", np.nan)
    mf_change_1m = _get_numeric(row, "mf_holding_change_1month_pct", np.nan)
    mf_change_2m = _get_numeric(row, "mf_holding_change_2month_pct", np.nan)
    mf_change_3m = _get_numeric(row, "mf_holding_change_3month_pct", np.nan)
    mf_change_4q = _get_numeric(row, "mf_holding_change_4qtr_pct", np.nan)
    mf_change_8q = _get_numeric(row, "mf_holding_change_8qtr_pct", np.nan)
    institutional_current = _get_numeric(row, "institutional_holding_current_qtr_pct", np.nan)
    institutional_change_4q = _get_numeric(row, "institutional_holding_change_4qtr_pct", np.nan)
    institutional_change_8q = _get_numeric(row, "institutional_holding_change_8qtr_pct", np.nan)
    sector_roe = _get_numeric(row, "sector_return_on_equity_roe", np.nan)
    industry_roe = _get_numeric(row, "industry_return_on_equity_roe", np.nan)
    sector_roa = _get_numeric(row, "sector_return_on_assets", np.nan)
    industry_roa = _get_numeric(row, "industry_return_on_assets", np.nan)
    sector_revenue_yoy = _get_numeric(row, "sector_revenue_growth_qtr_yoy_pct", np.nan)
    sector_profit_yoy = _get_numeric(row, "sector_net_profit_growth_qtr_yoy_pct", np.nan)
    sector_revenue_qoq = _get_numeric(row, "sector_revenue_growth_qtr_qoq_pct", np.nan)
    sector_profit_qoq = _get_numeric(row, "sector_net_profit_growth_qtr_qoq_pct", np.nan)
    roe_annual = _get_numeric(row, "roe_annual_pct", np.nan)
    roa_annual = _get_numeric(row, "roa_annual_pct", np.nan)
    sales_yoy_ratio = _get_numeric(row, "sales_yoy", np.nan)
    profit_yoy_ratio = _get_numeric(row, "profit_yoy", np.nan)
    debtor_days = _get_numeric(row, "debtor_days", np.nan)
    days_receivable = _get_numeric(row, "days_receivable_outstanding", np.nan)
    days_inventory = _get_numeric(row, "days_inventory_outstanding", np.nan)
    avg_working_capital_days_3y = _get_numeric(row, "avg_working_capital_days_3y", np.nan)
    inventory_turnover_ratio = _get_numeric(row, "inventory_turnover_ratio", np.nan)
    inventory_turnover_ratio_3y_back = _get_numeric(row, "inventory_turnover_ratio_3y_back", np.nan)
    cash_conversion_cycle_days = _get_numeric(row, "cash_conversion_cycle_days", np.nan)
    sma100 = _get_numeric(row, "day_sma100", np.nan)
    pivot_r3 = _get_numeric(row, "standard_resistance_r3", np.nan)
    vwap_day = _get_numeric(row, "vwap_day", np.nan)
    current_price = _get_numeric(row, "current_price", np.nan)
    day_high = _get_numeric(row, "day_high", np.nan)
    day_low = _get_numeric(row, "day_low", np.nan)
    month_low = _get_numeric(row, "month_low", np.nan)
    day_macd = _get_numeric(row, "day_macd", np.nan)
    day_macd_signal_line = _get_numeric(row, "day_macd_signal_line", np.nan)
    r2_to_price_diff = _get_numeric(row, "standard_r2_to_price_diff_pct", np.nan)
    r3_to_price_diff = _get_numeric(row, "standard_r3_to_price_diff_pct", np.nan)
    s3_to_price_diff = _get_numeric(row, "standard_s3_to_price_diff_pct", np.nan)
    month_change_pct = _get_numeric(row, "month_change_pct", np.nan)
    qtr_high = _get_numeric(row, "qtr_high", np.nan)
    qtr_low = _get_numeric(row, "qtr_low", np.nan)
    cash_from_financing_annual = _get_numeric(row, "cash_from_financing_annual_activity", np.nan)
    cash_from_investing_annual = _get_numeric(row, "cash_from_investing_activity_annual", np.nan)
    avg_debtor_days_3y = _get_numeric(row, "avg_debtor_days_3y", np.nan)
    days_payable_outstanding = _get_numeric(row, "days_payable_outstanding", np.nan)
    inventory_turnover_ratio_5y_back = _get_numeric(row, "inventory_turnover_ratio_5y_back", np.nan)
    sector_revenue_annual_yoy = _get_numeric(row, "sector_revenue_growth_annual_yoy_pct", np.nan)
    company_revenue_annual_yoy = _get_numeric(row, "revenue_growth_annual_yoy_pct", np.nan)
    sector_peg_ttm = _get_numeric(row, "sector_peg_ttm", np.nan)
    industry_peg_ttm = _get_numeric(row, "industry_peg_ttm", np.nan)

    strengths: List[str] = []
    confirmations: List[str] = []
    weaknesses: List[str] = []

    if not re.search(r"1-month beta|3-month beta", body_text, flags=re.IGNORECASE):
        beta_parts = []
        beta_values = []
        if not pd.isna(beta_1m):
            beta_parts.append(f"1-month beta of {_fmt_num(beta_1m)}")
            beta_values.append(float(beta_1m))
        if not pd.isna(beta_3m):
            beta_parts.append(f"3-month beta of {_fmt_num(beta_3m)}")
            beta_values.append(float(beta_3m))
        if beta_parts:
            max_beta = max(beta_values)
            min_beta = min(beta_values)
            sentence = "Short-horizon beta context shows " + " and ".join(beta_parts) + "."
            if max_beta >= 1.30:
                weaknesses.append(sentence + " This means the stock can swing harder than the benchmark in the near term.")
            elif max_beta <= 0.95 and min_beta <= 0.90:
                strengths.append(sentence + " That supports a steadier short-term risk profile.")
            else:
                confirmations.append(sentence + " That is close to a market-like short-term risk profile.")

    if not re.search(r"quarter price change|over the quarter|past quarter|three-month move", body_text, flags=re.IGNORECASE):
        if not pd.isna(qtr_change):
            q = float(qtr_change)
            sentence = f"Quarter price change stands at {_fmt_signed_pct(q)}."
            if q >= 10:
                confirmations.append(sentence + " That supports improving medium-term price persistence.")
            elif q <= -10:
                weaknesses.append(sentence + " That shows the broader three-month recovery is still fragile.")
            else:
                confirmations.append(sentence + " That indicates a measured, not overextended, three-month move.")

    if not re.search(r"quarterly operating margin|four quarters ago", body_text, flags=re.IGNORECASE):
        if not pd.isna(opm_qtr) and not pd.isna(opm_qtr_4q):
            delta = float(opm_qtr) - float(opm_qtr_4q)
            sentence = f"Quarterly operating margin stands at {_fmt_pct(opm_qtr)} versus {_fmt_pct(opm_qtr_4q)} four quarters ago."
            if delta >= 0.5:
                strengths.append(sentence + " That supports a cleaner operating-quality improvement story.")
            elif delta <= -0.5:
                weaknesses.append(sentence + " That indicates some deterioration in quarterly operating efficiency.")
            else:
                confirmations.append(sentence + " That suggests broadly stable quarterly operating efficiency.")

    if not re.search(r"Quarterly Growth Acceleration|revenue QoQ growth|net profit QoQ growth", body_text, flags=re.IGNORECASE):
        qoq_parts: List[str] = []
        positive_count = 0
        negative_count = 0
        if not pd.isna(revenue_qoq_growth):
            qoq_parts.append(f"revenue QoQ growth at {_fmt_signed_pct(revenue_qoq_growth)}")
            if float(revenue_qoq_growth) > 0:
                positive_count += 1
            elif float(revenue_qoq_growth) < 0:
                negative_count += 1
        if not pd.isna(net_profit_qoq_growth):
            qoq_parts.append(f"net profit QoQ growth at {_fmt_signed_pct(net_profit_qoq_growth)}")
            if float(net_profit_qoq_growth) > 0:
                positive_count += 1
            elif float(net_profit_qoq_growth) < 0:
                negative_count += 1
        if qoq_parts:
            sentence = "**Quarterly Growth Acceleration:** Additional sequential-growth context shows " + ", ".join(qoq_parts) + "."
            if positive_count >= 2 and (pd.isna(net_profit_qoq_growth) or pd.isna(revenue_qoq_growth) or float(net_profit_qoq_growth) >= float(revenue_qoq_growth)):
                confirmations.append(sentence + " That supports a constructive short-cycle earnings acceleration profile.")
            elif negative_count >= 2:
                weaknesses.append(sentence + " That signals a softer short-cycle operating trajectory and lowers follow-through confidence.")
            else:
                confirmations.append(sentence + " That adds useful sequential context beyond the year-on-year growth snapshot.")

    if not re.search(r"Quality Consistency|ROCE.*3-year average|ROE.*3-year average|operating margin.*5-year average", body_text, flags=re.IGNORECASE):
        quality_parts: List[str] = []
        supportive_count = 0
        weak_count = 0
        if not pd.isna(roce_current) and not pd.isna(roce_3y_avg):
            quality_parts.append(f"ROCE at {_fmt_pct(roce_current)} versus a 3-year average of {_fmt_pct(roce_3y_avg)}")
            if float(roce_current) >= float(roce_3y_avg):
                supportive_count += 1
            elif float(roce_current) < float(roce_3y_avg) - 2:
                weak_count += 1
        if not pd.isna(roe_current) and not pd.isna(roe_3y_avg):
            quality_parts.append(f"ROE at {_fmt_pct(roe_current)} versus a 3-year average of {_fmt_pct(roe_3y_avg)}")
            if float(roe_current) >= float(roe_3y_avg):
                supportive_count += 1
            elif float(roe_current) < float(roe_3y_avg) - 2:
                weak_count += 1
        if not pd.isna(opm_current) and not pd.isna(opm_5y_avg):
            quality_parts.append(f"operating margin at {_fmt_pct(opm_current)} versus a 5-year average of {_fmt_pct(opm_5y_avg)}")
            if float(opm_current) >= float(opm_5y_avg):
                supportive_count += 1
            elif float(opm_current) < float(opm_5y_avg) - 1:
                weak_count += 1
        if quality_parts:
            sentence = "**Quality Consistency:** Additional quality context shows " + ", ".join(quality_parts) + "."
            if supportive_count >= 2 and weak_count == 0:
                strengths.append(sentence + " That supports the case that quality is holding above its own medium-cycle base.")
            elif weak_count >= 2 and supportive_count == 0:
                weaknesses.append(sentence + " That suggests quality is running below its own medium-cycle reference base.")
            else:
                confirmations.append(sentence + " That adds useful consistency context beyond the headline profitability snapshot.")

    if not re.search(r"3-year average PE|historical days traded below the current PE|price-to-sales|price to sales", body_text, flags=re.IGNORECASE):
        valuation_parts: List[str] = []
        supportive = False
        demanding = False
        if not pd.isna(pe_ttm) and not pd.isna(pe_3y) and pe_3y > 0:
            discount = (float(pe_ttm) / float(pe_3y)) - 1
            valuation_parts.append(f"PE at {_fmt_num(pe_ttm)}x versus a 3-year average of {_fmt_num(pe_3y)}x")
            if discount <= -0.15:
                supportive = True
            elif discount >= 0.20:
                demanding = True
        if not pd.isna(days_below_pe):
            valuation_parts.append(f"only {_fmt_pct(days_below_pe)} of historical days traded below the current PE")
            if float(days_below_pe) <= 20:
                supportive = True
            elif float(days_below_pe) >= 70:
                demanding = True
        if not pd.isna(price_to_sales):
            valuation_parts.append(f"price-to-sales at {_fmt_num(price_to_sales)}x")
            if float(price_to_sales) <= 3:
                supportive = True
            elif float(price_to_sales) >= 8:
                demanding = True
        if valuation_parts:
            sentence = "**Valuation Depth:** Additional valuation context shows " + ", ".join(valuation_parts) + "."
            if supportive and not demanding:
                strengths.append(sentence + " That supports the view that valuation is still workable on a deeper history-aware lens.")
            elif demanding and not supportive:
                weaknesses.append(sentence + " That suggests valuation comfort is limited once broader history-aware context is included.")
            else:
                confirmations.append(sentence + " That adds useful depth beyond the basic PE snapshot.")

    if not re.search(r"Book-Value Valuation|historical days traded below the current price-to-book", body_text, flags=re.IGNORECASE):
        pbv_parts: List[str] = []
        supportive = False
        demanding = False
        if not pd.isna(price_to_book_value_adjusted):
            pbv_phrase = f"price-to-book at {_fmt_num(price_to_book_value_adjusted)}x"
            comparators: List[str] = []
            if not pd.isna(sector_price_to_book):
                comparators.append(f"sector {_fmt_num(sector_price_to_book)}x")
            if not pd.isna(industry_price_to_book):
                comparators.append(f"industry {_fmt_num(industry_price_to_book)}x")
            if comparators:
                if len(comparators) == 1:
                    pbv_phrase += f" versus {comparators[0]}"
                else:
                    pbv_phrase += f" versus {comparators[0]} and {comparators[1]}"
            pbv_parts.append(pbv_phrase)
            compare_values = [v for v in [sector_price_to_book, industry_price_to_book] if not pd.isna(v)]
            if compare_values:
                if float(price_to_book_value_adjusted) <= min(float(v) for v in compare_values):
                    supportive = True
                elif float(price_to_book_value_adjusted) >= max(float(v) for v in compare_values):
                    demanding = True
        if not pd.isna(days_below_pbv):
            pbv_parts.append(f"{_fmt_pct(days_below_pbv)} of historical days traded below the current price-to-book")
            if float(days_below_pbv) <= 20:
                supportive = True
            elif float(days_below_pbv) >= 70:
                demanding = True
        if pbv_parts:
            sentence = "**Book-Value Valuation:** Additional book-value context shows " + ", ".join(pbv_parts) + "."
            if supportive and not demanding:
                strengths.append(sentence + " That supports valuation comfort from an asset-value lens as well.")
            elif demanding and not supportive:
                weaknesses.append(sentence + " That suggests valuation is less comfortable once book-value context is included.")
            else:
                confirmations.append(sentence + " That adds a useful balance-sheet valuation lens alongside the PE and sales multiples.")

    if not re.search(r"price-to-CFO|price to CFO|price-to-FCF|price to FCF", body_text, flags=re.IGNORECASE):
        cashflow_val_parts: List[str] = []
        supportive = False
        demanding = False
        if not pd.isna(price_to_cfo):
            cashflow_val_parts.append(f"price-to-CFO at {_fmt_num(price_to_cfo)}x")
            if float(price_to_cfo) <= 15:
                supportive = True
            elif float(price_to_cfo) >= 30:
                demanding = True
        if not pd.isna(price_to_fcf):
            cashflow_val_parts.append(f"price-to-FCF at {_fmt_num(price_to_fcf)}x")
            if float(price_to_fcf) <= 20:
                supportive = True
            elif float(price_to_fcf) >= 40:
                demanding = True
        if cashflow_val_parts:
            sentence = "**Cash-Flow Valuation:** Additional cash-flow valuation context shows " + ", ".join(cashflow_val_parts) + "."
            if supportive and not demanding:
                strengths.append(sentence + " That supports valuation comfort from a cash-generation lens, not just an earnings lens.")
            elif demanding and not supportive:
                weaknesses.append(sentence + " That suggests valuation still looks demanding once cash-generation multiples are included.")
            else:
                confirmations.append(sentence + " That adds a useful cash-generation lens to the broader valuation picture.")

    if not re.search(r"relative strength versus sector|relative strength versus industry", body_text, flags=re.IGNORECASE):
        rs_parts: List[str] = []
        if not pd.isna(rr_sector_year):
            rs_parts.append(f"1-year relative strength versus sector at {_fmt_signed_pct(rr_sector_year)}")
        if not pd.isna(rr_industry_year):
            rs_parts.append(f"1-year relative strength versus industry at {_fmt_signed_pct(rr_industry_year)}")
        if rs_parts:
            sentence = "**Peer-Relative Context:** " + ", ".join(rs_parts) + "."
            positive_count = sum(1 for value in [rr_sector_year, rr_industry_year] if not pd.isna(value) and float(value) > 0)
            negative_count = sum(1 for value in [rr_sector_year, rr_industry_year] if not pd.isna(value) and float(value) < 0)
            if positive_count == len([v for v in [rr_sector_year, rr_industry_year] if not pd.isna(v)]) and positive_count > 0:
                confirmations.append(sentence + " That confirms the stock is also beating its closer peer set, not just the headline benchmark.")
            elif negative_count >= 1 and positive_count == 0:
                weaknesses.append(sentence + " That shows the stock is still lagging its closer peer group over the last year.")
            else:
                confirmations.append(sentence + " That gives a more complete peer-relative read on the move.")

    if not re.search(r"3-year relative strength versus sector|3-year relative strength versus industry", body_text, flags=re.IGNORECASE):
        rs3_parts: List[str] = []
        if not pd.isna(rr_sector_3y):
            rs3_parts.append(f"3-year relative strength versus sector at {_fmt_signed_pct(rr_sector_3y)}")
        if not pd.isna(rr_industry_3y):
            rs3_parts.append(f"3-year relative strength versus industry at {_fmt_signed_pct(rr_industry_3y)}")
        if rs3_parts:
            sentence = "**Long-Cycle Peer Leadership:** " + ", ".join(rs3_parts) + "."
            pos3 = sum(1 for value in [rr_sector_3y, rr_industry_3y] if not pd.isna(value) and float(value) > 0)
            neg3 = sum(1 for value in [rr_sector_3y, rr_industry_3y] if not pd.isna(value) and float(value) < 0)
            available3 = len([v for v in [rr_sector_3y, rr_industry_3y] if not pd.isna(v)])
            if available3 and pos3 == available3:
                strengths.append(sentence + " That reinforces the stock's longer-cycle leadership versus its closer peer set.")
            elif available3 and neg3 == available3:
                weaknesses.append(sentence + " That shows the stock still lacks longer-cycle leadership versus its sector and industry peer group.")
            else:
                confirmations.append(sentence + " That adds useful long-cycle peer-relative depth to the setup.")

    if not re.search(r"1-week relative strength versus sector|1-month relative strength versus sector|quarterly relative strength versus sector|1-week relative strength versus industry|1-month relative strength versus industry|quarterly relative strength versus industry|Peer Momentum Breadth", body_text, flags=re.IGNORECASE):
        sector_parts: List[str] = []
        industry_parts: List[str] = []
        peer_values: List[float] = []
        for label, value in [("1-week", rr_sector_week), ("1-month", rr_sector_month), ("quarter", rr_sector_quarter)]:
            if not pd.isna(value):
                sector_parts.append(f"{label} {_fmt_signed_pct(value)}")
                peer_values.append(float(value))
        for label, value in [("1-week", rr_industry_week), ("1-month", rr_industry_month), ("quarter", rr_industry_quarter)]:
            if not pd.isna(value):
                industry_parts.append(f"{label} {_fmt_signed_pct(value)}")
                peer_values.append(float(value))
        if sector_parts or industry_parts:
            sentence_parts: List[str] = []
            if sector_parts:
                sentence_parts.append("versus sector: " + ", ".join(sector_parts))
            if industry_parts:
                sentence_parts.append("versus industry: " + ", ".join(industry_parts))
            sentence = "**Peer Momentum Breadth:** " + "; ".join(sentence_parts) + "."
            pos_count = sum(1 for value in peer_values if value > 0)
            neg_count = sum(1 for value in peer_values if value < 0)
            available = len(peer_values)
            if available and pos_count >= max(4, available - 1):
                confirmations.append(sentence + " That shows the move is broadening beyond the headline benchmark into sector and industry peer momentum.")
            elif available and neg_count >= max(4, available - 1):
                weaknesses.append(sentence + " That shows the stock is still lagging its closer peer set across short and medium horizons.")
            else:
                confirmations.append(sentence + " That adds useful short-to-medium horizon peer breadth context.")

    if not re.search(r"5-year relative strength versus sector|5-year relative strength versus industry|Extended Peer Leadership", body_text, flags=re.IGNORECASE):
        rs5_parts: List[str] = []
        if not pd.isna(rr_sector_5y):
            rs5_parts.append(f"5-year relative strength versus sector at {_fmt_signed_pct(rr_sector_5y)}")
        if not pd.isna(rr_industry_5y):
            rs5_parts.append(f"5-year relative strength versus industry at {_fmt_signed_pct(rr_industry_5y)}")
        if rs5_parts:
            sentence = "**Extended Peer Leadership:** " + ", ".join(rs5_parts) + "."
            pos5 = sum(1 for value in [rr_sector_5y, rr_industry_5y] if not pd.isna(value) and float(value) > 0)
            neg5 = sum(1 for value in [rr_sector_5y, rr_industry_5y] if not pd.isna(value) and float(value) < 0)
            available5 = len([v for v in [rr_sector_5y, rr_industry_5y] if not pd.isna(v)])
            if available5 and pos5 == available5:
                strengths.append(sentence + " That reinforces durable peer leadership beyond the shorter-cycle evidence.")
            elif available5 and neg5 == available5:
                weaknesses.append(sentence + " That shows the stock still lacks extended-cycle peer leadership on a 5-year lens.")
            else:
                confirmations.append(sentence + " That adds useful extended peer-relative depth to the setup.")

    if not re.search(r"4-quarter FII holding change|4-quarter MF holding change|4-quarter institutional holding change|institutional over the last four quarters", body_text, flags=re.IGNORECASE):
        breadth_parts: List[str] = []
        if not pd.isna(fii_change_4q):
            breadth_parts.append(f"4-quarter FII holding change at {_fmt_signed_pct(fii_change_4q)}")
        if not pd.isna(mf_change_4q):
            breadth_parts.append(f"4-quarter MF holding change at {_fmt_signed_pct(mf_change_4q)}")
        if not pd.isna(institutional_change_4q):
            breadth_parts.append(f"4-quarter institutional holding change at {_fmt_signed_pct(institutional_change_4q)}")
        if breadth_parts:
            sentence = "**Ownership Breadth:** " + ", ".join(breadth_parts) + "."
            supportive = (not pd.isna(institutional_change_4q) and float(institutional_change_4q) >= 0.50) or (
                (not pd.isna(mf_change_4q) and float(mf_change_4q) > 0) and (pd.isna(fii_change_4q) or float(fii_change_4q) >= -0.25)
            )
            weak = (not pd.isna(institutional_change_4q) and float(institutional_change_4q) <= -0.50) or (
                (not pd.isna(fii_change_4q) and float(fii_change_4q) < 0) and (not pd.isna(mf_change_4q) and float(mf_change_4q) <= 0)
            )
            if supportive and not weak:
                strengths.append(sentence + " That supports broader sponsorship depth beyond just promoter stability.")
            elif weak and not supportive:
                weaknesses.append(sentence + " That indicates weaker ownership breadth and lowers conviction in sponsorship support.")
            else:
                confirmations.append(sentence + " That presents a mixed but useful read on broader sponsorship depth.")

    if not re.search(r"4-quarter promoter holding change|promoter over the last four quarters", body_text, flags=re.IGNORECASE):
        if not pd.isna(promoter_change_4q):
            sentence = f"**Ownership Durability:** Promoter holding change over the last four quarters stands at {_fmt_signed_pct(promoter_change_4q)}."
            if float(promoter_change_4q) >= 0.25:
                strengths.append(sentence + " That supports ownership stability and longer-cycle promoter conviction.")
            elif float(promoter_change_4q) <= -0.25:
                weaknesses.append(sentence + " That indicates some promoter dilution or reduced promoter participation over the last year.")
            else:
                confirmations.append(sentence + " That is broadly stable and useful as longer-cycle sponsorship context.")

    if not re.search(r"Long-Cycle Sponsorship|8-quarter promoter holding change|8-quarter FII holding change|8-quarter MF holding change|8-quarter institutional holding change", body_text, flags=re.IGNORECASE):
        sponsorship_parts: List[str] = []
        positive_count = 0
        negative_count = 0
        available_count = 0
        for label, value in [
            ("8-quarter promoter holding change", promoter_change_8q),
            ("8-quarter FII holding change", fii_change_8q),
            ("8-quarter MF holding change", mf_change_8q),
            ("8-quarter institutional holding change", institutional_change_8q),
        ]:
            if not pd.isna(value):
                sponsorship_parts.append(f"{label} at {_fmt_signed_pct(value)}")
                available_count += 1
                if float(value) > 0:
                    positive_count += 1
                elif float(value) < 0:
                    negative_count += 1
        if sponsorship_parts:
            sentence = "**Long-Cycle Sponsorship:** " + ", ".join(sponsorship_parts) + "."
            if available_count and positive_count >= max(2, available_count - 1):
                strengths.append(sentence + " That supports sponsorship durability across a multi-year ownership cycle.")
            elif available_count and negative_count >= max(2, available_count - 1):
                weaknesses.append(sentence + " That suggests longer-cycle sponsorship has softened across the ownership base.")
            else:
                confirmations.append(sentence + " That adds useful multi-year sponsorship context beyond the 4-quarter ownership snapshot.")

    if not re.search(r"Institutional Base|current institutional holding stands at|institutional base of professional ownership", body_text, flags=re.IGNORECASE):
        if not pd.isna(institutional_current):
            sentence = f"**Institutional Base:** Current institutional holding stands at {_fmt_pct(institutional_current)}."
            if float(institutional_current) >= 35:
                strengths.append(sentence + " That provides a meaningful base of professional ownership.")
            elif float(institutional_current) <= 5:
                weaknesses.append(sentence + " That indicates the stock still lacks a meaningful professional ownership base.")
            else:
                confirmations.append(sentence + " That adds useful context on the depth of the current sponsorship base.")

    if not re.search(r"Fresh Fund Flow Pulse|1-month MF holding change|2-month MF holding change|3-month MF holding change", body_text, flags=re.IGNORECASE):
        pulse_parts: List[str] = []
        pulse_values: List[float] = []
        for label, value in [("1-month", mf_change_1m), ("2-month", mf_change_2m), ("3-month", mf_change_3m)]:
            if not pd.isna(value):
                pulse_parts.append(f"{label} MF holding change at {_fmt_signed_pct(value)}")
                pulse_values.append(float(value))
        if pulse_parts:
            sentence = "**Fresh Fund Flow Pulse:** " + ", ".join(pulse_parts) + "."
            pos_count = sum(1 for value in pulse_values if value > 0)
            neg_count = sum(1 for value in pulse_values if value < 0)
            if pulse_values and pos_count >= max(2, len(pulse_values) - 1):
                confirmations.append(sentence + " That suggests recent fund-flow sponsorship has been improving in the monthly window.")
            elif pulse_values and neg_count >= max(2, len(pulse_values) - 1):
                weaknesses.append(sentence + " That suggests recent fund-flow sponsorship has cooled in the monthly window.")
            else:
                confirmations.append(sentence + " That adds useful short-horizon sponsorship pulse context.")


    if not re.search(r"Peer Profitability Benchmark|sector ROE|industry ROE|sector ROA|industry ROA", body_text, flags=re.IGNORECASE):
        benchmark_parts: List[str] = []
        if not pd.isna(sector_roe):
            benchmark_parts.append(f"sector ROE at {_fmt_pct(sector_roe)}")
        if not pd.isna(industry_roe):
            benchmark_parts.append(f"industry ROE at {_fmt_pct(industry_roe)}")
        if not pd.isna(sector_roa):
            benchmark_parts.append(f"sector ROA at {_fmt_pct(sector_roa)}")
        if not pd.isna(industry_roa):
            benchmark_parts.append(f"industry ROA at {_fmt_pct(industry_roa)}")
        if benchmark_parts:
            sentence = "**Peer Profitability Benchmark:** " + ", ".join(benchmark_parts) + "."
            strength_points = 0
            weakness_points = 0
            comparisons = 0
            for company_value, peer_values in [
                (roe_annual, [sector_roe, industry_roe]),
                (roa_annual, [sector_roa, industry_roa]),
            ]:
                if pd.isna(company_value):
                    continue
                valid_peers = [float(v) for v in peer_values if not pd.isna(v)]
                if not valid_peers:
                    continue
                comparisons += 1
                if float(company_value) >= max(valid_peers):
                    strength_points += 1
                elif float(company_value) <= min(valid_peers):
                    weakness_points += 1
            if comparisons and strength_points >= max(1, comparisons):
                strengths.append(sentence + " That reinforces that the business is outperforming its closer peer profitability benchmarks.")
            elif comparisons and weakness_points >= max(1, comparisons):
                weaknesses.append(sentence + " That suggests profitability still sits below its closer peer profitability benchmarks.")
            else:
                confirmations.append(sentence + " That adds useful peer-benchmark context around business quality and return strength.")

    if not re.search(r"Sector Growth Context|sector revenue growth at|sector profit growth at", body_text, flags=re.IGNORECASE):
        growth_parts: List[str] = []
        if not pd.isna(sector_revenue_yoy):
            growth_parts.append(f"sector revenue growth at {_fmt_pct(sector_revenue_yoy)} YoY")
        if not pd.isna(sector_profit_yoy):
            growth_parts.append(f"sector profit growth at {_fmt_pct(sector_profit_yoy)} YoY")
        if not pd.isna(sector_revenue_qoq):
            growth_parts.append(f"sector revenue growth at {_fmt_pct(sector_revenue_qoq)} QoQ")
        if not pd.isna(sector_profit_qoq):
            growth_parts.append(f"sector profit growth at {_fmt_pct(sector_profit_qoq)} QoQ")
        if growth_parts:
            sentence = "**Sector Growth Context:** " + ", ".join(growth_parts) + "."
            company_pairs = []
            if not pd.isna(sales_yoy_ratio) and not pd.isna(sector_revenue_yoy):
                company_pairs.append((float(sales_yoy_ratio) * 100.0, float(sector_revenue_yoy)))
            if not pd.isna(profit_yoy_ratio) and not pd.isna(sector_profit_yoy):
                company_pairs.append((float(profit_yoy_ratio) * 100.0, float(sector_profit_yoy)))
            if not pd.isna(revenue_qoq_growth) and not pd.isna(sector_revenue_qoq):
                company_pairs.append((float(revenue_qoq_growth), float(sector_revenue_qoq)))
            if not pd.isna(net_profit_qoq_growth) and not pd.isna(sector_profit_qoq):
                company_pairs.append((float(net_profit_qoq_growth), float(sector_profit_qoq)))
            outperform = sum(1 for company_value, sector_value in company_pairs if company_value >= sector_value + 3.0)
            underperform = sum(1 for company_value, sector_value in company_pairs if company_value <= sector_value - 3.0)
            if company_pairs and outperform >= max(2, len(company_pairs) // 2 + 1):
                confirmations.append(sentence + " That suggests the company is still holding up well against its sector growth backdrop.")
            elif company_pairs and underperform >= max(2, len(company_pairs) // 2 + 1):
                weaknesses.append(sentence + " That suggests the company is lagging its sector growth backdrop on several near-term measures.")
            else:
                confirmations.append(sentence + " That adds useful industry-cycle context around the company's current growth profile.")

    if not re.search(r"debtor days", body_text, flags=re.IGNORECASE):
        if not pd.isna(debtor_days):
            sentence = f"**Working-Capital Quality:** Debtor days stand at {_fmt_num(debtor_days, 1)}."
            if float(debtor_days) <= 30:
                strengths.append(sentence + " That supports cleaner receivables discipline and lowers working-capital drag risk.")
            elif float(debtor_days) >= 90:
                weaknesses.append(sentence + " That keeps receivables conversion on the weaker side and can slow cash realization.")
            else:
                confirmations.append(sentence + " That looks manageable, though not exceptionally tight.")

    if not re.search(r"days receivable outstanding|days inventory outstanding|3-year average working-capital days|inventory turnover.*3-year", body_text, flags=re.IGNORECASE):
        wc_depth_parts: List[str] = []
        supportive = False
        weak = False
        if not pd.isna(days_receivable):
            wc_depth_parts.append(f"days receivable outstanding at {_fmt_num(days_receivable, 1)}")
            if float(days_receivable) <= 45:
                supportive = True
            elif float(days_receivable) >= 90:
                weak = True
        if not pd.isna(days_inventory):
            wc_depth_parts.append(f"days inventory outstanding at {_fmt_num(days_inventory, 1)}")
            if float(days_inventory) <= 90:
                supportive = True
            elif float(days_inventory) >= 120:
                weak = True
        if not pd.isna(avg_working_capital_days_3y):
            wc_depth_parts.append(f"3-year average working-capital days at {_fmt_num(avg_working_capital_days_3y, 1)}")
            if not pd.isna(cash_conversion_cycle_days):
                if float(cash_conversion_cycle_days) <= float(avg_working_capital_days_3y):
                    supportive = True
                elif float(cash_conversion_cycle_days) >= float(avg_working_capital_days_3y) * 1.20:
                    weak = True
        if not pd.isna(inventory_turnover_ratio_3y_back):
            if not pd.isna(inventory_turnover_ratio):
                wc_depth_parts.append(
                    f"inventory turnover at {_fmt_num(inventory_turnover_ratio)}x versus a 3-year reference of {_fmt_num(inventory_turnover_ratio_3y_back)}x"
                )
                if float(inventory_turnover_ratio) >= float(inventory_turnover_ratio_3y_back) * 0.95:
                    supportive = True
                elif float(inventory_turnover_ratio) < float(inventory_turnover_ratio_3y_back) * 0.85:
                    weak = True
            else:
                wc_depth_parts.append(f"3-year inventory-turnover reference at {_fmt_num(inventory_turnover_ratio_3y_back)}x")
        if wc_depth_parts:
            sentence = "**Working-Capital Depth:** Additional operating-cycle context shows " + ", ".join(wc_depth_parts) + "."
            if supportive and not weak:
                strengths.append(sentence + " That reinforces operating-cycle discipline beyond the simpler debtor-days snapshot.")
            elif weak and not supportive:
                weaknesses.append(sentence + " That points to some operating-cycle softness once deeper working-capital context is included.")
            else:
                confirmations.append(sentence + " That adds useful operating-cycle depth beyond the headline cash-conversion view.")

    if not re.search(r"100-day SMA|day SMA100|SMA100", body_text, flags=re.IGNORECASE):
        if not pd.isna(sma100) and not pd.isna(current_price) and sma100 > 0 and current_price > 0:
            diff = (float(current_price) / float(sma100)) - 1
            sentence = f"**100-Day Structure:** The price is {_fmt_frac_pct(abs(diff))} {'above' if diff >= 0 else 'below'} the 100-day SMA of ₹{_fmt_num(sma100)}."
            if diff >= 0.03:
                confirmations.append(sentence + " That improves the intermediate trend-repair profile.")
            elif diff <= -0.05:
                weaknesses.append(sentence + " That shows the intermediate trend repair is not yet complete.")
            else:
                confirmations.append(sentence + " That places the stock near an important medium-term structure checkpoint.")

    if not re.search(r"VWAP", body_text, flags=re.IGNORECASE):
        if not pd.isna(vwap_day) and not pd.isna(current_price) and vwap_day > 0 and current_price > 0:
            diff = (float(current_price) / float(vwap_day)) - 1
            sentence = f"**VWAP Context:** The price is {_fmt_frac_pct(abs(diff))} {'above' if diff >= 0 else 'below'} the daily VWAP of ₹{_fmt_num(vwap_day)}."
            if diff >= 0.005:
                confirmations.append(sentence + " That supports a constructive intraday demand profile and cleaner price acceptance.")
            elif diff <= -0.005:
                weaknesses.append(sentence + " That suggests the stock closed below its average traded value zone and needs stronger follow-through.")
            else:
                confirmations.append(sentence + " That keeps the stock close to its average traded value zone for the day.")


    if not re.search(r"MACD Structure|MACD stands at|signal line", body_text, flags=re.IGNORECASE):
        if not pd.isna(day_macd) and not pd.isna(day_macd_signal_line):
            macd_spread = float(day_macd) - float(day_macd_signal_line)
            sentence = f"**MACD Structure:** MACD stands at ₹{_fmt_num(day_macd)} versus a signal line at ₹{_fmt_num(day_macd_signal_line)}."
            if macd_spread > 0:
                confirmations.append(sentence + " That supports positive technical momentum structure and reinforces the current trend-improvement phase.")
            elif macd_spread < 0:
                weaknesses.append(sentence + " That signals weaker trend momentum and reduces conviction in immediate follow-through.")
            else:
                confirmations.append(sentence + " That keeps the technical momentum balance neutral and worth monitoring for a cleaner crossover.")

    if not re.search(r"Extended Trade Map|distance to R2|distance to R3|distance to S3", body_text, flags=re.IGNORECASE):
        trade_parts: List[str] = []
        if not pd.isna(r2_to_price_diff):
            trade_parts.append(f"R2 is {_fmt_pct(abs(r2_to_price_diff))} above current price")
        if not pd.isna(r3_to_price_diff):
            trade_parts.append(f"R3 is {_fmt_pct(abs(r3_to_price_diff))} above current price")
        if not pd.isna(s3_to_price_diff):
            trade_parts.append(f"S3 is {_fmt_pct(abs(s3_to_price_diff))} below current price")
        if trade_parts:
            sentence = "**Extended Trade Map:** " + ", ".join(trade_parts) + "."
            confirmations.append(sentence + " That sharpens the secondary reward-to-risk map beyond the first trigger and invalidation levels.")

    if not re.search(r"Quarter Range Backdrop|quarter range spanning|1-month price change", body_text, flags=re.IGNORECASE):
        range_parts: List[str] = []
        supportive = False
        weak = False
        if not pd.isna(month_change_pct):
            range_parts.append(f"1-month price change at {_fmt_signed_pct(month_change_pct)}")
            if float(month_change_pct) >= 8:
                supportive = True
            elif float(month_change_pct) <= -8:
                weak = True
        if not pd.isna(qtr_high) and not pd.isna(qtr_low):
            range_parts.append(f"the quarter range spanning ₹{_fmt_num(qtr_low)} to ₹{_fmt_num(qtr_high)}")
            if not pd.isna(current_price) and float(qtr_high) > float(qtr_low):
                qpos = (float(current_price) - float(qtr_low)) / (float(qtr_high) - float(qtr_low))
                if qpos >= 0.70:
                    supportive = True
                elif qpos <= 0.30:
                    weak = True
        if range_parts:
            sentence = "**Quarter Range Backdrop:** Additional range context shows " + ", ".join(range_parts) + "."
            if supportive and not weak:
                confirmations.append(sentence + " That suggests the stock is holding the upper half of its recent range with supportive short-cycle persistence.")
            elif weak and not supportive:
                weaknesses.append(sentence + " That suggests the stock is still operating from a weaker part of its recent range structure.")
            else:
                confirmations.append(sentence + " That adds useful range-location context around the current setup.")


    if not re.search(r"Cash-Flow Route|cash from investing activity|cash from financing activity", body_text, flags=re.IGNORECASE):
        cf_parts: List[str] = []
        if not pd.isna(cash_from_investing_annual):
            cf_parts.append(f"annual cash from investing activity at ₹{_fmt_num(cash_from_investing_annual)} Cr")
        if not pd.isna(cash_from_financing_annual):
            cf_parts.append(f"annual cash from financing activity at ₹{_fmt_num(cash_from_financing_annual)} Cr")
        if cf_parts:
            sentence = "**Cash-Flow Route:** Additional cash-flow route context shows " + ", ".join(cf_parts) + "."
            if (not pd.isna(cash_from_investing_annual)) and (not pd.isna(cash_from_financing_annual)):
                investing = float(cash_from_investing_annual)
                financing = float(cash_from_financing_annual)
                if investing < 0 and financing <= 0:
                    strengths.append(sentence + " That suggests investment deployment without heavy dependence on fresh external financing.")
                elif investing < 0 and financing > 0:
                    confirmations.append(sentence + " That suggests the current investment cycle is being partly supported by financing inflows.")
                elif investing > 0 and financing < 0:
                    confirmations.append(sentence + " That suggests the business is harvesting investments while also reducing financing dependence.")
                else:
                    confirmations.append(sentence + " That adds useful capital-allocation and funding context around the current setup.")
            else:
                confirmations.append(sentence + " That adds useful capital-allocation and funding context around the current setup.")

    if not re.search(r"Working-Capital Baseline|3-year average debtor days|days payable outstanding|5-year reference", body_text, flags=re.IGNORECASE):
        baseline_parts: List[str] = []
        supportive = False
        weak = False
        if not pd.isna(avg_debtor_days_3y):
            baseline_parts.append(f"3-year average debtor days at {_fmt_num(avg_debtor_days_3y)}")
        if not pd.isna(days_payable_outstanding):
            baseline_parts.append(f"days payable outstanding at {_fmt_num(days_payable_outstanding)}")
        if not pd.isna(inventory_turnover_ratio_5y_back):
            if not pd.isna(inventory_turnover_ratio):
                baseline_parts.append(
                    f"inventory turnover at {_fmt_num(inventory_turnover_ratio)}x versus a 5-year reference of {_fmt_num(inventory_turnover_ratio_5y_back)}x"
                )
                if float(inventory_turnover_ratio) > float(inventory_turnover_ratio_5y_back):
                    supportive = True
                elif float(inventory_turnover_ratio) < float(inventory_turnover_ratio_5y_back) * 0.85:
                    weak = True
            else:
                baseline_parts.append(f"5-year inventory turnover reference at {_fmt_num(inventory_turnover_ratio_5y_back)}x")
        if not pd.isna(days_payable_outstanding) and not pd.isna(avg_debtor_days_3y):
            if float(days_payable_outstanding) > float(avg_debtor_days_3y):
                supportive = True
        if baseline_parts:
            sentence = "**Working-Capital Baseline:** Additional baseline context shows " + ", ".join(baseline_parts) + "."
            if supportive and not weak:
                confirmations.append(sentence + " That supports a workable operating-cycle base versus its own longer history.")
            elif weak and not supportive:
                weaknesses.append(sentence + " That suggests some operating-cycle efficiency has softened versus its longer baseline.")
            else:
                confirmations.append(sentence + " That adds useful baseline context around the company's operating-cycle profile.")

    if not re.search(r"PEG Peer Lens|sector PEG|industry PEG", body_text, flags=re.IGNORECASE):
        if not pd.isna(peg_ratio) and (not pd.isna(sector_peg_ttm) or not pd.isna(industry_peg_ttm)):
            peg_parts: List[str] = [f"PEG at {_fmt_num(peg_ratio)}"]
            if not pd.isna(sector_peg_ttm):
                peg_parts.append(f"sector PEG at {_fmt_num(sector_peg_ttm)}")
            if not pd.isna(industry_peg_ttm):
                peg_parts.append(f"industry PEG at {_fmt_num(industry_peg_ttm)}")
            sentence = "**PEG Peer Lens:** Additional growth-adjusted valuation context shows " + ", ".join(peg_parts) + "."
            if (not pd.isna(sector_peg_ttm)) and (not pd.isna(industry_peg_ttm)):
                peg_v = float(peg_ratio)
                if peg_v < float(sector_peg_ttm) and peg_v < float(industry_peg_ttm):
                    strengths.append(sentence + " That keeps the growth-adjusted valuation lens supportive versus peers.")
                elif peg_v > float(sector_peg_ttm) and peg_v > float(industry_peg_ttm):
                    weaknesses.append(sentence + " That makes the growth-adjusted valuation lens less comfortable versus peers.")
                else:
                    confirmations.append(sentence + " That keeps the growth-adjusted valuation picture broadly balanced versus peers.")
            else:
                confirmations.append(sentence + " That adds useful peer-relative context to the valuation picture.")

    if not re.search(r"Sector Annual Growth Context|sector annual revenue growth", body_text, flags=re.IGNORECASE):
        annual_parts: List[str] = []
        if not pd.isna(company_revenue_annual_yoy):
            annual_parts.append(f"company annual revenue growth at {_fmt_pct(company_revenue_annual_yoy)}")
        if not pd.isna(sector_revenue_annual_yoy):
            annual_parts.append(f"sector annual revenue growth at {_fmt_pct(sector_revenue_annual_yoy)}")
        if annual_parts:
            sentence = "**Sector Annual Growth Context:** Additional annual growth context shows " + ", ".join(annual_parts) + "."
            if (not pd.isna(company_revenue_annual_yoy)) and (not pd.isna(sector_revenue_annual_yoy)):
                if float(company_revenue_annual_yoy) > float(sector_revenue_annual_yoy):
                    strengths.append(sentence + " That suggests the business is still compounding ahead of its sector's annual revenue backdrop.")
                elif float(company_revenue_annual_yoy) < float(sector_revenue_annual_yoy):
                    weaknesses.append(sentence + " That suggests the business is currently trailing its sector's annual revenue backdrop.")
                else:
                    confirmations.append(sentence + " That places the business broadly in line with its sector's annual revenue backdrop.")
            else:
                confirmations.append(sentence + " That adds useful annual context to the growth picture.")

    if not re.search(r"Nifty Cross-Horizon Context|1-week relative strength vs Nifty 50|10-year relative strength vs Nifty 50", body_text, flags=re.IGNORECASE):
        nifty_parts: List[str] = []
        positive_count = 0
        negative_count = 0
        for label, value in [("1-week", rr_nifty50_week), ("quarter", rr_nifty50_quarter), ("1-year", rr_nifty50_year), ("10-year", rr_nifty50_10year)]:
            if not pd.isna(value):
                nifty_parts.append(f"{label} relative strength vs Nifty 50 at {_fmt_signed_pct(value)}")
                if float(value) > 0:
                    positive_count += 1
                elif float(value) < 0:
                    negative_count += 1
        if nifty_parts:
            sentence = "**Nifty Cross-Horizon Context:** Additional Nifty-relative context shows " + ", ".join(nifty_parts) + "."
            if positive_count >= 3 and negative_count == 0:
                confirmations.append(sentence + " That reinforces broad-market relative strength across multiple horizons.")
            elif negative_count >= 2 and positive_count == 0:
                weaknesses.append(sentence + " That suggests the stock is still struggling to outperform the broad benchmark across multiple horizons.")
            else:
                confirmations.append(sentence + " That adds useful broad-benchmark context around the current setup.")

    if not re.search(r"Decade Peer Leadership|10-year relative strength versus sector|10-year relative strength versus industry", body_text, flags=re.IGNORECASE):
        decade_parts: List[str] = []
        decade_positive = 0
        decade_negative = 0
        if not pd.isna(rr_sector_10year):
            decade_parts.append(f"10-year relative strength versus sector at {_fmt_signed_pct(rr_sector_10year)}")
            if float(rr_sector_10year) > 0:
                decade_positive += 1
            elif float(rr_sector_10year) < 0:
                decade_negative += 1
        if not pd.isna(rr_industry_10year):
            decade_parts.append(f"10-year relative strength versus industry at {_fmt_signed_pct(rr_industry_10year)}")
            if float(rr_industry_10year) > 0:
                decade_positive += 1
            elif float(rr_industry_10year) < 0:
                decade_negative += 1
        if decade_parts:
            sentence = "**Decade Peer Leadership:** " + ", ".join(decade_parts) + "."
            if decade_positive == len(decade_parts):
                strengths.append(sentence + " That reinforces decade-long leadership versus its closer peer set.")
            elif decade_negative == len(decade_parts):
                weaknesses.append(sentence + " That suggests the longer-cycle peer record remains weak despite any shorter-cycle repair.")
            else:
                confirmations.append(sentence + " That adds useful long-cycle peer context beyond the 3-year and 5-year lenses.")

    if not re.search(r"Fresh Promoter Signal|promoter holding change QoQ", body_text, flags=re.IGNORECASE):
        if not pd.isna(promoter_qoq):
            sentence = f"**Fresh Promoter Signal:** Promoter holding change QoQ stands at {_fmt_signed_pct(promoter_qoq)}."
            if float(promoter_qoq) > 0.05:
                confirmations.append(sentence + " That suggests promoters have been adding in the latest quarter.")
            elif float(promoter_qoq) < -0.05:
                weaknesses.append(sentence + " That suggests some near-term promoter supply and merits monitoring.")
            else:
                confirmations.append(sentence + " That keeps the latest-quarter promoter trend broadly stable.")

    if not re.search(r"Short-Term Range Context|day range spanning|month low at", body_text, flags=re.IGNORECASE):
        range_parts: List[str] = []
        supportive = False
        weak = False
        if day_high > 0 and day_low > 0 and day_high >= day_low:
            range_parts.append(f"day range spanning ₹{_fmt_num(day_low)} to ₹{_fmt_num(day_high)}")
            if current_price > 0 and day_high > day_low:
                intraday_pos = (float(current_price) - float(day_low)) / (float(day_high) - float(day_low))
                if intraday_pos >= 0.60:
                    supportive = True
                elif intraday_pos <= 0.30:
                    weak = True
        if month_low > 0:
            range_parts.append(f"month low at ₹{_fmt_num(month_low)}")
            if current_price > month_low * 1.05:
                supportive = True
        if range_parts:
            sentence = "**Short-Term Range Context:** Additional range context shows " + ", ".join(range_parts) + "."
            if supportive and not weak:
                confirmations.append(sentence + " That suggests price is holding away from recent downside anchors.")
            elif weak and not supportive:
                weaknesses.append(sentence + " That suggests price is still operating from a weaker part of its near-term range.")
            else:
                confirmations.append(sentence + " That adds useful near-term range-location context around the setup.")

    if strengths:
        bullet_text = "\n".join(f"* {s}" for s in strengths)
        text = _insert_before_heading(text, [r"(?:\*\*2\. What confirms the setup\*\*|2\. What confirms the setup)"], bullet_text)
    if confirmations:
        bullet_text = "\n".join(f"* {s}" for s in confirmations)
        text = _insert_before_heading(text, [r"(?:\*\*3\. What weakens the setup\*\*|3\. What weakens the setup)"], bullet_text)
    if weaknesses:
        bullet_text = "\n".join(f"* {s}" for s in weaknesses)
        text = _insert_before_heading(text, [r"(?:\*\*4\. Strategy fit\*\*|4\. Strategy fit)"], bullet_text)
    return text



def _ensure_batch5_peer_visibility(text: str, row: pd.Series) -> str:
    if not text:
        return text

    body_text = re.split(r"\n*Essential metrics table\n", text, maxsplit=1)[0]

    rr_sector_week = _get_numeric(row, "rr_sector_week_pct", np.nan)
    rr_sector_month = _get_numeric(row, "rr_sector_month_pct", np.nan)
    rr_sector_quarter = _get_numeric(row, "rr_sector_quarter_pct", np.nan)
    rr_sector_5y = _get_numeric(row, "rr_sector_5year_pct", np.nan)
    rr_industry_week = _get_numeric(row, "rr_industry_week_pct", np.nan)
    rr_industry_month = _get_numeric(row, "rr_industry_month_pct", np.nan)
    rr_industry_quarter = _get_numeric(row, "rr_industry_quarter_pct", np.nan)
    rr_industry_5y = _get_numeric(row, "rr_industry_5year_pct", np.nan)

    strengths: List[str] = []
    confirmations: List[str] = []
    weaknesses: List[str] = []

    if not re.search(r"Peer Momentum Breadth|1-week relative strength versus sector|1-month relative strength versus sector|quarterly relative strength versus sector|1-week relative strength versus industry|1-month relative strength versus industry|quarterly relative strength versus industry", body_text, flags=re.IGNORECASE):
        sector_parts: List[str] = []
        industry_parts: List[str] = []
        peer_values: List[float] = []
        for label, value in [("1-week", rr_sector_week), ("1-month", rr_sector_month), ("quarter", rr_sector_quarter)]:
            if not pd.isna(value):
                sector_parts.append(f"{label} {_fmt_signed_pct(value)}")
                peer_values.append(float(value))
        for label, value in [("1-week", rr_industry_week), ("1-month", rr_industry_month), ("quarter", rr_industry_quarter)]:
            if not pd.isna(value):
                industry_parts.append(f"{label} {_fmt_signed_pct(value)}")
                peer_values.append(float(value))
        if sector_parts or industry_parts:
            sentence_parts: List[str] = []
            if sector_parts:
                sentence_parts.append("versus sector: " + ", ".join(sector_parts))
            if industry_parts:
                sentence_parts.append("versus industry: " + ", ".join(industry_parts))
            sentence = "**Peer Momentum Breadth:** " + "; ".join(sentence_parts) + "."
            pos_count = sum(1 for value in peer_values if value > 0)
            neg_count = sum(1 for value in peer_values if value < 0)
            if pos_count > neg_count:
                confirmations.append(sentence + " That shows the move is broadening into sector and industry peer momentum across shorter horizons.")
            elif neg_count > pos_count:
                weaknesses.append(sentence + " That shows the stock is still lagging its closer peer set across short and medium horizons.")
            else:
                confirmations.append(sentence + " That adds useful short-to-medium horizon peer breadth context.")

    if not re.search(r"Extended Peer Leadership|5-year relative strength versus sector|5-year relative strength versus industry", body_text, flags=re.IGNORECASE):
        rs5_parts: List[str] = []
        values5: List[float] = []
        if not pd.isna(rr_sector_5y):
            rs5_parts.append(f"5-year relative strength versus sector at {_fmt_signed_pct(rr_sector_5y)}")
            values5.append(float(rr_sector_5y))
        if not pd.isna(rr_industry_5y):
            rs5_parts.append(f"5-year relative strength versus industry at {_fmt_signed_pct(rr_industry_5y)}")
            values5.append(float(rr_industry_5y))
        if rs5_parts:
            sentence = "**Extended Peer Leadership:** " + ", ".join(rs5_parts) + "."
            pos5 = sum(1 for v in values5 if v > 0)
            neg5 = sum(1 for v in values5 if v < 0)
            if pos5 == len(values5):
                strengths.append(sentence + " That reinforces durable peer leadership beyond the shorter-cycle evidence.")
            elif neg5 == len(values5):
                weaknesses.append(sentence + " That shows the stock still lacks extended-cycle peer leadership on a 5-year lens.")
            else:
                confirmations.append(sentence + " That adds useful extended peer-relative depth to the setup.")

    if strengths:
        bullet_text = "\n".join(f"* {s}" for s in strengths)
        text = _insert_before_heading(text, [r"(?:\*\*2\. What confirms the setup\*\*|2\. What confirms the setup)"], bullet_text)
    if confirmations:
        bullet_text = "\n".join(f"* {s}" for s in confirmations)
        text = _insert_before_heading(text, [r"(?:\*\*3\. What weakens the setup\*\*|3\. What weakens the setup)"], bullet_text)
    if weaknesses:
        bullet_text = "\n".join(f"* {s}" for s in weaknesses)
        text = _insert_before_heading(text, [r"(?:\*\*4\. Strategy fit\*\*|4\. Strategy fit)"], bullet_text)
    return text


def _inject_must_have_visibility(text: str, row: pd.Series) -> str:
    if not text:
        return text

    body_text = re.split(r"\n*Essential metrics table\n", text, maxsplit=1)[0]
    strengths: List[str] = []
    confirmations: List[str] = []
    weaknesses: List[str] = []

    current_ratio = _get_numeric(row, "current_ratio", np.nan)
    interest_coverage = _get_numeric(row, "interest_coverage", np.nan)
    sales_growth_3y = _get_numeric(row, "sales_growth_3y_pct", np.nan)
    sales_growth_5y = _get_numeric(row, "sales_growth_5y_pct", np.nan)
    profit_growth_3y = _get_numeric(row, "profit_growth_3y_pct", np.nan)
    eps_growth_3y = _get_numeric(row, "eps_growth_3y_pct", np.nan)
    eps_growth_5y = _get_numeric(row, "eps_growth_5y_pct", np.nan)
    year_1_high = _get_numeric(row, "year_1_high", np.nan)
    year_1_low = _get_numeric(row, "year_1_low", np.nan)
    current_price = _get_numeric(row, "current_price", np.nan)
    day_high = _get_numeric(row, "day_high", np.nan)
    day_low = _get_numeric(row, "day_low", np.nan)
    month_low = _get_numeric(row, "month_low", np.nan)

    if not re.search(r"current ratio|interest coverage", body_text, flags=re.IGNORECASE):
        liquidity_parts: List[str] = []
        if not pd.isna(current_ratio):
            liquidity_parts.append(f"current ratio of {_fmt_num(current_ratio)}")
        if not pd.isna(interest_coverage):
            liquidity_parts.append(f"interest coverage of {_fmt_num(interest_coverage)}x")
        if liquidity_parts:
            sentence = "**Balance-Sheet Liquidity:** The report also benefits from liquidity and debt-servicing context, with " + " and ".join(liquidity_parts) + "."
            good = ((pd.isna(current_ratio) or float(current_ratio) >= 1.5) and (pd.isna(interest_coverage) or float(interest_coverage) >= 3.0))
            bad = ((not pd.isna(current_ratio) and float(current_ratio) < 1.0) or (not pd.isna(interest_coverage) and float(interest_coverage) < 2.0))
            if good:
                strengths.append(sentence + " That supports balance-sheet flexibility.")
            elif bad:
                weaknesses.append(sentence + " That points to tighter balance-sheet comfort and lower debt-servicing headroom.")
            else:
                confirmations.append(sentence + " That looks adequate, though not exceptionally strong.")

    growth_regex = r"3-year sales growth|5-year sales growth|3-year profit growth|3-year EPS growth|5-year EPS growth|3Y sales growth|5Y sales growth|3Y profit growth|3Y EPS growth|5Y EPS growth"
    if not re.search(growth_regex, body_text, flags=re.IGNORECASE):
        growth_parts: List[str] = []
        growth_values: List[float] = []
        for label, value in [
            ("3Y sales growth", sales_growth_3y),
            ("5Y sales growth", sales_growth_5y),
            ("3Y profit growth", profit_growth_3y),
            ("3Y EPS growth", eps_growth_3y),
            ("5Y EPS growth", eps_growth_5y),
        ]:
            if not pd.isna(value):
                growth_parts.append(f"{label} at {_fmt_pct(value)}")
                growth_values.append(float(value))
        if growth_parts:
            sentence = "**Long-Cycle Growth:** Additional compounding context shows " + ", ".join(growth_parts) + "."
            positive_count = sum(1 for value in growth_values if value > 0)
            weak_count = sum(1 for value in growth_values if value <= 0)
            avg_growth = sum(growth_values) / len(growth_values)
            if positive_count >= max(3, len(growth_values) - 1) and avg_growth >= 12:
                strengths.append(sentence)
            elif weak_count >= max(2, len(growth_values) // 2) or avg_growth < 0:
                weaknesses.append(sentence + " That weakens the long-cycle compounding profile.")
            else:
                confirmations.append(sentence + " That is mixed but still useful long-horizon context.")

    if not re.search(r"52-week|1-year high|1-year low", body_text, flags=re.IGNORECASE):
        if not pd.isna(year_1_high) and not pd.isna(year_1_low) and year_1_high > 0 and year_1_low > 0:
            range_sentence = f"**52-Week Range Context:** The 52-week range spans ₹{_fmt_num(year_1_low)} to ₹{_fmt_num(year_1_high)}."
            if not pd.isna(current_price) and current_price > 0:
                below_high = (current_price / year_1_high) - 1
                above_low = (current_price / year_1_low) - 1
                if below_high >= -0.10:
                    confirmations.append(range_sentence + f" The stock is only {_fmt_frac_pct(abs(below_high))} below the 52-week high, which supports breakout proximity.")
                elif below_high <= -0.30:
                    weaknesses.append(range_sentence + f" The stock is still {_fmt_frac_pct(abs(below_high))} below the 52-week high, so the full one-year repair is not yet complete.")
                else:
                    confirmations.append(range_sentence + f" The price is {_fmt_frac_pct(abs(below_high))} below the 52-week high and {_fmt_frac_pct(above_low)} above the 52-week low, which gives useful one-year range context.")
            else:
                confirmations.append(range_sentence)

    if strengths:
        bullet_text = "\n".join(f"* {item}" for item in strengths)
        text = _insert_before_heading(text, [r"(?:\*\*2\. What confirms the setup\*\*|2\. What confirms the setup)"], bullet_text)
    if confirmations:
        bullet_text = "\n".join(f"* {item}" for item in confirmations)
        text = _insert_before_heading(text, [r"(?:\*\*3\. What weakens the setup\*\*|3\. What weakens the setup)"], bullet_text)
    if weaknesses:
        bullet_text = "\n".join(f"* {item}" for item in weaknesses)
        text = _insert_before_heading(text, [r"(?:\*\*4\. Strategy fit\*\*|4\. Strategy fit)"], bullet_text)
    return text


def _ensure_r3_entry_trigger_visibility(text: str, row: pd.Series) -> str:
    if not text:
        return text
    pivot_r3 = _get_numeric(row, "standard_resistance_r3", np.nan)
    if pd.isna(pivot_r3) or float(pivot_r3) <= 0:
        return text
    if re.search(r"\bpivot resistance R3\b|\bR3 at\b", text, flags=re.IGNORECASE):
        return text

    insertion = f" A stronger continuation could then target pivot resistance R3 at ₹{float(pivot_r3):,.2f}."

    patterns = [
        r"(\*\s*\*\*Entry Trigger:\*\*\s*.*?)(\n\*\s*\*\*Invalidation:\*\*)",
        r"(\*\s*\*\*Trigger:\*\*\s*.*?)(\n\*\s*\*\*Invalidation:\*\*)",
        r"(\*\s*\*\*Entry Trigger:\*\*\s*.*?)(\nInvalidation:)",
        r"(\*\s*\*\*Trigger:\*\*\s*.*?)(\nInvalidation:)",
        r"(Entry trigger:\s*.*?)(\nInvalidation:)",
        r"(Trigger:\s*.*?)(\nInvalidation:)",
        r"(\*\s*\*\*Trigger:\*\*\s*.*?)(\n\*\s*\*\*6\.)",
        r"(\*\s*\*\*Entry Trigger:\*\*\s*.*?)(\n\*\s*\*\*6\.)",
    ]
    for pattern in patterns:
        def _add_r3(match: re.Match) -> str:
            return f"{match.group(1)}{insertion}{match.group(2)}"

        updated, count = re.subn(pattern, _add_r3, text, count=1, flags=re.IGNORECASE | re.DOTALL)
        if count:
            return updated
    return text


def _ensure_s3_invalidation_visibility(text: str, row: pd.Series) -> str:
    if not text:
        return text
    pivot_s3 = _get_numeric(row, "standard_support_s3", np.nan)
    if pd.isna(pivot_s3) or float(pivot_s3) <= 0:
        return text
    if re.search(r"\bpivot support S3\b|\bS3 at\b", text, flags=re.IGNORECASE):
        return text

    insertion = f" A more decisive failure would come below pivot support S3 at ₹{float(pivot_s3):,.2f}."

    patterns = [
        r"(\*\s*\*\*Invalidation:\*\*\s*.*?)(\n\*\s*\*\*6\.)",
        r"(\*\s*\*\*Invalidation:\*\*\s*.*?)(\n6\. Confidence level)",
        r"(Invalidation:\s*.*?)(\n\*\s*\*\*6\.)",
        r"(Invalidation:\s*.*?)(\n6\. Confidence level)",
        r"(\*\s*\*\*Invalidation:\*\*\s*.*?)(\n\*\s*\*\*7\.)",
        r"(Invalidation:\s*.*?)(\n\*\s*\*\*7\.)",
    ]
    for pattern in patterns:
        def _add_s3(match: re.Match) -> str:
            return f"{match.group(1)}{insertion}{match.group(2)}"

        updated, count = re.subn(pattern, _add_s3, text, count=1, flags=re.IGNORECASE | re.DOTALL)
        if count:
            return updated
    return text


def _finalize_trigger_invalidation_consistency(text: str, row: pd.Series) -> str:
    if not text:
        return text

    pivot_r1 = _get_numeric(row, "standard_resistance_r1", np.nan)
    pivot_r2 = _get_numeric(row, "standard_resistance_r2", np.nan)
    pivot_r3 = _get_numeric(row, "standard_resistance_r3", np.nan)
    pivot_s1 = _get_numeric(row, "standard_support_s1", np.nan)
    pivot_s2 = _get_numeric(row, "standard_support_s2", np.nan)
    pivot_s3 = _get_numeric(row, "standard_support_s3", np.nan)
    month_high = _get_numeric(row, "month_high", np.nan)

    def _fmt_money(value: float) -> str:
        return f"₹{float(value):,.2f}"

    def _rebuild_trigger(match: re.Match) -> str:
        section = match.group(0)
        bullet_prefix = "* " if re.match(r"^\s*\*", section) else ""
        label = "Entry Trigger" if re.search(r"Entry Trigger", section, flags=re.IGNORECASE) else "Trigger"

        parts = []
        if not pd.isna(pivot_r1) and float(pivot_r1) > 0:
            if label == "Entry Trigger":
                parts.append(f"A decisive move above pivot resistance R1 at {_fmt_money(pivot_r1)} would confirm the breakout")
            else:
                parts.append(f"A decisive move above the immediate pivot resistance R1 ({_fmt_money(pivot_r1)}) on supportive volume would signal a breakout")
        else:
            parts.append("A decisive move above nearby resistance would confirm the breakout")

        if not pd.isna(month_high) and float(month_high) > 0:
            if label == "Entry Trigger":
                parts[-1] += f", with the monthly high at {_fmt_money(month_high)} as the immediate target"
            else:
                parts.append(f"The next upside objective is the monthly high of {_fmt_money(month_high)}")

        if not pd.isna(pivot_r2) and float(pivot_r2) > 0:
            if label == "Entry Trigger":
                parts.append(f"A sustained move could then target pivot resistance R2 at {_fmt_money(pivot_r2)}")
            else:
                parts.append(f"The next upside objective after that is pivot resistance R2 at {_fmt_money(pivot_r2)}")

        if not pd.isna(pivot_r3) and float(pivot_r3) > 0:
            if label == "Entry Trigger" and not pd.isna(pivot_r2) and float(pivot_r2) > 0:
                parts[-1] += f" and R3 at {_fmt_money(pivot_r3)}"
            else:
                parts.append(f"A stronger continuation could then target pivot resistance R3 at {_fmt_money(pivot_r3)}")

        sentence = ". ".join(p.rstrip('. ') for p in parts if p).strip() + "."
        return f"{bullet_prefix}**{label}:** {sentence}"

    def _rebuild_invalidation(match: re.Match) -> str:
        section = match.group(0)
        bullet_prefix = "* " if re.match(r"^\s*\*", section) else ""

        parts = []
        if not pd.isna(pivot_s1) and float(pivot_s1) > 0:
            parts.append(f"The initial invalidation level is pivot support S1 at {_fmt_money(pivot_s1)}")
        else:
            parts.append("The initial invalidation level sits near first support")

        if not pd.isna(pivot_s2) and float(pivot_s2) > 0:
            parts.append(f"A break below pivot support S2 at {_fmt_money(pivot_s2)} would weaken the setup")

        if not pd.isna(pivot_s3) and float(pivot_s3) > 0:
            parts.append(f"A fall to pivot support S3 at {_fmt_money(pivot_s3)} would represent a more decisive failure")

        sentence = ". ".join(p.rstrip('. ') for p in parts if p).strip() + "."
        return f"{bullet_prefix}**Invalidation:** {sentence}"

    trigger_patterns = [
        r"(?ms)^\*?\s*\*\*Entry Trigger:\*\*.*?(?=(?:\n\*?\s*\*\*Invalidation:\*\*|\n\*\*6\. Confidence level\*\*|\n6\. Confidence level|\Z))",
        r"(?ms)^\*?\s*\*\*Trigger:\*\*.*?(?=(?:\n\*?\s*\*\*Invalidation:\*\*|\n\*\*6\. Confidence level\*\*|\n6\. Confidence level|\Z))",
        r"(?ms)^Entry trigger:.*?(?=(?:\nInvalidation:|\n6\. Confidence level|\Z))",
        r"(?ms)^Trigger:.*?(?=(?:\nInvalidation:|\n6\. Confidence level|\Z))",
    ]
    for patt in trigger_patterns:
        new_text, count = re.subn(patt, _rebuild_trigger, text, count=1)
        if count:
            text = new_text
            break

    invalidation_patterns = [
        r"(?ms)^\*?\s*\*\*Invalidation:\*\*.*?(?=(?:\s+\*\*6\. Confidence level\*\*|\s+6\. Confidence level|\Z))",
        r"(?ms)^Invalidation:.*?(?=(?:\s+\*\*6\. Confidence level\*\*|\s+6\. Confidence level|\Z))",
    ]
    for patt in invalidation_patterns:
        new_text, count = re.subn(patt, _rebuild_invalidation, text, count=1)
        if count:
            text = new_text
            break

    return text




def _normalize_section_headings(text: str) -> str:
    if not text:
        return text

    # Normalize any wrapped or partially broken section headings first.
    wrapped_patterns = {
        r"(?is)\*{0,2}\s*1\.\s*Core\s+strengths\s*\*{0,2}": "**1. Core strengths**",
        r"(?is)\*{0,2}\s*2\.\s*What\s+confirms\s+the\s+setup\s*\*{0,2}": "**2. What confirms the setup**",
        r"(?is)\*{0,2}\s*3\.\s*What\s+weakens\s+the\s+setup\s*\*{0,2}": "**3. What weakens the setup**",
        r"(?is)\*{0,2}\s*4\.\s*Strategy\s+fit\s*\*{0,2}": "**4. Strategy fit**",
        r"(?is)\*{0,2}\s*5\.\s*Entry\s+trigger\s+and\s+invalidation\s*\*{0,2}": "**5. Entry trigger and invalidation**",
        r"(?is)\*{0,2}\s*6\.\s*Confidence\s+level\s*\*{0,2}": "**6. Confidence level**",
        r"(?is)\*{0,2}\s*7\.\s*Final\s+verdict\s*\*{0,2}": "**7. Final verdict**",
    }
    for patt, repl in wrapped_patterns.items():
        text = re.sub(patt, repl, text)

    heading_body = r"\*\*[1-7]\.\s+(?:Core strengths|What confirms the setup|What weakens the setup|Strategy fit|Entry trigger and invalidation|Confidence level|Final verdict)\*\*"

    # If a section heading got glued to the previous sentence, put it on its own line.
    text = re.sub(rf"(?<!\n)({heading_body})", r"\n\1", text)

    # Ensure each normalized section heading starts at the beginning of a line.
    text = re.sub(rf"\n?[ 	]*({heading_body})", r"\n\1", text)

    # Add a blank line before sections 6 and 7 when they immediately follow body text.
    text = re.sub(r"([^\n])\n(\*\*[67]\.\s+(?:Confidence level|Final verdict)\*\*)", r"\1\n\n\2", text)

    # Avoid accidental heading-body collisions like 'failure.**6. Confidence level**'.
    text = re.sub(r"([.!?])\s*(\*\*[67]\.\s+(?:Confidence level|Final verdict)\*\*)", r"\1\n\n\2", text)

    # Keep bullet labels in section 5 on their own lines.
    text = re.sub(r"(?<!\n)(\*\s*\*\*(?:Entry Trigger|Invalidation):\*\*)", r"\n\1", text)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sanitize_report_text(report_text: Optional[str], row: Optional[pd.Series] = None) -> Optional[str]:
    if not report_text:
        return report_text
    text = str(report_text)
    text = _normalize_section_headings(text)

    # Remove stray control characters that can leak in from regex replacement mishandling.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Drop parenthetical code-style annotations like (`volume_ratio_month`) or (price_vs_sma200_pct: -14.4%).
    code_paren = re.compile(r"\s*\((?:`?[a-z][a-z0-9_]*`?(?::\s*[^)]*)?)\)")
    text = code_paren.sub("", text)

    # Replace backticked field names with plain-English labels.
    backticked = re.compile(r"`([a-z][a-z0-9_]*)`")
    text = backticked.sub(lambda m: _humanize_field_name(m.group(1)), text)

    # Replace any remaining standalone known field references.
    for field_name, label in sorted(FIELD_NAME_LABELS.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(rf"\b{re.escape(field_name)}\b", label, text)

    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if row is not None:
        text = _inject_net_annual_cash_flow(text, row)
        text = _inject_trendlyne_durability(text, row)
        text = _inject_long_price_history(text, row)
        text = _inject_sensex_relative_strength(text, row)
        text = _inject_phase1_risk_ttm_context(text, row)
        text = _inject_annual_profitability_context(text, row)
        text = _inject_tier3_visibility(text, row)
        text = _inject_good_to_have_visibility(text, row)
        text = _ensure_batch5_peer_visibility(text, row)
        text = _ensure_r3_entry_trigger_visibility(text, row)
        text = _ensure_s3_invalidation_visibility(text, row)
        text = _finalize_trigger_invalidation_consistency(text, row)
        text = _inject_must_have_visibility(text, row)
        text = _normalize_section_headings(text)
    return _normalize_section_headings(text).strip()


def _infer_shortlist_horizon(strategy: str) -> str:
    strategy = str(strategy or "")
    for key, value in SHORTLIST_HORIZON_MAP.items():
        if key in strategy:
            return value
    return ""


UPSTREAM_DIAGNOSTIC_SCHEMA: Dict[str, List[str]] = {
    "identity": [
        "stock_name", "nse_code", "isin", "sector_name", "industry_name",
        "trade_date", "data_refresh_timestamp", "source_all_metrics_date",
        "source_shareholding_date", "current_price", "market_capitalization",
        "market_cap_bucket", "strategy", "primary_strategy_tag", "is_financial_like",
    ],
    "decision_summary": [
        "entry_quality_tag", "crowded_trend_flag", "analysis_confidence_score",
        "analysis_confidence_bucket", "data_completeness_score",
        "red_flag_count", "positive_flag_count", "red_flags", "positive_flags",
        "setup_quality_tag", "setup_confirmation_tag", "setup_risk_tag",
        "compounder_setup_flag", "earnings_reacceleration_flag", "cashflow_alignment_flag",
    ],
    "scores": [
        "swing_score", "short_term_score", "long_term_score",
        "growth_factor", "earnings_accel_factor", "business_quality_factor",
        "cashflow_quality_factor", "risk_factor", "valuation_factor",
        "ownership_factor", "trend_health_factor", "entry_timing_factor",
        "volume_factor", "momentum_factor", "technical_sanity_factor",
        "catalyst_proxy_factor", "quality_safety_score", "tradability_score",
        "swing_penalty", "short_term_penalty", "long_term_penalty",
        "swing_bucket_penalty", "short_term_bucket_penalty", "long_term_bucket_penalty",
    ],
    "volume_liquidity": [
        "day_volume", "week_volume_avg", "month_volume_avg",
        "volume_ratio_week", "volume_ratio_month", "volume_surge_strength", "volume_conviction_bucket", "volume_factor",
        "tradability_score", "day_change_pct", "week_change_pct", "month_change_pct",
        "vwap_day",
        "volume_confirmation_flag", "weak_volume_confirmation",
        "breakout_volume_confirmation_flag", "liquidity_risk_flag",
    ],
    "trend_structure": [
        "day_sma5", "day_sma30", "day_sma50", "day_sma100", "day_sma200",
        "price_vs_sma50_pct", "price_vs_sma200_pct", "not_overextended_raw",
        "distance_from_52w_high_pct", "day_rsi", "day_adx",
        "day_macd", "day_macd_signal_line", "macd_spread",
        "day_high", "day_low", "month_high", "month_low",
        "qtr_high", "qtr_low", "year_1_high", "year_1_low",
        "clean_trend_structure", "overextended_trend_flag",
        "crowded_breakout", "trend_confirmation_flag", "trend_structure_stage", "trend_extension_bucket", "room_to_month_high_pct",
    ],
    "pivot_structure": [
        "standard_pivot_point",
        "standard_resistance_r1", "standard_resistance_r2", "standard_resistance_r3",
        "standard_support_s1", "standard_support_s2", "standard_support_s3",
        "standard_r1_to_price_diff_pct", "standard_r2_to_price_diff_pct", "standard_r3_to_price_diff_pct",
        "standard_s1_to_price_diff_pct", "standard_s2_to_price_diff_pct", "standard_s3_to_price_diff_pct",
    ],
    "momentum_relative_strength": [
        "trendlyne_momentum_score", "prev_day_trendlyne_momentum_score",
        "prev_week_trendlyne_momentum_score", "prev_month_trendlyne_momentum_score",
        "rr_nifty50_week_pct", "rr_nifty50_month_pct", "rr_nifty50_quarter_pct",
        "rr_nifty50_year_pct", "rr_nifty50_3year_pct", "rr_nifty50_5year_pct", "rr_nifty50_10year_pct",
        "rr_sensex_week_pct", "rr_sensex_month_pct", "rr_sensex_quarter_pct", "rr_sensex_year_pct",
        "rr_sensex_3year_pct", "rr_sensex_5year_pct", "rr_sensex_10year_pct",
        "rr_sector_week_pct", "rr_sector_month_pct", "rr_sector_quarter_pct", "rr_sector_year_pct",
        "rr_sector_3year_pct", "rr_sector_5year_pct", "rr_sector_10year_pct",
        "rr_industry_week_pct", "rr_industry_month_pct", "rr_industry_quarter_pct", "rr_industry_year_pct",
        "rr_industry_3year_pct", "rr_industry_5year_pct", "rr_industry_10year_pct",
        "momentum_acceleration_flag", "relative_strength_confirmation_flag",
    ],
    "trendlyne_regime": [
        "trendlyne_durability_score", "trendlyne_valuation_score",
        "prev_day_trendlyne_durability_score", "prev_day_trendlyne_valuation_score",
        "prev_week_trendlyne_durability_score", "prev_week_trendlyne_valuation_score",
        "prev_month_trendlyne_durability_score", "prev_month_trendlyne_valuation_score",
        "dvm_classification_text",
    ],
    "technical_refinement": [
        "day_ema12", "day_ema20", "day_ema50", "day_ema100",
        "day_atr", "day_mfi", "day_roc21", "day_roc125",
        "normalized_momentum_score",
    ],
    "earnings_quality": [
        "sales_growth_3y_pct", "sales_growth_5y_pct",
        "profit_growth_3y_pct", "profit_growth_5y_pct",
        "eps_growth_3y_pct", "eps_growth_5y_pct",
        "sales_qoq", "sales_yoy", "profit_qoq", "profit_yoy",
        "revenue_qoq_growth_pct", "net_profit_qoq_growth_pct",
        "eps_qoq", "eps_yoy", "operating_leverage",
        "opm_current", "opm_last_year", "opm_5y_avg",
        "operating_profit_margin_qtr_pct", "operating_profit_margin_qtr_4qtr_ago_pct",
        "earnings_deceleration_flag", "margin_deterioration_flag",
        "high_quality_growth_flag",
    ],
    "quarterly_bases": [
        "sales_q_latest", "sales_q_prev", "sales_q_yoy_base",
        "profit_q_latest", "profit_q_prev", "profit_q_yoy_base",
        "eps_q_latest", "eps_q_prev", "eps_q_yoy_base",
    ],
    "business_quality": [
        "roce", "roce_3y_avg", "roce_5y_avg", "roe", "roe_3y_avg",
        "roe_annual_pct", "roa_annual_pct",
        "margin_quality_raw", "roce_quality_raw", "roe_quality_raw",
    ],
    "cashflow_balance_sheet": [
        "cfo_latest", "cfo_prev", "cfo_growth", "cfo_3y",
        "fcf_latest", "fcf_prev", "fcf_growth", "fcf_3y",
        "fcf_positive", "fcf_consistency",
        "debt_to_equity", "current_ratio", "interest_coverage", "altman_z_score",
        "weak_cash_conversion", "debt_risk_flag", "balance_sheet_risk_flag", "debt_trend_flag",
    ],
    "debt_history": [
        "debt_latest", "debt_prev", "debt_3y_back",
    ],
    "risk_profile": [
        "beta_1month", "beta_3month", "beta_1year", "beta_3year", "long_term_debt_to_equity_annual",
    ],
    "ttm_business_context": [
        "operating_revenue_ttm", "net_profit_ttm", "eps_ttm_growth_pct", "qtr_change_pct",
    ],
    "valuation_context": [
        "pe_ttm", "pe_3yr_average", "pe_5yr_average",
        "peg_ratio", "peg_ttm", "ev_ebitda", "earnings_yield",
        "price_to_fcf", "price_to_cfo", "price_to_sales", "price_to_book_value_adjusted",
        "days_traded_below_current_pe_pct", "days_traded_below_current_price_to_book_value_pct",
        "valuation_stretch_vs_history", "valuation_stretch_vs_sector",
        "valuation_stretch_vs_industry", "cheap_vs_history_flag", "valuation_regime",
        "sector_pe_ttm", "sector_peg_ttm", "industry_pe_ttm", "industry_peg_ttm",
        "sector_price_to_book_ttm", "industry_price_to_book_ttm",
    ],
    "ownership_sponsorship": [
        "promoter_holding_latest_pct", "promoter_holding_change_qoq_pct",
        "promoter_holding_change_4qtr_pct", "promoter_holding_change_8qtr_pct",
        "promoter_holding_pledge_percentage_qtr_pct", "promoter_pledge_change_qoq_pct",
        "fii_holding_current_qtr_pct", "fii_holding_change_qoq_pct",
        "fii_holding_change_4qtr_pct", "fii_holding_change_8qtr_pct",
        "mf_holding_current_qtr_pct", "mf_holding_change_qoq_pct",
        "mf_holding_change_1month_pct", "mf_holding_change_2month_pct",
        "mf_holding_change_3month_pct", "mf_holding_change_4qtr_pct",
        "mf_holding_change_8qtr_pct",
        "institutional_holding_current_qtr_pct", "institutional_holding_change_qoq_pct",
        "institutional_holding_change_4qtr_pct", "institutional_holding_change_8qtr_pct",
        "falling_institutional_support", "strong_sponsorship_flag",
        "promoter_pledge_risk", "ownership_trend_score",
    ],
    "forensic_quality": [
        "piotroski_score", "strong_piotroski_flag", "weak_piotroski_flag",
        "debtor_days", "avg_debtor_days_3y", "debtor_days_delta_vs_3y",
        "cash_conversion_cycle_days", "inventory_turnover_ratio",
        "inventory_turnover_ratio_3y_back", "inventory_turnover_ratio_5y_back", "inventory_turnover_trend",
        "working_capital_stress_flag", "debtor_deterioration_flag", "weak_inventory_trend_flag",
        "value_support_flag", "deep_value_support_flag",
    ],
    "working_capital_detail": [
        "days_receivable_outstanding", "days_inventory_outstanding", "days_payable_outstanding",
        "avg_working_capital_days_3y",
    ],
    "annual_context": [
        "operating_revenue_annual", "net_profit_annual",
        "cash_from_operating_activity_annual", "net_cash_flow_annual",
        "cash_from_investing_activity_annual", "cash_from_financing_annual_activity",
        "revenue_growth_annual_yoy_pct", "net_profit_annual_yoy_growth_pct", "basic_eps_ttm",
    ],
    "result_recency": [
        "latest_financial_result", "result_announced_date",
    ],
    "long_price_history": [
        "year_10_high", "year_10_low", "year_5_high", "year_5_low",
        "year_1_change_pct", "year_2_price_change_pct", "year_3_price_change_pct",
        "year_5_price_change_pct", "year_10_price_change_pct",
    ],
    "peer_context": [
        "sector_revenue_growth_qtr_yoy_pct", "sector_net_profit_growth_qtr_yoy_pct",
        "sector_revenue_growth_qtr_qoq_pct", "sector_net_profit_growth_qtr_qoq_pct",
        "sector_revenue_growth_annual_yoy_pct", "sector_pe_ttm", "sector_peg_ttm",
        "sector_price_to_book_ttm", "sector_return_on_equity_roe",
        "sector_return_on_assets", "industry_pe_ttm", "industry_peg_ttm",
        "industry_price_to_book_ttm", "industry_return_on_equity_roe",
        "industry_return_on_assets",
    ],
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _normalize_scalar(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, list, dict, tuple, set)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        if ";" in text:
            parts = [p.strip() for p in text.split(";") if p.strip()]
            if len(parts) > 1:
                return parts
    return value


def _collect_section(row: pd.Series, columns: List[str]) -> Dict[str, Any]:
    section: Dict[str, Any] = {}
    for col in columns:
        if col in row.index:
            value = _normalize_scalar(row.get(col))
            if value is not None:
                section[col] = value
    return section


def build_upstream_diagnostics(row: pd.Series, strategy: str) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    for section, columns in UPSTREAM_DIAGNOSTIC_SCHEMA.items():
        section_payload = _collect_section(row, columns)
        if section == "identity":
            section_payload.setdefault("strategy", strategy)
        if section_payload:
            diagnostics[section] = section_payload
    diagnostics["schema_expectations"] = {
        "required_upstream_outputs": {
            "decision_flags": [
                "weak_volume_confirmation", "crowded_breakout", "earnings_deceleration_flag",
                "valuation_stretch_vs_history", "valuation_stretch_vs_sector",
                "falling_institutional_support", "promoter_pledge_risk",
                "weak_cash_conversion", "low_confidence_data", "clean_trend_structure",
                "high_quality_growth_flag", "strong_sponsorship_flag",
            ],
            "summary_fields": [
                "red_flag_count", "positive_flag_count", "red_flags",
                "positive_flags", "analysis_confidence_score",
            ],
        }
    }
    return diagnostics


def _clean_extracted_value(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"^\*{1,2}", "", cleaned)
    cleaned = re.sub(r"\*{1,2}$", "", cleaned)
    cleaned = cleaned.strip().strip("'").strip('"').strip()
    return cleaned



def _annual_reporting_requirements(row: pd.Series) -> Optional[Dict[str, Any]]:
    try:
        annual_revenue = _normalize_scalar(row.get("operating_revenue_annual"))
        annual_profit = _normalize_scalar(row.get("net_profit_annual"))
        annual_cfo = _normalize_scalar(row.get("cash_from_operating_activity_annual"))
        net_annual_cf = _normalize_scalar(row.get("net_cash_flow_annual"))
    except Exception:
        return None

    required_fields: List[str] = []
    sentence_parts: List[str] = []

    if annual_revenue is not None:
        required_fields.append("annual revenue")
        sentence_parts.append(f"annual revenue of ₹{annual_revenue} Cr")
    if annual_profit is not None:
        required_fields.append("annual net profit")
        sentence_parts.append(f"annual net profit of ₹{annual_profit} Cr")
    if annual_cfo is not None:
        required_fields.append("annual operating cash flow")
        sentence_parts.append(f"annual operating cash flow of ₹{annual_cfo} Cr")
    if net_annual_cf is not None:
        try:
            material = abs(float(net_annual_cf)) >= 1
        except Exception:
            material = False
        if material:
            required_fields.append("net annual cash flow")
            sentence_parts.append(f"net annual cash flow of ₹{net_annual_cf} Cr")

    if not required_fields:
        return None

    return {
        "fields": required_fields,
        "instruction": (
            "When annual business-scale context is available and material, you MUST explicitly mention every field listed here "
            "in plain English inside Core strengths or What weakens the setup. Do not omit net annual cash flow when it is listed."
        ),
        "plain_english_targets": sentence_parts,
    }


def extract_report_summary_fields(report_text: Optional[str]) -> Dict[str, Optional[str]]:
    result = {"verdict_label": None, "best_horizon": None, "key_risk": None, "what_must_improve": None}
    if not report_text:
        return result
    patterns = {
        "verdict_label": r"(?im)^\s*\*{0,2}\s*Verdict Label:\s*\*{0,2}\s*(.+?)\s*$",
        "best_horizon": r"(?im)^\s*\*{0,2}\s*Best Horizon:\s*\*{0,2}\s*(.+?)\s*$",
        "key_risk": r"(?im)^\s*\*{0,2}\s*Key Risk:\s*\*{0,2}\s*(.+?)\s*$",
        "what_must_improve": r"(?im)^\s*\*{0,2}\s*What Must Improve:\s*\*{0,2}\s*(.+?)\s*$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report_text)
        if match:
            result[key] = _clean_extracted_value(match.group(1))
    return result


def build_prompt_payload(row: pd.Series, strategy: str) -> Dict[str, str]:
    diagnostics = build_upstream_diagnostics(row, strategy)
    annual_requirements = _annual_reporting_requirements(row)
    if annual_requirements:
        diagnostics["annual_reporting_requirements"] = annual_requirements
    system = (
        "You are an Indian equity research assistant. "
        "You are only polishing a pre-structured analyst decision payload into a readable report. "
        "Do not invent contracts, approvals, management commentary, macro views, or news. "
        "Do not change the core conclusion, verdict label, best horizon, key risk, or what must improve unless there is an obvious formatting issue. "
        "Write exactly these sections with headings: 1. Core strengths 2. What confirms the setup 3. What weakens the setup 4. Strategy fit 5. Entry trigger and invalidation 6. Confidence level 7. Final verdict. "
        "In the Final verdict section, end with exactly these four lines: "
        "'Verdict Label: <value provided>' 'Best Horizon: <value provided>' 'Key Risk: <value provided>' 'What Must Improve: <value provided>'. "
        "Use a professional but practical style. Be specific, compact, and grounded in the provided structured data and decision payload. "
        "Whenever a trigger is supported by the data, explicitly cite the raw numbers instead of generic labels. "
        "When available and material, explicitly surface TTM business scale, EPS TTM growth, beta profile, annual long-term debt-to-equity context, short-horizon beta, quarter price change, and quarterly operating margin versus four quarters ago. "
        "You must also explicitly surface current ratio, interest coverage, 3-year sales growth, 5-year sales growth, 3-year profit growth, 3-year EPS growth, 5-year EPS growth, and the 52-week high/low context whenever those fields are available. "
        "When available, preferentially incorporate the first-batch higher-value context fields in plain English: 3-year average PE, percentage of historical days traded below the current PE, price-to-sales, 1-year relative strength versus sector, 1-year relative strength versus industry, 4-quarter promoter holding change, debtor days, and the 100-day SMA. Use them to sharpen valuation, sponsorship, peer-relative, working-capital, and technical-repair context without making the report feel repetitive. "
        "When available, also incorporate the second-batch context fields in plain English: 3-year relative strength versus sector, 3-year relative strength versus industry, 4-quarter FII holding change, 4-quarter MF holding change, 4-quarter institutional holding change, the pivot point, pivot resistance R2, and pivot support S3. Use them to deepen longer-cycle peer leadership, sponsorship breadth, and trade-map clarity without making the report feel repetitive. "
        "When available, also incorporate the third-batch context fields in plain English: price-to-CFO, price-to-FCF, days receivable outstanding, days inventory outstanding, 3-year average working-capital days, and the 3-year inventory-turnover reference. Use them to deepen cash-flow valuation and operating-cycle discipline context without making the report feel repetitive. "
        "When available, also incorporate the fourth-batch context fields in plain English: revenue QoQ growth, net profit QoQ growth, 3-year average ROCE, 3-year average ROE, 5-year average operating margin, percentage of historical days traded below the current price-to-book, pivot resistance R3, and daily VWAP. Use them to sharpen short-cycle acceleration, quality-consistency, book-value valuation, and execution-map context without making the report feel repetitive. "
        "When available, also incorporate the fifth-batch context fields in plain English: 1-week, 1-month, and quarterly relative strength versus sector and industry, plus 5-year relative strength versus sector and industry. Use them to deepen peer-momentum breadth and extended peer leadership context without making the report feel repetitive. "
        "When available, also incorporate the sixth-batch context fields in plain English: 8-quarter promoter, FII, MF, and institutional holding changes, current institutional holding, and 1-month, 2-month, and 3-month MF holding changes. Use them to deepen long-cycle sponsorship durability, current ownership-base context, and fresh fund-flow pulse without making the report feel repetitive. "
        "When available, also incorporate the seventh-batch context fields in plain English: sector and industry ROE, sector and industry ROA, and sector revenue/profit growth across YoY and QoQ windows. Use them to deepen peer-profitability benchmarking and sector-growth context without making the report feel repetitive. "
        "When available, also incorporate the eighth-batch context fields in plain English: MACD versus signal line, distance to pivot resistance R2, distance to pivot resistance R3, distance to pivot support S3, 1-month price change, and the quarter high/low range. Use them to sharpen technical momentum structure, extend the trade map, and frame the quarter-range backdrop without making the report feel repetitive. "
        "In strengths, confirmations, and weaknesses, preferentially reference: pe_ttm, pe_3yr_average, pe_5yr_average, days_traded_below_current_pe_pct, price_to_sales, sector_pe_ttm, earnings_yield, ev_ebitda, roce_5y_avg, profit_yoy, profit_growth_5y_pct, sales_yoy, opm_current versus opm_last_year, cfo_latest, fcf_latest, cash_conversion_cycle_days, inventory_turnover_ratio, inventory_turnover_trend, debtor_days, debtor_days_delta_vs_3y, fii_holding_change_qoq_pct, mf_holding_change_qoq_pct, promoter_holding_change_qoq_pct, promoter_holding_change_4qtr_pct, fii_holding_change_4qtr_pct, mf_holding_change_4qtr_pct, institutional_holding_change_4qtr_pct, promoter_holding_change_8qtr_pct, fii_holding_change_8qtr_pct, mf_holding_change_1month_pct, mf_holding_change_2month_pct, mf_holding_change_3month_pct, mf_holding_change_8qtr_pct, institutional_holding_current_qtr_pct, institutional_holding_change_8qtr_pct, promoter_holding_pledge_percentage_qtr_pct, volume_ratio_month, volume_surge_strength, volume_conviction_bucket, rr_nifty50_month_pct, rr_sector_year_pct, rr_industry_year_pct, rr_sector_3year_pct, rr_industry_3year_pct, price_vs_sma200_pct, day_adx, distance_from_52w_high_pct, room_to_month_high_pct, day_sma100, standard_pivot_point, standard_resistance_r2, standard_support_s3, piotroski_score, peg_ratio, operating_leverage, debt_trend_flag, cashflow_alignment_flag, trend_structure_stage, trend_extension_bucket, compounder_setup_flag, earnings_reacceleration_flag, sector_return_on_equity_roe, industry_return_on_equity_roe, sector_return_on_assets, industry_return_on_assets, sector_revenue_growth_qtr_yoy_pct, sector_net_profit_growth_qtr_yoy_pct, sector_revenue_growth_qtr_qoq_pct, sector_net_profit_growth_qtr_qoq_pct, and is_financial_like. "
        "When pivot data is available, use standard_pivot_point, standard_resistance_r1, standard_resistance_r2, standard_resistance_r3, standard_support_s1, standard_support_s2, and standard_support_s3 in the Entry trigger and invalidation section instead of relying only on month high, month low, SMA50, and SMA200. "
        "If compounder_setup_flag is active, say that directly and explain it with supporting growth, ROCE, cash-flow, ownership, valuation, and trend evidence. "
        "If earnings_reacceleration_flag is active, say that directly and cite the recent versus historical growth and margin context. "
        "If cashflow_alignment_flag or debt_trend_flag is supportive, state that directly in plain English with the relevant numbers. "
        "When ownership is a strength or risk, always cite the exact promoter, FII, and MF QoQ changes, and include promoter pledge percentage when it is above zero. "
        "When ownership data is available, include one concise ownership sentence even if the evidence is only stable or mixed; do not omit it simply because the signal is not strongly positive or strongly negative. "
        "When forensic valuation is discussed, go beyond PE and PEG by explicitly using earnings_yield and ev_ebitda whenever they add context. "
        "If value_support_flag or deep_value_support_flag is active, name that signal directly instead of only implying it through valuation prose. "
        "If debt_trend_flag is Deleveraging, say that directly; if it is Leveraging Up while debt-to-equity is still low, explicitly say that leverage is rising from a low base. "
        "Explain working-capital quality through the exact driver mix such as cash conversion cycle, debtor days change, receivable days, inventory days, payable days, and inventory turnover trend, rather than using only a generic pressure/improvement label. "
        "Use quarterly raw base values when they improve clarity: sales_q_latest versus sales_q_yoy_base and sales_q_prev, profit_q_latest versus profit_q_yoy_base and profit_q_prev, and EPS base values when relevant. Also use qtr_change_pct, operating_profit_margin_qtr_pct, and operating_profit_margin_qtr_4qtr_ago_pct when they materially improve short-horizon execution or operating-quality context. "
        "When debt trend is discussed, use debt_latest, debt_prev, and debt_3y_back so the reader sees the raw debt path rather than only a label. "
        "Use annual context aggressively when it adds durability, especially operating_revenue_annual, net_profit_annual, cash_from_operating_activity_annual, net_cash_flow_annual, latest_financial_result, result_announced_date, roe_annual_pct, roa_annual_pct, and basic_eps_ttm. When annual business-scale fields are available and material, you MUST explicitly cite annual revenue, annual net profit, and annual operating cash flow in plain English; and you MUST also cite net annual cash flow whenever it is meaningfully positive or negative. When annual profitability fields are available and useful, you MUST explicitly cite annual ROE, annual ROA, and basic EPS (TTM) in plain English, and use them to clarify whether annual profitability quality is supportive or needs monitoring. Do not omit these annual figures merely because quarterly evidence is already present, and do not collapse them into vague phrases like business scale or cash-flow profile. If the payload includes annual_reporting_requirements, treat those fields as mandatory narrative inclusions. "
        "Use Tier 2 context when available: multi-horizon relative strength across 3-year, 5-year, and 10-year periods; Sensex-relative strength across week, month, quarter, year, 3-year, 5-year, and 10-year periods; Trendlyne durability and valuation scores; and price-to-book adjusted context versus sector and industry. Use these to separate durable long-cycle leaders from merely short-term movers. "
        "If long-horizon relative strength is strong versus Nifty 50 or Sensex, say that directly in strengths or confirmations. If it is persistently weak over multiple long horizons, say that directly in weaknesses. When Sensex-relative strength is mixed across short horizons, cite the exact week, month, quarter, and year values plainly instead of omitting them. If market-relative strength is strong but sector or industry relative strength is weaker, explain that mix plainly. "
        "If Trendlyne durability score is strong, you MUST explicitly cite it in plain English in Core strengths; if it is weak, you MUST explicitly cite it in What weakens the setup. Do not omit durability merely because other strength bullets already exist. If Trendlyne valuation score is weak, say that the stock still looks demanding on that regime lens even if simpler valuation comparisons look reasonable. "
        "When price_to_book_value_adjusted is available, compare it with sector_price_to_book_ttm and industry_price_to_book_ttm whenever that adds useful valuation context, especially for financials and asset-heavy businesses. "
        "Use Tier 3 technical refinement context when available: EMA structure (12/20/50/100-day), ATR as a volatility guide, Money Flow Index, 21-day and 125-day rate of change, normalized momentum score, and the DVM classification text. Use these to explain whether the move is an early repair, a mature trend, or a weak bounce. "
        "If EMA structure is constructive, you MUST explicitly say that in confirmations with the EMA stack context; if price is still below a key EMA such as the 100-day EMA, explain that as an unfinished trend repair. Do not omit EMA structure when 12/20/50/100-day EMA data is available and informative. "
        "When MFI, ROC, or normalized momentum materially strengthen or weaken the setup, you MUST cite the exact readings in plain English instead of generic momentum language. Always surface normalized momentum score when it is available and informative. If DVM classification is strong, weak, or materially mixed, mention it directly as a regime-style label. Use ATR explicitly as a volatility guide whenever it helps explain execution quality or risk. When beta_1month or beta_3month is available and informative, mention the short-horizon beta profile in plain English. "
        "If debt trend worsens but debt-to-equity remains objectively low, say that debt is rising from a low base instead of implying high absolute leverage. "
        "If is_financial_like equals 1, make the balance-sheet context sector-aware and avoid over-penalizing inventory or CCC style metrics. "
        "Anchor the Strategy fit section with trend_structure_stage whenever it is available. "
        "Never mention raw field names, snake_case variables, backticked identifiers, or internal column names anywhere in the report. Convert every metric reference into plain-English investor language. "
        "If shortlist inclusion and best horizon differ, explain it as a timeframe nuance: shortlist inclusion reflects the current setup, while best horizon reflects the more suitable holding period. Do not present that as a contradiction unless the data clearly conflicts."
    )
    user = json.dumps(diagnostics, indent=2, ensure_ascii=False, default=str)
    return {"system": system, "user": user}


def resolve_api_key(api_key: Optional[str] = None) -> str:
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    return os.getenv("DEEPSEEK_API_KEY", "").strip()


def call_deepseek(payload: Dict[str, str], api_key: str, model: str = DEEPSEEK_MODEL_DEFAULT, timeout: int = 90) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": payload["system"]},
            {"role": "user", "content": payload["user"]},
        ],
    }
    response = requests.post(DEEPSEEK_URL, headers=headers, json=body, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _get_numeric(row: pd.Series, col: str, default: float = 0.0) -> float:
    val = row.get(col, default)
    if _is_missing(val):
        return default
    try:
        return float(val)
    except Exception:
        return default


def _get_text(row: pd.Series, col: str, default: str = "") -> str:
    val = row.get(col, default)
    if _is_missing(val):
        return default
    return str(val).strip()


def _flagged(text: str, phrase: str) -> bool:
    return phrase.lower() in str(text).lower()


def _fmt_num(value: Any, decimals: int = 2, default: str = "NA") -> str:
    if _is_missing(value):
        return default
    try:
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return default


def _fmt_pct(value: Any, decimals: int = 1, default: str = "NA") -> str:
    if _is_missing(value):
        return default
    try:
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return default


def _fmt_frac_pct(value: Any, decimals: int = 1, default: str = "NA") -> str:
    if _is_missing(value):
        return default
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except Exception:
        return default


def _fmt_signed_pct(value: Any, decimals: int = 1, default: str = "NA") -> str:
    if _is_missing(value):
        return default
    try:
        return f"{float(value):+.{decimals}f}%"
    except Exception:
        return default


def _fmt_signed_frac_pct(value: Any, decimals: int = 1, default: str = "NA") -> str:
    if _is_missing(value):
        return default
    try:
        return f"{float(value) * 100:+.{decimals}f}%"
    except Exception:
        return default


def _fmt_date(value: Any, default: str = "NA") -> str:
    if _is_missing(value):
        return default
    try:
        ts = pd.to_datetime(value)
        return ts.strftime("%d %b %Y")
    except Exception:
        return str(value)


def _trendlyne_durability_bucket(value: Any) -> str:
    if _is_missing(value):
        return "unknown"
    score = float(value)
    if score > 55:
        return "strong"
    if score < 35:
        return "weak"
    return "neutral"


def _trendlyne_valuation_bucket(value: Any) -> str:
    if _is_missing(value):
        return "unknown"
    score = float(value)
    if score > 50:
        return "supportive"
    if score < 30:
        return "demanding"
    return "neutral"


def _period_label_pairs(values: Dict[str, Any]) -> List[tuple[str, float]]:
    order = ["3-year", "5-year", "10-year"]
    out: List[tuple[str, float]] = []
    for label in order:
        value = values.get(label)
        if not _is_missing(value):
            out.append((label, float(value)))
    return out


def _pick_relative_strength_periods(pairs: List[tuple[str, float]], positive: bool = True, limit: int = 2) -> List[tuple[str, float]]:
    filtered = [(label, value) for label, value in pairs if (value > 0 if positive else value < 0)]
    if not filtered:
        return []
    preferred_order = {"3-year": 0, "5-year": 1, "10-year": 2}
    filtered.sort(key=lambda x: preferred_order.get(x[0], 99))
    return filtered[:limit]




def _build_priority_good_to_have_metrics(row: pd.Series, strategy: str) -> List[tuple[str, str]]:
    strategy = str(strategy or "")
    candidates: List[tuple[str, Any]] = []

    if "LongTerm" in strategy:
        candidates = [
            ("PE (3Y Avg)", row.get("pe_3yr_average")),
            ("Days Below Current PE (%)", row.get("days_traded_below_current_pe_pct")),
            ("Days Below Current PBV (%)", row.get("days_traded_below_current_price_to_book_value_pct")),
            ("Price to Sales", row.get("price_to_sales")),
            ("Price to CFO", row.get("price_to_cfo")),
            ("Price to FCF", row.get("price_to_fcf")),
            ("ROCE (3Y Avg)", row.get("roce_3y_avg")),
            ("ROE (3Y Avg)", row.get("roe_3y_avg")),
            ("OPM (5Y Avg)", row.get("opm_5y_avg")),
            ("Sector ROE (%)", row.get("sector_return_on_equity_roe")),
            ("Industry ROE (%)", row.get("industry_return_on_equity_roe")),
            ("Sector Rev YoY (%)", row.get("sector_revenue_growth_qtr_yoy_pct")),
            ("Sector Profit YoY (%)", row.get("sector_net_profit_growth_qtr_yoy_pct")),
            ("Sector Rev Annual YoY (%)", row.get("sector_revenue_growth_annual_yoy_pct")),
            ("Sector PEG", row.get("sector_peg_ttm")),
            ("Industry PEG", row.get("industry_peg_ttm")),
            ("Avg Debtor Days (3Y)", row.get("avg_debtor_days_3y")),
            ("Payable Days", row.get("days_payable_outstanding")),
            ("Inv Turnover (5Y Ref)", row.get("inventory_turnover_ratio_5y_back")),
            ("RS vs Sector (3Y)", row.get("rr_sector_3year_pct")),
            ("RS vs Industry (3Y)", row.get("rr_industry_3year_pct")),
            ("RS vs Sector (5Y)", row.get("rr_sector_5year_pct")),
            ("RS vs Industry (5Y)", row.get("rr_industry_5year_pct")),
            ("Institutional Change (4Q)", row.get("institutional_holding_change_4qtr_pct")),
            ("Institutional Holding (%)", row.get("institutional_holding_current_qtr_pct")),
            ("Institutional Change (8Q)", row.get("institutional_holding_change_8qtr_pct")),
            ("FII Change (8Q)", row.get("fii_holding_change_8qtr_pct")),
            ("Promoter Change (QoQ)", row.get("promoter_holding_change_qoq_pct")),
            ("RS vs Nifty (1W)", row.get("rr_nifty50_week_pct")),
            ("RS vs Nifty (Qtr)", row.get("rr_nifty50_quarter_pct")),
            ("RS vs Nifty (1Y)", row.get("rr_nifty50_year_pct")),
            ("RS vs Nifty (10Y)", row.get("rr_nifty50_10year_pct")),
            ("RS vs Sector (10Y)", row.get("rr_sector_10year_pct")),
            ("RS vs Industry (10Y)", row.get("rr_industry_10year_pct")),
            ("Day High", row.get("day_high")),
            ("Day Low", row.get("day_low")),
            ("Month Low", row.get("month_low")),
            ("MACD", row.get("day_macd")),
            ("MACD Signal", row.get("day_macd_signal_line")),
            ("Month Change (%)", row.get("month_change_pct")),
            ("Quarter High", row.get("qtr_high")),
            ("Quarter Low", row.get("qtr_low")),
            ("Promoter Change (4Q)", row.get("promoter_holding_change_4qtr_pct")),
            ("Debtor Days", row.get("debtor_days")),
            ("Day SMA100", row.get("day_sma100")),
        ]
    elif "ShortTerm" in strategy or "Swing" in strategy:
        candidates = [
            ("Pivot Point", row.get("standard_pivot_point")),
            ("Pivot R2", row.get("standard_resistance_r2")),
            ("Pivot R3", row.get("standard_resistance_r3")),
            ("Pivot S3", row.get("standard_support_s3")),
            ("VWAP", row.get("vwap_day")),
            ("Day SMA100", row.get("day_sma100")),
            ("Revenue QoQ (%)", row.get("revenue_qoq_growth_pct")),
            ("Net Profit QoQ (%)", row.get("net_profit_qoq_growth_pct")),
            ("RS vs Sector (1M)", row.get("rr_sector_month_pct")),
            ("RS vs Industry (1M)", row.get("rr_industry_month_pct")),
            ("RS vs Sector (Qtr)", row.get("rr_sector_quarter_pct")),
            ("RS vs Industry (Qtr)", row.get("rr_industry_quarter_pct")),
            ("Sector Rev QoQ (%)", row.get("sector_revenue_growth_qtr_qoq_pct")),
            ("Sector Profit QoQ (%)", row.get("sector_net_profit_growth_qtr_qoq_pct")),
            ("Sector Rev Annual YoY (%)", row.get("sector_revenue_growth_annual_yoy_pct")),
            ("Sector PEG", row.get("sector_peg_ttm")),
            ("Industry PEG", row.get("industry_peg_ttm")),
            ("MACD", row.get("day_macd")),
            ("MACD Signal", row.get("day_macd_signal_line")),
            ("Dist to R2 (%)", row.get("standard_r2_to_price_diff_pct")),
            ("Dist to R3 (%)", row.get("standard_r3_to_price_diff_pct")),
            ("Dist to S3 (%)", row.get("standard_s3_to_price_diff_pct")),
            ("Month Change (%)", row.get("month_change_pct")),
            ("Quarter High", row.get("qtr_high")),
            ("Quarter Low", row.get("qtr_low")),
            ("Institutional Change (4Q)", row.get("institutional_holding_change_4qtr_pct")),
            ("FII Change (4Q)", row.get("fii_holding_change_4qtr_pct")),
            ("MF Change (4Q)", row.get("mf_holding_change_4qtr_pct")),
            ("Promoter Change (QoQ)", row.get("promoter_holding_change_qoq_pct")),
            ("RS vs Nifty (1W)", row.get("rr_nifty50_week_pct")),
            ("RS vs Nifty (Qtr)", row.get("rr_nifty50_quarter_pct")),
            ("RS vs Nifty (1Y)", row.get("rr_nifty50_year_pct")),
            ("Day High", row.get("day_high")),
            ("Day Low", row.get("day_low")),
            ("Month Low", row.get("month_low")),
        ]
    else:
        candidates = [
            ("PE (3Y Avg)", row.get("pe_3yr_average")),
            ("Days Below Current PE (%)", row.get("days_traded_below_current_pe_pct")),
            ("Days Below Current PBV (%)", row.get("days_traded_below_current_price_to_book_value_pct")),
            ("Price to Sales", row.get("price_to_sales")),
            ("Price to CFO", row.get("price_to_cfo")),
            ("Price to FCF", row.get("price_to_fcf")),
            ("ROCE (3Y Avg)", row.get("roce_3y_avg")),
            ("ROE (3Y Avg)", row.get("roe_3y_avg")),
            ("OPM (5Y Avg)", row.get("opm_5y_avg")),
            ("Sector ROE (%)", row.get("sector_return_on_equity_roe")),
            ("Industry ROE (%)", row.get("industry_return_on_equity_roe")),
            ("Sector Rev YoY (%)", row.get("sector_revenue_growth_qtr_yoy_pct")),
            ("Sector Profit YoY (%)", row.get("sector_net_profit_growth_qtr_yoy_pct")),
            ("Sector Rev Annual YoY (%)", row.get("sector_revenue_growth_annual_yoy_pct")),
            ("Sector PEG", row.get("sector_peg_ttm")),
            ("Industry PEG", row.get("industry_peg_ttm")),
            ("Avg Debtor Days (3Y)", row.get("avg_debtor_days_3y")),
            ("Payable Days", row.get("days_payable_outstanding")),
            ("Inv Turnover (5Y Ref)", row.get("inventory_turnover_ratio_5y_back")),
            ("RS vs Sector (3Y)", row.get("rr_sector_3year_pct")),
            ("RS vs Industry (3Y)", row.get("rr_industry_3year_pct")),
            ("Institutional Change (4Q)", row.get("institutional_holding_change_4qtr_pct")),
            ("Institutional Holding (%)", row.get("institutional_holding_current_qtr_pct")),
            ("Institutional Change (8Q)", row.get("institutional_holding_change_8qtr_pct")),
            ("Promoter Change (QoQ)", row.get("promoter_holding_change_qoq_pct")),
            ("RS vs Nifty (1W)", row.get("rr_nifty50_week_pct")),
            ("RS vs Nifty (Qtr)", row.get("rr_nifty50_quarter_pct")),
            ("RS vs Nifty (1Y)", row.get("rr_nifty50_year_pct")),
            ("RS vs Nifty (10Y)", row.get("rr_nifty50_10year_pct")),
            ("RS vs Sector (10Y)", row.get("rr_sector_10year_pct")),
            ("RS vs Industry (10Y)", row.get("rr_industry_10year_pct")),
            ("Day High", row.get("day_high")),
            ("Day Low", row.get("day_low")),
            ("Month Low", row.get("month_low")),
            ("MACD", row.get("day_macd")),
            ("MACD Signal", row.get("day_macd_signal_line")),
            ("Month Change (%)", row.get("month_change_pct")),
            ("Quarter High", row.get("qtr_high")),
            ("Quarter Low", row.get("qtr_low")),
            ("Promoter Change (4Q)", row.get("promoter_holding_change_4qtr_pct")),
            ("Debtor Days", row.get("debtor_days")),
            ("Day SMA100", row.get("day_sma100")),
            ("Pivot Point", row.get("standard_pivot_point")),
        ]

    rows: List[tuple[str, str]] = []
    pct_like = {
        "Days Below Current PE (%)", "Days Below Current PBV (%)", "Promoter Change (4Q)",
        "Revenue QoQ (%)", "Net Profit QoQ (%)",
        "RS vs Sector (1Y)", "RS vs Industry (1Y)",
        "RS vs Sector (1M)", "RS vs Industry (1M)",
        "RS vs Sector (Qtr)", "RS vs Industry (Qtr)",
        "RS vs Sector (3Y)", "RS vs Industry (3Y)",
        "RS vs Sector (5Y)", "RS vs Industry (5Y)",
        "Institutional Change (4Q)", "Institutional Holding (%)", "Institutional Change (8Q)",
        "Promoter Change (8Q)", "FII Change (8Q)", "MF Change (8Q)",
        "Promoter Change (QoQ)", "RS vs Nifty (1W)", "RS vs Nifty (Qtr)", "RS vs Nifty (1Y)", "RS vs Nifty (10Y)",
        "RS vs Sector (10Y)", "RS vs Industry (10Y)",
        "FII Change (4Q)", "MF Change (4Q)", "MF Change (1M)", "MF Change (2M)", "MF Change (3M)",
        "Sector ROE (%)", "Industry ROE (%)", "Sector Rev YoY (%)", "Sector Profit YoY (%)",
        "Sector Rev Annual YoY (%)",
        "Sector Rev QoQ (%)", "Sector Profit QoQ (%)",
        "Month Change (%)", "Dist to R2 (%)", "Dist to R3 (%)", "Dist to S3 (%)",
    }
    price_like = {"Day SMA100", "Pivot Point", "Pivot R2", "Pivot R3", "Pivot S3", "VWAP", "Quarter High", "Quarter Low", "MACD", "MACD Signal"}

    limit = 21 if "LongTerm" in strategy else 14

    for label, value in candidates:
        if _is_missing(value):
            continue
        if label in pct_like:
            rows.append((label, _fmt_pct(value)))
        elif label in price_like:
            rows.append((label, _fmt_num(value)))
        else:
            rows.append((label, _fmt_num(value)))
        if len(rows) >= limit:
            break
    return rows


def build_python_report_payload(row: pd.Series, strategy: str) -> Dict[str, Any]:
    stock_name = _get_text(row, "stock_name", "Unknown Stock")
    nse_code = _get_text(row, "nse_code", "")
    primary_strategy = _get_text(row, "primary_strategy_tag", "")
    entry_tag = _get_text(row, "entry_quality_tag", "")
    setup_tag = _get_text(row, "setup_confirmation_tag", "")
    confidence_score = _get_numeric(row, "analysis_confidence_score", 0)

    swing_score = _get_numeric(row, "swing_score", 0)
    short_score = _get_numeric(row, "short_term_score", 0)
    long_score = _get_numeric(row, "long_term_score", 0)
    score_map = {"Swing": swing_score, "Short Term": short_score, "Long Term": long_score}
    best_horizon = max(score_map, key=score_map.get)

    red_flags = _get_text(row, "red_flags", "")
    positive_flags = _get_text(row, "positive_flags", "")
    red_count = int(_get_numeric(row, "red_flag_count", 0))
    pos_count = int(_get_numeric(row, "positive_flag_count", 0))

    has_valuation = _flagged(red_flags, "Valuation stretched") or _get_numeric(row, "valuation_stretch_vs_history", 0) == 1
    has_weak_cash = _flagged(red_flags, "Weak cash conversion") or _get_numeric(row, "weak_cash_conversion", 0) == 1
    has_earnings_decel = _flagged(red_flags, "Earnings deceleration") or _get_numeric(row, "earnings_deceleration_flag", 0) == 1
    has_no_volume = _flagged(red_flags, "No volume confirmation") or _get_numeric(row, "weak_volume_confirmation", 0) == 1
    has_falling_inst = _flagged(red_flags, "Falling institutional support") or _get_numeric(row, "falling_institutional_support", 0) == 1
    has_pledge = _flagged(red_flags, "Promoter pledge risk") or _get_numeric(row, "promoter_pledge_risk", 0) == 1
    below_sma200 = _get_numeric(row, "price_vs_sma200_pct", 0) < 0
    working_cap_stress = _get_numeric(row, "working_capital_stress_flag", 0) == 1
    weak_inventory_trend = _get_numeric(row, "weak_inventory_trend_flag", 0) == 1

    current_price = _get_numeric(row, "current_price", 0)
    market_cap = _get_numeric(row, "market_capitalization", 0)
    pe_ttm = row.get("pe_ttm")
    pe_5y = row.get("pe_5yr_average")
    sector_pe = row.get("sector_pe_ttm")
    peg_ratio = row.get("peg_ratio") if not _is_missing(row.get("peg_ratio")) else row.get("peg_ttm")
    ev_ebitda = row.get("ev_ebitda")
    earnings_yield = row.get("earnings_yield")
    debt_to_equity = row.get("debt_to_equity")
    promoter_holding_latest = row.get("promoter_holding_latest_pct")
    roce_5y = row.get("roce_5y_avg")
    profit_yoy = row.get("profit_yoy")
    sales_yoy = row.get("sales_yoy")
    profit_growth_5y = row.get("profit_growth_5y_pct")
    cfo_latest = row.get("cfo_latest")
    fcf_latest = row.get("fcf_latest")
    debtor_delta = row.get("debtor_days_delta_vs_3y")
    cash_conversion_cycle_days = row.get("cash_conversion_cycle_days")
    inventory_turnover_ratio = row.get("inventory_turnover_ratio")
    inventory_turnover_trend = row.get("inventory_turnover_trend")
    fii_qoq = row.get("fii_holding_change_qoq_pct")
    mf_qoq = row.get("mf_holding_change_qoq_pct")
    promoter_qoq = row.get("promoter_holding_change_qoq_pct")
    pledge_pct = row.get("promoter_holding_pledge_percentage_qtr_pct")
    volume_ratio_month = row.get("volume_ratio_month")
    volume_surge_strength = row.get("volume_surge_strength")
    rr_nifty50_week = row.get("rr_nifty50_week_pct")
    rr_nifty50_month = row.get("rr_nifty50_month_pct")
    rr_nifty50_quarter = row.get("rr_nifty50_quarter_pct")
    rr_nifty50_year = row.get("rr_nifty50_year_pct")
    rr_nifty50_3year = row.get("rr_nifty50_3year_pct")
    rr_nifty50_5year = row.get("rr_nifty50_5year_pct")
    rr_nifty50_10year = row.get("rr_nifty50_10year_pct")
    rr_sector_3year = row.get("rr_sector_3year_pct")
    rr_sector_5year = row.get("rr_sector_5year_pct")
    rr_sector_10year = row.get("rr_sector_10year_pct")
    rr_industry_3year = row.get("rr_industry_3year_pct")
    rr_industry_5year = row.get("rr_industry_5year_pct")
    rr_industry_10year = row.get("rr_industry_10year_pct")
    trendlyne_durability_score = row.get("trendlyne_durability_score")
    trendlyne_valuation_score = row.get("trendlyne_valuation_score")
    dvm_classification_text = _get_text(row, "dvm_classification_text", "")
    day_ema12 = row.get("day_ema12")
    day_ema20 = row.get("day_ema20")
    day_ema50 = row.get("day_ema50")
    day_ema100 = row.get("day_ema100")
    day_atr = row.get("day_atr")
    day_mfi = row.get("day_mfi")
    day_roc21 = row.get("day_roc21")
    day_roc125 = row.get("day_roc125")
    normalized_momentum_score = row.get("normalized_momentum_score")
    price_to_book_value_adjusted = row.get("price_to_book_value_adjusted")
    sector_price_to_book_ttm = row.get("sector_price_to_book_ttm")
    industry_price_to_book_ttm = row.get("industry_price_to_book_ttm")
    price_vs_sma200 = row.get("price_vs_sma200_pct")
    opm_current = row.get("opm_current")
    opm_last_year = row.get("opm_last_year")
    day_adx = row.get("day_adx")
    distance_52w = row.get("distance_from_52w_high_pct")
    room_to_month_high = row.get("room_to_month_high_pct")
    piotroski_score = row.get("piotroski_score")
    operating_leverage = row.get("operating_leverage")
    debt_trend_flag = _get_text(row, "debt_trend_flag", "")
    trend_structure_stage = _get_text(row, "trend_structure_stage", "")
    trend_extension_bucket = _get_text(row, "trend_extension_bucket", "")
    volume_conviction_bucket = _get_text(row, "volume_conviction_bucket", "")
    is_financial_like = int(_get_numeric(row, "is_financial_like", 0)) == 1
    compounder_setup = int(_get_numeric(row, "compounder_setup_flag", 0)) == 1
    earnings_reacceleration = int(_get_numeric(row, "earnings_reacceleration_flag", 0)) == 1
    cashflow_alignment = int(_get_numeric(row, "cashflow_alignment_flag", 0)) == 1
    strong_piotroski = int(_get_numeric(row, "strong_piotroski_flag", 0)) == 1
    weak_piotroski = int(_get_numeric(row, "weak_piotroski_flag", 0)) == 1
    strong_sponsorship = int(_get_numeric(row, "strong_sponsorship_flag", 0)) == 1 or _flagged(positive_flags, "Strong sponsorship")
    value_support = int(_get_numeric(row, "value_support_flag", 0)) == 1
    deep_value_support = int(_get_numeric(row, "deep_value_support_flag", 0)) == 1
    day_rsi = row.get("day_rsi")
    sma50 = _get_numeric(row, "day_sma50", 0)
    sma200 = _get_numeric(row, "day_sma200", 0)
    day_high = _get_numeric(row, "day_high", 0)
    day_low = _get_numeric(row, "day_low", 0)
    month_high = _get_numeric(row, "month_high", 0)
    month_low = _get_numeric(row, "month_low", 0)
    pivot = _get_numeric(row, "standard_pivot_point", 0)
    pivot_r1 = _get_numeric(row, "standard_resistance_r1", 0)
    pivot_r2 = _get_numeric(row, "standard_resistance_r2", 0)
    pivot_s1 = _get_numeric(row, "standard_support_s1", 0)
    pivot_s2 = _get_numeric(row, "standard_support_s2", 0)
    pivot_s3 = _get_numeric(row, "standard_support_s3", 0)
    sales_q_latest = row.get("sales_q_latest")
    sales_q_prev = row.get("sales_q_prev")
    sales_q_yoy_base = row.get("sales_q_yoy_base")
    profit_q_latest = row.get("profit_q_latest")
    profit_q_prev = row.get("profit_q_prev")
    profit_q_yoy_base = row.get("profit_q_yoy_base")
    eps_q_latest = row.get("eps_q_latest")
    eps_q_prev = row.get("eps_q_prev")
    eps_q_yoy_base = row.get("eps_q_yoy_base")
    debt_latest = row.get("debt_latest")
    debt_prev = row.get("debt_prev")
    debt_3y_back = row.get("debt_3y_back")
    days_receivable = row.get("days_receivable_outstanding")
    days_inventory = row.get("days_inventory_outstanding")
    days_payable = row.get("days_payable_outstanding")
    avg_working_capital_days_3y = row.get("avg_working_capital_days_3y")
    operating_revenue_annual = row.get("operating_revenue_annual")
    net_profit_annual = row.get("net_profit_annual")
    cash_from_operating_activity_annual = row.get("cash_from_operating_activity_annual")
    net_cash_flow_annual = row.get("net_cash_flow_annual")
    cash_from_investing_activity_annual = row.get("cash_from_investing_activity_annual")
    cash_from_financing_annual_activity = row.get("cash_from_financing_annual_activity")
    revenue_growth_annual_yoy = row.get("revenue_growth_annual_yoy_pct")
    net_profit_annual_yoy_growth = row.get("net_profit_annual_yoy_growth_pct")
    latest_financial_result = row.get("latest_financial_result")
    result_announced_date = row.get("result_announced_date")

    verdict_label = "Watchlist Only"
    if red_count >= pos_count + 1 or setup_tag == "Weak Confirmation":
        verdict_label = "Avoid for Now"
    elif has_valuation and has_weak_cash and entry_tag == "Watch on Pullback":
        verdict_label = "Watchlist Only"
    elif best_horizon == "Long Term":
        if has_earnings_decel and (below_sma200 or has_no_volume):
            verdict_label = "Long-Term Accumulate"
        elif entry_tag == "Watch on Pullback":
            verdict_label = "Buy on Pullback"
        else:
            verdict_label = "Long-Term Accumulate"
    elif best_horizon in ["Swing", "Short Term"]:
        if entry_tag == "Watch on Pullback":
            verdict_label = "Buy on Pullback"
        elif has_valuation and has_weak_cash:
            verdict_label = "Tactical Buy"
        elif setup_tag == "Strong Confirmation" and entry_tag == "Clean Entry" and pos_count > red_count:
            verdict_label = "Tactical Buy"
        else:
            verdict_label = "Watchlist Only"

    if has_valuation and has_weak_cash:
        key_risk = "Momentum reversal from stretched valuation without cash-flow support."
        what_must_improve = "Cash flow generation must improve materially to justify the current valuation."
    elif has_valuation and best_horizon in ["Swing", "Short Term"]:
        key_risk = "Valuation is stretched, so even a small loss of momentum can trigger a sharp de-rating."
        what_must_improve = "Either valuation must cool off or earnings growth must accelerate further to justify the premium."
    elif has_weak_cash:
        key_risk = "Weak cash conversion undermines the quality of reported earnings and raises re-rating risk."
        what_must_improve = "Cash flow from operations and free cash flow need to recover and align with profit growth."
    elif working_cap_stress and not is_financial_like:
        key_risk = "Working-capital stress can erode the quality of earnings and tighten re-rating potential."
        what_must_improve = "Cash conversion cycle, debtor days, and inventory turnover need to stabilise or improve."
    elif has_no_volume:
        key_risk = "The move lacks strong volume confirmation, which reduces conviction in follow-through."
        what_must_improve = "Volume participation must improve meaningfully to validate the current setup."
    elif has_earnings_decel and below_sma200:
        key_risk = "Earnings deceleration combined with weak long-term structure can delay or fail the trend reversal."
        what_must_improve = "Sales and profit growth need to re-accelerate, and price must reclaim and hold above SMA200."
    elif has_earnings_decel:
        key_risk = "Earnings deceleration may weaken conviction and delay a stronger re-rating."
        what_must_improve = "Sales and profit growth need to re-accelerate over the next few quarters."
    elif has_falling_inst:
        key_risk = "Falling institutional support can limit follow-through and weaken sentiment."
        what_must_improve = "Institutional participation needs to stabilize or improve."
    elif has_pledge:
        key_risk = "Promoter pledge risk adds balance-sheet and sentiment pressure."
        what_must_improve = "Pledge levels need to reduce and ownership quality must improve."
    elif below_sma200:
        key_risk = "The stock still has weak long-term price structure despite improving near-term action."
        what_must_improve = "Price must reclaim and hold above SMA200 with confirmation from trend and volume."
    else:
        key_risk = "The setup still depends on continued price, volume, and earnings execution."
        what_must_improve = "Current strength must continue with confirmation from price, volume, and earnings."

    if confidence_score >= 90 and (has_valuation or has_weak_cash or has_earnings_decel or has_no_volume or entry_tag == "Watch on Pullback"):
        confidence_level_text = f"Data completeness is high and analysis confidence score is {confidence_score:.0f}, but conviction is conditional because meaningful contradictions are present."
    elif confidence_score >= 85:
        confidence_level_text = f"Data completeness is high and analysis confidence score is {confidence_score:.0f}. Conviction is high, supported by the balance of positive over negative signals."
    elif confidence_score >= 70:
        confidence_level_text = f"Analysis confidence score is {confidence_score:.0f}. Conviction is moderate and depends on the setup continuing to confirm."
    else:
        confidence_level_text = f"Analysis confidence score is {confidence_score:.0f}. Conviction is limited and the setup should be treated cautiously."

    quarter_sales_text = ""
    if not _is_missing(sales_q_latest) and not _is_missing(sales_q_yoy_base):
        quarter_sales_text = f"latest quarterly sales were ₹{_fmt_num(sales_q_latest)} Cr versus ₹{_fmt_num(sales_q_yoy_base)} Cr a year ago"
        if not _is_missing(sales_q_prev):
            quarter_sales_text += f" and ₹{_fmt_num(sales_q_prev)} Cr in the previous quarter"
    quarter_profit_text = ""
    if not _is_missing(profit_q_latest) and not _is_missing(profit_q_yoy_base):
        quarter_profit_text = f"latest quarterly profit was ₹{_fmt_num(profit_q_latest)} Cr versus ₹{_fmt_num(profit_q_yoy_base)} Cr a year ago"
        if not _is_missing(profit_q_prev):
            quarter_profit_text += f" and ₹{_fmt_num(profit_q_prev)} Cr in the previous quarter"
    debt_path_text = ""
    if not _is_missing(debt_latest):
        debt_path_text = f"debt stands at ₹{_fmt_num(debt_latest)} Cr"
        if not _is_missing(debt_prev):
            debt_path_text += f" versus ₹{_fmt_num(debt_prev)} Cr a year ago"
        if not _is_missing(debt_3y_back):
            debt_path_text += f" and ₹{_fmt_num(debt_3y_back)} Cr three years ago"
    wc_driver_parts = []
    if not _is_missing(days_receivable):
        wc_driver_parts.append(f"receivable days at {_fmt_num(days_receivable, 1)}")
    if not _is_missing(days_inventory):
        wc_driver_parts.append(f"inventory days at {_fmt_num(days_inventory, 1)}")
    if not _is_missing(days_payable):
        wc_driver_parts.append(f"payable days at {_fmt_num(days_payable, 1)}")
    wc_driver_text = ", ".join(wc_driver_parts)
    roe_annual = _get_numeric(row, "roe_annual_pct", np.nan)
    roa_annual = _get_numeric(row, "roa_annual_pct", np.nan)
    basic_eps_ttm = _get_numeric(row, "basic_eps_ttm", np.nan)
    annual_profitability_parts = []
    if not _is_missing(roe_annual):
        annual_profitability_parts.append(f"annual ROE at {_fmt_pct(roe_annual)}")
    if not _is_missing(roa_annual):
        annual_profitability_parts.append(f"annual ROA at {_fmt_pct(roa_annual)}")
    if not _is_missing(basic_eps_ttm):
        annual_profitability_parts.append(f"basic EPS (TTM) at ₹{_fmt_num(basic_eps_ttm)}")
    annual_profitability_text = ", ".join(annual_profitability_parts)
    annual_scale_parts = []
    if not _is_missing(operating_revenue_annual):
        annual_scale_parts.append(f"annual revenue at ₹{_fmt_num(operating_revenue_annual)} Cr")
    if not _is_missing(net_profit_annual):
        annual_scale_parts.append(f"annual net profit at ₹{_fmt_num(net_profit_annual)} Cr")
    if not _is_missing(cash_from_operating_activity_annual):
        annual_scale_parts.append(f"annual operating cash flow at ₹{_fmt_num(cash_from_operating_activity_annual)} Cr")
    if not _is_missing(net_cash_flow_annual) and abs(_get_numeric(row, "net_cash_flow_annual", 0)) >= 1:
        annual_scale_parts.append(f"net annual cash flow at ₹{_fmt_num(net_cash_flow_annual)} Cr")
    annual_scale_text = ", ".join(annual_scale_parts)
    annual_growth_text = ""
    if not _is_missing(revenue_growth_annual_yoy) or not _is_missing(net_profit_annual_yoy_growth):
        ag_parts = []
        if not _is_missing(revenue_growth_annual_yoy):
            ag_parts.append(f"annual revenue growth at {_fmt_pct(revenue_growth_annual_yoy)}")
        if not _is_missing(net_profit_annual_yoy_growth):
            ag_parts.append(f"annual profit growth at {_fmt_pct(net_profit_annual_yoy_growth)}")
        annual_growth_text = ", ".join(ag_parts)
    result_freshness_text = ""
    if not _is_missing(latest_financial_result) or not _is_missing(result_announced_date):
        freshness_parts = []
        if not _is_missing(latest_financial_result):
            freshness_parts.append(f"latest reported period {str(latest_financial_result)}")
        if not _is_missing(result_announced_date):
            freshness_parts.append(f"announced on {_fmt_date(result_announced_date)}")
        result_freshness_text = ", ".join(freshness_parts)
    if result_freshness_text:
        confidence_level_text += f" The operating evidence is anchored to {result_freshness_text}."

    long_horizon_nifty_pairs = _period_label_pairs({
        "3-year": rr_nifty50_3year,
        "5-year": rr_nifty50_5year,
        "10-year": rr_nifty50_10year,
    })
    long_horizon_sector_pairs = _period_label_pairs({
        "3-year": rr_sector_3year,
        "5-year": rr_sector_5year,
        "10-year": rr_sector_10year,
    })
    long_horizon_industry_pairs = _period_label_pairs({
        "3-year": rr_industry_3year,
        "5-year": rr_industry_5year,
        "10-year": rr_industry_10year,
    })
    nifty_rs_positive = _pick_relative_strength_periods(long_horizon_nifty_pairs, positive=True)
    nifty_rs_negative = _pick_relative_strength_periods(long_horizon_nifty_pairs, positive=False)
    sector_rs_negative = _pick_relative_strength_periods(long_horizon_sector_pairs, positive=False)
    industry_rs_negative = _pick_relative_strength_periods(long_horizon_industry_pairs, positive=False)
    long_horizon_rs_strength_text = ""
    long_horizon_rs_weak_text = ""
    long_horizon_rs_mixed_text = ""
    if len(nifty_rs_positive) >= 2:
        rs_parts = [f"{label.lower()} outperformance vs Nifty 50 at {_fmt_signed_pct(value)}" for label, value in nifty_rs_positive]
        long_horizon_rs_strength_text = "Long-horizon market-relative strength is supportive, with " + " and ".join(rs_parts) + "."
        mixed_elems = []
        if len(sector_rs_negative) >= 2:
            mixed_elems.append("sector")
        if len(industry_rs_negative) >= 2:
            mixed_elems.append("industry")
        if mixed_elems:
            long_horizon_rs_mixed_text = "Long-cycle outperformance is stronger versus the broad market than versus its own " + " and ".join(mixed_elems) + "."
    elif len(nifty_rs_negative) >= 2:
        rs_parts = [f"{label.lower()} underperformance vs Nifty 50 at {_fmt_pct(abs(value))}" for label, value in nifty_rs_negative]
        long_horizon_rs_weak_text = "Long-horizon market-relative strength is weak, with " + " and ".join(rs_parts) + "."

    year_5_high = _get_numeric(row, "year_5_high", np.nan)
    year_5_low = _get_numeric(row, "year_5_low", np.nan)
    year_10_high = _get_numeric(row, "year_10_high", np.nan)
    year_10_low = _get_numeric(row, "year_10_low", np.nan)
    year_1_change = _get_numeric(row, "year_1_change_pct", np.nan)
    year_2_change = _get_numeric(row, "year_2_price_change_pct", np.nan)
    year_3_change = _get_numeric(row, "year_3_price_change_pct", np.nan)
    year_5_change = _get_numeric(row, "year_5_price_change_pct", np.nan)
    year_10_change = _get_numeric(row, "year_10_price_change_pct", np.nan)

    long_price_range_text = ""
    range_parts = []
    if not pd.isna(year_5_low) and not pd.isna(year_5_high) and year_5_high > 0:
        range_parts.append(f"the 5-year range spans ₹{_fmt_num(year_5_low)} to ₹{_fmt_num(year_5_high)}")
    if not pd.isna(year_10_low) and not pd.isna(year_10_high) and year_10_high > 0:
        range_parts.append(f"the 10-year range spans ₹{_fmt_num(year_10_low)} to ₹{_fmt_num(year_10_high)}")
    if range_parts:
        long_price_range_text = " and ".join(range_parts)

    price_change_parts = []
    for label, value in [("1-year", year_1_change), ("2-year", year_2_change), ("3-year", year_3_change), ("5-year", year_5_change), ("10-year", year_10_change)]:
        if not pd.isna(value):
            price_change_parts.append(f"{label} return {_fmt_signed_pct(value)}")
    long_price_change_text = ", ".join(price_change_parts)

    long_high_confirm_text = ""
    long_high_weak_text = ""
    confirm_parts = []
    weak_parts = []
    if not pd.isna(current_price):
        if not pd.isna(year_5_high) and year_5_high > 0:
            dist5 = (current_price / year_5_high) - 1
            if dist5 >= -0.15:
                confirm_parts.append(f"only {_fmt_frac_pct(abs(dist5))} below the 5-year high of ₹{_fmt_num(year_5_high)}")
            elif dist5 <= -0.35:
                weak_parts.append(f"still {_fmt_frac_pct(abs(dist5))} below the 5-year high of ₹{_fmt_num(year_5_high)}")
        if not pd.isna(year_10_high) and year_10_high > 0:
            dist10 = (current_price / year_10_high) - 1
            if dist10 >= -0.15:
                confirm_parts.append(f"only {_fmt_frac_pct(abs(dist10))} below the 10-year high of ₹{_fmt_num(year_10_high)}")
            elif dist10 <= -0.35:
                weak_parts.append(f"still {_fmt_frac_pct(abs(dist10))} below the 10-year high of ₹{_fmt_num(year_10_high)}")
    if confirm_parts:
        long_high_confirm_text = "Long-cycle breakout context is constructive, with price " + " and ".join(confirm_parts) + "."
    if weak_parts:
        long_high_weak_text = "Long-cycle repair is incomplete, with the stock " + " and ".join(weak_parts) + "."

    durability_bucket = _trendlyne_durability_bucket(trendlyne_durability_score)
    valuation_bucket = _trendlyne_valuation_bucket(trendlyne_valuation_score)

    dvm_text_lower = dvm_classification_text.strip().lower()
    dvm_supportive = any(token in dvm_text_lower for token in ["strong performer", "under radar", "value stock", "turnaround potential"])
    dvm_weak = any(token in dvm_text_lower for token in ["weak stock", "momentum trap", "slowing down", "value trap", "expensive underperformer", "falling comet"])
    dvm_mixed = bool(dvm_text_lower) and not dvm_supportive and not dvm_weak

    ema_constructive = (
        (not _is_missing(day_ema12)) and (not _is_missing(day_ema20)) and (not _is_missing(day_ema50))
        and current_price > _get_numeric(row, "day_ema12", 0)
        and _get_numeric(row, "day_ema12", 0) >= _get_numeric(row, "day_ema20", 0)
        and _get_numeric(row, "day_ema20", 0) >= _get_numeric(row, "day_ema50", 0)
    )
    ema_fully_bullish = ema_constructive and (not _is_missing(day_ema100)) and current_price >= _get_numeric(row, "day_ema100", 0)
    ema_repair_incomplete = ema_constructive and (not _is_missing(day_ema100)) and current_price < _get_numeric(row, "day_ema100", 0)
    ema_weak = (
        (not _is_missing(day_ema20)) and (not _is_missing(day_ema50))
        and current_price < _get_numeric(row, "day_ema20", 0)
        and _get_numeric(row, "day_ema20", 0) < _get_numeric(row, "day_ema50", 0)
    )

    atr_pct = np.nan
    if current_price and not _is_missing(day_atr):
        try:
            atr_pct = (_get_numeric(row, "day_atr", 0) / current_price) * 100.0
        except Exception:
            atr_pct = np.nan

    mfi_supportive = (not _is_missing(day_mfi)) and 55 <= _get_numeric(row, "day_mfi", 0) <= 80
    mfi_overheated = (not _is_missing(day_mfi)) and _get_numeric(row, "day_mfi", 0) > 80
    mfi_weak = (not _is_missing(day_mfi)) and _get_numeric(row, "day_mfi", 100) < 40

    roc_dual_positive = (not _is_missing(day_roc21)) and (not _is_missing(day_roc125)) and _get_numeric(row, "day_roc21", 0) > 0 and _get_numeric(row, "day_roc125", 0) > 0
    roc_short_recovery = (not _is_missing(day_roc21)) and (not _is_missing(day_roc125)) and _get_numeric(row, "day_roc21", 0) > 0 and _get_numeric(row, "day_roc125", 0) < 0
    roc_both_negative = (not _is_missing(day_roc21)) and (not _is_missing(day_roc125)) and _get_numeric(row, "day_roc21", 0) < 0 and _get_numeric(row, "day_roc125", 0) < 0

    normalized_momentum_supportive = (not _is_missing(normalized_momentum_score)) and _get_numeric(row, "normalized_momentum_score", 0) >= 60
    normalized_momentum_weak = (not _is_missing(normalized_momentum_score)) and _get_numeric(row, "normalized_momentum_score", 100) < 40

    pb_support_text = ""
    pb_risk_text = ""
    if (not _is_missing(price_to_book_value_adjusted) and not _is_missing(sector_price_to_book_ttm)
        and _get_numeric(row, "price_to_book_value_adjusted", 9999) > 0 and _get_numeric(row, "sector_price_to_book_ttm", 0) > 0):
        pbv = _get_numeric(row, "price_to_book_value_adjusted", 0)
        sector_pb = _get_numeric(row, "sector_price_to_book_ttm", 0)
        industry_pb = _get_numeric(row, "industry_price_to_book_ttm", 0)
        comparator = []
        if sector_pb > 0:
            comparator.append(f"sector P/B at {_fmt_num(sector_price_to_book_ttm)}x")
        if industry_pb > 0:
            comparator.append(f"industry P/B at {_fmt_num(industry_price_to_book_ttm)}x")
        if comparator:
            if pbv <= sector_pb * 0.9 or (industry_pb > 0 and pbv <= industry_pb * 0.9):
                pb_support_text = f"price-to-book at {_fmt_num(price_to_book_value_adjusted)}x versus " + " and ".join(comparator)
            elif pbv >= sector_pb * 1.15 and (industry_pb <= 0 or pbv >= industry_pb * 1.15):
                pb_risk_text = f"price-to-book at {_fmt_num(price_to_book_value_adjusted)}x versus " + " and ".join(comparator)

    core_strengths: List[str] = []
    if is_financial_like:
        core_strengths.append("This is a financial-like business, so balance-sheet interpretation is sector-aware and inventory/CCC style checks are treated with caution.")
    if compounder_setup:
        core_strengths.append(
            f"Compounder setup is active: 5Y profit growth is {_fmt_pct(profit_growth_5y)} and 5Y ROCE is {_fmt_pct(roce_5y)}, with cash-flow alignment, ownership support, and a {trend_structure_stage or 'constructive'} trend structure."
        )
    if cashflow_alignment:
        cashflow_sentence = f"Cash-flow alignment signal is active, with CFO at ₹{_fmt_num(cfo_latest)} Cr and FCF at ₹{_fmt_num(fcf_latest)} Cr supporting the earnings profile."
        if annual_scale_text:
            cashflow_sentence += f" The annual business base is also visible, with {annual_scale_text}."
        core_strengths.append(cashflow_sentence)
    if _flagged(positive_flags, "High quality growth"):
        if not _is_missing(profit_yoy) and _get_numeric(row, "profit_yoy", 0) > 0:
            growth_sentence = f"Growth quality is supportive, with latest profit growth at {_fmt_frac_pct(profit_yoy)} and 5Y profit growth at {_fmt_pct(profit_growth_5y)}."
        else:
            growth_sentence = f"Historical growth quality remains strong, with 5Y profit growth at {_fmt_pct(profit_growth_5y)}, even though the latest quarter needs closer monitoring."
        if quarter_profit_text:
            growth_sentence += f" In raw terms, {quarter_profit_text}."
        core_strengths.append(growth_sentence)
    if not _is_missing(sales_yoy):
        if _get_numeric(row, "sales_yoy", 0) > 0:
            sales_sentence = f"Top-line growth is supportive, with sales YoY at {_fmt_frac_pct(sales_yoy)}."
            if quarter_sales_text:
                sales_sentence += f" In raw terms, {quarter_sales_text}."
            core_strengths.append(sales_sentence)
        elif _get_numeric(row, "profit_yoy", 0) > 0:
            caution_sentence = f"Profit is growing {_fmt_frac_pct(profit_yoy)}, but sales YoY is only {_fmt_frac_pct(sales_yoy)}, so the quality of the beat needs monitoring."
            if quarter_sales_text:
                caution_sentence += f" Quarterly sales context: {quarter_sales_text}."
            core_strengths.append(caution_sentence)
    if not _is_missing(opm_current) and not _is_missing(opm_last_year):
        opm_delta = _get_numeric(row, "opm_current", 0) - _get_numeric(row, "opm_last_year", 0)
        if opm_delta > 0:
            core_strengths.append(
                f"Operating margins are expanding: OPM improved to {_fmt_pct(opm_current)} from {_fmt_pct(opm_last_year)}, supporting operating leverage."
            )
    if not _is_missing(operating_leverage) and _get_numeric(row, "operating_leverage", 0) > 0:
        core_strengths.append(f"Operating leverage is positive, with profit growth running {_fmt_signed_frac_pct(operating_leverage)} above sales growth.")
    if annual_growth_text:
        core_strengths.append(f"Annual trend context is supportive, with {annual_growth_text}.")
    annual_metric_count = sum(
        1 for v in [operating_revenue_annual, net_profit_annual, cash_from_operating_activity_annual]
        if not _is_missing(v)
    ) + (1 if (not _is_missing(net_cash_flow_annual) and abs(_get_numeric(row, "net_cash_flow_annual", 0)) >= 1) else 0)
    annual_scale_should_surface = bool(annual_scale_text) and (
        annual_metric_count >= 3
        or bool(annual_growth_text)
        or cashflow_alignment
        or _get_numeric(row, "cash_from_operating_activity_annual", 0) > 0
        or abs(_get_numeric(row, "net_cash_flow_annual", 0)) >= 1
    )
    if annual_scale_should_surface and not cashflow_alignment:
        annual_sentence = f"Annual business scale adds durability, with {annual_scale_text}."
        if annual_growth_text:
            annual_sentence += f" This is supported by {annual_growth_text}."
        core_strengths.append(annual_sentence)
    annual_profitability_strong = False
    annual_profitability_weak = False
    if not _is_missing(roe_annual) and _get_numeric(row, "roe_annual_pct", 0) >= 15:
        annual_profitability_strong = True
    if not _is_missing(roa_annual):
        roa_val = _get_numeric(row, "roa_annual_pct", 0)
        if (is_financial_like and roa_val >= 1.0) or ((not is_financial_like) and roa_val >= 5.0):
            annual_profitability_strong = True
        if (is_financial_like and roa_val < 0.8) or ((not is_financial_like) and roa_val < 3.0):
            annual_profitability_weak = True
    if not _is_missing(roe_annual) and _get_numeric(row, "roe_annual_pct", 999) < 10:
        annual_profitability_weak = True
    if not _is_missing(basic_eps_ttm) and _get_numeric(row, "basic_eps_ttm", 1) <= 0:
        annual_profitability_weak = True
    if annual_profitability_text and annual_profitability_strong:
        core_strengths.append("Annual profitability context is supportive, with " + annual_profitability_text + ".")
    if long_horizon_rs_strength_text:
        core_strengths.append(long_horizon_rs_strength_text)
    if long_price_range_text:
        core_strengths.append(f"Long price-history context adds perspective: {long_price_range_text}.")
    if long_price_change_text:
        core_strengths.append(f"Long-cycle price performance remains visible, with {long_price_change_text}.")
    if durability_bucket == "strong" and valuation_bucket == "supportive":
        core_strengths.append(f"Trendlyne regime is supportive, with durability score {_fmt_num(trendlyne_durability_score, 0)} and valuation score {_fmt_num(trendlyne_valuation_score, 0)}, indicating durable fundamentals that are still competitively priced.")
    elif durability_bucket == "strong" and valuation_bucket == "neutral":
        core_strengths.append(f"Trendlyne durability score is strong at {_fmt_num(trendlyne_durability_score, 0)}, which supports long-cycle quality even though the valuation score of {_fmt_num(trendlyne_valuation_score, 0)} is only neutral.")
    elif durability_bucket == "neutral" and valuation_bucket == "supportive":
        core_strengths.append(f"Trendlyne valuation score is supportive at {_fmt_num(trendlyne_valuation_score, 0)}, indicating the stock is not fully priced for its current business profile.")
    if dvm_supportive:
        core_strengths.append(f"DVM classification is supportive at '{dvm_classification_text}', which reinforces the idea that the stock is behaving like a constructive trend rather than a weak bounce.")
    forensic_value_parts: List[str] = []
    if not _is_missing(pe_ttm) and not _is_missing(pe_5y) and _get_numeric(row, "pe_ttm", 999) < _get_numeric(row, "pe_5yr_average", 999):
        forensic_value_parts.append(f"PE at {_fmt_num(pe_ttm)}x versus 5Y average {_fmt_num(pe_5y)}x")
    if not _is_missing(sector_pe) and not _is_missing(pe_ttm) and _get_numeric(row, "pe_ttm", 999) < _get_numeric(row, "sector_pe_ttm", 999):
        forensic_value_parts.append(f"sector PE {_fmt_num(sector_pe)}x")
    if not _is_missing(peg_ratio) and _get_numeric(row, "peg_ratio", 999) <= 1.2:
        forensic_value_parts.append(f"PEG {_fmt_num(peg_ratio)}")
    if not _is_missing(earnings_yield) and _get_numeric(row, "earnings_yield", 0) >= 4:
        forensic_value_parts.append(f"earnings yield {_fmt_num(earnings_yield)}%")
    if not _is_missing(ev_ebitda) and _get_numeric(row, "ev_ebitda", 999) <= 12:
        forensic_value_parts.append(f"EV/EBITDA {_fmt_num(ev_ebitda)}x")
    if pb_support_text:
        forensic_value_parts.append(pb_support_text)
    if forensic_value_parts and (
        deep_value_support
        or value_support
        or len(forensic_value_parts) >= 2
        or (not _is_missing(earnings_yield) and _get_numeric(row, "earnings_yield", 0) >= 4)
        or (not _is_missing(ev_ebitda) and _get_numeric(row, "ev_ebitda", 999) <= 12)
    ):
        prefix = "Deep value support signal is active" if deep_value_support else ("Value support signal is active" if value_support else "Forensic valuation is supportive")
        core_strengths.append(f"{prefix}, with " + ", ".join(forensic_value_parts) + ".")
    if cashflow_alignment or (not _is_missing(cfo_latest) and _get_numeric(row, "cfo_latest", 0) > 0 and not _is_missing(fcf_latest) and _get_numeric(row, "fcf_latest", 0) > 0):
        core_strengths.append(f"Cash-flow profile is supportive, with CFO at ₹{_fmt_num(cfo_latest)} Cr and FCF at ₹{_fmt_num(fcf_latest)} Cr.")
    if debt_trend_flag == "Deleveraging":
        if not _is_missing(debt_to_equity) and _get_numeric(row, "debt_to_equity", 999) <= 0.5:
            debt_sentence = f"Deleveraging signal is active, and debt-to-equity is already low at {_fmt_num(debt_to_equity)}, which strengthens an already safe balance sheet."
        else:
            debt_sentence = f"Deleveraging signal is active, with debt-to-equity at {_fmt_num(debt_to_equity)} and improving balance-sheet flexibility."
        if debt_path_text:
            debt_sentence += f" Raw debt history shows that {debt_path_text}."
        core_strengths.append(debt_sentence)
    if strong_piotroski or (_get_numeric(row, "piotroski_score", 0) >= 7):
        core_strengths.append(f"Piotroski score is strong at {_fmt_num(piotroski_score, 0)}, supporting balance-sheet and operating quality.")
    working_cap_positive_parts: List[str] = []
    if not is_financial_like:
        if not _is_missing(cash_conversion_cycle_days) and 0 < _get_numeric(row, "cash_conversion_cycle_days", 999) <= 45:
            working_cap_positive_parts.append(f"cash conversion cycle at {_fmt_num(cash_conversion_cycle_days)} days")
        if not _is_missing(debtor_delta) and _get_numeric(row, "debtor_days_delta_vs_3y", 0) < 0:
            working_cap_positive_parts.append(f"debtor days improved by {_fmt_num(abs(_get_numeric(row, 'debtor_days_delta_vs_3y', 0)), 1)} days versus the 3-year average")
        if not _is_missing(inventory_turnover_trend) and _get_numeric(row, "inventory_turnover_trend", 0) > 0.2:
            working_cap_positive_parts.append(f"inventory turnover trend at {_fmt_num(inventory_turnover_trend)} versus historical averages")
    if working_cap_positive_parts:
        wc_sentence = "Working-capital quality is supportive, driven by " + ", ".join(working_cap_positive_parts) + "."
        if wc_driver_text and not is_financial_like:
            wc_sentence += f" The driver mix currently shows {wc_driver_text}."
        core_strengths.append(wc_sentence)
    ownership_parts_exact: List[str] = []
    if not _is_missing(promoter_holding_latest):
        ownership_parts_exact.append(f"promoter holding at {_fmt_pct(promoter_holding_latest)}")
    if not _is_missing(promoter_qoq):
        ownership_parts_exact.append(f"promoter change {_fmt_signed_pct(promoter_qoq)} QoQ")
    if not _is_missing(fii_qoq):
        ownership_parts_exact.append(f"FII change {_fmt_signed_pct(fii_qoq)} QoQ")
    if not _is_missing(mf_qoq):
        ownership_parts_exact.append(f"MF change {_fmt_signed_pct(mf_qoq)} QoQ")
    if not _is_missing(pledge_pct) and _get_numeric(row, "promoter_holding_pledge_percentage_qtr_pct", 0) > 0:
        ownership_parts_exact.append(f"pledge {_fmt_pct(pledge_pct)}")
    ownership_negative = (
        has_falling_inst
        or has_pledge
        or _get_numeric(row, "promoter_holding_change_qoq_pct", 0) < -0.05
        or (_get_numeric(row, "fii_holding_change_qoq_pct", 0) + _get_numeric(row, "mf_holding_change_qoq_pct", 0)) < -0.25
        or (_get_numeric(row, "fii_holding_change_qoq_pct", 0) < -0.5 and _get_numeric(row, "mf_holding_change_qoq_pct", 0) <= 0)
        or (_get_numeric(row, "mf_holding_change_qoq_pct", 0) < -0.5 and _get_numeric(row, "fii_holding_change_qoq_pct", 0) <= 0)
    )
    ownership_positive = (
        (strong_sponsorship
        or _get_numeric(row, "promoter_holding_change_qoq_pct", 0) > 0.05
        or (_get_numeric(row, "fii_holding_change_qoq_pct", 0) + _get_numeric(row, "mf_holding_change_qoq_pct", 0)) > 0.25)
        and not ownership_negative
        and _get_numeric(row, "fii_holding_change_qoq_pct", 0) > -0.5
        and _get_numeric(row, "mf_holding_change_qoq_pct", 0) > -0.5
    )
    ownership_mixed = (
        not ownership_positive
        and not ownership_negative
        and (strong_sponsorship
             or abs(_get_numeric(row, "promoter_holding_change_qoq_pct", 0)) >= 0.05
             or abs(_get_numeric(row, "fii_holding_change_qoq_pct", 0)) >= 0.25
             or abs(_get_numeric(row, "mf_holding_change_qoq_pct", 0)) >= 0.25)
    )
    if ownership_positive and ownership_parts_exact:
        core_strengths.append("Exact ownership evidence is supportive, with " + ", ".join(ownership_parts_exact) + ".")
    elif ownership_parts_exact and not ownership_negative:
        ownership_label = "mixed but broadly stable" if ownership_mixed else "broadly stable"
        core_strengths.append("Exact ownership evidence is " + ownership_label + ", with " + ", ".join(ownership_parts_exact) + ".")
    if not core_strengths:
        core_strengths.append("Business quality and signal balance are supportive, but the setup is not yet distinguished by a single standout metric.")

    confirmations: List[str] = []
    if _get_numeric(row, "volume_confirmation_flag", 0) == 1 or _get_numeric(row, "volume_ratio_month", 0) >= 1:
        confirmations.append(
            f"Volume confirmation supports the move, with current volume at {_fmt_num(volume_ratio_month)}x the 1-month average, surge strength at {_fmt_num(volume_surge_strength)}x, and conviction tagged as '{volume_conviction_bucket or 'Available'}'."
        )
    # Patch RR-H.QA1: three-band Nifty 1-month RS wording.
    # > +2%   -> supportive (confirmations)
    # -2..+2% -> mixed       (confirmations)
    # < -2%   -> weak        (deferred until weaknesses list exists below)
    rs_1m_value = _get_numeric(row, "rr_nifty50_month_pct", 0.0)
    rs_1m_present = (
        _get_numeric(row, "relative_strength_confirmation_flag", 0) == 1
        or rs_1m_value != 0.0
    )
    weak_rs_1m_sentence: Optional[str] = None
    if rs_1m_present:
        if rs_1m_value > _RS_NEUTRAL_BAND_PCT:
            confirmations.append(
                f"Relative strength versus Nifty 50 is supportive at {_fmt_signed_pct(rs_1m_value)} over the last month."
            )
        elif rs_1m_value < -_RS_NEUTRAL_BAND_PCT:
            weak_rs_1m_sentence = (
                f"Near-term relative strength versus Nifty 50 is weak at {_fmt_signed_pct(rs_1m_value)} over the last month."
            )
        else:
            confirmations.append(
                f"Near-term relative strength versus Nifty 50 is mixed at {_fmt_signed_pct(rs_1m_value)} over the last month."
            )
    if long_high_confirm_text:
        confirmations.append(long_high_confirm_text)
    if long_horizon_rs_mixed_text:
        confirmations.append(long_horizon_rs_mixed_text)
    if _get_numeric(row, "momentum_acceleration_flag", 0) == 1:
        confirmations.append("Momentum is accelerating versus its recent history.")
    if not _is_missing(day_adx) and _get_numeric(row, "day_adx", 0) >= 20:
        confirmations.append(f"Trend strength is genuine rather than sideways noise, with ADX at {_fmt_num(day_adx)}.")
    if trend_extension_bucket in {"Early", "Healthy"}:
        confirmations.append(f"Trend extension is currently tagged as '{trend_extension_bucket}', which is constructive rather than overextended.")
    if ema_fully_bullish:
        confirmations.append(f"EMA structure is fully constructive, with price above the 12-day, 20-day, 50-day, and 100-day EMAs.")
    elif ema_repair_incomplete:
        confirmations.append(f"EMA structure is constructive, with price above the 12-day, 20-day, and 50-day EMAs, though the 100-day EMA at ₹{_fmt_num(day_ema100)} remains the next structural hurdle.")
    elif ema_constructive:
        confirmations.append(f"EMA structure is constructive, with price holding above the 12-day, 20-day, and 50-day EMA cluster.")
    if normalized_momentum_supportive:
        confirmations.append(f"Normalized momentum score is supportive at {_fmt_num(normalized_momentum_score, 0)}, reinforcing trend quality.")
    if roc_dual_positive:
        confirmations.append(f"Rate-of-change readings are supportive, with 21-day ROC at {_fmt_signed_pct(day_roc21)} and 125-day ROC at {_fmt_signed_pct(day_roc125)}.")
    elif roc_short_recovery:
        confirmations.append(f"Short-term recovery is visible, with 21-day ROC at {_fmt_signed_pct(day_roc21)}, even though the 125-day ROC remains {_fmt_signed_pct(day_roc125)}.")
    if mfi_supportive:
        confirmations.append(f"Money flow is supportive, with MFI at {_fmt_num(day_mfi, 1)}, indicating healthy buying pressure without obvious exhaustion.")
    if not np.isnan(atr_pct) and atr_pct <= 4.5:
        confirmations.append(f"Volatility remains manageable, with ATR at ₹{_fmt_num(day_atr)} or about {_fmt_num(atr_pct,1)}% of price.")
    if dvm_supportive:
        confirmations.append(f"DVM regime classification is '{dvm_classification_text}', which is consistent with a constructive momentum profile.")
    elif dvm_mixed and dvm_classification_text:
        confirmations.append(f"DVM regime classification is '{dvm_classification_text}', which is broadly constructive but not yet a top-tier momentum regime.")
    if not _is_missing(distance_52w) and _get_numeric(row, "distance_from_52w_high_pct", 0) >= -0.10:
        confirmations.append(f"The stock is trading only {_fmt_frac_pct(abs(_get_numeric(row, 'distance_from_52w_high_pct', 0)))} below its 52-week high, keeping breakout proximity relevant.")
    if not _is_missing(room_to_month_high) and _get_numeric(row, "room_to_month_high_pct", 0) <= 0.03:
        confirmations.append(f"Price is only {_fmt_frac_pct(room_to_month_high)} below the monthly high, so a breakout trigger is nearby.")
    if pivot > 0 and current_price >= pivot:
        confirmations.append(f"Price is holding above the daily pivot point at ₹{pivot:,.2f}, which supports near-term structure.")
    elif pivot > 0 and pivot_r1 > current_price > 0:
        confirmations.append(f"Price is trading just below pivot resistance R1 at ₹{pivot_r1:,.2f}, which makes the next confirmation level well-defined.")
    if earnings_reacceleration:
        confirmations.append(
            f"Earnings reacceleration signal is active: recent profit growth at {_fmt_frac_pct(profit_yoy)} is running above the longer-term growth baseline, with margins holding or improving."
        )
    if deep_value_support:
        confirmations.append("Deep value support signal is active, adding a stronger margin-of-safety layer to the setup.")
    elif value_support:
        confirmations.append("Value support signal is active, so valuation support is helping the setup rather than fighting it.")
    if entry_tag:
        # Patch RR-H.QA1: when entry_quality_tag is "Insufficient Data",
        # surface an explicit, actionable caution rather than the bare
        # tag echo. Keeps reports for these stocks (RR-G.1: caution-only,
        # not blocker) while making the data gap visible to the reader.
        if entry_tag == "Insufficient Data":
            confirmations.append(
                "Entry quality is tagged as 'Insufficient Data'. "
                "Entry timing data is incomplete because 200-DMA or related "
                "long-horizon technical validation is unavailable. "
                "This is not a full-data failure. Confirm entry manually using "
                "available shorter moving averages, price structure, pivot levels, "
                "and risk levels."
            )
        else:
            confirmations.append(f"Entry quality is tagged as '{entry_tag}'.")
    if setup_tag:
        confirmations.append(f"Setup confirmation is tagged as '{setup_tag}'.")
    if not confirmations:
        confirmations.append("The setup has limited confirmation and still needs stronger participation from price, volume, or earnings.")

    weaknesses: List[str] = []
    # Patch RR-H.QA1: re-route the deferred "weak Nifty RS" sentence
    # (rr_nifty50_month_pct < -2%) into the weaknesses list. This was
    # computed inside the confirmations block above so the sign-correct
    # wording lands in the right report section.
    if weak_rs_1m_sentence is not None:
        weaknesses.append(weak_rs_1m_sentence)
    if has_valuation:
        weakness_parts = [f"PE is {_fmt_num(pe_ttm)}x"]
        if not _is_missing(pe_5y):
            weakness_parts.append(f"5Y average PE is {_fmt_num(pe_5y)}x")
        if not _is_missing(sector_pe):
            weakness_parts.append(f"sector PE is {_fmt_num(sector_pe)}x")
        if not _is_missing(peg_ratio):
            weakness_parts.append(f"PEG is {_fmt_num(peg_ratio)}")
        if not _is_missing(earnings_yield) and _get_numeric(row, "earnings_yield", 99) < 3:
            weakness_parts.append(f"earnings yield is only {_fmt_num(earnings_yield)}%")
        if not _is_missing(ev_ebitda) and _get_numeric(row, "ev_ebitda", 0) > 18:
            weakness_parts.append(f"EV/EBITDA is {_fmt_num(ev_ebitda)}x")
        weaknesses.append("Valuation is stretched: " + ", ".join(weakness_parts) + ".")
    elif (
        (not _is_missing(earnings_yield) and _get_numeric(row, "earnings_yield", 99) < 3)
        or (not _is_missing(ev_ebitda) and _get_numeric(row, "ev_ebitda", 0) > 18)
        or bool(pb_risk_text)
    ):
        forensic_risk_parts: List[str] = []
        if not _is_missing(earnings_yield) and _get_numeric(row, "earnings_yield", 99) < 3:
            forensic_risk_parts.append(f"earnings yield is only {_fmt_num(earnings_yield)}%")
        if not _is_missing(ev_ebitda) and _get_numeric(row, "ev_ebitda", 0) > 18:
            forensic_risk_parts.append(f"EV/EBITDA is {_fmt_num(ev_ebitda)}x")
        if pb_risk_text:
            forensic_risk_parts.append(pb_risk_text)
        weaknesses.append("Forensic valuation still looks demanding even if some simpler valuation lenses appear reasonable, with " + ", ".join(forensic_risk_parts) + ".")
    if durability_bucket == "weak":
        weaknesses.append(f"Trendlyne durability score is weak at {_fmt_num(trendlyne_durability_score, 0)}, which reduces conviction on long-cycle revenue, cash-flow, and balance-sheet consistency.")
    elif valuation_bucket == "demanding":
        weaknesses.append(f"Trendlyne valuation score is weak at {_fmt_num(trendlyne_valuation_score, 0)}, which suggests the stock still looks demanding on that regime lens despite any simpler history-based valuation support.")
    if long_horizon_rs_weak_text:
        weaknesses.append(long_horizon_rs_weak_text)
    if long_high_weak_text:
        weaknesses.append(long_high_weak_text)
    if has_weak_cash:
        cash_weak_sentence = f"Cash conversion is weak, with CFO at ₹{_fmt_num(cfo_latest)} Cr and FCF at ₹{_fmt_num(fcf_latest)} Cr."
        if not _is_missing(cash_from_operating_activity_annual) or not _is_missing(net_cash_flow_annual):
            annual_cash_parts = []
            if not _is_missing(cash_from_operating_activity_annual):
                annual_cash_parts.append(f"annual CFO at ₹{_fmt_num(cash_from_operating_activity_annual)} Cr")
            if not _is_missing(net_cash_flow_annual):
                annual_cash_parts.append(f"net annual cash flow at ₹{_fmt_num(net_cash_flow_annual)} Cr")
            cash_weak_sentence += " Annual cash context is " + ", ".join(annual_cash_parts) + "."
        weaknesses.append(cash_weak_sentence)
    if has_earnings_decel:
        earnings_sentence = f"Recent earnings momentum has decelerated, with sales YoY at {_fmt_frac_pct(sales_yoy)} and profit YoY at {_fmt_frac_pct(profit_yoy)}."
        raw_parts = []
        if quarter_sales_text:
            raw_parts.append(quarter_sales_text)
        if quarter_profit_text:
            raw_parts.append(quarter_profit_text)
        if raw_parts:
            earnings_sentence += " Raw quarterly bases show that " + " while ".join(raw_parts) + "."
        weaknesses.append(earnings_sentence)
    if not _is_missing(opm_current) and not _is_missing(opm_last_year):
        opm_delta = _get_numeric(row, "opm_current", 0) - _get_numeric(row, "opm_last_year", 0)
        if opm_delta < 0:
            weaknesses.append(f"Operating margin has contracted to {_fmt_pct(opm_current)} from {_fmt_pct(opm_last_year)}, which weakens quality of growth.")
    working_cap_negative_parts: List[str] = []
    if not is_financial_like:
        if not _is_missing(cash_conversion_cycle_days) and _get_numeric(row, "cash_conversion_cycle_days", 0) > 60:
            working_cap_negative_parts.append(f"cash conversion cycle at {_fmt_num(cash_conversion_cycle_days)} days")
        if not _is_missing(debtor_delta) and _get_numeric(row, "debtor_days_delta_vs_3y", 0) > 0:
            working_cap_negative_parts.append(f"debtor days higher by {_fmt_num(debtor_delta, 1)} days versus the 3-year average")
        if not _is_missing(inventory_turnover_trend) and (_get_numeric(row, "inventory_turnover_trend", 0) < 0 or weak_inventory_trend):
            working_cap_negative_parts.append(f"inventory turnover trend weaker by {_fmt_num(abs(_get_numeric(row, 'inventory_turnover_trend', 0)))} versus historical averages")
    if working_cap_negative_parts:
        wc_weak_sentence = "Working-capital pressure is being driven by " + ", ".join(working_cap_negative_parts) + "."
        if wc_driver_text and not is_financial_like:
            wc_weak_sentence += f" The underlying day-count mix is {wc_driver_text}."
            if not _is_missing(avg_working_capital_days_3y):
                wc_weak_sentence += f" Historical working-capital days baseline is {_fmt_num(avg_working_capital_days_3y, 1)}."
        weaknesses.append(wc_weak_sentence)
    if has_no_volume:
        weaknesses.append(f"Volume confirmation is weak or absent, with current volume only {_fmt_num(volume_ratio_month)}x the 1-month average and conviction tagged '{volume_conviction_bucket or 'Weak'}'.")
    if ema_weak:
        weaknesses.append(f"EMA structure is still weak, with price below the 20-day EMA and the 20-day EMA below the 50-day EMA, which suggests the repair is not yet complete.")
    elif ema_repair_incomplete:
        weaknesses.append(f"The EMA repair is incomplete because price is still below the 100-day EMA at ₹{_fmt_num(day_ema100)}, even though the shorter EMA structure has improved.")
    if mfi_overheated:
        weaknesses.append(f"Money flow looks crowded, with MFI at {_fmt_num(day_mfi, 1)}, which raises the risk of short-term exhaustion.")
    elif mfi_weak:
        weaknesses.append(f"Money flow remains soft, with MFI at {_fmt_num(day_mfi, 1)}, which suggests buying pressure is not yet fully convincing.")
    if normalized_momentum_weak:
        weaknesses.append(f"Normalized momentum score is weak at {_fmt_num(normalized_momentum_score, 0)}, which reduces conviction in trend durability.")
    if roc_both_negative:
        weaknesses.append(f"Rate-of-change readings are weak, with 21-day ROC at {_fmt_signed_pct(day_roc21)} and 125-day ROC at {_fmt_signed_pct(day_roc125)}, signalling pressure across both short and medium horizons.")
    elif roc_short_recovery:
        weaknesses.append(f"Momentum repair is still incomplete because 125-day ROC remains {_fmt_signed_pct(day_roc125)} even though 21-day ROC has recovered to {_fmt_signed_pct(day_roc21)}.")
    if dvm_weak:
        weaknesses.append(f"DVM classification is '{dvm_classification_text}', which is a weak regime label and lowers confidence that the move is structurally strong.")
    elif dvm_mixed and dvm_classification_text and "expensive" in dvm_text_lower:
        weaknesses.append(f"DVM classification is '{dvm_classification_text}', which suggests the stock is performing but already in a richer regime.")
    if not np.isnan(atr_pct) and atr_pct > 6:
        weaknesses.append(f"ATR is elevated at ₹{_fmt_num(day_atr)} or about {_fmt_num(atr_pct,1)}% of price, which implies higher execution risk and wider stop requirements.")
    if below_sma200:
        weaknesses.append(f"The stock remains {_fmt_frac_pct(abs(_get_numeric(row, 'price_vs_sma200_pct', 0)))} below SMA200, which weakens the long-term structure.")
    ownership_negative_parts: List[str] = []
    if not _is_missing(promoter_holding_latest):
        ownership_negative_parts.append(f"promoter holding at {_fmt_pct(promoter_holding_latest)}")
    if not _is_missing(promoter_qoq):
        ownership_negative_parts.append(f"promoter change {_fmt_signed_pct(promoter_qoq)} QoQ")
    if not _is_missing(fii_qoq):
        ownership_negative_parts.append(f"FII change {_fmt_signed_pct(fii_qoq)} QoQ")
    if not _is_missing(mf_qoq):
        ownership_negative_parts.append(f"MF change {_fmt_signed_pct(mf_qoq)} QoQ")
    if has_falling_inst and (_get_numeric(row, "fii_holding_change_qoq_pct", 0) < 0 or _get_numeric(row, "mf_holding_change_qoq_pct", 0) < 0):
        weaknesses.append("Exact ownership evidence is weaker, with " + ", ".join(ownership_negative_parts) + ".")
    elif ownership_mixed and ownership_negative_parts:
        weaknesses.append("Exact ownership evidence is mixed rather than one-way supportive, with " + ", ".join(ownership_negative_parts) + ".")
    elif not _is_missing(rr_nifty50_month) and _get_numeric(row, "rr_nifty50_month_pct", 0) < 0:
        weaknesses.append(f"Relative strength is weak, with the stock underperforming Nifty 50 by {_fmt_pct(abs(_get_numeric(row, 'rr_nifty50_month_pct', 0)))} over the last month.")
    if debt_trend_flag == "Leveraging Up":
        if not _is_missing(debt_to_equity) and _get_numeric(row, "debt_to_equity", 999) <= 0.5:
            debt_sentence = f"Debt is trending up, but from a low base: debt-to-equity remains only {_fmt_num(debt_to_equity)}. The direction needs monitoring more than the absolute leverage today."
        else:
            debt_sentence = f"Debt is trending up and debt-to-equity stands at {_fmt_num(debt_to_equity)}, which can limit balance-sheet flexibility if earnings do not accelerate."
        if debt_path_text:
            debt_sentence += f" Raw debt history indicates that {debt_path_text}."
        weaknesses.append(debt_sentence)
    if has_pledge and not _is_missing(pledge_pct):
        pledge_context = [part for part in ownership_negative_parts if not part.startswith("pledge ")]
        if pledge_context:
            weaknesses.append(f"Promoter pledge remains elevated at {_fmt_pct(pledge_pct)} of promoter holding. Exact ownership evidence around it shows " + ", ".join(pledge_context) + ".")
        else:
            weaknesses.append(f"Promoter pledge remains elevated at {_fmt_pct(pledge_pct)} of promoter holding.")
    if annual_profitability_text and annual_profitability_weak:
        weaknesses.append("Annual profitability context needs monitoring, with " + annual_profitability_text + ".")
    if weak_piotroski:
        weaknesses.append(f"Piotroski score is weak at {_fmt_num(piotroski_score, 0)}, which reduces conviction on balance-sheet quality.")
    if not weaknesses:
        weaknesses.append("No major structural weakness stands out beyond normal execution risk.")

    shortlist_horizon = _infer_shortlist_horizon(strategy)
    strategy_fit_text = (
        f"The stock is currently appearing in '{strategy}', while the strongest score profile points to a '{best_horizon}' holding horizon."
    )
    if shortlist_horizon and shortlist_horizon == best_horizon:
        strategy_fit_text = (
            f"The stock is appearing in '{strategy}', and that is consistent with its strongest score profile pointing to a '{best_horizon}' horizon."
        )
    elif shortlist_horizon:
        strategy_fit_text += (
            f" This is not a conflict: the shortlist reflects the current {shortlist_horizon.lower()} setup, while the best-horizon call reflects the more suitable holding period if the thesis plays out."
        )
    elif primary_strategy and primary_strategy == best_horizon:
        strategy_fit_text += " The primary strategy tag and the score profile are broadly aligned."
    elif primary_strategy:
        strategy_fit_text += (
            f" The primary strategy tag is '{primary_strategy}', but the score profile currently leans more toward '{best_horizon}'."
        )
    if trend_structure_stage:
        strategy_fit_text += f" Trend structure is currently classified as '{trend_structure_stage}'"
        if trend_extension_bucket:
            strategy_fit_text += f" with extension tagged '{trend_extension_bucket}'."
        else:
            strategy_fit_text += "."
    if is_financial_like:
        strategy_fit_text += " This is a financial-like stock, so balance-sheet interpretation is adjusted for sector context."

    resistance_candidates = []
    for label, level in [("pivot resistance R1", pivot_r1), ("month high", month_high), ("pivot resistance R2", pivot_r2)]:
        if level > current_price > 0:
            resistance_candidates.append((label, level))
    resistance_candidates = sorted(resistance_candidates, key=lambda x: x[1])
    if resistance_candidates:
        trigger_label, trigger_level = resistance_candidates[0]
        entry_trigger_text = f"Current price is ₹{current_price:,.2f}. Preferred trigger is a decisive move above {trigger_label} (₹{trigger_level:,.2f})."
        if len(resistance_candidates) > 1:
            alt_label, alt_level = resistance_candidates[1]
            entry_trigger_text += f" The next upside confirmation level after that is {alt_label} (₹{alt_level:,.2f})."
        if pivot_r2 > 0 and pivot_r2 > trigger_level:
            entry_trigger_text += f" If the move expands cleanly, pivot resistance R2 at ₹{pivot_r2:,.2f} becomes the next stretch objective."
        if _get_numeric(row, "standard_resistance_r3", 0) > 0 and _get_numeric(row, "standard_resistance_r3", 0) > max(trigger_level, pivot_r2):
            entry_trigger_text += f" A stronger continuation could then target pivot resistance R3 at ₹{_get_numeric(row, 'standard_resistance_r3', 0):,.2f}."
    else:
        entry_trigger_text = f"Current price is ₹{current_price:,.2f}. Preferred trigger is strength above nearby resistance or a constructive pullback toward SMA50 (₹{sma50:,.2f}) if the trend remains intact."
    if _get_numeric(row, "volume_confirmation_flag", 0) == 1:
        entry_trigger_text += f" Volume confirmation is already supportive at {_fmt_num(volume_ratio_month)}x the 1-month average."
    if not _is_missing(day_adx) and _get_numeric(row, "day_adx", 0) >= 20:
        entry_trigger_text += f" ADX at {_fmt_num(day_adx)} supports the validity of the ongoing trend."
    if pivot > 0:
        if current_price > pivot:
            entry_trigger_text += f" Price is also holding above the pivot point at ₹{pivot:,.2f}."
        elif current_price < pivot:
            entry_trigger_text += f" Price is still below the pivot point at ₹{pivot:,.2f}, so a clean reclaim would improve trigger quality."
    vwap_day = _get_numeric(row, "vwap_day", 0)
    if vwap_day > 0:
        if current_price > vwap_day:
            entry_trigger_text += f" Price is also above the daily VWAP at ₹{vwap_day:,.2f}, which supports cleaner intraday price acceptance."
        elif current_price < vwap_day:
            entry_trigger_text += f" Price remains below the daily VWAP at ₹{vwap_day:,.2f}, so stronger intraday acceptance would improve trigger quality."

    support_candidates = []
    for label, level in [("pivot support S1", pivot_s1), ("pivot point", pivot), ("SMA50", sma50), ("SMA200", sma200), ("month low", month_low), ("pivot support S2", pivot_s2), ("pivot support S3", pivot_s3)]:
        if level > 0 and level < current_price:
            support_candidates.append((label, level))
    support_candidates = sorted(set(support_candidates), key=lambda x: x[1], reverse=True)

    if support_candidates:
        initial_label, initial_level = support_candidates[0]
        deeper_candidates = [(label, level) for label, level in support_candidates[1:3] if level < initial_level]
        invalidation_text = f"Initial invalidation sits near {initial_label} (₹{initial_level:,.2f})."
        if deeper_candidates:
            deeper_text = " or ".join(f"{label} (₹{level:,.2f})" for label, level in deeper_candidates)
            invalidation_text += f" A deeper invalidation level is {deeper_text}."
        else:
            invalidation_text += " A close below this level would weaken the setup materially."
        if pivot_s3 > 0 and pivot_s3 < initial_level:
            invalidation_text += f" Pivot support S3 at ₹{pivot_s3:,.2f} represents a more extreme downside failure level if the setup breaks more decisively."
    else:
        invalidation_text = "No clean support-based invalidation level is available from the current structure."

    essential_metrics = [
        ("Current Price", f"₹{current_price:,.2f}"),
        ("Market Cap (Cr)", _fmt_num(market_cap)),
        ("PE (TTM)", _fmt_num(pe_ttm)),
        ("Sector PE", _fmt_num(sector_pe)),
        ("ROCE (5Y Avg)", _fmt_pct(roce_5y)),
        ("Sales Growth (3Y)", _fmt_pct(row.get("sales_growth_3y_pct"))),
        ("Sales Growth (5Y)", _fmt_pct(row.get("sales_growth_5y_pct"))),
        ("Profit Growth (3Y)", _fmt_pct(row.get("profit_growth_3y_pct"))),
        ("Profit Growth (5Y)", _fmt_pct(profit_growth_5y)),
        ("EPS Growth (3Y)", _fmt_pct(row.get("eps_growth_3y_pct"))),
        ("EPS Growth (5Y)", _fmt_pct(row.get("eps_growth_5y_pct"))),
        ("Debt to Equity", _fmt_num(row.get("debt_to_equity"))),
        ("Current Ratio", _fmt_num(row.get("current_ratio"))),
        ("Interest Coverage", _fmt_num(row.get("interest_coverage"))),
        ("Altman Z-Score", _fmt_num(row.get("altman_z_score"))),
        ("52-Week High", _fmt_num(row.get("year_1_high"))),
        ("52-Week Low", _fmt_num(row.get("year_1_low"))),
        ("Day SMA50", _fmt_num(row.get("day_sma50"))),
        ("Day SMA200", _fmt_num(row.get("day_sma200"))),
        ("RSI (Daily)", _fmt_num(day_rsi)),
    ]
    essential_metrics.extend(_build_priority_good_to_have_metrics(row, strategy))

    # --- Patch RR-H.QA1: Report Snapshot fields ---
    # Compact, deterministic 5-line top section. Lives in the report
    # text (and JSON 'report_text' field) so PDF and JSON stay in sync.
    # The wording maps directly to inputs the trainer can verify against
    # Reference_Report_Queue and the row's tags.
    def _trim_phrase(text: str, max_words: int = 18) -> str:
        if not text:
            return ""
        # Strip leading bullet/markup characters.
        cleaned = text.lstrip("-* \t").strip()
        # Trim at sentence boundary if a long sentence is present.
        for sep in (". ", "; "):
            if sep in cleaned:
                cleaned = cleaned.split(sep, 1)[0]
                break
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).rstrip(",;:") + "..."
        return cleaned

    if strategy == "Swing_Shortlist":
        bucket_label = "Swing"
    elif strategy == "ShortTerm_Shortlist":
        bucket_label = "Short Term"
    elif strategy == "LongTerm_Core_Shortlist":
        bucket_label = "Long Term Core"
    elif strategy == "LongTerm_Opp_Shortlist":
        bucket_label = "Long Term Opportunity"
    else:
        bucket_label = strategy.replace("_Shortlist", "").replace("_", " ") if strategy else "Unspecified"

    snapshot_why_generated = (
        f"Flagged for report by reference policy under {bucket_label}. "
        f"Verdict: {verdict_label or 'Watchlist Only'}."
    )

    snapshot_setup_type_parts = [verdict_label or "Watchlist Only"]
    if best_horizon:
        snapshot_setup_type_parts.append(f"{best_horizon} horizon")
    snapshot_setup_type = "; ".join(snapshot_setup_type_parts)

    snapshot_main_driver = (
        _trim_phrase(core_strengths[0]) if core_strengths
        else "No standout single driver — see core strengths section for the detailed mix."
    )

    # Action framing maps directly from entry_quality_tag.
    if entry_tag == "Insufficient Data":
        snapshot_action_framing = "Manual Confirmation Required"
        snapshot_main_caution = (
            "Entry timing data incomplete — confirm entry manually using available "
            "shorter moving averages, price structure, and risk levels."
        )
    elif entry_tag == "Clean Entry":
        snapshot_action_framing = "Clean Entry"
        snapshot_main_caution = (
            _trim_phrase(weaknesses[0]) if weaknesses else "No major caution flagged."
        )
    elif entry_tag == "Watch on Pullback":
        snapshot_action_framing = "Watch on Pullback"
        snapshot_main_caution = (
            _trim_phrase(weaknesses[0]) if weaknesses else "No major caution flagged."
        )
    elif entry_tag == "Crowded Trend":
        snapshot_action_framing = "Confirm Breakout"
        snapshot_main_caution = (
            _trim_phrase(weaknesses[0]) if weaknesses
            else "Entry is crowded — wait for breakout confirmation or pullback."
        )
    else:
        snapshot_action_framing = "Confirm Setup Manually"
        snapshot_main_caution = (
            _trim_phrase(weaknesses[0]) if weaknesses else "No major caution flagged."
        )

    return {
        "stock_name": stock_name,
        "nse_code": nse_code,
        "strategy": strategy,
        "verdict_label": verdict_label,
        "best_horizon": best_horizon,
        "key_risk": key_risk,
        "what_must_improve": what_must_improve,
        "core_strengths_bullets": core_strengths,
        "confirmations_bullets": confirmations,
        "weaknesses_bullets": weaknesses,
        "strategy_fit_text": strategy_fit_text,
        "entry_trigger_text": entry_trigger_text,
        "invalidation_text": invalidation_text,
        "confidence_level_text": confidence_level_text,
        "positive_flag_count": pos_count,
        "red_flag_count": red_count,
        "essential_metrics": essential_metrics,
        # Patch RR-H.QA1: snapshot fields used by render_python_report.
        "snapshot_why_generated": snapshot_why_generated,
        "snapshot_setup_type": snapshot_setup_type,
        "snapshot_main_driver": snapshot_main_driver,
        "snapshot_main_caution": snapshot_main_caution,
        "snapshot_action_framing": snapshot_action_framing,
    }



def render_python_report(payload: Dict[str, Any]) -> str:
    def bullets(items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    metric_rows = payload.get("essential_metrics", [])
    metric_table = "\n".join(f"| {label} | {value} |" for label, value in metric_rows)

    # --- Patch RR-H.QA1: Section 0 — Report Snapshot ---
    # Concise five-line preamble that surfaces the report's core
    # decision-relevant context up front. Backward-compatible: if
    # snapshot fields are missing (e.g. an older payload), the section
    # is omitted gracefully and the report opens with section 1.
    snapshot_keys = (
        "snapshot_why_generated",
        "snapshot_setup_type",
        "snapshot_main_driver",
        "snapshot_main_caution",
        "snapshot_action_framing",
    )
    has_snapshot = all(payload.get(k) for k in snapshot_keys)
    if has_snapshot:
        snapshot_text = (
            "0. Report Snapshot\n"
            f"- Why generated: {payload['snapshot_why_generated']}\n"
            f"- Setup: {payload['snapshot_setup_type']}\n"
            f"- Main driver: {payload['snapshot_main_driver']}\n"
            f"- Main caution: {payload['snapshot_main_caution']}\n"
            f"- Action framing: {payload['snapshot_action_framing']}\n\n"
        )
    else:
        snapshot_text = ""

    return (
        f"{snapshot_text}"
        f"1. Core strengths\n{bullets(payload['core_strengths_bullets'])}\n\n"
        f"2. What confirms the setup\n{bullets(payload['confirmations_bullets'])}\n\n"
        f"3. What weakens the setup\n{bullets(payload['weaknesses_bullets'])}\n\n"
        f"4. Strategy fit\n{payload['strategy_fit_text']}\n\n"
        f"5. Entry trigger and invalidation\nEntry trigger: {payload['entry_trigger_text']}\nInvalidation: {payload['invalidation_text']}\n\n"
        f"6. Confidence level\n{payload['confidence_level_text']} Positive flags: {payload['positive_flag_count']}. Red flags: {payload['red_flag_count']}.\n\n"
        f"7. Final verdict\nThe stock should be approached according to the structured signal balance described above.\n"
        f"Verdict Label: {payload['verdict_label']}\nBest Horizon: {payload['best_horizon']}\nKey Risk: {payload['key_risk']}\nWhat Must Improve: {payload['what_must_improve']}\n\n"
        f"Essential metrics table\n| Attribute | Value |\n| --- | --- |\n{metric_table}"
    )


def generate_reports(df: pd.DataFrame, strategy: str, call_api: bool = False, model: str = DEEPSEEK_MODEL_DEFAULT, api_key: Optional[str] = None) -> List[Dict]:
    resolved_api_key = resolve_api_key(api_key)
    reports: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        prompt = build_prompt_payload(row, strategy)
        python_payload = build_python_report_payload(row, strategy)
        item: Dict[str, Any] = {
            "stock_name": row.get("stock_name"),
            "nse_code": row.get("nse_code"),
            "strategy": strategy,
            "prompt_payload": prompt,
            "python_report_payload": python_payload,
            "report_text": None,
            "error": None,
            "verdict_label": python_payload["verdict_label"],
            "best_horizon": python_payload["best_horizon"],
            "key_risk": python_payload["key_risk"],
            "what_must_improve": python_payload["what_must_improve"],
            "report_source": "python_fallback" if not call_api else None,
            "fallback_used": not call_api,
            "fallback_reason": "API call disabled" if not call_api else None,
        }

        if call_api:
            if not resolved_api_key:
                item["error"] = "DeepSeek API key not found. Pass api_key=... or set DEEPSEEK_API_KEY."
                item["report_text"] = _sanitize_report_text(render_python_report(python_payload), row)
                item["report_source"] = "python_fallback"
                item["fallback_used"] = True
                item["fallback_reason"] = item["error"]
            else:
                try:
                    deepseek_user_payload = json.dumps(
                        {"upstream_diagnostics": build_upstream_diagnostics(row, strategy), "python_decision_payload": python_payload},
                        indent=2, ensure_ascii=False, default=str
                    )
                    deepseek_prompt = {"system": prompt["system"], "user": deepseek_user_payload}
                    item["report_text"] = _sanitize_report_text(call_deepseek(deepseek_prompt, api_key=resolved_api_key, model=model), row)
                    extracted = extract_report_summary_fields(item["report_text"])
                    for key, value in extracted.items():
                        if value:
                            item[key] = value
                    item["report_source"] = "deepseek"
                    item["fallback_used"] = False
                    item["fallback_reason"] = None
                except Exception as e:
                    item["error"] = str(e)
                    item["report_text"] = _sanitize_report_text(render_python_report(python_payload), row)
                    item["report_source"] = "python_fallback"
                    item["fallback_used"] = True
                    item["fallback_reason"] = str(e)

        if not item["report_text"]:
            item["report_text"] = _sanitize_report_text(render_python_report(python_payload), row)
            item["report_source"] = item["report_source"] or "python_fallback"
            item["fallback_used"] = True
            item["fallback_reason"] = item["fallback_reason"] or "Unknown fallback path"

        reports.append(item)

    return reports


def safe_filename(text: str) -> str:
    text = str(text or "unknown").strip()
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120] or "unknown"