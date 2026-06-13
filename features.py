
import numpy as np
import pandas as pd


# ======================================
# HELPERS
# ======================================

def _safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].mean(axis=1, skipna=True)


def _safe_growth(current: pd.Series, base: pd.Series) -> pd.Series:
    current = current.astype(float)
    base = base.astype(float)
    base_abs = base.abs().replace(0, np.nan)
    return (current - base) / base_abs


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def _first_existing(df: pd.DataFrame, names: list[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _contains_any(series: pd.Series, keywords: list[str]) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for kw in keywords:
        mask = mask | s.str.contains(kw, regex=False)
    return mask


def _series_from_bucket(buckets: pd.Series, mapping: dict[str, float], default: float) -> pd.Series:
    return buckets.map(mapping).fillna(default).astype(float)


def _join_labels(mask_map: list[tuple[str, pd.Series]]) -> pd.Series:
    idx = mask_map[0][1].index if mask_map else pd.RangeIndex(0)
    out = pd.Series("", index=idx, dtype="string")
    for label, mask in mask_map:
        label_mask = mask.fillna(False)
        out = np.where(
            label_mask,
            np.where(pd.Series(out, index=idx).astype(str).eq(""), label, pd.Series(out, index=idx).astype(str) + "; " + label),
            pd.Series(out, index=idx).astype(str)
        )
        out = pd.Series(out, index=idx, dtype="string")
    return out.replace("", pd.NA).fillna("None")


# ======================================
# SECTOR MAPPING
# ======================================

SECTOR_BUCKET_RULES = [
    ("Financials", ["financial", "bank", "nbfc", "finance", "housing finance", "insurance", "asset management", "broking", "capital market", "microfinance"]),
    ("Healthcare", ["pharma", "pharmaceutical", "healthcare", "hospital", "diagnostic", "biotech", "formulation", "api", "lifescience", "life science"]),
    ("Consumer", ["consumer", "fmcg", "retail", "apparel", "footwear", "jewellery", "jewelry", "restaurant", "food", "beverage", "durables", "personal care"]),
    ("Industrials", ["capital goods", "industrial", "engineering", "auto", "automobile", "machinery", "equipment", "manufacturing", "defence", "electrical", "electronics", "building products"]),
    ("Asset Heavy", ["utility", "power", "oil", "gas", "energy", "infra", "infrastructure", "telecom", "cement", "real estate", "construction materials", "logistics", "shipping", "port", "transport"]),
    ("Cyclicals", ["metal", "mining", "steel", "aluminium", "copper", "commodity", "chemical", "fertilizer", "paper", "sugar", "textile", "tyre", "rubber"]),
    ("Tech Services", ["software", "it services", "technology", "tech", "internet", "digital", "saas", "platform", "communication services"]),
]


def map_sector_rule_bucket(sector_name: object, industry_name: object) -> str:
    text = f"{sector_name or ''} | {industry_name or ''}".lower()
    for bucket, keywords in SECTOR_BUCKET_RULES:
        if any(k in text for k in keywords):
            return bucket
    return "Default"


# ======================================
# CORE FEATURE ENGINEERING
# ======================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cp = _col(out, "current_price").astype(float)

    # Sector mapping
    out["sector_rule_bucket"] = [
        map_sector_rule_bucket(sector, industry)
        for sector, industry in zip(_col(out, "sector_name", "").fillna(""), _col(out, "industry_name", "").fillna(""))
    ]

    # Long-term growth
    out["sales_growth_long"] = _safe_mean(out, ["sales_growth_3y_pct", "sales_growth_5y_pct"])
    out["profit_growth_long"] = _safe_mean(out, ["profit_growth_3y_pct", "profit_growth_5y_pct"])
    out["eps_growth_long"] = _safe_mean(out, ["eps_growth_3y_pct", "eps_growth_5y_pct"])

    # Earnings acceleration
    out["sales_qoq"] = _safe_growth(_col(out, "sales_q_latest"), _col(out, "sales_q_prev"))
    out["sales_yoy"] = _safe_growth(_col(out, "sales_q_latest"), _col(out, "sales_q_yoy_base"))
    out["profit_qoq"] = _safe_growth(_col(out, "profit_q_latest"), _col(out, "profit_q_prev"))
    out["profit_yoy"] = _safe_growth(_col(out, "profit_q_latest"), _col(out, "profit_q_yoy_base"))
    out["eps_qoq"] = _safe_growth(_col(out, "eps_q_latest"), _col(out, "eps_q_prev"))
    out["eps_yoy"] = _safe_growth(_col(out, "eps_q_latest"), _col(out, "eps_q_yoy_base"))
    out["operating_leverage"] = out["profit_yoy"] - out["sales_yoy"]

    # Business quality
    out["roce_quality_raw"] = _safe_mean(out, ["roce", "roce_3y_avg", "roce_5y_avg"])
    out["roe_quality_raw"] = _safe_mean(out, ["roe", "roe_3y_avg"])
    out["margin_quality_raw"] = _safe_mean(out, ["opm_current", "opm_last_year", "opm_5y_avg"])

    # Cash flow
    out["cfo_growth"] = _safe_growth(_col(out, "cfo_latest"), _col(out, "cfo_prev"))
    out["fcf_growth"] = _safe_growth(_col(out, "fcf_latest"), _col(out, "fcf_prev"))
    out["fcf_positive"] = (_col(out, "fcf_latest") > 0).astype(float)
    out["fcf_consistency"] = (_col(out, "fcf_3y") > 0).astype(float)

    # Ownership
    out["ownership_flow_raw"] = _safe_mean(
        out,
        ["promoter_holding_change_qoq_pct", "fii_holding_change_qoq_pct", "mf_holding_change_qoq_pct"],
    )

    # Trend health
    out["price_vs_sma5_pct"] = (cp / _col(out, "day_sma5").replace(0, np.nan)) - 1
    out["price_vs_sma30_pct"] = (cp / _col(out, "day_sma30").replace(0, np.nan)) - 1
    out["price_vs_sma50_pct"] = (cp / _col(out, "day_sma50").replace(0, np.nan)) - 1
    out["price_vs_sma100_pct"] = (cp / _col(out, "day_sma100").replace(0, np.nan)) - 1
    out["price_vs_sma200_pct"] = (cp / _col(out, "day_sma200").replace(0, np.nan)) - 1
    out["macd_spread"] = _col(out, "day_macd") - _col(out, "day_macd_signal_line")

    # Entry timing
    out["distance_from_52w_high_pct"] = (cp / _col(out, "year_1_high").replace(0, np.nan)) - 1
    out["volume_ratio_week"] = _col(out, "day_volume") / _col(out, "week_volume_avg").replace(0, np.nan)
    out["volume_ratio_month"] = _col(out, "day_volume") / _col(out, "month_volume_avg").replace(0, np.nan)
    out["volume_ratio_blended"] = _safe_mean(out, ["volume_ratio_week", "volume_ratio_month"])

    # Valuation helper
    out["cashflow_multiple"] = _col(out, "price_to_fcf").where(_col(out, "price_to_fcf").notna(), _col(out, "price_to_cfo"))

    # Technical sanity
    out["not_overextended_raw"] = 1 - ((out["price_vs_sma200_pct"] - 0.20).clip(lower=0) / 0.40)
    out["not_overextended_raw"] = out["not_overextended_raw"].clip(lower=0, upper=1)

    # Momentum history
    out["momentum_delta_day"] = _col(out, "trendlyne_momentum_score") - _col(out, "prev_day_trendlyne_momentum_score")
    out["momentum_delta_week"] = _col(out, "trendlyne_momentum_score") - _col(out, "prev_week_trendlyne_momentum_score")
    out["momentum_delta_month"] = _col(out, "trendlyne_momentum_score") - _col(out, "prev_month_trendlyne_momentum_score")

    # Valuation context
    pe_ttm = _col(out, "pe_ttm")
    out["pe_vs_3y_avg_pct"] = (pe_ttm / _col(out, "pe_3yr_average").replace(0, np.nan)) - 1
    out["pe_vs_5y_avg_pct"] = (pe_ttm / _col(out, "pe_5yr_average").replace(0, np.nan)) - 1
    out["pe_vs_sector_pct"] = (pe_ttm / _col(out, "sector_pe_ttm").replace(0, np.nan)) - 1
    out["pe_vs_industry_pct"] = (pe_ttm / _col(out, "industry_pe_ttm").replace(0, np.nan)) - 1

    peg_ratio = _first_existing(out, ["peg_ratio", "peg_ttm"], default=np.nan)
    out["peg_vs_sector_pct"] = (peg_ratio / _col(out, "sector_peg_ttm").replace(0, np.nan)) - 1
    out["peg_vs_industry_pct"] = (peg_ratio / _col(out, "industry_peg_ttm").replace(0, np.nan)) - 1

    # Margin context
    out["margin_change_vs_last_year"] = _col(out, "opm_current") - _col(out, "opm_last_year")
    out["margin_change_vs_5y_avg"] = _col(out, "opm_current") - _col(out, "opm_5y_avg")

    # Sector relative earnings context
    out["sales_vs_sector_yoy_gap"] = out["sales_yoy"] - (_col(out, "sector_revenue_growth_qtr_yoy_pct") / 100.0)
    out["profit_vs_sector_yoy_gap"] = out["profit_yoy"] - (_col(out, "sector_net_profit_growth_qtr_yoy_pct") / 100.0)

    # Market cap bucket
    def market_cap_bucket(mc):
        if pd.isna(mc):
            return "Unknown"
        if mc >= 20000:
            return "Large Cap"
        if mc >= 5000:
            return "Mid Cap"
        if mc >= 1000:
            return "Small Cap"
        return "Micro Cap"

    out["market_cap_bucket"] = _col(out, "market_capitalization").apply(market_cap_bucket)

    # --------- New derived columns from current DB ----------
    out["sma50_to_sma200_spread_pct"] = (_col(out, "day_sma50") / _col(out, "day_sma200").replace(0, np.nan)) - 1
    out["volume_surge_strength"] = pd.concat([out["volume_ratio_week"], out["volume_ratio_month"]], axis=1).max(axis=1, skipna=True)
    out["volume_conviction_bucket"] = np.select(
        [out["volume_surge_strength"].fillna(0) >= 1.50,
         out["volume_surge_strength"].fillna(0) >= 1.00,
         out["volume_surge_strength"].fillna(0) >= 0.70],
        ["Strong", "Adequate", "Weak"],
        default="Absent",
    )
    out["trend_structure_stage"] = np.select(
        [
            (cp > _col(out, "day_sma50")) & (_col(out, "day_sma50") > _col(out, "day_sma200")),
            (cp > _col(out, "day_sma50")) & (_col(out, "day_sma50") <= _col(out, "day_sma200")),
            (cp < _col(out, "day_sma50")) & (cp >= _col(out, "day_sma200")),
            (cp < _col(out, "day_sma50")) & (cp < _col(out, "day_sma200")),
            (out["price_vs_sma50_pct"].abs() <= 0.03),
        ],
        ["Strong Uptrend", "Early Uptrend", "Weak Structure", "Downtrend", "Neutral"],
        default="Neutral",
    )
    out["trend_extension_bucket"] = np.select(
        [
            (out["price_vs_sma50_pct"].fillna(-999) > 0.18) & (_col(out, "day_rsi").fillna(0) > 72),
            (out["price_vs_sma50_pct"].fillna(-999) > 0.12) | (_col(out, "day_rsi").fillna(0) > 68),
            (out["price_vs_sma50_pct"].fillna(-999) >= 0.03) & (out["price_vs_sma50_pct"].fillna(-999) <= 0.12) & (_col(out, "day_rsi").fillna(0).between(55, 68)),
            (out["price_vs_sma50_pct"].fillna(-999) >= 0.00) & (out["price_vs_sma50_pct"].fillna(-999) < 0.03),
        ],
        ["Crowded", "Extended", "Healthy", "Early"],
        default="Below SMA50",
    )
    out["room_to_month_high_pct"] = ((_col(out, "month_high") - cp) / cp.replace(0, np.nan)).clip(lower=0)

    out["earnings_reacceleration_flag"] = (
        (
            (out["profit_yoy"].fillna(-999) > ((out["profit_growth_long"].fillna(0) / 100.0) + 0.05)) |
            (out["eps_yoy"].fillna(-999) > ((out["eps_growth_long"].fillna(0) / 100.0) + 0.05))
        ) &
        ((out["profit_qoq"].fillna(0) > 0) | (out["eps_qoq"].fillna(0) > 0)) &
        (out["operating_leverage"].fillna(0) > 0) &
        (out["margin_change_vs_last_year"].fillna(0) >= 0)
    ).astype(int)

    out["cashflow_alignment_flag"] = (
        (out["profit_yoy"].fillna(0) > 0) &
        (_col(out, "cfo_latest").fillna(0) > 0) &
        ((out["cfo_growth"].fillna(-999) > 0) | (out["fcf_growth"].fillna(-999) > 0)) &
        ((out["fcf_positive"].fillna(0) == 1) | (out["fcf_growth"].fillna(-999) > 0))
    ).astype(int)

    debt_change_1y = _safe_growth(_col(out, "debt_latest"), _col(out, "debt_prev"))
    debt_change_3y = _safe_growth(_col(out, "debt_latest"), _col(out, "debt_3y_back"))
    out["debt_trend_flag"] = np.select(
        [(debt_change_1y.fillna(0) < 0) & (debt_change_3y.fillna(0) < 0),
         (debt_change_1y.fillna(0) > 0) & (debt_change_3y.fillna(0) > 0)],
        ["Deleveraging", "Leveraging Up"],
        default="Stable",
    )

    peg_now = _first_existing(out, ["peg_ratio", "peg_ttm"], default=np.nan)
    stretched_val = (
        (out["pe_vs_3y_avg_pct"].fillna(-999) > 0.35) |
        (out["pe_vs_5y_avg_pct"].fillna(-999) > 0.35) |
        (out["pe_vs_sector_pct"].fillna(-999) > 0.30) |
        (out["pe_vs_industry_pct"].fillna(-999) > 0.30) |
        (peg_now.fillna(0) > 2.5) |
        (_col(out, "price_to_fcf").fillna(0) > 60) |
        (_col(out, "price_to_cfo").fillna(0) > 40)
    )
    cheap_val = (((out["pe_vs_3y_avg_pct"].fillna(999) < -0.15) | (out["pe_vs_5y_avg_pct"].fillna(999) < -0.15)) &
                 (peg_now.fillna(1.5) <= 1.5))
    reasonable_val = ((~stretched_val) &
                      ((out["pe_vs_3y_avg_pct"].fillna(999).abs() <= 0.15) | (out["pe_vs_5y_avg_pct"].fillna(999).abs() <= 0.15)) &
                      (peg_now.fillna(2.0) <= 2.0))
    out["valuation_regime"] = np.select([stretched_val, cheap_val, reasonable_val], ["Stretched", "Cheap", "Reasonable"], default="Full")
    ownership_delta = (
        0.25 * _col(out, "promoter_holding_change_qoq_pct").fillna(0) +
        0.35 * _col(out, "fii_holding_change_qoq_pct").fillna(0) +
        0.40 * _col(out, "mf_holding_change_qoq_pct").fillna(0)
    )
    out["ownership_trend_score"] = (50 + (ownership_delta * 20)).clip(0, 100)

    # Phase 1 scoring migration derived fields
    roe_base = _first_existing(out, ["roe", "roe_annual_pct", "roe_annual"], default=np.nan)
    roa_base = _first_existing(out, ["roa", "roa_annual_pct", "roa_annual"], default=np.nan)
    out["roe_vs_sector"] = roe_base - _col(out, "sector_return_on_equity_roe")
    out["roe_vs_industry"] = roe_base - _col(out, "industry_return_on_equity_roe")
    out["roa_vs_sector"] = roa_base - _col(out, "sector_return_on_assets")
    out["roa_vs_industry"] = roa_base - _col(out, "industry_return_on_assets")

    qtr_span = (_col(out, "qtr_high") - _col(out, "qtr_low")).replace(0, np.nan)
    day_span = (_col(out, "day_high") - _col(out, "day_low")).replace(0, np.nan)
    out["qtr_range_position"] = ((cp - _col(out, "qtr_low")) / qtr_span).clip(lower=0, upper=1)
    out["day_close_strength"] = ((cp - _col(out, "day_low")) / day_span).clip(lower=0, upper=1)
    out["peg_gap_vs_sector"] = peg_now - _col(out, "sector_peg_ttm")
    out["peg_gap_vs_industry"] = peg_now - _col(out, "industry_peg_ttm")

    if "standard_r1_to_price_diff_pct" not in out.columns:
        out["standard_r1_to_price_diff_pct"] = ((_col(out, "standard_resistance_r1") - cp) / cp.replace(0, np.nan))
    if "standard_r2_to_price_diff_pct" not in out.columns:
        out["standard_r2_to_price_diff_pct"] = ((_col(out, "standard_resistance_r2") - cp) / cp.replace(0, np.nan))
    if "standard_r3_to_price_diff_pct" not in out.columns:
        out["standard_r3_to_price_diff_pct"] = ((_col(out, "standard_resistance_r3") - cp) / cp.replace(0, np.nan))
    if "standard_s1_to_price_diff_pct" not in out.columns:
        out["standard_s1_to_price_diff_pct"] = ((cp - _col(out, "standard_support_s1")) / cp.replace(0, np.nan))
    if "standard_s2_to_price_diff_pct" not in out.columns:
        out["standard_s2_to_price_diff_pct"] = ((cp - _col(out, "standard_support_s2")) / cp.replace(0, np.nan))
    if "standard_s3_to_price_diff_pct" not in out.columns:
        out["standard_s3_to_price_diff_pct"] = ((cp - _col(out, "standard_support_s3")) / cp.replace(0, np.nan))

    # Screener-enriched forensic columns
    out["debtor_days_delta_vs_3y"] = _col(out, "debtor_days") - _col(out, "avg_debtor_days_3y")
    out["inventory_turnover_delta_vs_3y"] = _col(out, "inventory_turnover_ratio") - _col(out, "inventory_turnover_ratio_3y_back")
    out["inventory_turnover_delta_vs_5y"] = _col(out, "inventory_turnover_ratio") - _col(out, "inventory_turnover_ratio_5y_back")
    out["inventory_turnover_trend"] = _safe_mean(out, ["inventory_turnover_delta_vs_3y", "inventory_turnover_delta_vs_5y"])

    sector_text = _col(out, "sector_name", "").fillna("").astype(str) + " | " + _col(out, "industry_name", "").fillna("").astype(str)
    out["is_financial_like"] = _contains_any(
        sector_text,
        ["financial", "bank", "nbfc", "insurance", "asset management", "microfinance", "lending", "broking", "capital market", "housing finance"],
    ).astype(int)

    out["working_capital_stress_flag"] = (
        (_col(out, "cash_conversion_cycle_days").fillna(0) > 120) |
        (out["debtor_days_delta_vs_3y"].fillna(0) > 20) |
        (out["inventory_turnover_trend"].fillna(0) < -1.0)
    ).astype(int)
    out["debtor_deterioration_flag"] = (out["debtor_days_delta_vs_3y"].fillna(0) > 20).astype(int)
    out["weak_inventory_trend_flag"] = (out["inventory_turnover_trend"].fillna(0) < -1.0).astype(int)
    out["weak_piotroski_flag"] = (_col(out, "piotroski_score").fillna(0) <= 3).astype(int)
    out["strong_piotroski_flag"] = (_col(out, "piotroski_score").fillna(0) >= 7).astype(int)
    out["value_support_flag"] = (((_col(out, "earnings_yield").fillna(0) >= 4) | (_col(out, "ev_ebitda").fillna(999) <= 12))).astype(int)
    out["deep_value_support_flag"] = (((_col(out, "earnings_yield").fillna(0) >= 6) & (_col(out, "ev_ebitda").fillna(999) <= 10))).astype(int)

    out["compounder_setup_flag"] = (
        ((out["sales_growth_long"].fillna(0) > 15) & (out["profit_growth_long"].fillna(0) > 15) & (_col(out, "roce_5y_avg").fillna(0) >= 15)).astype(int) &
        (out["cashflow_alignment_flag"] == 1) &
        (out["debt_trend_flag"] != "Leveraging Up") &
        (out["ownership_trend_score"].fillna(0) >= 50) &
        (out["valuation_regime"].isin(["Cheap", "Reasonable", "Full"])) &
        (out["trend_structure_stage"].isin(["Strong Uptrend", "Early Uptrend", "Neutral"]))
    ).astype(int)

    # Neutralize inventory/CCC flags for financials
    fin_mask = out["is_financial_like"] == 1
    for col in ["working_capital_stress_flag", "debtor_deterioration_flag", "weak_inventory_trend_flag"]:
        out.loc[fin_mask, col] = 0

    return out


# ======================================
# ADVANCED FILTER LAYER
# ======================================

def add_advanced_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Quality safety score with sector-aware neutralization for financials
    dte = _col(out, "debt_to_equity").astype(float)
    altman = _col(out, "altman_z_score").astype(float)
    int_cov = _col(out, "interest_coverage").astype(float)

    debt_component = (1 - (dte / 2).clip(0, 1)) * 100
    altman_component = (altman.clip(0, 5) / 5) * 100
    interest_component = (int_cov.clip(0, 10) / 10) * 100

    fin_mask = _col(out, "is_financial_like", 0).fillna(0).astype(int) == 1
    debt_component = np.where(fin_mask, 60, debt_component)
    altman_component = np.where(fin_mask, 60, altman_component)
    interest_component = np.where(fin_mask, 60, interest_component)

    out["quality_safety_score"] = (
        0.30 * debt_component +
        0.25 * altman_component +
        0.20 * (out["fcf_positive"].fillna(0) * 100) +
        0.15 * (out["fcf_consistency"].fillna(0) * 100) +
        0.10 * interest_component
    )
    out["quality_safety_score"] = out["quality_safety_score"].clip(0, 100)

    # Tradability score
    volume_score = (_col(out, "day_volume") / _col(out, "month_volume_avg").replace(0, np.nan)).clip(0, 3) / 3
    volume_score = volume_score.fillna(0)
    size_score = (_col(out, "market_capitalization") / 20000).clip(0, 1)
    out["tradability_score"] = (0.60 * volume_score * 100 + 0.40 * size_score * 100).clip(0, 100)

    # Sector-aware thresholds
    buckets = _col(out, "sector_rule_bucket", "Default")
    volume_confirm_thr = _series_from_bucket(buckets, {"Financials":0.85,"Asset Heavy":0.95,"Cyclicals":1.00,"Industrials":0.95,"Consumer":0.90,"Healthcare":0.90,"Tech Services":0.85,"Default":0.95}, 0.95)
    weak_volume_thr = _series_from_bucket(buckets, {"Financials":0.65,"Asset Heavy":0.70,"Cyclicals":0.75,"Industrials":0.75,"Consumer":0.70,"Healthcare":0.70,"Tech Services":0.65,"Default":0.70}, 0.70)
    hist_valuation_thr = _series_from_bucket(buckets, {"Financials":0.45,"Asset Heavy":0.30,"Cyclicals":0.20,"Industrials":0.35,"Consumer":0.55,"Healthcare":0.50,"Tech Services":0.50,"Default":0.35}, 0.35)
    sector_valuation_thr = _series_from_bucket(buckets, {"Financials":0.40,"Asset Heavy":0.25,"Cyclicals":0.20,"Industrials":0.30,"Consumer":0.45,"Healthcare":0.40,"Tech Services":0.40,"Default":0.30}, 0.30)
    margin_deterioration_thr = _series_from_bucket(buckets, {"Financials":-999.0,"Asset Heavy":-4.0,"Cyclicals":-5.0,"Industrials":-3.0,"Consumer":-2.0,"Healthcare":-2.5,"Tech Services":-2.0,"Default":-3.0}, -3.0)
    debt_thr = _series_from_bucket(buckets, {"Financials":999.0,"Asset Heavy":1.50,"Cyclicals":1.20,"Industrials":1.00,"Consumer":0.80,"Healthcare":0.80,"Tech Services":0.60,"Default":1.00}, 1.00)
    current_ratio_thr = _series_from_bucket(buckets, {"Financials":-999.0,"Asset Heavy":0.90,"Cyclicals":1.00,"Industrials":1.00,"Consumer":1.10,"Healthcare":1.10,"Tech Services":1.00,"Default":1.00}, 1.00)
    interest_cov_thr = _series_from_bucket(buckets, {"Financials":-999.0,"Asset Heavy":1.5,"Cyclicals":2.0,"Industrials":2.5,"Consumer":3.0,"Healthcare":3.0,"Tech Services":2.5,"Default":2.5}, 2.5)
    altman_thr = _series_from_bucket(buckets, {"Financials":-999.0,"Asset Heavy":1.6,"Cyclicals":1.8,"Industrials":1.8,"Consumer":2.0,"Healthcare":2.0,"Tech Services":1.8,"Default":1.8}, 1.8)

    # Core diagnostic flags
    out["volume_confirmation_flag"] = (out["volume_ratio_blended"].fillna(0) >= volume_confirm_thr).astype(int)
    out["weak_volume_confirmation"] = (out["volume_ratio_blended"].fillna(0) < weak_volume_thr).astype(int)
    out["breakout_volume_confirmation_flag"] = (
        (out["distance_from_52w_high_pct"] >= -0.05) &
        (out["price_vs_sma50_pct"] > 0) &
        (out["volume_ratio_blended"].fillna(0) >= (volume_confirm_thr + 0.15))
    ).astype(int)

    liq_floor = np.where(_col(out, "market_cap_bucket").eq("Micro Cap"), 25, np.where(_col(out, "market_cap_bucket").eq("Small Cap"), 30, 20))
    out["liquidity_risk_flag"] = (_col(out, "tradability_score") < liq_floor).astype(int)

    out["momentum_acceleration_flag"] = (((_col(out, "momentum_delta_week") > 0) | (_col(out, "momentum_delta_month") > 0))).astype(int)
    rs_sum = (
        _col(out, "rr_nifty50_week_pct").fillna(0).gt(0).astype(int) +
        _col(out, "rr_nifty50_month_pct").fillna(0).gt(0).astype(int) +
        _col(out, "rr_nifty50_quarter_pct").fillna(0).gt(0).astype(int) +
        _col(out, "rr_nifty50_year_pct").fillna(0).gt(0).astype(int)
    )
    out["relative_strength_confirmation_flag"] = (rs_sum >= 2).astype(int)

    out["valuation_stretch_vs_history"] = (
        (out["pe_vs_3y_avg_pct"].fillna(-999) > hist_valuation_thr) |
        (out["pe_vs_5y_avg_pct"].fillna(-999) > hist_valuation_thr)
    ).astype(int)
    out["valuation_stretch_vs_sector"] = (
        (out["pe_vs_sector_pct"].fillna(-999) > sector_valuation_thr) |
        (out["peg_vs_sector_pct"].fillna(-999) > sector_valuation_thr)
    ).astype(int)
    out["valuation_stretch_vs_industry"] = (
        (out["pe_vs_industry_pct"].fillna(-999) > sector_valuation_thr) |
        (out["peg_vs_industry_pct"].fillna(-999) > sector_valuation_thr)
    ).astype(int)
    out["cheap_vs_history_flag"] = (
        (out["pe_vs_3y_avg_pct"].fillna(999) < -0.15) |
        (out["pe_vs_5y_avg_pct"].fillna(999) < -0.15)
    ).astype(int)

    out["earnings_deceleration_flag"] = (
        (((out["profit_yoy"].fillna(0) < 0.10) & (out["profit_growth_long"].fillna(0) / 100.0 > 0.20)) |
         ((out["eps_yoy"].fillna(0) < 0.10) & (out["eps_growth_long"].fillna(0) / 100.0 > 0.20)))
    ).astype(int)
    out["margin_deterioration_flag"] = (
        (out["margin_change_vs_last_year"].fillna(0) < margin_deterioration_thr) |
        (out["margin_change_vs_5y_avg"].fillna(0) < margin_deterioration_thr)
    ).astype(int)
    out["high_quality_growth_flag"] = (
        (out["sales_growth_long"].fillna(0) > 15) &
        (out["profit_growth_long"].fillna(0) > 15) &
        (out["roce_quality_raw"].fillna(0) > 15)
    ).astype(int)

    inst_change_4q = _first_existing(out, ["institutional_holding_change_4qtr_pct"], default=np.nan)
    inst_change_qoq = _first_existing(out, ["institutional_holding_change_qoq_pct"], default=np.nan)
    out["falling_institutional_support"] = (
        (_col(out, "fii_holding_change_qoq_pct").fillna(0) < 0) &
        (_col(out, "mf_holding_change_qoq_pct").fillna(0) < 0) &
        (inst_change_qoq.fillna(0) <= 0)
    ).astype(int)
    out["strong_sponsorship_flag"] = (
        (_col(out, "fii_holding_change_qoq_pct").fillna(0) > 0) |
        (_col(out, "mf_holding_change_qoq_pct").fillna(0) > 0) |
        (inst_change_4q.fillna(0) > 0)
    ).astype(int)

    pledge = _col(out, "promoter_holding_pledge_percentage_qtr_pct").fillna(0)
    pledge_change = _col(out, "promoter_pledge_change_qoq_pct").fillna(0)
    out["promoter_pledge_risk"] = ((pledge > 5) | (pledge_change > 0)).astype(int)
    out["debt_risk_flag"] = (_col(out, "debt_to_equity").fillna(0) > debt_thr).astype(int)
    out["balance_sheet_risk_flag"] = (
        (_col(out, "current_ratio").fillna(999) < current_ratio_thr) |
        (_col(out, "interest_coverage").fillna(999) < interest_cov_thr) |
        (_col(out, "altman_z_score").fillna(999) < altman_thr)
    ).astype(int)
    out["weak_cash_conversion"] = (
        (out["fcf_positive"].fillna(0) == 0) |
        (out["fcf_consistency"].fillna(0) == 0) |
        (out["cfo_growth"].fillna(0) < -0.10)
    ).astype(int)
    out["crowded_trend_flag"] = (
        (out["price_vs_sma200_pct"] > 0.30) &
        (_col(out, "day_rsi") > 68) &
        (out["distance_from_52w_high_pct"] > -0.05)
    ).astype(int)

    # Additional structure flags
    out["clean_trend_structure"] = (
        (out["price_vs_sma50_pct"].fillna(-999) > 0) &
        (out["price_vs_sma200_pct"].fillna(-999) > 0) &
        (out["sma50_to_sma200_spread_pct"].fillna(-999) > 0)
    ).astype(int)
    out["overextended_trend_flag"] = out["trend_extension_bucket"].isin(["Extended", "Crowded"]).astype(int)
    out["crowded_breakout"] = out["crowded_trend_flag"].astype(int)
    out["trend_confirmation_flag"] = (
        (out["volume_confirmation_flag"] == 1) &
        (out["relative_strength_confirmation_flag"] == 1) &
        (out["price_vs_sma50_pct"].fillna(-999) > 0)
    ).astype(int)

    # Confidence
    confidence_fields = [
        "current_price","market_capitalization","sales_growth_5y_pct","profit_growth_5y_pct","eps_growth_5y_pct",
        "roce_5y_avg","roe","opm_current","cfo_latest","fcf_latest","debt_to_equity","interest_coverage",
        "altman_z_score","pe_ttm","peg_ratio","price_to_fcf","day_rsi","day_adx","day_sma50","day_sma200",
        "day_volume","week_volume_avg","month_volume_avg","promoter_holding_latest_pct",
        "piotroski_score","earnings_yield","ev_ebitda",
    ]
    existing_conf = [c for c in confidence_fields if c in out.columns]
    if existing_conf:
        out["data_completeness_score"] = out[existing_conf].notna().mean(axis=1) * 100
    else:
        out["data_completeness_score"] = 0.0

    base_conf = out["data_completeness_score"].copy()
    base_conf = base_conf - (out["liquidity_risk_flag"] * 10)
    base_conf = base_conf - (out["weak_volume_confirmation"] * 5)
    base_conf = base_conf - (out["earnings_deceleration_flag"] * 6)
    base_conf = base_conf - (out["weak_cash_conversion"] * 8)
    base_conf = base_conf - (out["valuation_stretch_vs_history"] * 5)
    base_conf = base_conf - (out["valuation_stretch_vs_sector"] * 5)
    base_conf = base_conf - (out["valuation_stretch_vs_industry"] * 4)
    base_conf = base_conf - (out["working_capital_stress_flag"] * 4)
    base_conf = base_conf - (out["weak_piotroski_flag"] * 5)
    out["_base_conf_pre_tags"] = base_conf.clip(0, 100)

    # Setup tags
    def entry_quality(row):
        if row.get("crowded_trend_flag", 0) == 1:
            return "Crowded Trend"
        rsi = row.get("day_rsi")
        p200 = row.get("price_vs_sma200_pct")
        if pd.isna(rsi) or pd.isna(p200):
            return "Insufficient Data"
        if rsi <= 65 and p200 <= 0.20:
            return "Clean Entry"
        if rsi > 65 and p200 <= 0.30:
            return "Watch on Pullback"
        return "Neutral"

    out["entry_quality_tag"] = out.apply(entry_quality, axis=1)
    out["setup_quality_tag"] = np.select(
        [
            (out["high_quality_growth_flag"] == 1) & (out["quality_safety_score"] >= 65),
            (out["quality_safety_score"] >= 50),
        ],
        ["High Quality", "Acceptable Quality"],
        default="Fragile Quality",
    )
    out["setup_confirmation_tag"] = np.select(
        [
            (out["volume_confirmation_flag"] == 1) & (out["relative_strength_confirmation_flag"] == 1) & (out["momentum_acceleration_flag"] == 1),
            (out["volume_confirmation_flag"] == 1) | (out["relative_strength_confirmation_flag"] == 1),
        ],
        ["Strong Confirmation", "Partial Confirmation"],
        default="Weak Confirmation",
    )
    out["setup_risk_tag"] = np.select(
        [
            (out["crowded_trend_flag"] == 1) | (out["valuation_stretch_vs_history"] == 1) | (out["promoter_pledge_risk"] == 1),
            (out["earnings_deceleration_flag"] == 1) | (out["weak_cash_conversion"] == 1) | (out["falling_institutional_support"] == 1),
        ],
        ["Elevated Risk", "Moderate Risk"],
        default="Contained Risk",
    )

    # Final confidence
    base_conf = out["_base_conf_pre_tags"].copy()
    base_conf = base_conf - (out["setup_confirmation_tag"].eq("Partial Confirmation").astype(int) * 5)
    base_conf = base_conf - (out["setup_confirmation_tag"].eq("Weak Confirmation").astype(int) * 10)
    base_conf = base_conf - (out["entry_quality_tag"].eq("Watch on Pullback").astype(int) * 4)
    base_conf = base_conf - (out["entry_quality_tag"].eq("Crowded Trend").astype(int) * 6)
    base_conf = base_conf - ((out["price_vs_sma200_pct"].fillna(0) < 0).astype(int) * 4)
    out["analysis_confidence_score"] = base_conf.clip(0, 100)
    out = out.drop(columns=["_base_conf_pre_tags"])
    out["analysis_confidence_bucket"] = np.select(
        [out["analysis_confidence_score"] >= 85, out["analysis_confidence_score"] >= 70, out["analysis_confidence_score"] >= 50],
        ["High", "Moderate", "Low"],
        default="Very Low",
    )

    # Final penalty-style flags expected by reporting/audit
    out["weak_volume_flag"] = out["weak_volume_confirmation"].astype(int)
    out["no_volume_confirmation_flag"] = (out["volume_confirmation_flag"] == 0).astype(int)
    out["crowded_trend_flag_final"] = out["crowded_trend_flag"].astype(int)
    out["negative_profit_yoy_flag"] = (out["profit_yoy"].fillna(0) < 0).astype(int)
    out["negative_eps_yoy_flag"] = (out["eps_yoy"].fillna(0) < 0).astype(int)
    out["negative_profit_and_eps_yoy_flag"] = (
        (out["negative_profit_yoy_flag"] == 1) & (out["negative_eps_yoy_flag"] == 1)
    ).astype(int)
    out["high_debt_flag"] = out["debt_risk_flag"].astype(int)
    out["weak_altman_flag"] = (_col(out, "altman_z_score").fillna(999) < altman_thr).astype(int)
    out["double_negative_fcf_flag"] = ((_col(out, "fcf_latest").fillna(0) < 0) & (_col(out, "fcf_prev").fillna(0) < 0)).astype(int)
    out["weak_roce_5y_flag"] = (_col(out, "roce_5y_avg").fillna(0) < 12).astype(int)
    out["negative_profit_growth_5y_flag"] = (_col(out, "profit_growth_5y_pct").fillna(0) < 0).astype(int)
    out["very_overextended_sma200_flag"] = (out["price_vs_sma200_pct"].fillna(0) > 0.35).astype(int)
    out["overextended_sma200_flag"] = (out["price_vs_sma200_pct"].fillna(0) > 0.25).astype(int)
    out["rsi_overheat_flag"] = (_col(out, "day_rsi").fillna(0) > 70).astype(int)
    out["negative_fii_and_mf_flag"] = (
        (_col(out, "fii_holding_change_qoq_pct").fillna(0) < 0) & (_col(out, "mf_holding_change_qoq_pct").fillna(0) < 0)
    ).astype(int)
    out["valuation_stretch_flag"] = (
        (out["valuation_stretch_vs_history"] == 1) |
        (out["valuation_stretch_vs_sector"] == 1) |
        (out["valuation_stretch_vs_industry"] == 1)
    ).astype(int)
    out["promoter_pledge_risk_flag"] = out["promoter_pledge_risk"].astype(int)
    out["falling_institutional_support_flag"] = out["falling_institutional_support"].astype(int)
    out["weak_cash_conversion_flag"] = out["weak_cash_conversion"].astype(int)
    out["low_confidence_data_flag"] = (out["analysis_confidence_score"] < 70).astype(int)
    out["liquidity_risk_flag_final"] = out["liquidity_risk_flag"].astype(int)
    out["margin_deterioration_flag_final"] = out["margin_deterioration_flag"].astype(int)
    out["earnings_deceleration_flag_final"] = out["earnings_deceleration_flag"].astype(int)
    out["valuation_and_cashflow_contradiction_flag"] = (
        (out["valuation_stretch_flag"] == 1) & (out["weak_cash_conversion_flag"] == 1)
    ).astype(int)

    # Positive / red flag summaries for reports
    red_map = [
        ("Weak volume confirmation", out["weak_volume_flag"] == 1),
        ("No volume confirmation", out["no_volume_confirmation_flag"] == 1),
        ("Crowded trend", out["crowded_trend_flag_final"] == 1),
        ("Negative profit YoY", out["negative_profit_yoy_flag"] == 1),
        ("Negative EPS YoY", out["negative_eps_yoy_flag"] == 1),
        ("High debt", out["high_debt_flag"] == 1),
        ("Weak Altman", out["weak_altman_flag"] == 1),
        ("Double negative FCF", out["double_negative_fcf_flag"] == 1),
        ("Weak ROCE 5Y", out["weak_roce_5y_flag"] == 1),
        ("Negative profit growth 5Y", out["negative_profit_growth_5y_flag"] == 1),
        ("Overextended vs SMA200", out["overextended_sma200_flag"] == 1),
        ("RSI overheat", out["rsi_overheat_flag"] == 1),
        ("Negative FII and MF flows", out["negative_fii_and_mf_flag"] == 1),
        ("Valuation stretched", out["valuation_stretch_flag"] == 1),
        ("Promoter pledge risk", out["promoter_pledge_risk_flag"] == 1),
        ("Falling institutional support", out["falling_institutional_support_flag"] == 1),
        ("Weak cash conversion", out["weak_cash_conversion_flag"] == 1),
        ("Working-capital stress", out["working_capital_stress_flag"] == 1),
        ("Debtor deterioration", out["debtor_deterioration_flag"] == 1),
        ("Weak inventory trend", out["weak_inventory_trend_flag"] == 1),
        ("Weak Piotroski", out["weak_piotroski_flag"] == 1),
        ("Low confidence data", out["low_confidence_data_flag"] == 1),
        ("Margin deterioration", out["margin_deterioration_flag_final"] == 1),
        ("Earnings deceleration", out["earnings_deceleration_flag_final"] == 1),
    ]
    positive_map = [
        ("Volume confirmation", out["volume_confirmation_flag"] == 1),
        ("Breakout volume confirmation", out["breakout_volume_confirmation_flag"] == 1),
        ("Momentum acceleration", out["momentum_acceleration_flag"] == 1),
        ("Relative strength confirmation", out["relative_strength_confirmation_flag"] == 1),
        ("Cheap vs history", out["cheap_vs_history_flag"] == 1),
        ("High quality growth", out["high_quality_growth_flag"] == 1),
        ("Strong sponsorship", out["strong_sponsorship_flag"] == 1),
        ("Cash-flow alignment", out["cashflow_alignment_flag"] == 1),
        ("Deleveraging trend", out["debt_trend_flag"].eq("Deleveraging")),
        ("Strong Piotroski", out["strong_piotroski_flag"] == 1),
        ("Value support", out["value_support_flag"] == 1),
        ("Deep value support", out["deep_value_support_flag"] == 1),
        ("Clean trend structure", out["clean_trend_structure"] == 1),
        ("Trend confirmation", out["trend_confirmation_flag"] == 1),
        ("Earnings reacceleration", out["earnings_reacceleration_flag"] == 1),
        ("Compounder setup", out["compounder_setup_flag"] == 1),
    ]
    out["red_flags"] = _join_labels(red_map)
    out["positive_flags"] = _join_labels(positive_map)
    out["red_flag_count"] = sum(mask.astype(int) for _, mask in red_map)
    out["positive_flag_count"] = sum(mask.astype(int) for _, mask in positive_map)

    return out
