from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StockScoreConfig:
    # Main blocks
    weight_rs_sector: float = 0.35
    weight_momentum: float = 0.25
    weight_trend: float = 0.20
    weight_risk_quality: float = 0.20

    # RS vs sector
    rs_w_3m: float = 0.30
    rs_w_6m: float = 0.45
    rs_w_12m: float = 0.25
    rs_scale_3m: float = 0.08
    rs_scale_6m: float = 0.12
    rs_scale_12m: float = 0.20

    # Momentum absolute
    mom_w_3m: float = 0.30
    mom_w_6m: float = 0.40
    mom_w_12_1: float = 0.30
    mom_scale_3m: float = 0.15
    mom_scale_6m: float = 0.25
    mom_scale_12_1: float = 0.35

    # Trend absolute
    trend_w_ma100: float = 0.30
    trend_w_ma200: float = 0.40
    trend_w_slope200: float = 0.30
    trend_scale_ma100: float = 0.10
    trend_scale_ma200: float = 0.15
    trend_scale_slope200: float = 0.30

    # Risk quality
    risk_w_downside_vol: float = 0.35
    risk_w_max_drawdown: float = 0.30
    risk_w_tracking_error: float = 0.35
    risk_window_days: int = 126

    # Eligibility / selection
    min_history_days: int = 260
    require_positive_rs6m: bool = False
    require_price_above_ma200: bool = False
    top_per_sector: int = 2
    max_total_names: int = 10

    # Optimizer export window
    optimizer_lookback_months: int = 12

    def validate(self) -> None:
        groups = {
            "Main": [self.weight_rs_sector, self.weight_momentum, self.weight_trend, self.weight_risk_quality],
            "RS": [self.rs_w_3m, self.rs_w_6m, self.rs_w_12m],
            "Momentum": [self.mom_w_3m, self.mom_w_6m, self.mom_w_12_1],
            "Trend": [self.trend_w_ma100, self.trend_w_ma200, self.trend_w_slope200],
            "Risk": [self.risk_w_downside_vol, self.risk_w_max_drawdown, self.risk_w_tracking_error],
        }
        for name, vals in groups.items():
            if not np.isclose(sum(vals), 1.0, atol=1e-8):
                raise ValueError(f"I pesi del blocco {name} devono sommare a 1. Attuale: {sum(vals):.4f}")
        if self.top_per_sector < 1:
            raise ValueError("Top per settore deve essere >= 1")
        if self.max_total_names < 1:
            raise ValueError("Numero massimo titoli deve essere >= 1")
        if self.min_history_days < 200:
            raise ValueError("Per la metodologia corrente sono consigliati almeno 200 giorni di storico.")

    def to_dict(self) -> dict:
        return asdict(self)


def _monthly_last_series(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    if s.empty:
        return s
    return s.groupby(s.index.to_period("M")).last()


def _monthly_last_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_index().groupby(df.index.to_period("M")).last()


def _bounded_score(x: pd.Series | pd.DataFrame, scale: float):
    return 50.0 + 50.0 * np.tanh(x / scale)


def _rank_low_is_good(df: pd.DataFrame) -> pd.DataFrame:
    # Lowest raw value receives the highest percentile score.
    return df.rank(axis=1, pct=True, ascending=False, method="average") * 100.0


def _rank_high_is_good(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, ascending=True, method="average") * 100.0


def _rolling_max_drawdown(prices: pd.Series, window: int) -> pd.Series:
    def _mdd(x: np.ndarray) -> float:
        if len(x) < 2 or not np.isfinite(x).all():
            return np.nan
        peak = np.maximum.accumulate(x)
        dd = x / peak - 1.0
        return float(np.nanmin(dd))
    return prices.rolling(window, min_periods=max(60, int(window * 0.65))).apply(_mdd, raw=True)


def _ensure_mapping(mapping: pd.DataFrame, stocks: Iterable[str], sectors: Iterable[str]) -> pd.DataFrame:
    required = {"Asset", "Sector_Benchmark"}
    if not required.issubset(mapping.columns):
        raise ValueError("La mappatura deve contenere le colonne Asset e Sector_Benchmark.")
    m = mapping[["Asset", "Sector_Benchmark"]].copy()
    m["Asset"] = m["Asset"].astype(str)
    m["Sector_Benchmark"] = m["Sector_Benchmark"].astype(str)
    m = m.drop_duplicates("Asset", keep="last")
    missing_assets = sorted(set(stocks) - set(m["Asset"]))
    if missing_assets:
        raise ValueError("Manca la mappatura settore per: " + ", ".join(missing_assets[:10]))
    bad = m[~m["Sector_Benchmark"].isin(list(sectors))]
    if not bad.empty:
        vals = bad["Sector_Benchmark"].unique().tolist()
        raise ValueError("Benchmark settoriali non presenti nei dati: " + ", ".join(vals[:10]))
    return m


def compute_stock_score(
    stock_prices: pd.DataFrame,
    sector_prices: pd.DataFrame,
    mapping: pd.DataFrame,
    cfg: Optional[StockScoreConfig] = None,
    eligible_sectors: Optional[List[str]] = None,
) -> Dict[str, object]:
    cfg = cfg or StockScoreConfig()
    cfg.validate()

    stocks = stock_prices.copy().sort_index()
    sectors = sector_prices.copy().sort_index()
    stocks = stocks[~stocks.index.duplicated(keep="last")].apply(pd.to_numeric, errors="coerce")
    sectors = sectors[~sectors.index.duplicated(keep="last")].apply(pd.to_numeric, errors="coerce")
    stocks = stocks.dropna(axis=1, how="all")
    sectors = sectors.dropna(axis=1, how="all")

    if stocks.empty:
        raise ValueError("Nessuna serie azionaria disponibile.")
    if sectors.empty:
        raise ValueError("Nessuna serie benchmark settoriale disponibile.")

    mapping = _ensure_mapping(mapping, stocks.columns, sectors.columns)
    if eligible_sectors:
        eligible_set = set(eligible_sectors)
        mapping = mapping[mapping["Sector_Benchmark"].isin(eligible_set)].copy()
        keep_assets = [a for a in stocks.columns if a in set(mapping["Asset"])]
        stocks = stocks[keep_assets]
        if stocks.empty:
            raise ValueError("Nessun titolo appartiene ai settori eleggibili selezionati.")

    assets = list(stocks.columns)
    map_dict = mapping.set_index("Asset")["Sector_Benchmark"].to_dict()

    # Build component containers indexed by monthly PeriodIndex.
    component_series: Dict[str, Dict[str, pd.Series]] = {
        k: {} for k in [
            "rs_3m", "rs_6m", "rs_12m", "rs_score",
            "mom_3m", "mom_6m", "mom_12_1", "momentum_score",
            "above_ma100", "above_ma200", "slope_ma200", "trend_score",
            "downside_vol", "max_drawdown", "tracking_error",
        ]
    }
    observations = {}

    for asset in assets:
        sector = map_dict[asset]
        pair = pd.concat(
            [stocks[asset].rename("stock"), sectors[sector].rename("sector")], axis=1
        ).dropna().sort_index()
        observations[asset] = len(pair)
        if len(pair) < max(180, int(cfg.min_history_days * 0.7)):
            continue

        mp = _monthly_last_df(pair)
        stock_m = mp["stock"]
        sector_m = mp["sector"]

        # RS vs own sector: absolute excess return, 50 = in line with sector.
        stock_3 = stock_m.pct_change(3, fill_method=None)
        stock_6 = stock_m.pct_change(6, fill_method=None)
        stock_12 = stock_m.pct_change(12, fill_method=None)
        sec_3 = sector_m.pct_change(3, fill_method=None)
        sec_6 = sector_m.pct_change(6, fill_method=None)
        sec_12 = sector_m.pct_change(12, fill_method=None)
        rs3 = stock_3 - sec_3
        rs6 = stock_6 - sec_6
        rs12 = stock_12 - sec_12
        rs_score = (
            cfg.rs_w_3m * _bounded_score(rs3, cfg.rs_scale_3m)
            + cfg.rs_w_6m * _bounded_score(rs6, cfg.rs_scale_6m)
            + cfg.rs_w_12m * _bounded_score(rs12, cfg.rs_scale_12m)
        )

        # Absolute momentum: 50 = roughly flat on each horizon.
        mom3 = stock_3
        mom6 = stock_6
        mom12_1 = stock_m.shift(1) / stock_m.shift(12) - 1.0
        momentum_score = (
            cfg.mom_w_3m * _bounded_score(mom3, cfg.mom_scale_3m)
            + cfg.mom_w_6m * _bounded_score(mom6, cfg.mom_scale_6m)
            + cfg.mom_w_12_1 * _bounded_score(mom12_1, cfg.mom_scale_12_1)
        )

        # Trend on daily prices.
        sp = pair["stock"]
        ma100 = sp.rolling(100, min_periods=80).mean()
        ma200 = sp.rolling(200, min_periods=160).mean()
        above100_d = sp / ma100 - 1.0
        above200_d = sp / ma200 - 1.0
        slope200_d = (ma200 / ma200.shift(20)) ** (252.0 / 20.0) - 1.0
        above100 = _monthly_last_series(above100_d)
        above200 = _monthly_last_series(above200_d)
        slope200 = _monthly_last_series(slope200_d)
        trend_score = (
            cfg.trend_w_ma100 * _bounded_score(above100, cfg.trend_scale_ma100)
            + cfg.trend_w_ma200 * _bounded_score(above200, cfg.trend_scale_ma200)
            + cfg.trend_w_slope200 * _bounded_score(slope200, cfg.trend_scale_slope200)
        )

        # Risk quality on daily returns: lower downside vol / drawdown / idiosyncratic TE is better.
        rets = pair.pct_change(fill_method=None).dropna()
        sr = rets["stock"]
        er = rets["stock"] - rets["sector"]
        neg = sr.clip(upper=0.0)
        downside = (neg.pow(2).rolling(cfg.risk_window_days, min_periods=80).mean() * 252.0) ** 0.5
        maxdd = _rolling_max_drawdown(pair["stock"], cfg.risk_window_days)
        te = er.rolling(cfg.risk_window_days, min_periods=80).std(ddof=1) * np.sqrt(252.0)

        downside_m = _monthly_last_series(downside)
        maxdd_m = _monthly_last_series(maxdd)
        te_m = _monthly_last_series(te)

        vals = {
            "rs_3m": rs3,
            "rs_6m": rs6,
            "rs_12m": rs12,
            "rs_score": rs_score,
            "mom_3m": mom3,
            "mom_6m": mom6,
            "mom_12_1": mom12_1,
            "momentum_score": momentum_score,
            "above_ma100": above100,
            "above_ma200": above200,
            "slope_ma200": slope200,
            "trend_score": trend_score,
            "downside_vol": downside_m,
            "max_drawdown": maxdd_m,
            "tracking_error": te_m,
        }
        for k, v in vals.items():
            component_series[k][asset] = v

    def to_frame(name: str) -> pd.DataFrame:
        if not component_series[name]:
            return pd.DataFrame()
        return pd.concat(component_series[name], axis=1).sort_index()

    components = {k: to_frame(k) for k in component_series}
    if components["rs_score"].empty:
        raise ValueError("Storico insufficiente per calcolare lo Stock Score.")

    # Risk block is cross-sectional: reward lower stock-specific risk among current candidates.
    downside_score = _rank_low_is_good(components["downside_vol"])
    maxdd_score = _rank_high_is_good(components["max_drawdown"])
    te_score = _rank_low_is_good(components["tracking_error"])
    risk_quality_score = (
        cfg.risk_w_downside_vol * downside_score
        + cfg.risk_w_max_drawdown * maxdd_score
        + cfg.risk_w_tracking_error * te_score
    )

    common_index = components["rs_score"].index
    for name in ["momentum_score", "trend_score"]:
        common_index = common_index.union(components[name].index)
    common_index = common_index.union(risk_quality_score.index)

    rs_score = components["rs_score"].reindex(common_index)
    momentum_score = components["momentum_score"].reindex(common_index)
    trend_score = components["trend_score"].reindex(common_index)
    risk_quality_score = risk_quality_score.reindex(common_index)

    stock_score = (
        cfg.weight_rs_sector * rs_score
        + cfg.weight_momentum * momentum_score
        + cfg.weight_trend * trend_score
        + cfg.weight_risk_quality * risk_quality_score
    )

    # Eligibility masks.
    coverage_ok = pd.DataFrame(True, index=stock_score.index, columns=stock_score.columns)
    for a in stock_score.columns:
        if observations.get(a, 0) < cfg.min_history_days:
            coverage_ok[a] = False
    eligible = coverage_ok.copy()
    if cfg.require_positive_rs6m:
        eligible &= components["rs_6m"].reindex(stock_score.index, columns=stock_score.columns) > 0
    if cfg.require_price_above_ma200:
        eligible &= components["above_ma200"].reindex(stock_score.index, columns=stock_score.columns) > 0
    stock_score_eligible = stock_score.where(eligible)

    # Historical rank among the candidates still eligible each month.
    rank = stock_score_eligible.rank(axis=1, ascending=False, method="min")

    # Selection history: top K within each sector, then optional overall cap.
    selection = build_stock_selection(stock_score_eligible, mapping, cfg)

    current = current_stock_table(
        stock_score=stock_score,
        eligible_score=stock_score_eligible,
        rank=rank,
        components=components,
        risk_quality_score=risk_quality_score,
        mapping=mapping,
        observations=observations,
        selection=selection,
    )

    audit = build_stock_audit(
        stock_prices=stocks,
        stock_score=stock_score_eligible,
        rs_score=rs_score,
        momentum_score=momentum_score,
        trend_score=trend_score,
        risk_quality_score=risk_quality_score,
        selection=selection,
        mapping=mapping,
        cfg=cfg,
    )

    optimizer = build_optimizer_export(stocks, current, cfg)

    return {
        "config": cfg,
        "mapping": mapping,
        "stock_prices": stocks,
        "sector_prices": sectors,
        "components": components,
        "risk_quality_score": risk_quality_score,
        "stock_score": stock_score,
        "eligible_score": stock_score_eligible,
        "rank": rank,
        "selection": selection,
        "current_scores": current,
        "audit": audit,
        **optimizer,
    }


def build_stock_selection(score: pd.DataFrame, mapping: pd.DataFrame, cfg: StockScoreConfig) -> pd.DataFrame:
    sector_map = mapping.set_index("Asset")["Sector_Benchmark"].to_dict()
    rows: List[dict] = []
    for dt in score.index:
        s = score.loc[dt].dropna().sort_values(ascending=False)
        if s.empty:
            continue
        picked: List[Tuple[str, float, str]] = []
        for sector in sorted(set(sector_map.get(a) for a in s.index if a in sector_map)):
            names = [a for a in s.index if sector_map.get(a) == sector]
            for a in names[: cfg.top_per_sector]:
                picked.append((a, float(s[a]), sector))
        picked = sorted(picked, key=lambda x: x[1], reverse=True)[: cfg.max_total_names]
        for i, (asset, val, sector) in enumerate(picked, start=1):
            rows.append({
                "Signal_Month": dt.to_timestamp("M"),
                "Selection_Rank": i,
                "Asset": asset,
                "Sector_Benchmark": sector,
                "Stock_Score": val,
            })
    return pd.DataFrame(rows)


def current_stock_table(
    stock_score: pd.DataFrame,
    eligible_score: pd.DataFrame,
    rank: pd.DataFrame,
    components: Dict[str, pd.DataFrame],
    risk_quality_score: pd.DataFrame,
    mapping: pd.DataFrame,
    observations: Dict[str, int],
    selection: pd.DataFrame,
) -> pd.DataFrame:
    valid = stock_score.dropna(how="all")
    if valid.empty:
        raise ValueError("Nessuno score corrente disponibile.")
    dt = valid.index.max()
    assets = list(stock_score.columns)
    sector_map = mapping.set_index("Asset")["Sector_Benchmark"].to_dict()

    def val(frame: pd.DataFrame, a: str):
        try:
            return frame.loc[dt, a]
        except Exception:
            return np.nan

    selected_now = set()
    if not selection.empty:
        latest = pd.Timestamp(selection["Signal_Month"].max()).to_period("M")
        if latest == dt:
            selected_now = set(selection.loc[pd.to_datetime(selection["Signal_Month"]).dt.to_period("M") == dt, "Asset"])

    rows = []
    for a in assets:
        rows.append({
            "Rank": val(rank, a),
            "Asset": a,
            "Sector_Benchmark": sector_map.get(a, ""),
            "Selected": a in selected_now,
            "Eligible": pd.notna(val(eligible_score, a)),
            "Stock_Score": val(stock_score, a),
            "RS_vs_Sector_Score": val(components["rs_score"], a),
            "Momentum_Score": val(components["momentum_score"], a),
            "Trend_Score": val(components["trend_score"], a),
            "Risk_Quality_Score": val(risk_quality_score, a),
            "RS_3M": val(components["rs_3m"], a),
            "RS_6M": val(components["rs_6m"], a),
            "RS_12M": val(components["rs_12m"], a),
            "Momentum_3M": val(components["mom_3m"], a),
            "Momentum_6M": val(components["mom_6m"], a),
            "Momentum_12_1": val(components["mom_12_1"], a),
            "Above_MA100": val(components["above_ma100"], a),
            "Above_MA200": val(components["above_ma200"], a),
            "Slope_MA200": val(components["slope_ma200"], a),
            "Downside_Vol_126d": val(components["downside_vol"], a),
            "Max_Drawdown_126d": val(components["max_drawdown"], a),
            "Tracking_Error_vs_Sector_126d": val(components["tracking_error"], a),
            "Observations": observations.get(a, 0),
            "As_Of": dt.to_timestamp("M"),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["Selected", "Stock_Score"], ascending=[False, False], na_position="last").reset_index(drop=True)


def build_stock_audit(
    stock_prices: pd.DataFrame,
    stock_score: pd.DataFrame,
    rs_score: pd.DataFrame,
    momentum_score: pd.DataFrame,
    trend_score: pd.DataFrame,
    risk_quality_score: pd.DataFrame,
    selection: pd.DataFrame,
    mapping: pd.DataFrame,
    cfg: StockScoreConfig,
) -> Dict[str, object]:
    # 1) Average monthly Spearman correlations among score blocks.
    blocks = {
        "RS_vs_Sector": rs_score,
        "Momentum": momentum_score,
        "Trend": trend_score,
        "Risk_Quality": risk_quality_score,
    }
    corr_sum = None
    corr_n = 0
    for dt in stock_score.index:
        x = pd.DataFrame({k: v.loc[dt] if dt in v.index else np.nan for k, v in blocks.items()}).dropna()
        if len(x) >= 4:
            c = x.corr(method="spearman")
            corr_sum = c if corr_sum is None else corr_sum.add(c, fill_value=0)
            corr_n += 1
    avg_corr = corr_sum / corr_n if corr_sum is not None and corr_n else pd.DataFrame()

    # 2) Information coefficient: score at month t vs next-month stock return.
    mp = _monthly_last_df(stock_prices)
    fwd = mp.shift(-1) / mp - 1.0
    ic_rows = []
    for dt in stock_score.index.intersection(fwd.index):
        x = pd.concat([stock_score.loc[dt].rename("Score"), fwd.loc[dt].rename("Fwd1M")], axis=1).dropna()
        if len(x) >= 4:
            ic = x["Score"].corr(x["Fwd1M"], method="spearman")
            ic_rows.append({"Month": dt.to_timestamp("M"), "IC_Spearman": ic, "N": len(x)})
    ic_df = pd.DataFrame(ic_rows)

    # 3) Diagnostic EW selection backtest vs all uploaded candidates.
    sel_map: Dict[pd.Period, List[str]] = {}
    if not selection.empty:
        tmp = selection.copy()
        tmp["Period"] = pd.to_datetime(tmp["Signal_Month"]).dt.to_period("M")
        sel_map = tmp.groupby("Period")["Asset"].apply(list).to_dict()
    bt_rows = []
    prev_w: Dict[str, float] = {}
    for dt in sorted(sel_map):
        if dt not in mp.index:
            continue
        pos = mp.index.get_loc(dt)
        if isinstance(pos, slice) or pos + 1 >= len(mp.index):
            continue
        nxt = mp.index[pos + 1]
        names = [a for a in sel_map[dt] if a in mp.columns and pd.notna(mp.loc[dt, a]) and pd.notna(mp.loc[nxt, a])]
        all_names = [a for a in mp.columns if pd.notna(mp.loc[dt, a]) and pd.notna(mp.loc[nxt, a])]
        if not names:
            continue
        rsel = float(np.mean([mp.loc[nxt, a] / mp.loc[dt, a] - 1 for a in names]))
        rall = float(np.mean([mp.loc[nxt, a] / mp.loc[dt, a] - 1 for a in all_names])) if all_names else np.nan
        w = {a: 1.0 / len(names) for a in names}
        union = set(prev_w) | set(w)
        turnover = 0.5 * sum(abs(w.get(a, 0) - prev_w.get(a, 0)) for a in union) if prev_w else 1.0
        bt_rows.append({
            "Signal_Month": dt.to_timestamp("M"),
            "Return_Month": nxt.to_timestamp("M"),
            "Selected_EW_Return": rsel,
            "All_Candidates_EW_Return": rall,
            "Excess": rsel - rall if pd.notna(rall) else np.nan,
            "Turnover": turnover,
            "N_Selected": len(names),
            "Selected": ", ".join(names),
        })
        prev_w = w
    bt = pd.DataFrame(bt_rows)
    if not bt.empty:
        bt["Selected_Index"] = (1 + bt["Selected_EW_Return"]).cumprod() * 100
        bt["All_Index"] = (1 + bt["All_Candidates_EW_Return"]).cumprod() * 100

    # 4) Current weight sensitivity of the selection rule.
    sensitivity = current_weight_sensitivity(
        stock_score=stock_score,
        blocks=blocks,
        mapping=mapping,
        cfg=cfg,
        n_sims=300,
        perturbation=0.20,
        seed=42,
    )

    stats = {}
    if not ic_df.empty:
        stats.update({
            "Mean_IC": float(ic_df["IC_Spearman"].mean()),
            "Median_IC": float(ic_df["IC_Spearman"].median()),
            "Positive_IC_Rate": float((ic_df["IC_Spearman"] > 0).mean()),
            "IC_Months": int(len(ic_df)),
        })
    if not bt.empty:
        sel_growth = float((1 + bt["Selected_EW_Return"]).prod() - 1)
        all_growth = float((1 + bt["All_Candidates_EW_Return"]).prod() - 1)
        stats.update({
            "Diagnostic_Selected_Cumulative": sel_growth,
            "Diagnostic_All_Candidates_Cumulative": all_growth,
            "Diagnostic_Excess_Cumulative": sel_growth - all_growth,
            "Avg_Turnover": float(bt["Turnover"].mean()),
        })
    stats.update(sensitivity)

    return {
        "average_block_correlation": avg_corr,
        "ic_history": ic_df,
        "selection_backtest": bt,
        "stats": stats,
    }


def current_weight_sensitivity(
    stock_score: pd.DataFrame,
    blocks: Dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    cfg: StockScoreConfig,
    n_sims: int = 300,
    perturbation: float = 0.20,
    seed: int = 42,
) -> Dict[str, float]:
    valid = stock_score.dropna(how="all")
    if valid.empty:
        return {"Selection_Stability": np.nan}
    dt = valid.index.max()
    base_score = stock_score.loc[dt].dropna()
    if base_score.empty:
        return {"Selection_Stability": np.nan}

    mapping_dict = mapping.set_index("Asset")["Sector_Benchmark"].to_dict()

    def select(s: pd.Series) -> set:
        picked = []
        for sec in sorted(set(mapping_dict.get(a) for a in s.index if a in mapping_dict)):
            names = [a for a in s.sort_values(ascending=False).index if mapping_dict.get(a) == sec]
            picked.extend(names[: cfg.top_per_sector])
        picked = sorted(picked, key=lambda a: s.get(a, -np.inf), reverse=True)[: cfg.max_total_names]
        return set(picked)

    base_sel = select(base_score)
    if not base_sel:
        return {"Selection_Stability": np.nan}

    block_names = ["RS_vs_Sector", "Momentum", "Trend", "Risk_Quality"]
    base_w = np.array([
        cfg.weight_rs_sector,
        cfg.weight_momentum,
        cfg.weight_trend,
        cfg.weight_risk_quality,
    ], dtype=float)
    rng = np.random.default_rng(seed)
    overlaps = []
    for _ in range(n_sims):
        mult = rng.uniform(1 - perturbation, 1 + perturbation, size=len(base_w))
        w = base_w * mult
        w = w / w.sum()
        score = None
        for wi, name in zip(w, block_names):
            frame = blocks[name]
            if dt not in frame.index:
                continue
            s = frame.loc[dt]
            score = wi * s if score is None else score.add(wi * s, fill_value=0)
        if score is None:
            continue
        sel = select(score.dropna())
        if sel:
            overlaps.append(len(sel & base_sel) / max(len(base_sel), 1))
    return {
        "Selection_Stability": float(np.mean(overlaps)) if overlaps else np.nan,
        "Sensitivity_Simulations": int(len(overlaps)),
    }


def build_optimizer_export(stock_prices: pd.DataFrame, current: pd.DataFrame, cfg: StockScoreConfig) -> Dict[str, pd.DataFrame]:
    selected = current[current["Selected"] & current["Eligible"]].sort_values("Stock_Score", ascending=False)
    assets = selected["Asset"].tolist()
    if not assets:
        return {
            "optimizer_candidates": selected,
            "optimizer_prices_12m": pd.DataFrame(),
            "optimizer_returns_12m": pd.DataFrame(),
        }
    as_of = stock_prices[assets].dropna(how="all").index.max()
    start = as_of - pd.DateOffset(months=cfg.optimizer_lookback_months)
    px = stock_prices.loc[(stock_prices.index > start) & (stock_prices.index <= as_of), assets].copy()
    rets = px.pct_change(fill_method=None)
    return {
        "optimizer_candidates": selected,
        "optimizer_prices_12m": px,
        "optimizer_returns_12m": rets,
    }


def stock_result_to_excel_bytes(result: Dict[str, object]) -> bytes:
    bio = BytesIO()
    cfg: StockScoreConfig = result["config"]
    audit: Dict[str, object] = result["audit"]

    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame({"Parametro": list(cfg.to_dict().keys()), "Valore": list(cfg.to_dict().values())}).to_excel(
            writer, sheet_name="Config", index=False
        )
        result["current_scores"].to_excel(writer, sheet_name="Current_Scores", index=False)
        result["selection"].to_excel(writer, sheet_name="Selection_History", index=False)
        result["mapping"].to_excel(writer, sheet_name="Mapping", index=False)
        result["optimizer_candidates"].to_excel(writer, sheet_name="Optimizer_Candidates", index=False)
        result["optimizer_prices_12m"].to_excel(writer, sheet_name="Optimizer_Prices_12M")
        result["optimizer_returns_12m"].to_excel(writer, sheet_name="Optimizer_Returns_12M")
        result["stock_prices"].to_excel(writer, sheet_name="Clean_Stock_Prices")
        result["sector_prices"].to_excel(writer, sheet_name="Clean_Sector_Prices")
        audit["average_block_correlation"].to_excel(writer, sheet_name="Audit_Block_Corr")
        audit["ic_history"].to_excel(writer, sheet_name="Audit_IC", index=False)
        audit["selection_backtest"].to_excel(writer, sheet_name="Audit_EW_Backtest", index=False)
        pd.DataFrame([audit["stats"]]).to_excel(writer, sheet_name="Audit_Stats", index=False)

        # Historical core score sheets in long form.
        hist = []
        for period in result["stock_score"].index:
            for asset in result["stock_score"].columns:
                val = result["stock_score"].loc[period, asset]
                if pd.notna(val):
                    hist.append({
                        "Month": period.to_timestamp("M"),
                        "Asset": asset,
                        "Stock_Score": val,
                        "Rank": result["rank"].loc[period, asset] if period in result["rank"].index else np.nan,
                    })
        pd.DataFrame(hist).to_excel(writer, sheet_name="Score_History", index=False)

    bio.seek(0)
    return bio.read()
