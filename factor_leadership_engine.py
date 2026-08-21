
from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class FactorLeadershipConfig:
    weight_6m: float = 0.40
    weight_3m: float = 0.30
    weight_12m: float = 0.20
    weight_rs_world: float = 0.10

    # Relative Strength: 50 = in linea con World.
    # tanh rende lo score stabile e limita l'impatto degli outlier.
    rs_horizon_months: int = 6
    rs_scale: float = 0.10

    diagnostic_top_n: int = 3
    transaction_cost_bps: float = 0.0

    def validate(self):
        s = self.weight_6m + self.weight_3m + self.weight_12m + self.weight_rs_world
        if not np.isclose(s, 1.0):
            raise ValueError(f"I pesi del LeaderScore devono sommare a 1. Attuale: {s:.4f}")
        if self.rs_horizon_months not in (3, 6, 12):
            raise ValueError("L'orizzonte RS deve essere 3, 6 o 12 mesi.")
        if self.rs_scale <= 0:
            raise ValueError("RS scale deve essere > 0.")
        if self.diagnostic_top_n < 1:
            raise ValueError("Top N diagnostico deve essere almeno 1.")

    def to_dict(self):
        return asdict(self)


def _monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().resample("ME").last().dropna(how="all")


def _rank_0_100(df: pd.DataFrame) -> pd.DataFrame:
    # Ranking cross-sectional. Il migliore tende a 100.
    return df.rank(axis=1, pct=True, method="average") * 100.0


def _rs_score(excess_return: pd.DataFrame | pd.Series, scale: float) -> pd.DataFrame | pd.Series:
    # 50 = benchmark; ~73 = +0.5*scale; ~88 = +scale; simmetrico sotto 50.
    return 50.0 + 50.0 * np.tanh(excess_return / scale)


def _classify(score: float) -> str:
    if pd.isna(score):
        return "n.d."
    if score >= 75:
        return "Strong Leader"
    if score >= 60:
        return "Leader"
    if score >= 40:
        return "Neutral"
    if score >= 25:
        return "Laggard"
    return "Weak"


def _trend_label(r3, r6, r12, rs6):
    vals = [r3, r6, r12]
    positive = sum(pd.notna(x) and x > 0 for x in vals)
    if positive == 3 and pd.notna(rs6) and rs6 > 0:
        return "Leadership confermata"
    if positive >= 2 and pd.notna(rs6) and rs6 > 0:
        return "Positiva"
    if positive >= 2:
        return "Mista"
    if positive <= 1 and pd.notna(rs6) and rs6 < 0:
        return "Debole"
    return "Neutrale"


def compute_factor_leadership(
    prices: pd.DataFrame,
    benchmark: str,
    cfg: Optional[FactorLeadershipConfig] = None,
) -> Dict:
    cfg = cfg or FactorLeadershipConfig()
    cfg.validate()

    prices = prices.copy().sort_index()
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' non trovato.")

    factors = [c for c in prices.columns if c != benchmark]
    if len(factors) < 2:
        raise ValueError("Servono almeno 2 ETF/fattori oltre al benchmark.")

    mp = _monthly_prices(prices)
    if len(mp) < 13:
        raise ValueError("Servono almeno 13 osservazioni mensili (~1 anno) per il LeaderScore.")

    # Returns
    ret3 = mp[factors].pct_change(3)
    ret6 = mp[factors].pct_change(6)
    ret12 = mp[factors].pct_change(12)

    b3 = mp[benchmark].pct_change(3)
    b6 = mp[benchmark].pct_change(6)
    b12 = mp[benchmark].pct_change(12)

    rs3 = ret3.sub(b3, axis=0)
    rs6 = ret6.sub(b6, axis=0)
    rs12 = ret12.sub(b12, axis=0)

    rank3 = _rank_0_100(ret3)
    rank6 = _rank_0_100(ret6)
    rank12 = _rank_0_100(ret12)

    rs_by_h = {3: rs3, 6: rs6, 12: rs12}
    rs_selected = rs_by_h[cfg.rs_horizon_months]
    rs_score = _rs_score(rs_selected, cfg.rs_scale)

    # Non re-rankiamo il risultato finale: così il 10% RS conserva
    # l'informazione "sopra/sotto MSCI World" e non duplica il puro ranking.
    leader_score = (
        cfg.weight_6m * rank6
        + cfg.weight_3m * rank3
        + cfg.weight_12m * rank12
        + cfg.weight_rs_world * rs_score
    )

    latest = leader_score.dropna(how="all").index.max()
    current = pd.DataFrame({
        "Factor": factors,
        "LeaderScore": leader_score.loc[latest, factors].values,
        "Rank_6M": rank6.loc[latest, factors].values,
        "Rank_3M": rank3.loc[latest, factors].values,
        "Rank_12M": rank12.loc[latest, factors].values,
        "RS_World_Score": rs_score.loc[latest, factors].values,
        "Return_3M": ret3.loc[latest, factors].values,
        "Return_6M": ret6.loc[latest, factors].values,
        "Return_12M": ret12.loc[latest, factors].values,
        "RS_vs_World_3M": rs3.loc[latest, factors].values,
        "RS_vs_World_6M": rs6.loc[latest, factors].values,
        "RS_vs_World_12M": rs12.loc[latest, factors].values,
    })
    current["Rank"] = current["LeaderScore"].rank(ascending=False, method="min").astype("Int64")
    current["Leadership"] = current["LeaderScore"].map(_classify)
    current["Trend_Status"] = [
        _trend_label(r3_, r6_, r12_, rs6_)
        for r3_, r6_, r12_, rs6_ in zip(
            current["Return_3M"], current["Return_6M"], current["Return_12M"], current["RS_vs_World_6M"]
        )
    ]
    current["As_Of"] = latest
    current = current.sort_values(["Rank", "Factor"]).reset_index(drop=True)

    # Long monthly history
    history_rows = []
    valid_dates = leader_score.dropna(how="all").index
    for dt in valid_dates:
        for factor in factors:
            sc = leader_score.loc[dt, factor]
            if pd.isna(sc):
                continue
            history_rows.append({
                "Month": dt,
                "Factor": factor,
                "LeaderScore": float(sc),
                "Rank": int(leader_score.loc[dt].rank(ascending=False, method="min").loc[factor]),
                "Rank_6M": float(rank6.loc[dt, factor]),
                "Rank_3M": float(rank3.loc[dt, factor]),
                "Rank_12M": float(rank12.loc[dt, factor]),
                "RS_World_Score": float(rs_score.loc[dt, factor]),
                "Return_3M": float(ret3.loc[dt, factor]) if pd.notna(ret3.loc[dt, factor]) else np.nan,
                "Return_6M": float(ret6.loc[dt, factor]) if pd.notna(ret6.loc[dt, factor]) else np.nan,
                "Return_12M": float(ret12.loc[dt, factor]) if pd.notna(ret12.loc[dt, factor]) else np.nan,
                "RS_vs_World_6M": float(rs6.loc[dt, factor]) if pd.notna(rs6.loc[dt, factor]) else np.nan,
            })
    history = pd.DataFrame(history_rows)

    # Diagnostic: Information Coefficient vs next month
    monthly_returns = mp.pct_change()
    ic_rows = []
    for dt in valid_dates:
        pos = monthly_returns.index.searchsorted(dt, side="right")
        if pos >= len(monthly_returns.index):
            continue
        next_dt = monthly_returns.index[pos]
        s = leader_score.loc[dt, factors]
        fwd = monthly_returns.loc[next_dt, factors]
        aligned = pd.concat([s.rename("Score"), fwd.rename("Next_Return")], axis=1).dropna()
        if len(aligned) >= 3:
            ic = aligned["Score"].corr(aligned["Next_Return"], method="spearman")
            ic_rows.append({"Signal_Month": dt, "Return_Month": next_dt, "IC_Spearman": ic})
    ic_history = pd.DataFrame(ic_rows)

    # Diagnostic top-N EW vs all factor EW vs benchmark
    bt_rows = []
    prev_top = set()
    for dt in valid_dates:
        pos = monthly_returns.index.searchsorted(dt, side="right")
        if pos >= len(monthly_returns.index):
            continue
        next_dt = monthly_returns.index[pos]

        ranked = leader_score.loc[dt, factors].dropna().sort_values(ascending=False)
        top = ranked.head(min(cfg.diagnostic_top_n, len(ranked))).index.tolist()
        if not top:
            continue

        top_ret = monthly_returns.loc[next_dt, top].dropna().mean()
        all_ret = monthly_returns.loc[next_dt, factors].dropna().mean()
        bench_ret = monthly_returns.loc[next_dt, benchmark]

        current_top = set(top)
        if prev_top:
            turnover = len(current_top.symmetric_difference(prev_top)) / (2 * max(len(current_top), 1))
        else:
            turnover = 1.0
        tc = turnover * cfg.transaction_cost_bps / 10000.0
        top_ret_net = top_ret - tc if pd.notna(top_ret) else np.nan

        bt_rows.append({
            "Signal_Month": dt,
            "Return_Month": next_dt,
            "TopN_Return": top_ret_net,
            "AllFactors_Return": all_ret,
            "Benchmark_Return": bench_ret,
            "Turnover_Approx": turnover,
            "Top_Factors": ", ".join(top),
        })
        prev_top = current_top

    backtest = pd.DataFrame(bt_rows)
    if not backtest.empty:
        backtest["TopN_Index"] = (1 + backtest["TopN_Return"].fillna(0)).cumprod() * 100
        backtest["AllFactors_Index"] = (1 + backtest["AllFactors_Return"].fillna(0)).cumprod() * 100
        backtest["Benchmark_Index"] = (1 + backtest["Benchmark_Return"].fillna(0)).cumprod() * 100

    # Persistence / top overlap
    persist_rows = []
    prev = None
    for dt in valid_dates:
        top = set(leader_score.loc[dt, factors].dropna().nlargest(min(cfg.diagnostic_top_n, len(factors))).index)
        if prev is not None and top:
            overlap = len(top & prev) / len(top)
            persist_rows.append({"Month": dt, "TopN_Overlap_vs_Prior": overlap})
        prev = top
    persistence = pd.DataFrame(persist_rows)

    stats = {
        "As_Of": latest,
        "Mean_IC": float(ic_history["IC_Spearman"].mean()) if not ic_history.empty else np.nan,
        "Median_IC": float(ic_history["IC_Spearman"].median()) if not ic_history.empty else np.nan,
        "Positive_IC_Ratio": float((ic_history["IC_Spearman"] > 0).mean()) if not ic_history.empty else np.nan,
        "Mean_TopN_Overlap": float(persistence["TopN_Overlap_vs_Prior"].mean()) if not persistence.empty else np.nan,
        "Current_TopN": current.head(cfg.diagnostic_top_n)["Factor"].tolist(),
    }

    return {
        "config": cfg.to_dict(),
        "benchmark": benchmark,
        "factors": factors,
        "monthly_prices": mp,
        "current_ranking": current,
        "history": history,
        "ic_history": ic_history,
        "diagnostic_backtest": backtest,
        "persistence": persistence,
        "stats": stats,
    }


def evaluate_markowitz_alignment(
    current_ranking: pd.DataFrame,
    weights: pd.DataFrame,
) -> Dict:
    """
    weights columns: Factor, Weight
    Weight can be 0-1 or 0-100. It is normalized internally.
    """
    if weights is None or weights.empty:
        return {
            "table": pd.DataFrame(),
            "spearman_score_weight": np.nan,
            "top3_weight_share": np.nan,
            "weighted_leader_score": np.nan,
        }

    w = weights[["Factor", "Weight"]].copy()
    w["Weight"] = pd.to_numeric(w["Weight"], errors="coerce").fillna(0.0)
    if w["Weight"].max() > 1.5:
        w["Weight"] = w["Weight"] / 100.0

    total = w["Weight"].sum()
    if total > 0:
        w["Weight"] = w["Weight"] / total

    tab = current_ranking.merge(w, on="Factor", how="left")
    tab["Weight"] = tab["Weight"].fillna(0.0)
    valid = tab[["LeaderScore", "Weight"]].dropna()

    corr = (
        valid["LeaderScore"].corr(valid["Weight"], method="spearman")
        if len(valid) >= 3 and valid["Weight"].nunique() > 1 else np.nan
    )
    top3 = set(tab.nsmallest(3, "Rank")["Factor"])
    top3_share = float(tab.loc[tab["Factor"].isin(top3), "Weight"].sum())
    weighted_score = float((tab["LeaderScore"].fillna(0) * tab["Weight"]).sum())

    return {
        "table": tab.sort_values("Rank"),
        "spearman_score_weight": corr,
        "top3_weight_share": top3_share,
        "weighted_leader_score": weighted_score,
    }


def factor_result_to_excel_bytes(result: Dict, alignment: Optional[Dict] = None) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        result["current_ranking"].to_excel(writer, sheet_name="Current_Ranking", index=False)
        result["history"].to_excel(writer, sheet_name="Monthly_History", index=False)
        result["ic_history"].to_excel(writer, sheet_name="IC_Audit", index=False)
        result["diagnostic_backtest"].to_excel(writer, sheet_name="Diagnostic_Backtest", index=False)
        result["persistence"].to_excel(writer, sheet_name="Persistence", index=False)
        pd.DataFrame([result["config"]]).to_excel(writer, sheet_name="Config", index=False)
        pd.DataFrame([result["stats"]]).to_excel(writer, sheet_name="Stats", index=False)
        if alignment and isinstance(alignment.get("table"), pd.DataFrame) and not alignment["table"].empty:
            alignment["table"].to_excel(writer, sheet_name="Markowitz_Alignment", index=False)
    bio.seek(0)
    return bio.read()
