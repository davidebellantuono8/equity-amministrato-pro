from __future__ import annotations

import json
import csv
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sector_rotation_engine import (
    ModelConfig,
    detect_benchmark,
    export_analysis_excel,
    import_price_files,
    run_analysis,
)
from stock_score_engine import StockScoreConfig, compute_stock_score, stock_result_to_excel_bytes
from factor_leadership_engine import (
    FactorLeadershipConfig,
    compute_factor_leadership,
    evaluate_markowitz_alignment,
    factor_result_to_excel_bytes,
)


APP_TITLE = "CDG Quant Suite"
APP_VERSION = "3.2"

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _save_uploads_to_temp(uploaded_files: Iterable) -> tuple[tempfile.TemporaryDirectory, list[Path]]:
    td = tempfile.TemporaryDirectory()
    paths = []
    used = set()
    for i, uf in enumerate(uploaded_files):
        name = Path(uf.name).name or f"upload_{i}.csv"
        stem, suffix = Path(name).stem, Path(name).suffix
        candidate = name
        j = 1
        while candidate.lower() in used:
            candidate = f"{stem}_{j}{suffix}"
            j += 1
        used.add(candidate.lower())
        p = Path(td.name) / candidate
        p.write_bytes(uf.getvalue())
        paths.append(p)
    return td, paths


def import_uploaded_price_files(uploaded_files) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not uploaded_files:
        raise ValueError("Seleziona almeno un file CSV/Excel.")
    td, paths = _save_uploads_to_temp(uploaded_files)
    try:
        bundle = import_price_files(paths)
        return bundle.prices, bundle.metadata, bundle.warnings
    finally:
        td.cleanup()


def sector_excel_bytes(result, metadata=None) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "CDG_Sector_Rotation.xlsx"
        export_analysis_excel(p, result, metadata)
        return p.read_bytes()


def fmt_pct(x):
    return "" if pd.isna(x) else f"{x:.2%}"


def fmt_num(x):
    return "" if pd.isna(x) else f"{x:.1f}"


def csv_it_bytes(df: pd.DataFrame, index: bool = True) -> bytes:
    """
    CSV compatibile con il formato abituale:
    separatore virgola, decimali con virgola; i campi numerici contenenti
    la virgola decimale vengono quotati automaticamente.
    """
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.strftime("%d/%m/%Y")
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime("%d/%m/%Y")
    return out.to_csv(
        index=index,
        sep=",",
        decimal=",",
        quoting=csv.QUOTE_MINIMAL,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")


def show_top_banner(title: str, subtitle: str):
    st.markdown(
        f"""
        <div style="padding:18px 22px;border-radius:16px;background:linear-gradient(90deg,#0B3A6E,#0B66C3);color:white;margin-bottom:14px;">
            <div style="font-size:28px;font-weight:750;">{title}</div>
            <div style="font-size:14px;opacity:.90;margin-top:4px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_color(v):
    if pd.isna(v):
        return ""
    if v >= 70:
        return "background-color: #dff3e4"
    if v <= 30:
        return "background-color: #fde2e2"
    return "background-color: #fff7d6"


# -----------------------------------------------------------------------------
# Sector Rotation page
# -----------------------------------------------------------------------------

def sector_rotation_page():
    show_top_banner(
        "CDG Sector Rotation",
        "3 anni daily → Composite Score → Top 5 settori → export 12 mesi daily per Markowitz / Black-Litterman",
    )

    with st.expander("Metodo operativo", expanded=False):
        st.markdown(
            """
            **Input consigliato:** almeno 3 anni di prezzi giornalieri dei 10 ETF settoriali + MSCI World.  
            **Score:** Momentum 45% + Relative Strength 25% + Trend 20% + Risk-adjusted Momentum 10%.  
            **Output operativo:** Top 5. L'ottimizzatore esterno lavora poi sui Top 5 con **1 anno di dati daily**, Mean-Variance + Black-Litterman, peso 0–40%, ribilanciamento mensile.
            """
        )

    uploads = st.file_uploader(
        "Carica uno o più CSV / Excel di prezzi",
        type=["csv", "txt", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="sector_uploads",
    )
    c1, c2 = st.columns([1, 5])
    with c1:
        import_clicked = st.button("Importa dati", type="primary", key="sector_import")
    if import_clicked:
        try:
            prices, metadata, warnings = import_uploaded_price_files(uploads)
            st.session_state["sector_prices"] = prices
            st.session_state["sector_metadata"] = metadata
            st.session_state["sector_warnings"] = warnings
            st.session_state.pop("sector_result", None)
            st.success(f"Importate {prices.shape[1]} serie e {prices.shape[0]} date.")
        except Exception as e:
            st.error(str(e))

    if "sector_prices" not in st.session_state:
        st.info("Carica i dati. Il formato può essere un singolo file wide oppure più file CSV/Excel.")
        return

    prices: pd.DataFrame = st.session_state["sector_prices"]
    metadata: pd.DataFrame = st.session_state.get("sector_metadata", pd.DataFrame())
    warnings = st.session_state.get("sector_warnings", [])
    if warnings:
        with st.expander(f"Avvisi import ({len(warnings)})"):
            for w in warnings:
                st.warning(w)

    with st.expander("Controllo dati importati", expanded=False):
        a, b, c = st.columns(3)
        a.metric("Serie", prices.shape[1])
        b.metric("Prima data", prices.index.min().strftime("%d/%m/%Y"))
        c.metric("Ultima data", prices.index.max().strftime("%d/%m/%Y"))
        st.dataframe(metadata, use_container_width=True, hide_index=True)

    benchmark_default = detect_benchmark(prices.columns)
    default_idx = list(prices.columns).index(benchmark_default) if benchmark_default in prices.columns else 0

    st.subheader("Parametri")
    p1, p2, p3, p4, p5 = st.columns(5)
    benchmark = p1.selectbox("Benchmark", list(prices.columns), index=default_idx, key="sector_benchmark")
    top_n = p2.number_input("Top settori", min_value=1, max_value=max(1, prices.shape[1]-1), value=5, step=1)
    buffer = p3.number_input("Buffer ranking", min_value=0, max_value=10, value=2, step=1)
    gap = p4.number_input("Gap sostituzione score", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    lookback = p5.number_input("Export optimizer (mesi)", min_value=6, max_value=36, value=12, step=1)

    with st.expander("Pesi dello score", expanded=False):
        w1, w2, w3, w4 = st.columns(4)
        wm = w1.number_input("Momentum", value=0.45, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        wrs = w2.number_input("Relative Strength", value=0.25, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        wt = w3.number_input("Trend", value=0.20, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        wra = w4.number_input("Risk-adjusted", value=0.10, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        m1, m2, m3, m4 = st.columns(4)
        mw3 = m1.number_input("Momentum 3M", value=0.30, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        mw6 = m2.number_input("Momentum 6M", value=0.40, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        mw12 = m3.number_input("Momentum 12M", value=0.20, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        mw121 = m4.number_input("Momentum 12-1", value=0.10, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")

    f1, f2 = st.columns(2)
    positive_6m = f1.checkbox("Escludi Momentum 6M ≤ 0", value=False)
    above_ma = f2.checkbox("Escludi prezzo sotto MA200", value=False)

    run_clicked = st.button("ESEGUI CDG SECTOR ROTATION", type="primary", use_container_width=True)
    if run_clicked:
        try:
            cfg = ModelConfig(
                top_n=int(top_n),
                selection_buffer=int(buffer),
                replacement_score_gap=float(gap),
                covariance_lookback_months=int(lookback),
                weight_momentum=float(wm),
                weight_relative_strength=float(wrs),
                weight_trend=float(wt),
                weight_risk_adjusted=float(wra),
                mom_w_3m=float(mw3),
                mom_w_6m=float(mw6),
                mom_w_12m=float(mw12),
                mom_w_12_1=float(mw121),
                require_positive_6m=positive_6m,
                require_price_above_ma200=above_ma,
            )
            result = run_analysis(prices, benchmark, cfg)
            st.session_state["sector_result"] = result
            latest = result["composite_selection"]["Signal_Date"].max()
            top5 = result["composite_selection"].loc[
                result["composite_selection"]["Signal_Date"] == latest, "Asset"
            ].tolist()
            st.session_state["sector_top5"] = top5
            st.session_state["sector_benchmark_selected"] = benchmark
            st.success("Modello completato.")
        except Exception as e:
            st.exception(e)

    if "sector_result" not in st.session_state:
        return

    result = st.session_state["sector_result"]
    current = result["current_scores"].copy()
    top5 = st.session_state.get("sector_top5", [])

    tabs = st.tabs(["Ranking corrente", "Top 5", "Backtest diagnostico", "Input Optimizer", "Download"])
    with tabs[0]:
        cols = [
            "Rank", "Asset", "Composite_Score", "Momentum_Score", "Relative_Strength_Score",
            "Trend_Score", "Risk_Adjusted_Score", "Momentum_3M", "Momentum_6M", "Momentum_12M",
            "RS_6M", "Above_MA200",
        ]
        show = current[cols].copy()
        st.dataframe(
            show.style.map(score_color, subset=["Composite_Score"]),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        st.markdown("### Settori selezionati")
        st.write(" → ".join(top5) if top5 else "Nessuna selezione")
        if not current.empty:
            chart_df = current.set_index("Asset")[["Composite_Score"]].sort_values("Composite_Score", ascending=True)
            st.bar_chart(chart_df)

    with tabs[2]:
        st.caption("Il backtest interno è solo un controllo signal-only Equal Weight; l'allocazione operativa viene effettuata nel tuo Markowitz/Black-Litterman.")
        st.dataframe(result["metrics"], use_container_width=True, hide_index=True)
        bt = result["backtests"]
        if not bt.empty:
            idx = bt.pivot_table(index="Return_Date", columns="Strategy", values="Portfolio_Index", aggfunc="first")
            bench = bt.groupby("Return_Date")["Benchmark_Index"].first().rename(benchmark)
            idx = idx.join(bench, how="outer")
            st.line_chart(idx)

    with tabs[3]:
        st.markdown("### Top 5 – 12 mesi daily per Portfolio Optimizer")
        st.dataframe(result["optimizer_input"], use_container_width=True, hide_index=True)
        st.caption("Il file prezzi contiene i Top 5 e il benchmark sugli ultimi 12 mesi (o il lookback scelto sopra).")
        st.dataframe(result["optimizer_daily_prices"].tail(12), use_container_width=True)

    with tabs[4]:
        xlsx = sector_excel_bytes(result, metadata)
        st.download_button(
            "Scarica analisi completa Excel",
            data=xlsx,
            file_name="CDG_Sector_Rotation_Streamlit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        csv_bytes = csv_it_bytes(result["optimizer_daily_prices"], index=True)
        st.download_button(
            "Scarica Top 5 – prezzi 12M CSV",
            data=csv_bytes,
            file_name="CDG_Top5_Optimizer_Prices_12M.csv",
            mime="text/csv",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Stock Score page
# -----------------------------------------------------------------------------

def _initial_mapping(assets: list[str], sector_cols: list[str], old: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    old_map = {}
    if old is not None and not old.empty and {"Asset", "Sector_Benchmark"}.issubset(old.columns):
        old_map = old.set_index("Asset")["Sector_Benchmark"].to_dict()
    default_sector = sector_cols[0] if sector_cols else ""
    return pd.DataFrame({
        "Asset": assets,
        "Sector_Benchmark": [old_map.get(a, default_sector) if old_map.get(a, default_sector) in sector_cols else default_sector for a in assets],
    })


def stock_score_page():
    show_top_banner(
        "CDG Stock Score",
        "Ufficio Studi → settori CDG Top 5 → RS vs settore + Momentum + Trend + Risk Quality → candidati per Markowitz",
    )

    st.markdown(
        """
        **Stock Score di default** = **35% RS vs proprio settore + 25% Momentum + 20% Trend + 20% Risk Quality**.  
        Il peso del settore nel portafoglio ETF **non entra nello Stock Score**: il settore serve come benchmark di forza relativa, mentre l'allocazione finale dei titoli resta al Markowitz.
        """
    )

    st.subheader("1. Dati azionari")
    stock_uploads = st.file_uploader(
        "Carica i prezzi dei titoli selezionati dall'Ufficio Studi (uno o più CSV/Excel)",
        type=["csv", "txt", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="stock_uploads",
    )
    if st.button("Importa titoli", type="primary"):
        try:
            stock_prices, stock_meta, stock_warn = import_uploaded_price_files(stock_uploads)
            st.session_state["stock_prices"] = stock_prices
            st.session_state["stock_metadata"] = stock_meta
            st.session_state["stock_warnings"] = stock_warn
            st.session_state.pop("stock_result", None)
            st.success(f"Importati {stock_prices.shape[1]} titoli.")
        except Exception as e:
            st.error(str(e))

    if "stock_prices" not in st.session_state:
        st.info("Prima importa le serie giornaliere dei titoli.")
        return

    stock_prices: pd.DataFrame = st.session_state["stock_prices"]
    if st.session_state.get("stock_warnings"):
        with st.expander("Avvisi import titoli"):
            for w in st.session_state["stock_warnings"]:
                st.warning(w)

    st.subheader("2. Benchmark settoriali")
    sector_in_session = "sector_prices" in st.session_state
    
    # Inizializza il default di use_session solo la prima volta
    if "stock_use_session" not in st.session_state:
        st.session_state["stock_use_session"] = sector_in_session

    use_session = st.checkbox(
        "Usa i dati già caricati nel CDG Sector Rotation",
        key="stock_use_session",
        disabled=not sector_in_session,
    )
    
    if use_session and sector_in_session:
        sector_prices = st.session_state["sector_prices"]
        st.session_state["stock_sector_prices"] = sector_prices
        st.success("Benchmark settoriali collegati dalla pagina CDG Sector Rotation.")
    else:
        sector_uploads = st.file_uploader(
            "Carica ETF/indici settoriali (e, se vuoi, MSCI World)",
            type=["csv", "txt", "xlsx", "xls", "xlsm"],
            accept_multiple_files=True,
            key="stock_sector_uploads",
        )
        if st.button("Importa benchmark settoriali"):
            try:
                sector_prices, _, _ = import_uploaded_price_files(sector_uploads)
                st.session_state["stock_sector_prices"] = sector_prices
                st.success(f"Importate {sector_prices.shape[1]} serie benchmark.")
            except Exception as e:
                st.error(str(e))

    if "stock_sector_prices" not in st.session_state:
        st.info("Collega o importa i benchmark settoriali prima di procedere.")
        return
    sector_prices = st.session_state["stock_sector_prices"]

    # Exclude broad world benchmark from sector mapping by default when identifiable.
    broad = detect_benchmark(sector_prices.columns)
    sector_cols = [c for c in sector_prices.columns if c != broad]
    if not sector_cols:
        sector_cols = list(sector_prices.columns)

    st.subheader("3. Mappatura titolo → settore")
    # Ricalcola la mappatura base solo quando cambiano titoli o colonne settoriali.
    _mapping_inputs_key = (tuple(stock_prices.columns), tuple(sector_cols))
    if (st.session_state.get("_stock_mapping_inputs_key") != _mapping_inputs_key
            or "_stock_mapping_base" not in st.session_state):
        st.session_state["_stock_mapping_inputs_key"] = _mapping_inputs_key
        st.session_state["_stock_mapping_base"] = _initial_mapping(
            list(stock_prices.columns),
            sector_cols,
            st.session_state.get("stock_mapping"),
        )
        # Svuota gli edit precedenti se cambiano gli input base
        if "stock_mapping_editor" in st.session_state:
            del st.session_state["stock_mapping_editor"]

    # Visualizza la tabella fornendo SOLO il base stabile. Non assegniamo l'output
    # direttamente in session_state per evitare loop continui di re-render in stlite.
    st.data_editor(
        st.session_state["_stock_mapping_base"],
        hide_index=True,
        use_container_width=True,
        disabled=["Asset"],
        column_config={
            "Asset": st.column_config.TextColumn("Titolo"),
            "Sector_Benchmark": st.column_config.SelectboxColumn(
                "Benchmark settoriale",
                options=sector_cols,
                required=True,
            ),
        },
        key="stock_mapping_editor",
    )

    # Ricostruisce il mapping finale applicando gli edit salvati nel widget state
    mapping = st.session_state["_stock_mapping_base"].copy()
    edits = st.session_state.get("stock_mapping_editor", {}).get("edited_rows", {})
    for row_idx, row_edits in edits.items():
        for col, val in row_edits.items():
            mapping.iat[row_idx, mapping.columns.get_loc(col)] = val
            
    st.session_state["stock_mapping"] = mapping

    top5_from_sector = [s for s in st.session_state.get("sector_top5", []) if s in sector_cols]
    default_eligible = top5_from_sector if top5_from_sector else sector_cols
    
    # Previene l'errore di Streamlit sui default se cambiano i sector_cols
    curr_eligible = st.session_state.get("stock_eligible_sectors", default_eligible)
    valid_curr_eligible = [s for s in curr_eligible if s in sector_cols]
    if not valid_curr_eligible:
        valid_curr_eligible = default_eligible

    eligible = st.multiselect(
        "Settori eleggibili (di norma i Top 5 del CDG Sector Rotation)",
        options=sector_cols,
        default=valid_curr_eligible,
        key="stock_eligible_sectors",
    )

    st.subheader("4. Parametri Stock Score")
    p1, p2, p3, p4 = st.columns(4)
    wrs = p1.number_input("RS vs settore", min_value=0.0, max_value=1.0, value=0.35, step=0.05, format="%.2f", key="ss_wrs")
    wm = p2.number_input("Momentum", min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.2f", key="ss_wm")
    wt = p3.number_input("Trend", min_value=0.0, max_value=1.0, value=0.20, step=0.05, format="%.2f", key="ss_wt")
    wr = p4.number_input("Risk Quality", min_value=0.0, max_value=1.0, value=0.20, step=0.05, format="%.2f", key="ss_wr")

    s1, s2, s3, s4 = st.columns(4)
    top_per_sector = s1.number_input("Top titoli per settore", min_value=1, max_value=10, value=2, step=1, key="ss_top_per_sector")
    max_total = s2.number_input("Massimo titoli finali", min_value=1, max_value=30, value=10, step=1, key="ss_max_total")
    min_days = s3.number_input("Minimo giorni storico", min_value=200, max_value=800, value=260, step=20, key="ss_min_days")
    optimizer_months = s4.number_input("Lookback export optimizer", min_value=6, max_value=36, value=12, step=1, key="ss_opt_months")

    f1, f2 = st.columns(2)
    require_rs = f1.checkbox("Escludi titoli con RS 6M vs settore ≤ 0", value=False, key="ss_req_rs")
    require_ma = f2.checkbox("Escludi titoli sotto MA200", value=False, key="ss_req_ma")

    with st.expander("Dettaglio dei quattro blocchi", expanded=False):
        st.markdown(
            """
            **RS vs settore – 35%**  
            3M 30% · 6M 45% · 12M 25%. 50 = in linea con il proprio ETF settoriale; sopra 50 = sovraperformance.

            **Momentum – 25%**  
            3M 30% · 6M 40% · 12-1 30%. Valuta la forza assoluta del titolo, con maggiore enfasi al medio termine.

            **Trend – 20%**  
            Distanza da MA100 30% · distanza da MA200 40% · pendenza MA200 30%.

            **Risk Quality – 20%**  
            Downside volatility 126 giorni 35% · Max Drawdown 126 giorni 30% · Tracking Error vs settore 126 giorni 35%.  
            Qui un rischio idiosincratico più contenuto riceve un punteggio maggiore.
            """
        )

    if st.button("ESEGUI CDG STOCK SCORE", type="primary", use_container_width=True):
        try:
            cfg = StockScoreConfig(
                weight_rs_sector=float(wrs),
                weight_momentum=float(wm),
                weight_trend=float(wt),
                weight_risk_quality=float(wr),
                top_per_sector=int(top_per_sector),
                max_total_names=int(max_total),
                min_history_days=int(min_days),
                optimizer_lookback_months=int(optimizer_months),
                require_positive_rs6m=require_rs,
                require_price_above_ma200=require_ma,
            )
            result = compute_stock_score(
                stock_prices=stock_prices,
                sector_prices=sector_prices,
                mapping=mapping,
                cfg=cfg,
                eligible_sectors=eligible,
            )
            st.session_state["stock_result"] = result
            st.success("Stock Score completato.")
        except Exception as e:
            st.exception(e)

    if "stock_result" not in st.session_state:
        return

    result = st.session_state["stock_result"]
    current = result["current_scores"].copy()
    audit = result["audit"]

    selected = current[current["Selected"] & current["Eligible"]].copy()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Titoli analizzati", int(current["Eligible"].sum()))
    k2.metric("Titoli selezionati", int(selected.shape[0]))
    if audit["stats"].get("Mean_IC") is not None:
        k3.metric("IC medio 1M", f"{audit['stats'].get('Mean_IC', np.nan):.3f}")
    else:
        k3.metric("IC medio 1M", "n.d.")
    stab = audit["stats"].get("Selection_Stability", np.nan)
    k4.metric("Stabilità selezione ±20% pesi", "n.d." if pd.isna(stab) else f"{stab:.1%}")

    tabs = st.tabs(["Ranking", "Candidati Markowitz", "Audit score", "Storico selezione", "Download"])

    with tabs[0]:
        show_cols = [
            "Rank", "Asset", "Sector_Benchmark", "Selected", "Eligible", "Stock_Score",
            "RS_vs_Sector_Score", "Momentum_Score", "Trend_Score", "Risk_Quality_Score",
            "RS_6M", "Momentum_6M", "Above_MA200", "Max_Drawdown_126d",
            "Tracking_Error_vs_Sector_126d",
        ]
        st.dataframe(
            current[show_cols].style.map(score_color, subset=["Stock_Score"]),
            use_container_width=True,
            hide_index=True,
        )
        if not current.empty:
            chart = current.set_index("Asset")[["Stock_Score"]].dropna().sort_values("Stock_Score", ascending=True)
            st.bar_chart(chart)

    with tabs[1]:
        st.markdown("### Universo da passare al Markowitz")
        st.dataframe(
            selected[[
                "Asset", "Sector_Benchmark", "Stock_Score", "RS_vs_Sector_Score", "Momentum_Score",
                "Trend_Score", "Risk_Quality_Score"
            ]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Il peso finale dei titoli non viene determinato dallo Stock Score. Questi sono i candidati; il Portfolio Optimizer usa i 12 mesi daily.")
        st.dataframe(result["optimizer_prices_12m"].tail(15), use_container_width=True)

    with tabs[2]:
        st.markdown("### Correlazione media fra i blocchi")
        corr = audit["average_block_correlation"]
        if not corr.empty:
            st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
        else:
            st.info("Storico insufficiente per la matrice di audit.")

        st.markdown("### Capacità predittiva diagnostica – IC 1 mese")
        ic = audit["ic_history"]
        if not ic.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("IC medio", f"{ic['IC_Spearman'].mean():.3f}")
            c2.metric("IC mediano", f"{ic['IC_Spearman'].median():.3f}")
            c3.metric("Mesi IC positivo", f"{(ic['IC_Spearman'] > 0).mean():.1%}")
            st.line_chart(ic.set_index("Month")[["IC_Spearman"]])
        st.caption("L'IC è un audit dello score sui titoli caricati; non replica il backtest reale dell'Ufficio Studi se la lista storica dei titoli non è point-in-time.")

        bt = audit["selection_backtest"]
        if not bt.empty:
            st.markdown("### Backtest diagnostico Equal Weight della selezione")
            st.line_chart(bt.set_index("Return_Month")[["Selected_Index", "All_Index"]])
            st.dataframe(bt.tail(12), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.dataframe(result["selection"], use_container_width=True, hide_index=True)

    with tabs[4]:
        xlsx = stock_result_to_excel_bytes(result)
        st.download_button(
            "Scarica analisi Stock Score Excel",
            data=xlsx,
            file_name="CDG_Stock_Score.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            "Scarica candidati Markowitz – prezzi 12M CSV",
            data=csv_it_bytes(result["optimizer_prices_12m"], index=True),
            file_name="CDG_StockScore_Optimizer_Prices_12M.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Scarica mapping titolo-settore CSV",
            data=csv_it_bytes(result["mapping"], index=False),
            file_name="CDG_Stock_Mapping.csv",
            mime="text/csv",
            use_container_width=True,
        )




# -----------------------------------------------------------------------------
# Factor Leadership Monitor page
# -----------------------------------------------------------------------------

def factor_leadership_page():
    show_top_banner(
        "CDG Factor Leadership Monitor",
        "Controllo indipendente della leadership Smart Beta: 40% 6M + 30% 3M + 20% 12M + 10% RS vs MSCI World",
    )

    st.markdown(
        """
        Il **Factor Leadership Monitor non assegna i pesi** e non sostituisce Markowitz.
        Serve a verificare se l'allocazione Smart Beta sta effettivamente privilegiando
        i fattori che mostrano la leadership più forte.

        **LeaderScore = 40% Rank 6M + 30% Rank 3M + 20% Rank 12M + 10% RS vs MSCI World.**

        Per evitare una duplicazione matematica del momentum, la componente **RS vs World**
        non viene semplicemente ri-rankata: **50 = in linea con MSCI World**, sopra 50 =
        sovraperformance, sotto 50 = sottoperformance.
        """
    )

    uploads = st.file_uploader(
        "Carica ETF Smart Beta + MSCI World (CSV / Excel, anche più file)",
        type=["csv", "txt", "xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="factor_uploads",
    )

    if st.button("Importa dati Factor Leadership", type="primary", key="factor_import"):
        try:
            prices, metadata, warnings = import_uploaded_price_files(uploads)
            st.session_state["factor_prices"] = prices
            st.session_state["factor_metadata"] = metadata
            st.session_state["factor_warnings"] = warnings
            st.session_state.pop("factor_result", None)
            st.success(f"Importate {prices.shape[1]} serie e {prices.shape[0]} date.")
        except Exception as e:
            st.error(str(e))

    if "factor_prices" not in st.session_state:
        st.info("Input consigliato: 3 anni di prezzi giornalieri. Per il solo score corrente bastano circa 13 mesi.")
        return

    prices: pd.DataFrame = st.session_state["factor_prices"]
    metadata = st.session_state.get("factor_metadata", pd.DataFrame())
    warnings = st.session_state.get("factor_warnings", [])
    if warnings:
        with st.expander(f"Avvisi import ({len(warnings)})"):
            for w in warnings:
                st.warning(w)

    with st.expander("Controllo dati importati", expanded=False):
        a, b, c = st.columns(3)
        a.metric("Serie", prices.shape[1])
        b.metric("Prima data", prices.index.min().strftime("%d/%m/%Y"))
        c.metric("Ultima data", prices.index.max().strftime("%d/%m/%Y"))
        if isinstance(metadata, pd.DataFrame) and not metadata.empty:
            st.dataframe(metadata, use_container_width=True, hide_index=True)

    benchmark_default = detect_benchmark(prices.columns)
    default_idx = list(prices.columns).index(benchmark_default) if benchmark_default in prices.columns else 0

    st.subheader("Parametri del monitor")
    p1, p2, p3, p4 = st.columns(4)
    benchmark = p1.selectbox("Benchmark", list(prices.columns), index=default_idx, key="factor_benchmark")
    rs_horizon = p2.selectbox("RS vs World – orizzonte", [3, 6, 12], index=1, format_func=lambda x: f"{x} mesi")
    rs_scale = p3.number_input(
        "RS scale",
        min_value=0.02,
        max_value=0.50,
        value=0.10,
        step=0.01,
        format="%.2f",
        help="10% significa che una sovra/sottoperformance di circa 10 punti porta lo score RS verso le zone estreme senza renderlo illimitato."
    )
    diagnostic_top_n = p4.number_input("Top N diagnostico", min_value=1, max_value=max(1, prices.shape[1]-1), value=3, step=1)

    max_factor_n = max(1, prices.shape[1] - 1)
    shortlist_default = min(4, max_factor_n)
    shortlist_n = st.number_input(
        "Top N LeaderScore da esportare per un test Markowitz",
        min_value=1,
        max_value=max_factor_n,
        value=shortlist_default,
        step=1,
        help=(
            "Non assegna pesi. Crea una shortlist dei factor leader da passare "
            "al Portfolio Optimizer. Default Top 4."
        ),
    )

    with st.expander("Pesi LeaderScore", expanded=False):
        w1, w2, w3, w4 = st.columns(4)
        w6 = w1.number_input("6 mesi", min_value=0.0, max_value=1.0, value=0.40, step=0.05, format="%.2f")
        w3m = w2.number_input("3 mesi", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.2f")
        w12 = w3.number_input("12 mesi", min_value=0.0, max_value=1.0, value=0.20, step=0.05, format="%.2f")
        wrs = w4.number_input("RS vs World", min_value=0.0, max_value=1.0, value=0.10, step=0.05, format="%.2f")

    if st.button("ESEGUI FACTOR LEADERSHIP MONITOR", type="primary", use_container_width=True):
        try:
            cfg = FactorLeadershipConfig(
                weight_6m=float(w6),
                weight_3m=float(w3m),
                weight_12m=float(w12),
                weight_rs_world=float(wrs),
                rs_horizon_months=int(rs_horizon),
                rs_scale=float(rs_scale),
                diagnostic_top_n=int(diagnostic_top_n),
            )
            result = compute_factor_leadership(prices, benchmark, cfg)
            st.session_state["factor_result"] = result
            st.success("Factor Leadership Monitor completato.")
        except Exception as e:
            st.exception(e)

    if "factor_result" not in st.session_state:
        return

    result = st.session_state["factor_result"]
    current = result["current_ranking"].copy()
    stats = result["stats"]

    k1, k2, k3, k4 = st.columns(4)
    if not current.empty:
        k1.metric("Leader attuale", str(current.iloc[0]["Factor"]))
        k2.metric("LeaderScore", f"{current.iloc[0]['LeaderScore']:.1f}")
    k3.metric("IC medio 1M", "n.d." if pd.isna(stats["Mean_IC"]) else f"{stats['Mean_IC']:.3f}")
    k4.metric("Persistenza Top N", "n.d." if pd.isna(stats["Mean_TopN_Overlap"]) else f"{stats['Mean_TopN_Overlap']:.1%}")

    tabs = st.tabs([
        "Ranking corrente",
        "Confronto Markowitz",
        "Storico leadership",
        "Audit",
        "Download",
    ])

    with tabs[0]:
        show_cols = [
            "Rank", "Factor", "LeaderScore", "Leadership", "Trend_Status",
            "Rank_6M", "Rank_3M", "Rank_12M", "RS_World_Score",
            "Return_3M", "Return_6M", "Return_12M",
            "RS_vs_World_3M", "RS_vs_World_6M", "RS_vs_World_12M",
        ]
        st.dataframe(
            current[show_cols].style.map(score_color, subset=["LeaderScore"]),
            use_container_width=True,
            hide_index=True,
        )
        chart = current.set_index("Factor")[["LeaderScore"]].sort_values("LeaderScore", ascending=True)
        st.bar_chart(chart)

        st.caption(
            "Interpretazione: ≥75 Strong Leader · 60–74 Leader · 40–59 Neutral · "
            "25–39 Laggard · <25 Weak. La soglia non determina automaticamente un acquisto/vendita."
        )

        shortlist = current.head(int(shortlist_n)).copy()
        st.markdown(f"### Shortlist Top {int(shortlist_n)} LeaderScore per test Markowitz")
        st.dataframe(
            shortlist[["Rank", "Factor", "LeaderScore", "Leadership", "Trend_Status"]],
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "La shortlist è un filtro di eleggibilità, non un portafoglio. "
            "I pesi finali restano determinati da Black-Litterman / Markowitz 1Y daily 0–40%. "
            "Confronta il risultato con il Markowitz eseguito sull'intero universo."
        )

    with tabs[1]:
        st.markdown("### Inserisci i pesi prodotti dal tuo Markowitz")
        st.caption(
            "Il monitor non modifica i pesi. Questa tabella misura soltanto se l'allocazione "
            "corrente è coerente con il ranking di leadership."
        )
        default_weights = pd.DataFrame({
            "Factor": current["Factor"].tolist(),
            "Weight": [0.0] * len(current),
        })
        old_weights = st.session_state.get("factor_markowitz_weights")
        if isinstance(old_weights, pd.DataFrame) and set(old_weights.get("Factor", [])) == set(default_weights["Factor"]):
            default_weights = old_weights

        edited = st.data_editor(
            default_weights,
            hide_index=True,
            use_container_width=True,
            disabled=["Factor"],
            column_config={
                "Factor": st.column_config.TextColumn("ETF / Fattore"),
                "Weight": st.column_config.NumberColumn(
                    "Peso Markowitz",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.1,
                    format="%.2f%%",
                    help="Inserisci i pesi in percentuale, es. 40 per 40%."
                ),
            },
            key="factor_weights_editor",
        )
        st.session_state["factor_markowitz_weights"] = edited
        alignment = evaluate_markowitz_alignment(current, edited)
        st.session_state["factor_alignment"] = alignment

        a1, a2, a3 = st.columns(3)
        corr = alignment["spearman_score_weight"]
        a1.metric("Correlazione Score/Peso", "n.d." if pd.isna(corr) else f"{corr:.2f}")
        a2.metric("Peso nei Top 3 leader", f"{alignment['top3_weight_share']:.1%}")
        a3.metric("LeaderScore medio ponderato", f"{alignment['weighted_leader_score']:.1f}")

        if not alignment["table"].empty:
            tab = alignment["table"][["Rank", "Factor", "LeaderScore", "Leadership", "Weight"]].copy()
            tab["Weight"] = tab["Weight"] * 100.0
            st.dataframe(tab, use_container_width=True, hide_index=True)

            active = set(
                alignment["table"].loc[
                    alignment["table"]["Weight"] > 1e-8, "Factor"
                ].astype(str)
            )
            leaders = set(current.head(int(shortlist_n))["Factor"].astype(str))
            overlap = active & leaders
            missed = leaders - active
            outside = active - leaders

            d1, d2, d3 = st.columns(3)
            d1.metric(
                f"Overlap con Top {int(shortlist_n)}",
                f"{len(overlap)}/{len(leaders)}" if leaders else "n.d.",
            )
            d2.metric("Leader non pesati", len(missed))
            d3.metric("Attivi fuori shortlist", len(outside))

            if missed:
                st.write("**Leader Top N con peso Markowitz = 0:** " + ", ".join(sorted(missed)))
            if outside:
                st.write("**Fattori attivi fuori shortlist:** " + ", ".join(sorted(outside)))

            top_factor = str(current.iloc[0]["Factor"]) if not current.empty else ""
            top_factor_weight = float(
                alignment["table"].loc[
                    alignment["table"]["Factor"].astype(str) == top_factor, "Weight"
                ].sum()
            )
            if top_factor and top_factor_weight <= 1e-8:
                st.warning(
                    f"Il leader n.1 ({top_factor}) ha peso 0%. Non è automaticamente un errore: "
                    "Markowitz può considerarlo ridondante per correlazioni, volatilità o rendimenti "
                    "impliciti Black-Litterman. Se vuoi testare una vera factor rotation, prova anche "
                    f"un secondo Markowitz limitato ai Top {int(shortlist_n)} LeaderScore."
                )

        if pd.notna(corr):
            if corr >= 0.50:
                st.success("Allocazione fortemente coerente con la leadership fattoriale.")
            elif corr >= 0.10:
                st.info("Allocazione moderatamente coerente con la leadership.")
            elif corr > -0.10:
                st.warning("Allocazione quasi neutrale rispetto al ranking di leadership.")
            else:
                st.warning("Markowitz sta andando in senso opposto alla leadership: verifica se il motivo è diversificazione/rischio.")

    with tabs[2]:
        hist = result["history"]
        pivot = hist.pivot_table(index="Month", columns="Factor", values="Rank", aggfunc="first")
        if not pivot.empty:
            st.markdown("### Rank mensile")
            rank_view = pivot.tail(24).copy()
            st.dataframe(
                rank_view,
                use_container_width=True,
            )

        score_pivot = hist.pivot_table(index="Month", columns="Factor", values="LeaderScore", aggfunc="first")
        if not score_pivot.empty:
            st.markdown("### Evoluzione LeaderScore")
            st.line_chart(score_pivot.tail(24))

    with tabs[3]:
        st.markdown("### Information Coefficient – mese successivo")
        ic = result["ic_history"]
        if not ic.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("IC medio", f"{ic['IC_Spearman'].mean():.3f}")
            c2.metric("IC mediano", f"{ic['IC_Spearman'].median():.3f}")
            c3.metric("Mesi IC positivo", f"{(ic['IC_Spearman'] > 0).mean():.1%}")
            st.line_chart(ic.set_index("Signal_Month")[["IC_Spearman"]])
        else:
            st.info("Storico insufficiente per l'audit IC.")

        bt = result["diagnostic_backtest"]
        if not bt.empty:
            st.markdown(f"### Backtest diagnostico Top {int(diagnostic_top_n)} Equal Weight")
            idx = bt.set_index("Return_Month")[["TopN_Index", "AllFactors_Index", "Benchmark_Index"]]
            st.line_chart(idx)
            st.caption(
                "Questo backtest è solo diagnostico: il portafoglio operativo resta quello prodotto dal tuo Markowitz/Black-Litterman."
            )

        pers = result["persistence"]
        if not pers.empty:
            st.markdown("### Persistenza della leadership")
            st.metric("Overlap medio Top N mese su mese", f"{pers['TopN_Overlap_vs_Prior'].mean():.1%}")

    with tabs[4]:
        alignment = st.session_state.get("factor_alignment")
        xlsx = factor_result_to_excel_bytes(result, alignment)
        st.download_button(
            "Scarica report Factor Leadership Excel",
            data=xlsx,
            file_name="CDG_Factor_Leadership_Monitor.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        st.download_button(
            "Scarica ranking corrente CSV",
            data=csv_it_bytes(current, index=False),
            file_name="CDG_Factor_Leadership_Current.csv",
            mime="text/csv",
            use_container_width=True,
        )

        shortlist_factors = current.head(int(shortlist_n))["Factor"].tolist()
        daily_factor_prices = prices[shortlist_factors].copy().sort_index()
        if not daily_factor_prices.empty:
            last_date = daily_factor_prices.index.max()
            cutoff = last_date - pd.DateOffset(years=1)
            shortlist_12m = daily_factor_prices.loc[
                daily_factor_prices.index >= cutoff
            ].copy()

            st.download_button(
                f"Scarica Top {int(shortlist_n)} LeaderScore – prezzi 12M daily per Markowitz",
                data=csv_it_bytes(shortlist_12m, index=True),
                file_name=f"CDG_Factor_Top{int(shortlist_n)}_Markowitz_Prices_12M.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(
                "Il file contiene soltanto gli ETF/fattori della shortlist. "
                "MSCI World è escluso dall'universo da ottimizzare."
            )


# -----------------------------------------------------------------------------
# Methodology page
# -----------------------------------------------------------------------------

def methodology_page():
    show_top_banner("Metodologia", "Regole operative della CDG Quant Suite")
    st.markdown(
        """
        ## Flusso operativo mensile

        **1. CDG Sector Rotation**  
        3 anni di dati daily → 10 ETF settoriali + MSCI World → Composite Score → **Top 5 settori**.

        **2. CDG Stock Score**  
        Lista titoli dell'Ufficio Studi → tieni i titoli appartenenti ai Top 5 settori → calcola **RS vs settore + Momentum + Trend + Risk Quality** → scegli i migliori titoli per ciascun settore.

        **3. CDG Factor Leadership Monitor (Smart Beta)**  
        ETF Smart Beta + MSCI World → **40% 6M + 30% 3M + 20% 12M + 10% RS vs World** → ranking di leadership.  
        Uso base: monitor indipendente del Markowitz. Uso tattico da testare: **LeaderScore → Top 4/5 → Markowitz 1Y daily 0–40%**.  

        **4. Portfolio Optimizer**  
        Passa i candidati selezionati con **12 mesi di prezzi giornalieri** → Mean-Variance + Black-Litterman → costruisci il basket mensile.

        ## Principio metodologico
        Il modello separa volutamente tre funzioni:

        - **Research:** qualità/fondamentali, demandata all'Ufficio Studi.
        - **Selection:** leadership settoriale e leadership del singolo titolo, demandata ai due score CDG.
        - **Allocation:** pesi finali, demandati all'ottimizzatore.

        Lo Stock Score **non replica il peso settoriale** risultante dal portafoglio ETF. Il benchmark settoriale serve per misurare la forza relativa del titolo; Markowitz resta libero di determinare i pesi finali.
        """
    )


# -----------------------------------------------------------------------------
# Main navigation
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.caption(f"Streamlit v{APP_VERSION}")
    page = st.radio(
        "Modulo",
        ["CDG Sector Rotation", "CDG Stock Score", "Factor Leadership Monitor", "Metodologia"],
        index=0,
    )
    st.divider()
    st.caption("Configurazione operativa: dati daily, ribilanciamento mensile, score separato dall'allocazione Markowitz.")

if page == "CDG Sector Rotation":
    sector_rotation_page()
elif page == "CDG Stock Score":
    stock_score_page()
elif page == "Factor Leadership Monitor":
    factor_leadership_page()
else:
    methodology_page()
