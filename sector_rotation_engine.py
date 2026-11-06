from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import math
import re
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ModelConfig:
    top_n: int = 5
    selection_buffer: int = 2
    replacement_score_gap: float = 5.0

    require_positive_6m: bool = False
    require_price_above_ma200: bool = False

    # Composite score
    weight_momentum: float = 0.45
    weight_relative_strength: float = 0.25
    weight_trend: float = 0.20
    weight_risk_adjusted: float = 0.10

    # Momentum block
    mom_w_3m: float = 0.30
    mom_w_6m: float = 0.40
    mom_w_12m: float = 0.20
    mom_w_12_1: float = 0.10

    # Relative strength block
    rs_w_3m: float = 0.30
    rs_w_6m: float = 0.45
    rs_w_12m: float = 0.25
    # Scale parameters used to map absolute excess returns vs benchmark to 0-100.
    rs_scale_3m: float = 0.06
    rs_scale_6m: float = 0.10
    rs_scale_12m: float = 0.15

    # Trend block
    trend_w_ma100: float = 0.30
    trend_w_ma200: float = 0.40
    trend_w_slope200: float = 0.30

    # Risk engine / optimizer inputs
    covariance_lookback_months: int = 12
    ewma_lambda: float = 0.97
    expected_return_proxy_min: float = -0.02
    expected_return_proxy_max: float = 0.08

    # Backtest
    transaction_cost_bps: float = 0.0
    risk_free_rate: float = 0.0

    def validate(self) -> None:
        if self.top_n < 1:
            raise ValueError("Top N deve essere almeno 1.")
        if self.selection_buffer < 0:
            raise ValueError("Il buffer non può essere negativo.")
        if self.replacement_score_gap < 0:
            raise ValueError("Il gap di sostituzione non può essere negativo.")
        if not (0 < self.ewma_lambda < 1):
            raise ValueError("EWMA lambda deve essere compreso tra 0 e 1.")

        groups = {
            "Composite": [
                self.weight_momentum,
                self.weight_relative_strength,
                self.weight_trend,
                self.weight_risk_adjusted,
            ],
            "Momentum": [self.mom_w_3m, self.mom_w_6m, self.mom_w_12m, self.mom_w_12_1],
            "Relative Strength": [self.rs_w_3m, self.rs_w_6m, self.rs_w_12m],
            "Trend": [self.trend_w_ma100, self.trend_w_ma200, self.trend_w_slope200],
        }
        for name, vals in groups.items():
            s = float(sum(vals))
            if not np.isclose(s, 1.0, atol=1e-8):
                raise ValueError(f"I pesi {name} devono sommare a 1. Attuale: {s:.4f}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImportBundle:
    prices: pd.DataFrame
    metadata: pd.DataFrame
    warnings: List[str]


# ============================================================
# IMPORTER: CSV / EXCEL / MULTI-FILE
# ============================================================

_DATE_HINTS = {
    "date", "data", "datum", "day", "giorno", "datetime", "timestamp", "time"
}
_ID_HINTS = {
    "ticker", "symbol", "isin", "asset", "instrument", "security", "name", "nome",
    "etf", "titolo", "fund", "fondo"
}

# Smart-beta / factor descriptors.
# These words distinguish a factor ETF from a plain broad MSCI World benchmark.
_FACTOR_STYLE_PATTERNS = [
    r"\bmomentum\b",
    r"\bvalue\b",
    r"\bquality\b",
    r"\bminimum volatility\b",
    r"\bmin\.?\s*vol\b",
    r"\blow volatility\b",
    r"\bmultifactor\b",
    r"\bmulti[-\s]?factor\b",
    r"\bequal weight\b",
    r"\bequal-weight\b",
    r"\bhigh dividend\b",
    r"\bdividend\b",
    r"\bsize\b",
    r"\bsmall cap\b",
    r"\bsmall-cap\b",
    r"\benhanced\b",
    r"\bfactor\b",
    r"\bminimum variance\b",
    r"\bmin\.?\s*variance\b",
]


def _has_factor_style(name: str) -> bool:
    n = _norm(name)
    return any(re.search(pat, n) for pat in _FACTOR_STYLE_PATTERNS)


_PRICE_PRIORITY = [
    "adjusted close", "adj close", "adjclose", "adjusted", "close", "closing price",
    "last", "price", "prezzo", "nav", "value", "valore"
]
_EXCLUDE_NUMERIC_HINTS = {
    "volume", "volumi", "turnover", "open interest", "oi", "bid", "ask", "spread",
    "return", "performance", "perf", "change", "variazione", "percent", "pct", "%",
    "high", "low", "open", "apertura", "massimo", "minimo"
}
_OHLC_HINTS = {"open", "high", "low", "close", "volume", "adj close", "adjusted close"}

_SECTOR_ALIASES = [
    (r"consumer\s+discretionary", "Consumer Discretionary"),
    (r"communication\s+services?", "Communication Services"),
    (r"consumer\s+staples?|con\.?\s*staples?", "Consumer Staples"),
    (r"information\s+tech(?:nology)?|inform\.\s*tech|technology", "Information Technology"),
    (r"health\s*care|healthcare", "Health Care"),
    (r"financials?", "Financials"),
    (r"industrials?", "Industrials"),
    (r"materials?", "Materials"),
    (r"utilities", "Utilities"),
    (r"energy", "Energy"),
    (r"real\s+estate", "Real Estate"),
]


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _clean_col_name(s: object) -> str:
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", s)


def _parse_number_one(v: object) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)

    s = str(v).strip()
    if not s or s.lower() in {"nan", "na", "n/a", "null", "none", "-", "--"}:
        return np.nan

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    # Remove currencies, spaces and common non-numeric symbols, but keep separators/sign/exponent.
    s = s.replace("\u00a0", "").replace(" ", "").replace("'", "")
    s = re.sub(r"[^0-9,\.\-+eE]", "", s)
    if not s:
        return np.nan

    if "," in s and "." in s:
        # Last separator is normally the decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and 1 <= len(parts[1]) <= 6:
            s = parts[0].replace(".", "") + "." + parts[1]
        else:
            s = s.replace(",", "")
    elif s.count(".") > 1:
        parts = s.split(".")
        # Multiple dots: assume thousands separators except the last when plausible.
        if 1 <= len(parts[-1]) <= 6:
            s = "".join(parts[:-1]) + "." + parts[-1]
        else:
            s = "".join(parts)

    try:
        x = float(s)
        return -x if neg else x
    except Exception:
        return np.nan


def parse_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    return s.map(_parse_number_one).astype(float)


def parse_date_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    raw = s.astype(str).str.strip()
    nonempty = raw[(raw != "") & (~raw.str.lower().isin(["nan", "none", "nat"]))]
    dayfirst = True
    if not nonempty.empty:
        # Infer unambiguous slash dates when possible.
        sample = nonempty.head(100)
        m = sample.str.extract(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
        if not m.dropna().empty:
            a = pd.to_numeric(m[0], errors="coerce")
            b = pd.to_numeric(m[1], errors="coerce")
            if (b > 12).any() and not (a > 12).any():
                dayfirst = False
            elif (a > 12).any():
                dayfirst = True

    # format='mixed' available in modern pandas; fallback if necessary.
    try:
        out = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst, format="mixed")
    except TypeError:
        out = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst)
    return out


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        n = _norm(c)
        if n in _DATE_HINTS or any(h in n for h in ["date", "data", "datetime", "timestamp", "trading day"]):
            return c

    best = None
    best_ratio = 0.0
    for c in df.columns:
        raw = df[c]
        # Ordinary numeric price columns are easily misread by pandas as nanosecond timestamps.
        # Only allow numeric date candidates when they look like Excel serial dates.
        if pd.api.types.is_numeric_dtype(raw):
            num = pd.to_numeric(raw, errors="coerce").dropna()
            if num.empty or not ((num > 20000) & (num < 80000)).mean() >= 0.75:
                continue
        parsed = parse_date_series(raw)
        ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        if ratio > best_ratio and ratio >= 0.75:
            best, best_ratio = c, ratio
    return best


def _numeric_candidates(df: pd.DataFrame, date_col: str) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for c in df.columns:
        if c == date_col:
            continue
        parsed = parse_numeric_series(df[c])
        ratio = parsed.notna().mean() if len(parsed) else 0
        if ratio >= 0.65 and parsed.notna().sum() >= 3:
            out[c] = parsed
    return out


def _pick_price_column(candidates: Dict[str, pd.Series]) -> Optional[str]:
    if not candidates:
        return None
    norms = {_norm(c): c for c in candidates}
    for hint in _PRICE_PRIORITY:
        for n, orig in norms.items():
            if n == hint or hint in n:
                return orig
    return next(iter(candidates))


def _auto_alias(name: str, source_stem: str = "", alias_names: bool = True) -> str:
    n = _norm(name)

    # 0) Single stocks: keep the real name ("Infineon Technologies AG" must not become
    #    "Information Technology"); only generic price headers fall back to the file name.
    if not alias_names:
        if n in {"close", "adj close", "adjusted close", "price", "prezzo", "last", "nav", "value", "valore"} and source_stem:
            return source_stem
        return _clean_col_name(name)

    # 1) Sector ETFs: preserve the sector alias even if "MSCI World" is in the fund name.
    for pat, alias in _SECTOR_ALIASES:
        if re.search(pat, n):
            return alias

    # 2) Smart-beta/factor ETFs: NEVER collapse them to "MSCI World".
    #    Examples: MSCI World Momentum, Value, Quality, Minimum Volatility, etc.
    if "msci world" in n and _has_factor_style(n):
        return _clean_col_name(name)

    # 3) Only a plain broad MSCI World series is aliased as the benchmark.
    if "msci world" in n:
        return "MSCI World"

    generic = {
        "close", "adj close", "adjusted close", "price", "prezzo", "last", "nav", "value", "valore"
    }
    if n in generic and source_stem:
        return source_stem
    return _clean_col_name(name)


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 2
    while f"{base} ({i})" in used:
        i += 1
    x = f"{base} ({i})"
    used.add(x)
    return x


def _read_csv_raw(path: Path) -> pd.DataFrame:
    errors = []
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            # sep=None lets Python's CSV sniffer handle comma/semicolon/tab.
            return pd.read_csv(path, sep=None, engine="python", dtype=object, encoding=enc)
        except Exception as e:
            errors.append(f"{enc}: {e}")
    raise ValueError("Impossibile leggere il CSV. " + " | ".join(errors[-2:]))


def _parse_table(df: pd.DataFrame, source: str, sheet: str = "", alias_names: bool = True) -> Tuple[List[pd.DataFrame], List[dict], List[str]]:
    warnings_list: List[str] = []
    if df is None or df.empty:
        return [], [], [f"{source}{' / ' + sheet if sheet else ''}: foglio vuoto."]

    df = df.copy()
    df.columns = [_clean_col_name(c) for c in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        return [], [], [f"{source}: nessun dato utile."]

    date_col = _detect_date_column(df)
    if date_col is None:
        return [], [], [f"{source}{' / ' + sheet if sheet else ''}: colonna data non riconosciuta."]

    dates = parse_date_series(df[date_col])
    valid_dates = dates.notna()
    if valid_dates.mean() < 0.60:
        return [], [], [f"{source}: troppe date non riconosciute nella colonna '{date_col}'."]

    numeric = _numeric_candidates(df, date_col)
    if not numeric:
        return [], [], [f"{source}: nessuna colonna prezzo numerica riconosciuta."]

    # Long format detection: date + identifier + one price field.
    id_col = None
    for c in df.columns:
        if c == date_col:
            continue
        n = _norm(c)
        if n in _ID_HINTS or any(h in n for h in _ID_HINTS):
            if c not in numeric:
                id_col = c
                break

    price_col = _pick_price_column(numeric)
    if id_col is not None and price_col is not None and df[id_col].nunique(dropna=True) > 1:
        tmp = pd.DataFrame({"Date": dates, "Instrument": df[id_col].astype(str), "Price": numeric[price_col]})
        tmp = tmp.dropna(subset=["Date", "Price"])
        if not tmp.empty:
            p = tmp.pivot_table(index="Date", columns="Instrument", values="Price", aggfunc="last").sort_index()
            frames, meta = [], []
            for c in p.columns:
                one = p[[c]].copy()
                frames.append(one)
                meta.append({
                    "original_name": str(c),
                    "suggested_name": _auto_alias(str(c), Path(source).stem, alias_names),
                    "source_file": source,
                    "sheet": sheet,
                    "source_column": price_col,
                    "format": "long",
                })
            return frames, meta, warnings_list

    # If table looks like OHLC data, import only the preferred price column.
    col_norms = {_norm(c): c for c in numeric}
    ohlc_count = sum(any(h == n or h in n for h in _OHLC_HINTS) for n in col_norms)
    single_series_mode = ohlc_count >= 2

    frames: List[pd.DataFrame] = []
    meta: List[dict] = []

    if single_series_mode:
        chosen = _pick_price_column(numeric)
        if chosen is None:
            return [], [], [f"{source}: file OHLC riconosciuto ma nessun prezzo utilizzabile."]
        x = pd.DataFrame({chosen: numeric[chosen].values}, index=dates)
        x = x[~x.index.isna()].sort_index()
        frames.append(x)
        meta.append({
            "original_name": chosen,
            "suggested_name": _auto_alias(chosen, Path(source).stem, alias_names),
            "source_file": source,
            "sheet": sheet,
            "source_column": chosen,
            "format": "single/OHLC",
        })
        return frames, meta, warnings_list

    # Wide format: each numeric column is an instrument, excluding obvious non-price measures.
    imported = 0
    for c, series in numeric.items():
        n = _norm(c)
        if any(h in n for h in _EXCLUDE_NUMERIC_HINTS):
            # Don't exclude a genuine instrument just because its long name happens to contain e.g. 'turnover'.
            if len(n.split()) <= 4:
                continue
        x = pd.DataFrame({c: series.values}, index=dates)
        x = x[~x.index.isna()].sort_index()
        if x[c].notna().sum() < 3:
            continue
        frames.append(x)
        meta.append({
            "original_name": c,
            "suggested_name": _auto_alias(c, Path(source).stem, alias_names),
            "source_file": source,
            "sheet": sheet,
            "source_column": c,
            "format": "wide",
        })
        imported += 1

    if imported == 0:
        warnings_list.append(f"{source}: nessuna serie prezzo importata dopo i filtri.")
    return frames, meta, warnings_list


def import_price_files(paths: Iterable[str | Path], alias_names: bool = True) -> ImportBundle:
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("Nessun file selezionato.")

    raw_series: List[pd.DataFrame] = []
    raw_meta: List[dict] = []
    all_warnings: List[str] = []

    for path in paths:
        if not path.exists():
            all_warnings.append(f"File non trovato: {path}")
            continue
        ext = path.suffix.lower()
        try:
            if ext == ".csv" or ext == ".txt":
                df = _read_csv_raw(path)
                frames, meta, warns = _parse_table(df, path.name, alias_names=alias_names)
                raw_series.extend(frames)
                raw_meta.extend(meta)
                all_warnings.extend(warns)
            elif ext in {".xlsx", ".xls", ".xlsm"}:
                book = pd.read_excel(path, sheet_name=None, dtype=object)
                for sheet_name, df in book.items():
                    frames, meta, warns = _parse_table(df, path.name, str(sheet_name), alias_names=alias_names)
                    raw_series.extend(frames)
                    raw_meta.extend(meta)
                    all_warnings.extend(warns)
            else:
                all_warnings.append(f"Formato non supportato ignorato: {path.name}")
        except Exception as e:
            all_warnings.append(f"Errore importando {path.name}: {e}")

    if not raw_series:
        raise ValueError("Non sono state riconosciute serie prezzo nei file selezionati.")

    used: set[str] = set()
    combined: Optional[pd.DataFrame] = None
    meta_rows: List[dict] = []

    for frame, meta in zip(raw_series, raw_meta):
        orig_col = frame.columns[0]
        alias = _unique_name(meta["suggested_name"], used)
        x = frame.rename(columns={orig_col: alias})
        x = x[~x.index.duplicated(keep="last")].sort_index()
        combined = x if combined is None else combined.join(x, how="outer")

        s = x[alias].dropna()
        meta_rows.append({
            "Name": alias,
            "Original_Name": meta["original_name"],
            "Source_File": meta["source_file"],
            "Sheet": meta["sheet"],
            "Source_Column": meta["source_column"],
            "Detected_Format": meta["format"],
            "Start": s.index.min() if not s.empty else pd.NaT,
            "End": s.index.max() if not s.empty else pd.NaT,
            "Observations": int(s.shape[0]),
        })

    assert combined is not None
    combined = combined.sort_index()
    combined.index.name = "Date"
    combined = combined.dropna(axis=0, how="all")

    metadata = pd.DataFrame(meta_rows)
    return ImportBundle(prices=combined, metadata=metadata, warnings=all_warnings)


def import_price_folder(folder: str | Path, recursive: bool = False) -> ImportBundle:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError("Cartella non valida.")
    globber = folder.rglob if recursive else folder.glob
    files = []
    for pattern in ("*.csv", "*.txt", "*.xlsx", "*.xls", "*.xlsm"):
        files.extend(globber(pattern))
    files = sorted(set(files))
    if not files:
        raise ValueError("Nessun CSV/Excel trovato nella cartella.")
    return import_price_files(files)


def detect_benchmark(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    if not cols:
        return None

    # Exact normalized alias created by the importer gets top priority.
    for c in cols:
        if _norm(c) == "msci world":
            return c

    # Prefer a plain broad MSCI World ETF/Index.
    # Explicitly exclude sector AND smart-beta/factor variants.
    for c in cols:
        n = _norm(c)
        is_sector = any(re.search(pat, n) for pat, _ in _SECTOR_ALIASES)
        if "msci world" in n and not is_sector and not _has_factor_style(n):
            return c

    # Generic "World" fallback, again excluding obvious factor/style and sector series.
    for c in cols:
        n = _norm(c)
        is_sector = any(re.search(pat, n) for pat, _ in _SECTOR_ALIASES)
        if "world" in n and not is_sector and not _has_factor_style(n):
            return c

    # Do not silently choose a factor ETF as benchmark.
    return None


# ============================================================
# SIGNAL ENGINE
# ============================================================

def _monthly_last(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end observations with the ACTUAL last available date as index."""
    if prices.empty:
        return prices.copy()
    periods = prices.index.to_period("M")
    grouped = prices.groupby(periods).last()
    last_dates = pd.Series(prices.index, index=prices.index).groupby(periods).max()
    grouped.index = pd.DatetimeIndex(last_dates.values)
    grouped.index.name = "Date"
    return grouped.sort_index()


def _pct_rank_0_100(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, method="average") * 100.0


def _sample_daily_to_monthly(daily: pd.DataFrame, monthly_dates: pd.DatetimeIndex) -> pd.DataFrame:
    # Monthly dates come from the same master price calendar, but reindex(method=ffill)
    # also handles instruments imported from different trading calendars.
    return daily.sort_index().reindex(monthly_dates, method="ffill")


def _per_asset_daily(daily: pd.DataFrame, fn, ffill_limit: int = 10) -> pd.DataFrame:
    """Apply ``fn`` to each column on its own valid observations, then realign to the master
    calendar (short forward-fill covers dates when only other instruments traded)."""
    out = {}
    for c in daily.columns:
        s = daily[c].dropna()
        if s.empty:
            out[c] = pd.Series(np.nan, index=daily.index)
            continue
        out[c] = fn(s).reindex(daily.index).ffill(limit=ffill_limit)
    return pd.DataFrame(out, index=daily.index, columns=daily.columns)


def _diagnose_empty_selection(prices: pd.DataFrame, signals: Dict[str, pd.DataFrame],
                              benchmark: str, cfg: ModelConfig) -> str:
    comp = signals["composite_score"]
    if comp.notna().any().any():
        filtri = []
        if cfg.require_positive_6m:
            filtri.append("'Escludi Momentum 6M ≤ 0'")
        if cfg.require_price_above_ma200:
            filtri.append("'Escludi prezzo sotto MA200'")
        return ("Nessun settore supera i filtri di eleggibilità in nessuna data"
                + (f" ({' e '.join(filtri)})" if filtri else "") + ". Disattiva i filtri e riprova.")

    mp = signals["monthly_prices"]
    lines = [
        "Composite Score non calcolabile in nessuna data: ogni componente deve essere disponibile "
        "per almeno un settore nello stesso mese.",
        f"Storico: {prices.index.min():%d/%m/%Y} → {prices.index.max():%d/%m/%Y} "
        f"({len(mp)} mesi; servono almeno 13 mesi, consigliati 3 anni).",
    ]
    checks = {
        "Momentum (3/6/12M)": "momentum_score",
        f"Relative Strength vs benchmark '{benchmark}'": "rs_score",
        "Trend (MA100/MA200, ≥160 osservazioni giornaliere)": "trend_score",
        "Risk-adjusted (vol 126 giorni)": "risk_adj_score",
    }
    for label, key in checks.items():
        ok = signals[key].notna().any(axis=1).sum()
        lines.append(f"- {label}: {'OK' if ok else 'MANCANTE'} ({ok} mesi validi)")

    bench_obs = prices[benchmark].dropna()
    lines.append(
        f"Benchmark '{benchmark}': {len(bench_obs)} osservazioni"
        + (f" dal {bench_obs.index.min():%d/%m/%Y}" if not bench_obs.empty else "")
        + ". Verifica che sia la serie corretta (es. MSCI World) e con storico sufficiente."
    )
    short = [c for c in prices.columns if c != benchmark and prices[c].notna().sum() < 260]
    if short:
        lines.append("Serie con meno di ~1 anno di dati giornalieri: " + ", ".join(map(str, short)) +
                     ". Se i file sono settimanali/mensili, servono prezzi giornalieri.")
    return "\n".join(lines)


def compute_signals(prices: pd.DataFrame, benchmark: str, cfg: ModelConfig) -> Dict[str, pd.DataFrame]:
    cfg.validate()
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' non presente nei dati.")
    assets = [c for c in prices.columns if c != benchmark]
    if len(assets) < cfg.top_n:
        raise ValueError(f"Hai {len(assets)} settori utilizzabili ma Top N={cfg.top_n}.")

    p = prices.sort_index().copy()
    mp = _monthly_last(p)
    mr = mp.pct_change(fill_method=None)

    # Momentum
    mom_1m = mp[assets].pct_change(1, fill_method=None)
    mom_3m = mp[assets].pct_change(3, fill_method=None)
    mom_6m = mp[assets].pct_change(6, fill_method=None)
    mom_12m = mp[assets].pct_change(12, fill_method=None)
    mom_12_1 = mp[assets].shift(1) / mp[assets].shift(12) - 1

    momentum_raw = (
        cfg.mom_w_3m * mom_3m
        + cfg.mom_w_6m * mom_6m
        + cfg.mom_w_12m * mom_12m
        + cfg.mom_w_12_1 * mom_12_1
    )
    momentum_score = _pct_rank_0_100(momentum_raw)

    # Relative Strength vs benchmark
    bench_3m = mp[benchmark].pct_change(3, fill_method=None)
    bench_6m = mp[benchmark].pct_change(6, fill_method=None)
    bench_12m = mp[benchmark].pct_change(12, fill_method=None)

    rs_3m = mom_3m.sub(bench_3m, axis=0)
    rs_6m = mom_6m.sub(bench_6m, axis=0)
    rs_12m = mom_12m.sub(bench_12m, axis=0)
    rs_raw = cfg.rs_w_3m * rs_3m + cfg.rs_w_6m * rs_6m + cfg.rs_w_12m * rs_12m

    # IMPORTANT: do not cross-sectionally rank simple excess returns here.
    # Subtracting the same benchmark from every sector can otherwise make the RS ranking
    # almost redundant with momentum. We instead score ABSOLUTE out/under-performance
    # around zero using a bounded tanh transform. 50 = in line with benchmark.
    def rs_bounded(x: pd.DataFrame, scale: float) -> pd.DataFrame:
        return 50.0 + 50.0 * np.tanh(x / scale)

    rs_score = (
        cfg.rs_w_3m * rs_bounded(rs_3m, cfg.rs_scale_3m)
        + cfg.rs_w_6m * rs_bounded(rs_6m, cfg.rs_scale_6m)
        + cfg.rs_w_12m * rs_bounded(rs_12m, cfg.rs_scale_12m)
    )

    # Trend on daily data. Each asset is computed on its OWN trading calendar: with mixed
    # calendars (EU/US ETFs, indices quoted on holidays/weekends) the union index has many
    # NaN rows and a 200-row window would never reach min_periods -> Trend always NaN.
    daily_assets = p[assets]

    def _above_ma(window: int, min_p: int):
        return lambda s: s / s.rolling(window, min_periods=min_p).mean() - 1

    def _slope_ma200(s: pd.Series) -> pd.Series:
        m = s.rolling(200, min_periods=160).mean()
        return (m / m.shift(20)) ** (252.0 / 20.0) - 1

    above_ma100 = _per_asset_daily(daily_assets, _above_ma(100, 80))
    above_ma200 = _per_asset_daily(daily_assets, _above_ma(200, 160))
    slope_ma200 = _per_asset_daily(daily_assets, _slope_ma200)

    monthly_dates = mp.index
    above_ma100_m = _sample_daily_to_monthly(above_ma100, monthly_dates)
    above_ma200_m = _sample_daily_to_monthly(above_ma200, monthly_dates)
    slope_ma200_m = _sample_daily_to_monthly(slope_ma200, monthly_dates)
    trend_score = (
        cfg.trend_w_ma100 * _pct_rank_0_100(above_ma100_m)
        + cfg.trend_w_ma200 * _pct_rank_0_100(above_ma200_m)
        + cfg.trend_w_slope200 * _pct_rank_0_100(slope_ma200_m)
    )

    # Risk-adjusted momentum: 6M return / realized vol (126 trading days)
    daily_vol_126 = _per_asset_daily(
        daily_assets,
        lambda s: s.pct_change(fill_method=None).rolling(126, min_periods=80).std(ddof=1) * np.sqrt(252),
    )
    vol_126_m = _sample_daily_to_monthly(daily_vol_126, monthly_dates)
    risk_adj_raw = mom_6m / vol_126_m.replace(0, np.nan)
    risk_adj_score = _pct_rank_0_100(risk_adj_raw)

    composite_raw = (
        cfg.weight_momentum * momentum_score
        + cfg.weight_relative_strength * rs_score
        + cfg.weight_trend * trend_score
        + cfg.weight_risk_adjusted * risk_adj_score
    )
    composite_score = _pct_rank_0_100(composite_raw)

    return {
        "monthly_prices": mp,
        "monthly_returns": mr,
        "mom_1m": mom_1m,
        "mom_3m": mom_3m,
        "mom_6m": mom_6m,
        "mom_12m": mom_12m,
        "mom_12_1": mom_12_1,
        "momentum_raw": momentum_raw,
        "momentum_score": momentum_score,
        "rs_3m": rs_3m,
        "rs_6m": rs_6m,
        "rs_12m": rs_12m,
        "rs_raw": rs_raw,
        "rs_score": rs_score,
        "above_ma100": above_ma100_m,
        "above_ma200": above_ma200_m,
        "slope_ma200": slope_ma200_m,
        "trend_score": trend_score,
        "vol_126": vol_126_m,
        "risk_adj_raw": risk_adj_raw,
        "risk_adj_score": risk_adj_score,
        "composite_raw": composite_raw,
        "composite_score": composite_score,
    }


# ============================================================
# SELECTION / TURNOVER CONTROL
# ============================================================

def build_selection(
    signals: Dict[str, pd.DataFrame],
    cfg: ModelConfig,
    ranking_key: str = "composite_score",
) -> pd.DataFrame:
    score = signals[ranking_key]
    mom6 = signals["mom_6m"].reindex(score.index)
    above200 = signals["above_ma200"].reindex(score.index)

    rows: List[dict] = []
    incumbents: List[str] = []

    for dt in score.index:
        s = score.loc[dt].dropna().sort_values(ascending=False)
        if s.empty:
            continue

        eligible: List[str] = []
        for a in s.index:
            if cfg.require_positive_6m:
                if pd.isna(mom6.loc[dt, a]) or mom6.loc[dt, a] <= 0:
                    continue
            if cfg.require_price_above_ma200:
                if pd.isna(above200.loc[dt, a]) or above200.loc[dt, a] <= 0:
                    continue
            eligible.append(a)

        if not eligible:
            incumbents = []
            continue

        ranks = {a: i + 1 for i, a in enumerate(eligible)}
        # Incumbents can survive only within Top N + buffer.
        selected = [
            a for a in incumbents
            if a in ranks and ranks[a] <= cfg.top_n + cfg.selection_buffer
        ]

        # If too many survived (can happen after top_n changes), keep strongest.
        selected = sorted(selected, key=lambda a: s[a], reverse=True)[:cfg.top_n]

        # Fill empty slots immediately.
        for cand in eligible:
            if len(selected) >= cfg.top_n:
                break
            if cand not in selected:
                selected.append(cand)

        # Strong entrants can replace the weakest incumbent/candidate only if the score gap
        # is sufficiently large. This avoids flip-flop around the cut-off.
        for cand in eligible[: cfg.top_n]:
            if cand in selected:
                continue
            weakest = min(selected, key=lambda a: s[a])
            if float(s[cand] - s[weakest]) >= cfg.replacement_score_gap:
                selected.remove(weakest)
                selected.append(cand)

        # Guarantee capacity and sort for presentation.
        selected = sorted(selected, key=lambda a: s[a], reverse=True)[:cfg.top_n]
        incumbents = selected.copy()

        for pos, asset in enumerate(selected, start=1):
            rows.append({
                "Signal_Date": dt,
                "Selected_Rank": pos,
                "Raw_Rank": ranks.get(asset, np.nan),
                "Asset": asset,
                "Ranking_Method": ranking_key,
                "Ranking_Score": float(s[asset]),
                "Composite_Score": float(signals["composite_score"].loc[dt, asset])
                    if pd.notna(signals["composite_score"].loc[dt, asset]) else np.nan,
                "Momentum_6M": float(mom6.loc[dt, asset]) if pd.notna(mom6.loc[dt, asset]) else np.nan,
                "Above_MA200": float(above200.loc[dt, asset]) if pd.notna(above200.loc[dt, asset]) else np.nan,
            })

    return pd.DataFrame(rows)


# ============================================================
# BACKTEST
# ============================================================

def _selection_map(selection: pd.DataFrame) -> Dict[pd.Timestamp, List[str]]:
    if selection.empty:
        return {}
    return selection.groupby("Signal_Date")["Asset"].apply(list).to_dict()


def backtest_selection_ew(
    signals: Dict[str, pd.DataFrame],
    selection: pd.DataFrame,
    benchmark: str,
    cfg: ModelConfig,
    strategy_name: str,
) -> pd.DataFrame:
    mp = signals["monthly_prices"]
    sel_map = _selection_map(selection)
    if not sel_map:
        return pd.DataFrame()

    dates = list(mp.index)
    date_pos = {d: i for i, d in enumerate(dates)}
    rows = []
    prev_weights: Dict[str, float] = {}

    for signal_date in sorted(sel_map):
        if signal_date not in date_pos:
            continue
        i = date_pos[signal_date]
        if i + 1 >= len(dates):
            continue
        next_date = dates[i + 1]
        assets = sel_map[signal_date]

        available = [
            a for a in assets
            if a in mp.columns
            and pd.notna(mp.loc[signal_date, a])
            and pd.notna(mp.loc[next_date, a])
            and mp.loc[signal_date, a] != 0
        ]
        if not available:
            continue

        w = {a: 1.0 / len(available) for a in available}
        union = set(prev_weights) | set(w)
        turnover = 0.5 * sum(abs(w.get(a, 0.0) - prev_weights.get(a, 0.0)) for a in union)
        if not prev_weights:
            turnover = 1.0

        asset_rets = [mp.loc[next_date, a] / mp.loc[signal_date, a] - 1 for a in available]
        gross_ret = float(np.mean(asset_rets))
        net_ret = gross_ret - turnover * cfg.transaction_cost_bps / 10000.0

        bench_ret = (
            float(mp.loc[next_date, benchmark] / mp.loc[signal_date, benchmark] - 1)
            if pd.notna(mp.loc[signal_date, benchmark]) and pd.notna(mp.loc[next_date, benchmark])
            else np.nan
        )

        rows.append({
            "Strategy": strategy_name,
            "Signal_Date": signal_date,
            "Return_Date": next_date,
            "Portfolio_Return": net_ret,
            "Benchmark_Return": bench_ret,
            "Excess_Return": net_ret - bench_ret if pd.notna(bench_ret) else np.nan,
            "Turnover": turnover,
            "N_Assets": len(available),
            "Assets": ", ".join(available),
        })
        prev_weights = w

    bt = pd.DataFrame(rows)
    if not bt.empty:
        bt["Portfolio_Index"] = (1 + bt["Portfolio_Return"]).cumprod() * 100
        bt["Benchmark_Index"] = (1 + bt["Benchmark_Return"]).cumprod() * 100
    return bt


def backtest_all_assets_ew(
    signals: Dict[str, pd.DataFrame], benchmark: str, evaluation_dates: Iterable[pd.Timestamp]
) -> pd.DataFrame:
    mp = signals["monthly_prices"]
    assets = [c for c in mp.columns if c != benchmark]
    evaluation_dates = list(evaluation_dates)
    rows = []

    for d in evaluation_dates:
        if d not in mp.index:
            continue
        i = mp.index.get_loc(d)
        if isinstance(i, slice) or i == 0:
            continue
        prev = mp.index[i - 1]
        available = [
            a for a in assets
            if pd.notna(mp.loc[prev, a]) and pd.notna(mp.loc[d, a]) and mp.loc[prev, a] != 0
        ]
        if not available:
            continue
        r = float(np.mean([mp.loc[d, a] / mp.loc[prev, a] - 1 for a in available]))
        br = float(mp.loc[d, benchmark] / mp.loc[prev, benchmark] - 1)
        rows.append({
            "Strategy": "All Sectors Equal Weight",
            "Signal_Date": prev,
            "Return_Date": d,
            "Portfolio_Return": r,
            "Benchmark_Return": br,
            "Excess_Return": r - br,
            "Turnover": 0.0,
            "N_Assets": len(available),
            "Assets": ", ".join(available),
        })

    bt = pd.DataFrame(rows)
    if not bt.empty:
        bt["Portfolio_Index"] = (1 + bt["Portfolio_Return"]).cumprod() * 100
        bt["Benchmark_Index"] = (1 + bt["Benchmark_Return"]).cumprod() * 100
    return bt


def _perf_stats(r: pd.Series, rf: float = 0.0) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {k: np.nan for k in [
            "Cumulative_Return", "Ann_Return", "Ann_Vol", "Sharpe", "Max_Drawdown",
            "Best_Month", "Worst_Month", "Positive_Months"
        ]}
    growth = float((1 + r).prod())
    ann_return = growth ** (12.0 / len(r)) - 1
    ann_vol = float(r.std(ddof=1) * np.sqrt(12))
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return {
        "Cumulative_Return": growth - 1,
        "Ann_Return": ann_return,
        "Ann_Vol": ann_vol,
        "Sharpe": sharpe,
        "Max_Drawdown": float(dd.min()),
        "Best_Month": float(r.max()),
        "Worst_Month": float(r.min()),
        "Positive_Months": float((r > 0).mean()),
    }


def metrics_for_backtest(bt: pd.DataFrame, strategy_name: str, cfg: ModelConfig) -> pd.DataFrame:
    if bt.empty:
        return pd.DataFrame()
    p = _perf_stats(bt["Portfolio_Return"], cfg.risk_free_rate)
    b = _perf_stats(bt["Benchmark_Return"], cfg.risk_free_rate)
    aligned = bt[["Portfolio_Return", "Benchmark_Return"]].dropna()
    if len(aligned) >= 2:
        excess = aligned["Portfolio_Return"] - aligned["Benchmark_Return"]
        te = float(excess.std(ddof=1) * np.sqrt(12))
        corr = float(aligned.corr().iloc[0, 1])
        var_b = float(aligned["Benchmark_Return"].var(ddof=1))
        beta = float(aligned.cov().iloc[0, 1] / var_b) if var_b > 0 else np.nan
        hit = float((excess > 0).mean())
    else:
        te = corr = beta = hit = np.nan
    ir = (p["Ann_Return"] - b["Ann_Return"]) / te if pd.notna(te) and te > 0 else np.nan

    return pd.DataFrame([{
        "Strategy": strategy_name,
        **p,
        "Benchmark_Ann_Return": b["Ann_Return"],
        "Ann_Excess_Return": p["Ann_Return"] - b["Ann_Return"],
        "Tracking_Error": te,
        "Information_Ratio": ir,
        "Beta": beta,
        "Correlation": corr,
        "Hit_Rate_vs_Benchmark": hit,
        "Avg_Turnover": float(bt["Turnover"].mean()) if "Turnover" in bt else np.nan,
        "Months": int(len(bt)),
    }])


# ============================================================
# RISK ENGINE / OPTIMIZER INPUTS
# ============================================================

def _daily_returns_window(prices: pd.DataFrame, assets: List[str], as_of: pd.Timestamp, months: int) -> pd.DataFrame:
    start = as_of - pd.DateOffset(months=months)
    r = prices.loc[(prices.index > start) & (prices.index <= as_of), assets].pct_change(fill_method=None)
    return r


def _ledoit_wolf_cov(daily_returns: pd.DataFrame) -> pd.DataFrame:
    cols = list(daily_returns.columns)
    r = daily_returns.dropna(axis=0, how="any")
    if len(r) < 20:
        return daily_returns.cov() * 252
    if HAS_SKLEARN:
        model = LedoitWolf().fit(r.values)
        cov = model.covariance_ * 252
        return pd.DataFrame(cov, index=cols, columns=cols)
    sample = r.cov().values
    target = np.diag(np.diag(sample))
    cov = (0.8 * sample + 0.2 * target) * 252
    return pd.DataFrame(cov, index=cols, columns=cols)


def _ewma_cov(daily_returns: pd.DataFrame, lam: float) -> pd.DataFrame:
    cols = list(daily_returns.columns)
    r = daily_returns.dropna(axis=0, how="any")
    if len(r) < 2:
        return r.cov() * 252
    x = r.values
    n = len(x)
    weights = np.array([(1 - lam) * lam ** (n - 1 - i) for i in range(n)], dtype=float)
    weights /= weights.sum()
    mu = np.average(x, axis=0, weights=weights)
    xc = x - mu
    cov = (xc * weights[:, None]).T @ xc * 252
    return pd.DataFrame(cov, index=cols, columns=cols)


def expected_return_proxy(current_scores: pd.Series, cfg: ModelConfig) -> pd.Series:
    s = current_scores.astype(float)
    if s.max() == s.min():
        return pd.Series((cfg.expected_return_proxy_min + cfg.expected_return_proxy_max) / 2, index=s.index)
    z = (s - s.min()) / (s.max() - s.min())
    return cfg.expected_return_proxy_min + z * (cfg.expected_return_proxy_max - cfg.expected_return_proxy_min)


def build_risk_engine(
    prices: pd.DataFrame,
    signals: Dict[str, pd.DataFrame],
    selection: pd.DataFrame,
    benchmark: str,
    cfg: ModelConfig,
) -> Dict[str, pd.DataFrame]:
    if selection.empty:
        raise ValueError("Nessuna selezione disponibile per costruire gli input dell'ottimizzatore.")
    as_of = pd.Timestamp(selection["Signal_Date"].max())
    current_sel = selection[selection["Signal_Date"] == as_of].sort_values("Selected_Rank")
    assets = current_sel["Asset"].tolist()

    dr = _daily_returns_window(prices, assets, as_of, cfg.covariance_lookback_months)
    sample = dr.cov() * 252
    lw = _ledoit_wolf_cov(dr)
    ewma = _ewma_cov(dr, cfg.ewma_lambda)
    corr = dr.corr()

    current_score = signals["composite_score"].loc[as_of, assets]
    proxy = expected_return_proxy(current_score, cfg)
    vol = pd.Series(np.sqrt(np.maximum(np.diag(lw.reindex(index=assets, columns=assets).values), 0)), index=assets)

    inp = pd.DataFrame({
        "Asset": assets,
        "Composite_Score": current_score.reindex(assets).values,
        "Expected_Return_Proxy": proxy.reindex(assets).values,
        "Vol_LedoitWolf_Ann": vol.reindex(assets).values,
        "Momentum_Score": signals["momentum_score"].loc[as_of, assets].values,
        "Relative_Strength_Score": signals["rs_score"].loc[as_of, assets].values,
        "Trend_Score": signals["trend_score"].loc[as_of, assets].values,
        "Risk_Adjusted_Score": signals["risk_adj_score"].loc[as_of, assets].values,
        "Momentum_3M": signals["mom_3m"].loc[as_of, assets].values,
        "Momentum_6M": signals["mom_6m"].loc[as_of, assets].values,
        "Momentum_12M": signals["mom_12m"].loc[as_of, assets].values,
        "RS_6M": signals["rs_6m"].loc[as_of, assets].values,
        "Above_MA200": signals["above_ma200"].loc[as_of, assets].values,
        "As_Of": as_of,
    })

    start = as_of - pd.DateOffset(months=cfg.covariance_lookback_months)
    optimizer_prices = prices.loc[(prices.index > start) & (prices.index <= as_of), assets + [benchmark]].copy()

    return {
        "optimizer_input": inp,
        "cov_ledoit_wolf": lw,
        "cov_ewma": ewma,
        "cov_sample": sample,
        "correlation": corr,
        "optimizer_daily_prices": optimizer_prices,
        "optimizer_daily_returns": dr,
    }


# ============================================================
# CURRENT TABLES / FULL ANALYSIS
# ============================================================

def current_scores_table(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cs = signals["composite_score"].dropna(how="all")
    if cs.empty:
        raise ValueError("Serie storica insufficiente per calcolare il Composite Score (servono circa 12 mesi).")
    dt = cs.index.max()
    table = pd.DataFrame({
        "Asset": cs.columns,
        "Composite_Score": signals["composite_score"].loc[dt].values,
        "Momentum_Score": signals["momentum_score"].loc[dt].values,
        "Relative_Strength_Score": signals["rs_score"].loc[dt].values,
        "Trend_Score": signals["trend_score"].loc[dt].values,
        "Risk_Adjusted_Score": signals["risk_adj_score"].loc[dt].values,
        "Momentum_1M": signals["mom_1m"].loc[dt].values,
        "Momentum_3M": signals["mom_3m"].loc[dt].values,
        "Momentum_6M": signals["mom_6m"].loc[dt].values,
        "Momentum_12M": signals["mom_12m"].loc[dt].values,
        "Momentum_12_1": signals["mom_12_1"].loc[dt].values,
        "RS_3M": signals["rs_3m"].loc[dt].values,
        "RS_6M": signals["rs_6m"].loc[dt].values,
        "RS_12M": signals["rs_12m"].loc[dt].values,
        "Above_MA100": signals["above_ma100"].loc[dt].values,
        "Above_MA200": signals["above_ma200"].loc[dt].values,
        "Slope_MA200": signals["slope_ma200"].loc[dt].values,
        "Vol_126d_Ann": signals["vol_126"].loc[dt].values,
        "As_Of": dt,
    })
    table["Rank"] = table["Composite_Score"].rank(ascending=False, method="min").astype("Int64")
    cols = ["Rank"] + [c for c in table.columns if c != "Rank"]
    return table[cols].sort_values("Rank")


def _signals_long(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["momentum_score", "rs_score", "trend_score", "risk_adj_score", "composite_score"]
    out = None
    for k in keys:
        x = signals[k].copy()
        x.index.name = "Date"
        x = x.reset_index().melt(id_vars="Date", var_name="Asset", value_name=k)
        out = x if out is None else out.merge(x, on=["Date", "Asset"], how="outer")
    return out.sort_values(["Date", "composite_score"], ascending=[True, False])


def run_analysis(prices: pd.DataFrame, benchmark: str, cfg: Optional[ModelConfig] = None) -> Dict[str, object]:
    cfg = cfg or ModelConfig()
    cfg.validate()
    prices = prices.copy().sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")

    signals = compute_signals(prices, benchmark, cfg)
    composite_selection = build_selection(signals, cfg, "composite_score")
    if composite_selection.empty:
        raise ValueError(_diagnose_empty_selection(prices, signals, benchmark, cfg))
    momentum_selection = build_selection(signals, cfg, "momentum_score")

    bt_comp = backtest_selection_ew(signals, composite_selection, benchmark, cfg, "Composite Top N EW")
    bt_mom = backtest_selection_ew(signals, momentum_selection, benchmark, cfg, "Momentum Top N EW")

    # Use the composite evaluation window for a fair apples-to-apples comparison.
    eval_dates = bt_comp["Return_Date"].tolist() if not bt_comp.empty else []
    bt_all = backtest_all_assets_ew(signals, benchmark, eval_dates)

    # Restrict momentum to the same return dates if possible.
    if eval_dates and not bt_mom.empty:
        bt_mom = bt_mom[bt_mom["Return_Date"].isin(eval_dates)].reset_index(drop=True)
        if not bt_mom.empty:
            bt_mom["Portfolio_Index"] = (1 + bt_mom["Portfolio_Return"]).cumprod() * 100
            bt_mom["Benchmark_Index"] = (1 + bt_mom["Benchmark_Return"]).cumprod() * 100

    metrics = pd.concat([
        metrics_for_backtest(bt_comp, "Composite Top N EW", cfg),
        metrics_for_backtest(bt_mom, "Momentum Top N EW", cfg),
        metrics_for_backtest(bt_all, "All Sectors Equal Weight", cfg),
    ], ignore_index=True)

    backtests = pd.concat([bt_comp, bt_mom, bt_all], ignore_index=True, sort=False)
    risk = build_risk_engine(prices, signals, composite_selection, benchmark, cfg)
    current = current_scores_table(signals)

    return {
        "config": cfg,
        "benchmark": benchmark,
        "prices": prices,
        "signals": signals,
        "signals_long": _signals_long(signals),
        "current_scores": current,
        "composite_selection": composite_selection,
        "momentum_selection": momentum_selection,
        "backtests": backtests,
        "metrics": metrics,
        **risk,
    }


# ============================================================
# EXPORT TO EXCEL (used locally by the desktop app)
# ============================================================

def export_analysis_excel(path: str | Path, result: Dict[str, object], metadata: Optional[pd.DataFrame] = None) -> Path:
    """Export a polished workbook. Uses openpyxl on the user's local machine."""
    from openpyxl import load_workbook
    from openpyxl.chart import LineChart, Reference
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        path = path.with_suffix(".xlsx")

    cfg: ModelConfig = result["config"]
    benchmark: str = result["benchmark"]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Summary / config
        summary = pd.DataFrame({
            "Parametro": [
                "Benchmark", "Top N", "Selection buffer", "Replacement score gap",
                "Momentum weight", "Relative Strength weight", "Trend weight", "Risk-adjusted weight",
                "Covariance lookback months", "EWMA lambda", "Transaction cost bps",
                "Dati da", "Dati a", "Numero serie"
            ],
            "Valore": [
                benchmark, cfg.top_n, cfg.selection_buffer, cfg.replacement_score_gap,
                cfg.weight_momentum, cfg.weight_relative_strength, cfg.weight_trend, cfg.weight_risk_adjusted,
                cfg.covariance_lookback_months, cfg.ewma_lambda, cfg.transaction_cost_bps,
                result["prices"].index.min(), result["prices"].index.max(), result["prices"].shape[1]
            ]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)
        result["current_scores"].to_excel(writer, sheet_name="Current_Scores", index=False)
        result["metrics"].to_excel(writer, sheet_name="Metrics", index=False)
        result["composite_selection"].to_excel(writer, sheet_name="Composite_Selection", index=False)
        result["momentum_selection"].to_excel(writer, sheet_name="Momentum_Selection", index=False)
        result["signals_long"].to_excel(writer, sheet_name="Monthly_Scores", index=False)
        result["optimizer_input"].to_excel(writer, sheet_name="Optimizer_Input", index=False)
        result["cov_ledoit_wolf"].to_excel(writer, sheet_name="Cov_LedoitWolf")
        result["cov_ewma"].to_excel(writer, sheet_name="Cov_EWMA")
        result["cov_sample"].to_excel(writer, sheet_name="Cov_Sample")
        result["correlation"].to_excel(writer, sheet_name="Correlation")
        result["optimizer_daily_prices"].to_excel(writer, sheet_name="Optimizer_Prices_12M")
        result["optimizer_daily_returns"].to_excel(writer, sheet_name="Optimizer_Returns_12M")
        result["backtests"].to_excel(writer, sheet_name="Backtest", index=False)
        result["prices"].to_excel(writer, sheet_name="Clean_Prices")
        if metadata is not None and not metadata.empty:
            metadata.to_excel(writer, sheet_name="Import_Metadata", index=False)

        # Wide backtest index sheet for easy charting.
        bt = result["backtests"]
        if not bt.empty:
            wide = bt.pivot_table(index="Return_Date", columns="Strategy", values="Portfolio_Return", aggfunc="first")
            # Add benchmark using first available benchmark return by date.
            br = bt.groupby("Return_Date")["Benchmark_Return"].first().rename(benchmark)
            wide = wide.join(br, how="outer").sort_index()
            idx = (1 + wide.fillna(0)).cumprod() * 100
            idx.to_excel(writer, sheet_name="Backtest_Index")

    wb = load_workbook(path)
    blue = "0B66C3"
    light_blue = "DCEEFF"
    dark = "1F2937"

    # Generic formatting
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor=blue)
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 24

        max_col = min(ws.max_column, 40)
        for col_idx in range(1, max_col + 1):
            letter = get_column_letter(col_idx)
            max_len = 0
            for cell in ws[letter][: min(ws.max_row, 300)]:
                v = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(v))
            ws.column_dimensions[letter].width = min(max(max_len + 2, 11), 34)

        # Percentage-like columns
        for cell in ws[1]:
            h = str(cell.value or "").lower()
            if any(k in h for k in ["return", "vol", "drawdown", "momentum_", "rs_", "above_", "slope", "turnover", "hit_rate", "correlation", "beta"]):
                if "score" not in h and "rank" not in h and "correlation" not in h and "beta" not in h:
                    for row in range(2, ws.max_row + 1):
                        ws.cell(row, cell.column).number_format = "0.00%"

    # Conditional color scale for score columns.
    if "Current_Scores" in wb.sheetnames:
        ws = wb["Current_Scores"]
        headers = {c.value: c.column for c in ws[1]}
        for h in ["Composite_Score", "Momentum_Score", "Relative_Strength_Score", "Trend_Score", "Risk_Adjusted_Score"]:
            if h in headers and ws.max_row >= 2:
                col = get_column_letter(headers[h])
                ws.conditional_formatting.add(
                    f"{col}2:{col}{ws.max_row}",
                    ColorScaleRule(start_type="min", start_color="F8696B", mid_type="percentile", mid_value=50,
                                   mid_color="FFEB84", end_type="max", end_color="63BE7B")
                )

    # Add a backtest line chart.
    if "Backtest_Index" in wb.sheetnames:
        ws = wb["Backtest_Index"]
        if ws.max_row > 2 and ws.max_column > 1:
            chart = LineChart()
            chart.title = "Backtest comparativo (base 100)"
            chart.y_axis.title = "Indice"
            chart.x_axis.title = "Data"
            data = Reference(ws, min_col=2, max_col=ws.max_column, min_row=1, max_row=ws.max_row)
            cats = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            chart.height = 10
            chart.width = 20
            ws.add_chart(chart, f"{get_column_letter(ws.max_column + 2)}2")

    wb.save(path)
    return path


def save_config_json(path: str | Path, cfg: ModelConfig, benchmark: str = "") -> Path:
    path = Path(path)
    payload = {"benchmark": benchmark, "config": cfg.to_dict()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_config_json(path: str | Path) -> Tuple[ModelConfig, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = ModelConfig(**payload.get("config", {}))
    return cfg, payload.get("benchmark", "")
