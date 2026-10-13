from __future__ import annotations

import difflib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from sector_rotation_engine import _SECTOR_ALIASES, _norm


DEFAULT_MARKET_DATA_FILE = "market_data_settori.csv"

# Settori Morningstar (market data) -> alias GICS usati dall'importatore per gli ETF settoriali.
MORNINGSTAR_TO_GICS = {
    "technology": "Information Technology",
    "financial services": "Financials",
    "healthcare": "Health Care",
    "consumer cyclical": "Consumer Discretionary",
    "consumer defensive": "Consumer Staples",
    "basic materials": "Materials",
    "industrials": "Industrials",
    "energy": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "communication services": "Communication Services",
}

STATUS_FOUND = "Trovato"
STATUS_FUZZY = "Da verificare"
STATUS_NO_BENCH = "Benchmark non caricato"
STATUS_NOT_FOUND = "Non trovato"
STATUS_MANUAL = "Manuale"

_ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")

# Forme societarie e parole "rumore" ignorate nel confronto dei nomi.
_LEGAL_TOKENS = {
    "plc", "ag", "sa", "spa", "nv", "se", "inc", "corp", "corporation", "co", "ltd", "limited",
    "llc", "lp", "ab", "asa", "as", "oyj", "oy", "sab", "de", "cv", "bv", "kgaa", "gmbh", "kk",
    "holding", "holdings", "group", "the", "company", "cl", "class", "a", "b", "c", "reg", "shs",
    "ord", "adr", "sca", "sapa", "srl", "pcl", "tbk", "bhd", "nyrt", "dd", "ad", "aps", "hf",
    # articoli/preposizioni e parole generiche ("Banca Monte dei Paschi di Siena", "Delta Air Lines (New)")
    "di", "dei", "del", "della", "delle", "degli", "du", "des", "la", "le", "les", "of", "and", "und",
    "et", "y", "new", "compagnie", "cie",
}
_FUZZY_CUTOFF = 0.88

_COL_HINTS = {
    "name": ["nome", "name", "titolo", "security", "company", "descrizione"],
    "isin": ["isin"],
    "sector": ["settore", "sector"],
    "industry": ["industria", "industry"],
}


def normalize_company_name(name: object) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t not in _LEGAL_TOKENS]
    return " ".join(tokens)


def gics_from_sector_label(sector: object) -> Optional[str]:
    n = _norm(sector)
    if not n or n == "nan":
        return None
    if n in MORNINGSTAR_TO_GICS:
        return MORNINGSTAR_TO_GICS[n]
    for pat, alias in _SECTOR_ALIASES:
        if re.search(pat, n):
            return alias
    return None


def _find_col(columns: Iterable[str], kind: str) -> Optional[str]:
    cols = list(columns)
    for hint in _COL_HINTS[kind]:
        for c in cols:
            if _norm(c) == hint:
                return c
    for hint in _COL_HINTS[kind]:
        for c in cols:
            if hint in _norm(c):
                return c
    return None


def _standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Riconosce le colonne Nome / ISIN / Settore / Industria anche se l'header non è sulla prima riga."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if _find_col(df.columns, "name") is None or _find_col(df.columns, "sector") is None:
        for i in range(min(10, len(df))):
            header = [str(v).strip() for v in df.iloc[i].tolist()]
            if _find_col(header, "name") is not None and _find_col(header, "sector") is not None:
                df = df.iloc[i + 1:].copy()
                df.columns = header
                break
    name_col = _find_col(df.columns, "name")
    sector_col = _find_col(df.columns, "sector")
    if name_col is None or sector_col is None:
        raise ValueError("Market data: colonne 'Nome' e 'Settore' non trovate.")
    isin_col = _find_col(df.columns, "isin")
    industry_col = _find_col(df.columns, "industry")
    out = pd.DataFrame({
        "Nome": df[name_col].astype(str).str.strip(),
        "ISIN": df[isin_col].astype(str).str.strip().str.upper() if isin_col else "",
        "Settore": df[sector_col].astype(str).str.strip(),
        "Industria": df[industry_col].astype(str).str.strip() if industry_col else "",
    })
    out = out[(out["Nome"] != "") & (out["Nome"].str.lower() != "nan")]
    out = out[(out["Settore"] != "") & (out["Settore"].str.lower() != "nan")]
    return out.drop_duplicates("Nome", keep="first").reset_index(drop=True)


def load_market_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        raw = pd.read_excel(path, dtype=object)
    else:
        raw = pd.read_csv(path, dtype=object, sep=None, engine="python")
    return _standardize(raw)


def load_default_market_data() -> Optional[pd.DataFrame]:
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        p = base / DEFAULT_MARKET_DATA_FILE
        if p.exists():
            return load_market_data(p)
    return None


def _tokens_cover(short: list[str], long: list[str], allow_prefix: bool) -> bool:
    """True se ogni token di `short` trova un token distinto in `long` (uguale, o abbreviazione se allow_prefix)."""
    free = list(long)
    for a in short:
        for i, m in enumerate(free):
            if a == m or (allow_prefix and len(a) >= 3 and m.startswith(a)):
                del free[i]
                break
        else:
            return False
    return True


def _token_match(key: str, name_keys: list[str]) -> Optional[str]:
    """Abbinamento per parole: 'Infineon Tech' -> 'Infineon Technologies', 'Adobe Systems' -> 'Adobe'.
    Accettato solo se il candidato è unico."""
    toks = key.split()
    if sum(len(t) for t in toks) < 5:
        return None
    hits = []
    for k in name_keys:
        kt = k.split()
        if _tokens_cover(toks, kt, allow_prefix=True) or (
            sum(len(t) for t in kt) >= 5 and _tokens_cover(kt, toks, allow_prefix=False)
        ):
            hits.append(k)
            if len(hits) > 1:
                return None
    return hits[0] if hits else None


def _resolve_benchmark(gics: Optional[str], sector_cols: list[str]) -> Optional[str]:
    if gics is None:
        return None
    for c in sector_cols:
        if c == gics:
            return c
    for c in sector_cols:
        if gics_from_sector_label(c) == gics:
            return c
    return None


def suggest_sectors(
    assets: list[str],
    sector_cols: list[str],
    market_data: Optional[pd.DataFrame],
    stock_metadata: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Per ogni titolo: settore market data, benchmark suggerito e stato dell'abbinamento."""
    rows = []
    md = market_data if market_data is not None else pd.DataFrame(columns=["Nome", "ISIN", "Settore"])
    by_isin = {i: r for i, r in zip(md["ISIN"], md.to_dict("records")) if isinstance(i, str) and _ISIN_RE.fullmatch(i)}
    by_name: dict[str, dict] = {}
    for r in md.to_dict("records"):
        by_name.setdefault(normalize_company_name(r["Nome"]), r)
    name_keys = [k for k in by_name if k]

    # Testo extra dal metadata d'import (nome originale, file, foglio): utile se l'ISIN o il nome
    # completo stanno nel nome file anziché nell'intestazione della colonna.
    extra_text: dict[str, str] = {}
    original_names: dict[str, list[str]] = {}
    if stock_metadata is not None and not stock_metadata.empty and "Name" in stock_metadata.columns:
        for _, m in stock_metadata.iterrows():
            asset_name = str(m["Name"])
            # File/foglio identificano il titolo solo se contengono una sola serie.
            fields = ["Original_Name"]
            if str(m.get("Detected_Format", "")) == "single/OHLC":
                fields += ["Source_File", "Sheet"]
            extra_text[asset_name] = " ".join(str(m.get(c, "")) for c in fields)
            original_names.setdefault(asset_name, [])
            for c in fields:
                v = str(m.get(c, "") or "")
                if c == "Source_File":
                    v = Path(v).stem
                if v and v.lower() != "nan":
                    original_names[asset_name].append(v)

    for asset in assets:
        hit, how = None, None
        isins = _ISIN_RE.findall(f"{asset} {extra_text.get(asset, '')}".upper())
        for code in isins:
            if code in by_isin:
                hit, how = by_isin[code], "isin"
                break
        if hit is None:
            candidates = [asset] + original_names.get(asset, [])
            for cand in candidates:
                key = normalize_company_name(cand)
                if key and key in by_name:
                    hit, how = by_name[key], "name"
                    break
            if hit is None:
                for cand in candidates:
                    key = normalize_company_name(cand)
                    close = difflib.get_close_matches(key, name_keys, n=1, cutoff=_FUZZY_CUTOFF) if key else []
                    if close:
                        hit, how = by_name[close[0]], "fuzzy"
                        break
            if hit is None:
                for cand in candidates:
                    key = normalize_company_name(cand)
                    k = _token_match(key, name_keys) if key else None
                    if k:
                        hit, how = by_name[k], "fuzzy"
                        break

        if hit is None:
            rows.append({"Asset": asset, "MD_Nome": "", "MD_Settore": "", "Suggested_Benchmark": None, "Match_Status": STATUS_NOT_FOUND})
            continue
        bench = _resolve_benchmark(gics_from_sector_label(hit["Settore"]), sector_cols)
        if bench is None:
            status = STATUS_NO_BENCH
        elif how == "fuzzy":
            status = STATUS_FUZZY
        else:
            status = STATUS_FOUND
        rows.append({
            "Asset": asset,
            "MD_Nome": hit["Nome"],
            "MD_Settore": hit["Settore"],
            "Suggested_Benchmark": bench,
            "Match_Status": status,
        })
    return pd.DataFrame(rows, columns=["Asset", "MD_Nome", "MD_Settore", "Suggested_Benchmark", "Match_Status"])
