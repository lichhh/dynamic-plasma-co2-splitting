"""Loaders for the electricity-price, RegD-signal and FR-price series
(Supplementary Note 5, Supplementary Table 7).

All series are read from data/plasma_data_sources.xlsx, built by
scripts/data_prep/build_data_workbook.py:
    DE_LU_price          ENTSO-E Germany-Luxembourg day-ahead, 2025 (1 h and 15 min)
    Hou_Pan_price        ERCOT Houston / Panhandle, 2020 (day-ahead 1 h, real-time 15 min)
    PJM_RegD_signal_2s   PJM RegD signal, 2 s, July 2020 (time of day x date)
    PJM_FR_price_hourly  PJM regulation market clearing price (mcp), hourly, July 2020

The function bodies are taken from the two scheduling notebooks without change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .parameters import SchedulingData

REPO = Path(__file__).resolve().parents[1]
DATA_XLSX = REPO / "data" / "plasma_data_sources.xlsx"

# Market key -> (loader arguments, time step in h). Keys match the results/ folders.
MARKETS = {
    "de_dam_1h":  ("DE", 1, 1.0),
    "de_dam_15m": ("DE", 2, 0.25),
    "hou_dam":    ("ERCOT", "DAM_Hou", 1.0),
    "hou_rtm":    ("ERCOT", "RTM_Hou", 0.25),
    "pan_dam":    ("ERCOT", "DAM_Pan", 1.0),
    "pan_rtm":    ("ERCOT", "RTM_Pan", 0.25),
}


def load_entsoe_de_lu(
    xlsx_path: str,
    sequence: int = 1,
    fx_eur_to_usd: float = 1.14,
    month: Optional[int] = None,
    sheet: str = "DE_LU_price",
) -> SchedulingData:
    """Load ENTSO-E Germany-Luxembourg DAM prices from the unified xlsx.

    The DE_LU_price sheet keeps two rows per 15-min timestamp:
      Sequence 1 -> hourly DAM (the same value repeats 4 times within an hour)
      Sequence 2 -> 15-min DAM (varies each quarter)

    sequence=1 returns one row per hour (8760 for 2025).
    sequence=2 returns one row per 15 min (35040 for 2025).
    Both are converted EUR -> USD with `fx_eur_to_usd`.

    `month`: optional 1..12 selector slicing to one calendar month of 2025
    (non-leap year). None = full year.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if sequence not in (1, 2):
        raise ValueError(f"sequence must be 1 or 2; got {sequence}")
    sub = df[df["Sequence"] == f"Sequence {sequence}"]
    eur = sub["Day-ahead Price (EUR/MWh)"].astype(float).to_numpy()

    # Sequence 1 in the source is stored at 15-min cadence with repeats; pick
    # every 4th value to get a true hourly series (avoids 4x duplicate rows).
    if sequence == 1:
        eur = eur[::4]

    if month is not None:
        if not (1 <= month <= 12):
            raise ValueError(f"month must be 1..12; got {month}")
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 2025 non-leap
        rows_per_hour = 1 if sequence == 1 else 4
        start = sum(days_per_month[: month - 1]) * 24 * rows_per_hour
        n     = days_per_month[month - 1]         * 24 * rows_per_hour
        eur   = eur[start: start + n]

    usd = eur * fx_eur_to_usd
    time_steps = [f"t{i+1}" for i in range(len(usd))]
    return SchedulingData(
        time_steps=time_steps,
        electricity_price=pd.DataFrame({
            "t": time_steps,
            "price_USD_per_MWh": usd,
        }),
    )


def load_hou_pan(
    xlsx_path: str,
    series: str = "DAM_Hou",
    fx_to_usd: float = 1.0,
    month: Optional[int] = None,
    sheet: str = "Hou_Pan_price",
) -> SchedulingData:
    """Load 2020 ERCOT Houston / Panhandle prices from the unified xlsx
    (Hou_Pan_price sheet, subset of Nature Energy 2024 MOESM5).

    Available series (already in USD/MWh):
        'DAM_Hou'  Day-Ahead Market, Hou cluster, 1 h     -> 8784 full year
        'DAM_Pan'  Day-Ahead Market, Pan cluster, 1 h     -> 8784 full year
        'RTM_Hou'  Real-Time Market, Hou cluster, 15 min  -> 35136 full year
        'RTM_Pan'  Real-Time Market, Pan cluster, 15 min  -> 35136 full year

    `month`: optional 1..12 selector. If given, the returned series is sliced
    to that calendar month of 2020 (leap year).
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    col_map = {
        "DAM_Hou": "DAM_price_Hou (USD/MWh)",
        "DAM_Pan": "DAM_price_Pan (USD/MWh)",
        "RTM_Hou": "RTM_Hou_price (USD/MWh)",
        "RTM_Pan": "Pan_price (USD/MWh)",
    }
    if series not in col_map:
        raise ValueError(f"series must be one of {list(col_map)}; got {series!r}")
    price = df[col_map[series]].dropna().astype(float).to_numpy() * fx_to_usd

    if month is not None:
        if not (1 <= month <= 12):
            raise ValueError(f"month must be 1..12; got {month}")
        days_per_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        rows_per_hour = 4 if series.startswith("RTM") else 1
        start_row = sum(days_per_month[: month - 1]) * 24 * rows_per_hour
        n_rows    = days_per_month[month - 1]            * 24 * rows_per_hour
        price = price[start_row: start_row + n_rows]

    time_steps = [f"t{i+1}" for i in range(len(price))]
    return SchedulingData(
        time_steps=time_steps,
        electricity_price=pd.DataFrame({
            "t": time_steps,
            "price_USD_per_MWh": price,
        }),
    )


def load_market(market: str, month: Optional[int] = None,
                xlsx_path: str | Path = DATA_XLSX) -> tuple[SchedulingData, float]:
    """Return (price series, time step in h) for one of the six market keys."""
    source, arg, dt_h = MARKETS[market]
    if source == "DE":
        data = load_entsoe_de_lu(str(xlsx_path), sequence=arg, month=month)
    else:
        data = load_hou_pan(str(xlsx_path), series=arg, month=month)
    return data, dt_h


def load_pjm_regd(
    xlsx_path: str,
    sheet: str = "PJM_RegD_signal_2s",
    dt_h: float = 1.0,
) -> pd.DataFrame:
    """Aggregate PJM RegD 2-second data into per-`dt_h` blocks.

    The sheet is wide-format: column 0 is time-of-day (00:00:00, 00:00:02, ...),
    columns 1..N are dates (one per day). Each cell is the normalised RegD signal
    in [-1, 1].

    Output rows per day = 24 / dt_h  (e.g. 24 for 1 h, 96 for 15 min).

    Returns DataFrame with columns:
        t            sequential step label ('t1', 't2', ...)
        a_bar        mean RegD signal over the step (used for actual DAM cost)
        a_rms        RMS over the step (rough measure of tracking effort)
        a_max_abs    peak |a| over the step
        max_drift_h  max |cumulative integral| from start of step, in h-of-unit-bandwidth
                     (multiply by R_h MW and mix_kmol_per_MWh to get kmol storage need)
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    points_per_2s   = 1800 * 24                         # 43200 2-s rows per full day
    points_per_step = int(round(dt_h * 3600 / 2))       # 1800 for 1 h, 450 for 15 min
    if points_per_2s % points_per_step != 0:
        raise ValueError(f"dt_h={dt_h} h does not divide a 24 h day evenly into 2 s blocks")
    steps_per_day = points_per_2s // points_per_step

    vals = df.iloc[:points_per_2s, 1:].to_numpy(dtype=float)     # (43200, N_days)
    n_days = vals.shape[1]

    dtau = 2.0 / 3600.0    # 2 s in hours
    rows = []
    for d in range(n_days):
        for s in range(steps_per_day):
            seg = vals[s * points_per_step: (s + 1) * points_per_step, d]
            cum = np.cumsum(seg) * dtau                   # h of unit bandwidth
            rows.append({
                "a_bar":       float(seg.mean()),
                "a_rms":       float(np.sqrt((seg ** 2).mean())),
                "a_max_abs":   float(np.max(np.abs(seg))),
                "max_drift_h": float(np.max(np.abs(cum))),
            })
    out = pd.DataFrame(rows)
    out.insert(0, "t", [f"t{i+1}" for i in range(len(out))])
    return out


def load_regd_2s_flat(xlsx_path: str | Path = DATA_XLSX,
                      sheet: str = "PJM_RegD_signal_2s") -> np.ndarray:
    """The full July 2020 RegD signal as one day-major 2-s array (used for the replay)."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    n_full = 1800 * 24                                 # 43,200 rows per day
    vals = df.iloc[:n_full, 1:].to_numpy(dtype=float)  # (43200, 31)
    return vals.flatten(order="F")                     # day-major


def load_fr_price(
    xlsx_path: str,
    price_col: str = "rmccp_d",
    sheet: str = "PJM_FR_price_hourly",
) -> pd.DataFrame:
    """Read the hourly PJM regulation market clearing price from the unified xlsx
    (PJM_FR_price_hourly sheet). The column is named rmccp_d for historical
    reasons; it holds mcp = reg_ccp + reg_pcp (see convert_pjm_reserve_to_fr_price.py).
    Returns ['t', 'price_USDpMWh']."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if price_col not in df.columns:
        raise ValueError(f"{price_col!r} not in {list(df.columns)}")
    price = df[price_col].astype(float).to_numpy()
    return pd.DataFrame({
        "t": [f"t{i+1}" for i in range(len(price))],
        "price_USDpMWh": price,
    })


def _resample_to_length(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """Expand or truncate a per-row DataFrame to exactly target_n rows."""
    n = len(df)
    if n == target_n:
        return df.reset_index(drop=True)
    if target_n > n and target_n % n == 0:
        factor = target_n // n
        return df.loc[df.index.repeat(factor)].reset_index(drop=True)
    if n > target_n and n % target_n == 0:
        step = n // target_n
        return df.iloc[::step].iloc[:target_n].reset_index(drop=True)
    return df.iloc[:min(n, target_n)].reset_index(drop=True)


def load_case4_market(market: str, month: int = 7,
                      xlsx_path: str | Path = DATA_XLSX) -> tuple[SchedulingData, float]:
    """Attach the PJM RegD statistics and the hourly FR price to one market's
    July prices (run_case4_all_markets.py: build_case4_data)."""
    dam_obj, dt_h = load_market(market, month=month, xlsx_path=xlsx_path)
    dam = dam_obj.electricity_price.copy()

    regd = load_pjm_regd(str(xlsx_path), sheet="PJM_RegD_signal_2s", dt_h=dt_h)
    fr   = load_fr_price(str(xlsx_path), price_col="rmccp_d", sheet="PJM_FR_price_hourly")

    # Resample hourly FR up to DAM resolution if needed
    if len(fr) != len(dam):
        fr = _resample_to_length(fr, len(dam))

    n = min(len(dam), len(regd), len(fr))
    dam  = dam.iloc[:n].reset_index(drop=True)
    regd = regd.iloc[:n].reset_index(drop=True)
    fr   = fr.iloc[:n].reset_index(drop=True)
    t_labels = [f"t{i + 1}" for i in range(n)]
    dam["t"]  = t_labels
    regd["t"] = t_labels
    fr["t"]   = t_labels

    data = SchedulingData(
        time_steps=t_labels,
        electricity_price=dam,
        regd_stats=regd,
        fr_price=fr,
    )
    return data, dt_h
