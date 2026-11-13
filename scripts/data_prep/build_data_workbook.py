"""Build data/plasma_data_sources.xlsx from the original downloads (Supplementary Note 5).

Sheets
    README               sheet directory and units
    DE_LU_price          ENTSO-E Germany-Luxembourg day-ahead price, 2025
                         (Sequence 1 = 1 h, Sequence 2 = 15 min), EUR/MWh
    Hou_Pan_price        ERCOT Houston and Panhandle day-ahead (1 h) and real-time
                         (15 min) prices, 2020, USD/MWh; sheet 'Fig. 1a and Fig. 4a'
                         of the Source Data of Wang et al., Nat. Energy 9, 1064 (2024)
    PJM_RegD_signal_2s   PJM RegD ('Dynamic') regulation signal, 2 s, July 2020
    PJM_FR_price_hourly  PJM regulation market clearing price, hourly, July 2020
                         (output of convert_pjm_reserve_to_fr_price.py)

    python scripts/data_prep/build_data_workbook.py \
        --de-csv GUI_ENERGY_PRICES_202412312300-202512312300.csv \
        --ercot-xlsx 41560_2024_1578_MOESM5_ESM.xlsx \
        --regd-xlsx PJM_RegD_July_2020_raw.xlsx \
        --fr-csv pjm_FR_price_real_July2020.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def build_readme() -> pd.DataFrame:
    return pd.DataFrame([
        {"sheet": "DE_LU_price",
         "description": "ENTSO-E Germany-Luxembourg day-ahead price, full year 2025",
         "columns": "MTU (CET/CEST), Sequence (1=hourly, 2=15-min), Day-ahead Price (EUR/MWh)",
         "rows": "~70,000",
         "loaded_by": "load_entsoe_de_lu()"},
        {"sheet": "Hou_Pan_price",
         "description": "ERCOT Houston + Panhandle DAM and RTM prices, full year 2020",
         "columns": "DAM_price_Hou, DAM_price_Pan, RTM_Hou_price, Pan_price (USD/MWh)",
         "rows": "8,784 hourly + 35,136 quarter-hourly",
         "loaded_by": "load_hou_pan()"},
        {"sheet": "PJM_RegD_signal_2s",
         "description": "PJM Dynamic (RegD) regulation signal, 2-second resolution, July 2020",
         "columns": "time-of-day (rows) x date (cols); values in [-1,1]",
         "rows": "43,200 x 31 days",
         "loaded_by": "load_pjm_regd()"},
        {"sheet": "PJM_FR_price_hourly",
         "description": "PJM regulation market clearing price (mcp = capability + performance), hourly, July 2020",
         "columns": "datetime_ept, rmccp_d (USD/MWh; holds mcp)",
         "rows": "744",
         "loaded_by": "load_fr_price()"},
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--de-csv", type=Path, required=True)
    ap.add_argument("--ercot-xlsx", type=Path, required=True)
    ap.add_argument("--regd-xlsx", type=Path, required=True)
    ap.add_argument("--fr-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "plasma_data_sources.xlsx")
    args = ap.parse_args()

    de = pd.read_csv(args.de_csv)[["MTU (CET/CEST)", "Sequence", "Day-ahead Price (EUR/MWh)"]]
    hp = pd.read_excel(args.ercot_xlsx, sheet_name="Fig. 1a and Fig. 4a")
    regd = pd.read_excel(args.regd_xlsx, sheet_name="Dynamic")
    fr = pd.read_csv(args.fr_csv)

    with pd.ExcelWriter(args.out, engine="openpyxl") as w:
        build_readme().to_excel(w, sheet_name="README", index=False)
        de.to_excel(w, sheet_name="DE_LU_price", index=False)
        hp.to_excel(w, sheet_name="Hou_Pan_price", index=False)
        regd.to_excel(w, sheet_name="PJM_RegD_signal_2s", index=False)
        fr.to_excel(w, sheet_name="PJM_FR_price_hourly", index=False)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
