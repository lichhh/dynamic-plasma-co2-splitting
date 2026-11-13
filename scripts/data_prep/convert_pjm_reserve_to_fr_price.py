"""Extract the hourly PJM regulation market clearing price for July 2020 from the
PJM Data Miner 2 export 'reserve_market_results' (service REG).

mcp (= reg_ccp + reg_pcp, capability plus performance clearing price) is used as
the FR capacity price in eq. S29b. It is written to a column named rmccp_d, which
the Case 4 loader expects, sliced to July 1-31, 2020 EPT (744 hourly rows) to align
with the RegD signal.

    python scripts/data_prep/convert_pjm_reserve_to_fr_price.py \
        --src reserve_market_results.csv --out pjm_FR_price_real_July2020.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("pjm_FR_price_real_July2020.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    df["datetime_ept"] = pd.to_datetime(df["datetime_beginning_ept"])
    df = df[df["service"] == "REG"]                       # safety filter
    df = df.sort_values("datetime_ept").reset_index(drop=True)

    # Slice to July 1 00:00 - July 31 23:00 EPT inclusive
    start = pd.Timestamp("2020-07-01 00:00")
    end   = pd.Timestamp("2020-07-31 23:00")
    mask  = (df["datetime_ept"] >= start) & (df["datetime_ept"] <= end)
    sub   = df.loc[mask, ["datetime_ept", "mcp"]].copy()
    sub.rename(columns={"mcp": "rmccp_d"}, inplace=True)

    print(f"July rows    : {len(sub):,d}  (expected 744)")
    print(f"Price stats  : min {sub['rmccp_d'].min():.2f}  "
          f"mean {sub['rmccp_d'].mean():.2f}  "
          f"max {sub['rmccp_d'].max():.2f}  USD/MWh")
    sub.to_csv(args.out, index=False, date_format="%Y-%m-%d %H:%M")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
