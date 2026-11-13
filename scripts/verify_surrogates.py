"""Evaluate the three ALAMO models on the 189 measured operating points and report
R2 and RMSE per temperature (Supplementary Table 1, Supplementary Fig. 16).

    python scripts/verify_surrogates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from plasmaco.surrogate import co2_conversion, current_A     # noqa: E402


def main() -> None:
    df = pd.read_csv(REPO / "data" / "alamo_training_data.csv")
    print(f"{'T (C)':>5s} {'n':>4s} {'R2':>7s} {'RMSE (%)':>9s} {'max |model - table|':>20s}")
    for T, g in df.groupby("T_C"):
        I = current_A(g["power_kW"].to_numpy())
        pred = co2_conversion(T, I, g["flow_mL_min"].to_numpy())
        meas = g["conversion_measured_pct"].to_numpy()
        r2 = 1.0 - np.sum((meas - pred) ** 2) / np.sum((meas - meas.mean()) ** 2)
        rmse = float(np.sqrt(np.mean((meas - pred) ** 2)))
        dev = float(np.max(np.abs(pred - g["conversion_alamo_pct"].to_numpy())))
        print(f"{T:>5d} {len(g):>4d} {r2:>7.3f} {rmse:>9.2f} {dev:>20.2e}")


if __name__ == "__main__":
    main()
