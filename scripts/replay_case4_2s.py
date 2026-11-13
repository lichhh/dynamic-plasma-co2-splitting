"""Replay every Case 4 solution against the 2-s PJM RegD signal and report how
often the plasma target was clipped and the buffers left their bounds
(Supplementary Fig. 30 caption, Supplementary Note 6).

    python scripts/replay_case4_2s.py

Reads results/{market}/case4/{schedule,design}.csv; writes results/replay_2s_summary.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from plasmaco.data import MARKETS, load_regd_2s_flat       # noqa: E402
from plasmaco.replay import replay_2s                      # noqa: E402


def main() -> None:
    flat = load_regd_2s_flat()
    print(f"RegD signal: {len(flat):,d} samples at 2 s ({len(flat) / 43200:.0f} days)\n")
    rows = []
    for market, (_, _, dt_h) in MARKETS.items():
        case_dir = REPO / "results" / market / "case4"
        if not (case_dir / "schedule.csv").exists():
            print(f"  {market}: no Case 4 solution, skipped")
            continue
        sch = pd.read_csv(case_dir / "schedule.csv")
        des = pd.read_csv(case_dir / "design.csv").iloc[0]
        *_, st = replay_2s(sch, des, flat, dt_h)
        n = st["total_2s"]
        row = {
            "market": market,
            "dt_h": dt_h,
            "steps_2s": n,
            "plasma_target_clipped_pct": st["pct_clip"],
            "mix_buffer_out_of_bounds_pct": 100 * (st["n_mix_hi"] + st["n_mix_lo"]) / n,
            "co2_buffer_out_of_bounds_pct": 100 * (st["n_co2_hi"] + st["n_co2_lo"]) / n,
            "S_mix_min_kmol": st["S_mix_range"][0], "S_mix_max_kmol": st["S_mix_range"][1],
            "S_bar_mix_kmol": st["S_bar_mix"],
            "S_CO2_min_kmol": st["S_CO2_range"][0], "S_CO2_max_kmol": st["S_CO2_range"][1],
            "S_bar_CO2_kmol": st["S_bar_CO2"],
        }
        rows.append(row)
        print(f"  {market:<11s} clipped {row['plasma_target_clipped_pct']:.3f}%  "
              f"mix buffer {row['mix_buffer_out_of_bounds_pct']:.3f}%  "
              f"CO2 buffer {row['co2_buffer_out_of_bounds_pct']:.3f}%")
    out = REPO / "results" / "replay_2s_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
