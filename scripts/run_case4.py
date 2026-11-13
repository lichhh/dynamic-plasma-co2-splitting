"""Solve Case 4 (Case 3 plus frequency regulation) for July 2020.

The same PJM RegD signal and hourly PJM regulation clearing price are applied in
all three markets (Supplementary Note 5), with each market's own July prices.

    python scripts/run_case4.py
    python scripts/run_case4.py --markets de_dam_1h hou_dam pan_dam

Writes results/{market}/case4/{schedule,design,costs,metrics}.csv
(Supplementary Table 10, Fig. 5d, Supplementary Figs. 21-26 and 30).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from plasmaco import ProcessParameters                       # noqa: E402
from plasmaco.data import MARKETS, load_case4_market         # noqa: E402
from plasmaco.model_case4 import build_case4, export_solution, solve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    ap.add_argument("--time-limit", type=float, default=None, help="solver time limit, s")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    args = ap.parse_args()

    summary = []
    for market in args.markets:
        data, dt_h = load_case4_market(market, month=7)
        pi = data.electricity_price["price_USD_per_MWh"].to_numpy()
        print(f"\n=== Case 4 / {market}: {len(data.time_steps)} steps x {dt_h} h, "
              f"mean price {pi.mean():.2f} USD/MWh ===")
        params = ProcessParameters()
        params.FR_enabled = True
        _, _, sym = build_case4(data, params, dt_h=dt_h, solver="cplex")
        res = solve(sym, time_limit_s=args.time_limit, log=False)
        print(f"  {res['model_status']} / {res['solve_status']}  "
              f"LCOC {res['LCOC_USD_per_t_CO']:,.2f} USD/t CO")
        sol = export_solution(sym, csv_dir=str(args.out / market / "case4"))
        summary.append((market, res["LCOC_USD_per_t_CO"], float(sol["costs"]["fr_revenue"])))

    print(f"\n{'market':<12s} {'LCOC (MILP objective)':>22s} {'FR revenue USD/yr':>18s}")
    for market, lcoc, fr in summary:
        print(f"{market:<12s} {lcoc:>22,.2f} {fr:>18,.0f}")


if __name__ == "__main__":
    main()
