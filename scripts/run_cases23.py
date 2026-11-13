"""Solve Cases 2 and 3 for the six market-resolution series.

Full year (Supplementary Table 9, Figs. 5b,c,e,f, Supplementary Figs. 21-26, 31):
    python scripts/run_cases23.py
July 2020 slice, for comparison with Case 4 (Supplementary Table 10, Fig. 5d):
    python scripts/run_cases23.py --month 7
Subsets:
    python scripts/run_cases23.py --markets hou_dam pan_dam --cases 3

Writes results/{market}/case{2,3}[_jul]/{schedule,design,costs,metrics}.csv.
Full-year 15-min models have ~35,000 time steps and take several hours each.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from plasmaco import ProcessParameters                      # noqa: E402
from plasmaco.data import MARKETS, load_market              # noqa: E402
from plasmaco.model_case23 import build_model, export_solution, solve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--markets", nargs="+", default=list(MARKETS), choices=list(MARKETS))
    ap.add_argument("--cases", nargs="+", type=int, default=[2, 3], choices=[2, 3])
    ap.add_argument("--month", type=int, default=None,
                    help="calendar month to slice (7 = July); default full year")
    ap.add_argument("--time-limit", type=float, default=None, help="solver time limit, s")
    ap.add_argument("--out", type=Path, default=REPO / "results")
    args = ap.parse_args()

    suffix = "" if args.month is None else ("_jul" if args.month == 7 else f"_m{args.month:02d}")
    summary = []
    for market in args.markets:
        data, dt_h = load_market(market, month=args.month)
        pi = data.electricity_price["price_USD_per_MWh"].to_numpy()
        print(f"\n=== {market}: {len(data.time_steps)} steps x {dt_h} h, "
              f"mean price {pi.mean():.2f} USD/MWh ===")
        for case in args.cases:
            params = ProcessParameters()
            _, _, sym = build_model(data, params, case=case, dt_h=dt_h, solver="cplex")
            res = solve(sym, time_limit_s=args.time_limit, log=False)
            print(f"  Case {case}: {res['model_status']} / {res['solve_status']}  "
                  f"LCOC {res['LCOC_USD_per_t_CO']:,.2f} USD/t CO")
            export_solution(sym, csv_dir=str(args.out / market / f"case{case}{suffix}"))
            summary.append((market, case, res["LCOC_USD_per_t_CO"]))

    print(f"\n{'market':<12s} {'case':>4s} {'LCOC (MILP objective)':>22s}")
    for market, case, lcoc in summary:
        print(f"{market:<12s} {case:>4d} {lcoc:>22,.2f}")


if __name__ == "__main__":
    main()
