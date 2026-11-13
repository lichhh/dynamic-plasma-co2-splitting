"""Collect design variables and LCOC components for Cases 1-4 in every market
(Supplementary Tables 9 and 10). Case 1 is computed in closed form (plasmaco.case1).

    python scripts/make_tables.py

Reads results/{market}/case*/; writes results/summary_tables.xlsx with sheets
Full_year (Cases 1-3) and July_2020 (Cases 1-4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from plasmaco.case1 import case1_lcoc                      # noqa: E402

RES_DIR  = REPO / "results"
OUT_XLSX = RES_DIR / "summary_tables.xlsx"

MARKETS = [
    ("de_dam_1h",  "Germany",   "1 h"),
    ("de_dam_15m", "Germany",   "15 min"),
    ("hou_dam",    "Houston",   "1 h"),
    ("hou_rtm",    "Houston",   "15 min"),
    ("pan_dam",    "Panhandle", "1 h"),
    ("pan_rtm",    "Panhandle", "15 min"),
]

# (metrics key, column name)
DESIGN_COLS = [
    ("N_R",          "Plasma modules"),
    ("K_PSA",        "PSA capacity (kmol/h)"),
    ("K_comp",       "Compressor capacity (kmol/h)"),
    ("S_bar_CO2",    "CO2 tank (kmol)"),
    ("S_bar_mix",    "Mix tank (kmol)"),
]
COST_COLS = [
    ("LCOC_USDpt",              "LCOC (USD/t CO)"),
    ("LCOC_capex_USDpt",        "Capital (USD/t)"),
    ("LCOC_fixed_opex_USDpt",   "Fixed O&M (USD/t)"),
    ("LCOC_electricity_USDpt",  "Electricity (USD/t)"),
    ("LCOC_co2_USDpt",          "CO2 feed (USD/t)"),
    ("LCOC_carbon_USDpt",       "Solid carbon (USD/t)"),
    ("LCOC_fr_credit_USDpt",    "FR credit (USD/t)"),
    ("elec_baseline_USDpyr",    "Electricity, continuous (M USD/yr)"),
    ("elec_reduction_pct",      "Electricity cost reduction (%)"),
    ("pi_weighted_paid_USDpMWh", "Price paid (USD/MWh)"),
    ("FR_revenue_USDpyr",       "FR revenue (k USD/yr)"),
]


def read_case(case_dir: Path) -> pd.Series | None:
    """Merge design.csv and metrics.csv into one Series and rescale every LCOC
    component from the delivered output (~99.5% of target, within the production
    tolerance) to the design target, so all cases share one denominator."""
    m_path = case_dir / "metrics.csv"
    d_path = case_dir / "design.csv"
    if not m_path.exists():
        return None
    s = pd.read_csv(m_path, index_col=0)["value"].astype(float)
    if d_path.exists():
        for k, v in pd.read_csv(d_path).iloc[0].items():
            s[k] = float(v)
    scale = float(s["M_CO_t_per_yr_actual"]) / float(s["M_CO_t_per_yr_target"])
    for key in s.index:
        if key.startswith("LCOC_") and key.endswith("_USDpt"):
            s[key] = s[key] * scale
    return s


def derive_c1(s_case2: pd.Series) -> pd.Series:
    """Case 1 from the Case 2 run on the same price series: capital, fixed O&M and
    electricity from plasmaco.case1; CO2 feed and solid carbon copied from Case 2."""
    s = s_case2.copy()
    c1 = case1_lcoc(
        elec_baseline_USDpyr=float(s["elec_baseline_USDpyr"]),
        pi_unweighted_USDpMWh=float(s["pi_unweighted_USDpMWh"]),
    )
    for k in ("N_R", "K_PSA", "K_comp", "S_bar_CO2", "S_bar_mix",
              "LCOC_capex_USDpt", "LCOC_fixed_opex_USDpt", "LCOC_electricity_USDpt"):
        s[k] = c1[k]
    s["LCOC_USDpt"] = (c1["LCOC_capex_USDpt"] + c1["LCOC_fixed_opex_USDpt"]
                       + c1["LCOC_electricity_USDpt"]
                       + float(s["LCOC_co2_USDpt"]) + float(s["LCOC_carbon_USDpt"]))
    s["elec_reduction_pct"] = 0.0
    s["pi_weighted_paid_USDpMWh"] = float(s["pi_unweighted_USDpMWh"])
    return s


def collect_row(loc: str, res: str, case: int, s: pd.Series | None) -> dict:
    row = {"Market": loc, "Resolution": res, "Case": case}
    if s is None:
        return row
    for key, name in DESIGN_COLS + COST_COLS:
        v = s.get(key, None)
        if v is None:
            row[name] = None
        elif key == "elec_baseline_USDpyr":
            row[name] = float(v) / 1e6
        elif key == "FR_revenue_USDpyr":
            row[name] = float(v) / 1e3
        else:
            row[name] = float(v)
    return row


def build_table(suffix: str, with_case4: bool) -> pd.DataFrame:
    rows = []
    for folder, loc, res in MARKETS:
        s2 = read_case(RES_DIR / folder / f"case2{suffix}")
        s3 = read_case(RES_DIR / folder / f"case3{suffix}")
        rows.append(collect_row(loc, res, 1, derive_c1(s2) if s2 is not None else None))
        rows.append(collect_row(loc, res, 2, s2))
        rows.append(collect_row(loc, res, 3, s3))
        if with_case4:
            rows.append(collect_row(loc, res, 4, read_case(RES_DIR / folder / "case4")))
    return pd.DataFrame(rows)


def main() -> None:
    full_year = build_table("", with_case4=False)
    july = build_table("_jul", with_case4=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        full_year.to_excel(w, sheet_name="Full_year", index=False)
        july.to_excel(w, sheet_name="July_2020", index=False)
    cols = ["Market", "Resolution", "Case", "Plasma modules", "LCOC (USD/t CO)",
            "Electricity cost reduction (%)"]
    with pd.option_context("display.width", 140, "display.float_format", "{:.1f}".format):
        print("Full year\n", full_year[cols].to_string(index=False))
        print("\nJuly 2020\n", july[cols].to_string(index=False))
    print(f"\nwrote {OUT_XLSX.relative_to(REPO)}")


if __name__ == "__main__":
    main()
