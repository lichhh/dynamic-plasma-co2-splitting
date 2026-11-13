"""Capacity-sizing and scheduling MILP for Case 4: Case 3 plus frequency regulation
(Supplementary Note 4, "FR representation").

Exported from plasma_FR_scheduling_gamspy.ipynb (cells 10, 12) without changes to
the model, solver settings or post-processing.
"""
from __future__ import annotations

import math
import sys
from typing import Optional

import gamspy as gp
import numpy as np
import pandas as pd

from .parameters import ProcessParameters, SchedulingData


def build_case4(
    data: SchedulingData,
    params: ProcessParameters,
    dt_h: float = 1.0,
    annualization_factor: Optional[float] = None,
    solver: str = "cplex",
    N_R_max: int = 60,
    K_PSA_max_factor: float = 1.5,
    K_comp_max_factor: float = 1.5,
    warm_start: bool = True,
):
    """Hour-level MILP for Case 4: DAM + FR with integrated process flexibility.

    Core assumptions are IDENTICAL to Case 2 / Case 3:
      * Plasma at the fixed operating point (single alpha_plasma, F_feed per
        module, P_module). A module is either fully ON at P_module or OFF.
      * Single-component intermediate storage S_mix (no CO / CO2 composition
        split — same as Case 3).
      * No vent streams; integer rounding absorbed by the production-tolerance band.

    FR is added on top of Case 3 by introducing:
        R_h            FR capacity committed in hour h [MW]
        eq_FR_up       R_h <= P_module * (N_R - n_op[h])                     [ignite idle module]
        eq_FR_down     R_h <= P_module * n_op[h] * (1 - alpha_pl_down)       [shut a running module]
        eq_FR_S_up     R_h * max_drift_h * k_mix <= S_bar_mix - S_mix[h]     [intermediate storage room up]
        eq_FR_S_down   R_h * max_drift_h * k_mix <= S_mix[h]                 [intermediate storage room down]
    DAM electricity uses the actual hourly average power (PDF eq 20):
        P_actual_h = P_module * n_op[h] + R_h * a_bar[h] + a_comp * F_comp[h]
    FR revenue is subtracted from cost: sum_h lambda_FR[h] * R_h * dt  (annualised).

    With params.FR_enabled = False (or no regd_stats), R_h is forced to 0 and
    the model collapses to Case 3 exactly.
    """
    T_int = len(data.time_steps)
    if annualization_factor is None:
        annualization_factor = params.operating_hours_per_year / (T_int * dt_h)

    F_PSA_rated  = params.F_PSA_rated_kmolph
    F_comp_rated = params.F_comp_rated_kmolph
    F_CO_target_kmol = params.F_CO_product_rated_kmolph * T_int * dt_h
    k_mix = params.mix_kmol_per_MWh

    fr_active = params.FR_enabled and data.regd_stats is not None

    m = gp.Container()

    # ============================== Sets ==============================
    t  = gp.Set(m, "t", records=data.time_steps, description="hourly scheduling steps")
    tt = gp.Alias(m, "tt", t)
    succ = gp.Set(
        m, "succ", domain=[t, tt],
        records=[(data.time_steps[i], data.time_steps[(i + 1) % T_int])
                 for i in range(T_int)],
        description="cyclic successor: succ[t, t+1]",
    )

    # ============================ Parameters ===========================
    dt_p         = gp.Parameter(m, "dt",            records=dt_h)
    w_year       = gp.Parameter(m, "w_year",        records=annualization_factor)
    F_feed_p     = gp.Parameter(m, "F_feed",        records=params.F_feed_kmolph)
    one_plus_a   = gp.Parameter(m, "one_plus_a",    records=1.0 + params.alpha_plasma)
    P_module_p   = gp.Parameter(m, "P_module_MW",   records=params.P_module_MW)
    a_comp_MW    = gp.Parameter(m, "a_comp_MW",     records=params.a_comp_kWh_kmol / 1000.0)
    alpha_down_p = gp.Parameter(m, "alpha_down",    records=params.alpha_down)
    alpha_CO_p   = gp.Parameter(m, "alpha_PSA_CO",  records=params.alpha_PSA_CO)
    alpha_CO2_p  = gp.Parameter(m, "alpha_PSA_CO2", records=params.alpha_PSA_CO2)
    pi = gp.Parameter(
        m, "pi", domain=t,
        records=data.electricity_price.rename(columns={"price_USD_per_MWh": "value"}),
        description="DAM price [USD/MWh]",
    )

    # FR per-hour signal stats + price (zeros if FR disabled)
    if fr_active:
        a_bar_records   = data.regd_stats[["t", "a_bar"]].rename(columns={"a_bar": "value"})
        drift_records   = data.regd_stats[["t", "max_drift_h"]].rename(columns={"max_drift_h": "value"})
        # Storage budget uses the LARGER of (actual drift, safety floor): if a
        # quiet hour has tiny realised drift we still want a conservative cushion.
        drift_records   = drift_records.copy()
        drift_records["value"] = drift_records["value"].clip(lower=params.FR_drift_floor_h)
        if data.fr_price is not None:
            fr_price_records = data.fr_price[["t", "price_USDpMWh"]].rename(columns={"price_USDpMWh": "value"})
        else:
            fr_price_records = pd.DataFrame({"t": data.time_steps, "value": params.c_FR_USDpMWh})
    else:
        zeros = pd.DataFrame({"t": data.time_steps, "value": 0.0})
        a_bar_records   = zeros
        drift_records   = zeros
        fr_price_records = zeros

    a_bar_p   = gp.Parameter(m, "a_bar",       domain=t, records=a_bar_records,
                             description="hour-mean RegD signal")
    drift_p   = gp.Parameter(m, "max_drift_h", domain=t, records=drift_records,
                             description="max cumulative RegD excursion [h of unit-bandwidth]")
    lam_FR_p  = gp.Parameter(m, "lam_FR",      domain=t, records=fr_price_records,
                             description="FR price [USD/(MW.h)]")

    # ============= Design variables =============
    N_R       = gp.Variable(m, "N_R",       type="Integer",  description="plasma modules installed")
    K_PSA     = gp.Variable(m, "K_PSA",     type="Positive", description="rated PSA throughput [kmol/h]")
    K_comp    = gp.Variable(m, "K_comp",    type="Positive", description="rated compressor capacity [kmol/h]")
    F_supply  = gp.Variable(m, "F_supply",  type="Positive", description="constant CO2 supply rate [kmol/h]")
    S_bar_CO2 = gp.Variable(m, "S_bar_CO2", type="Positive", description="feed CO2 storage cap [kmol]")
    S_bar_mix = gp.Variable(m, "S_bar_mix", type="Positive", description="intermediate storage cap [kmol]")
    N_R.up[...]    = N_R_max
    K_PSA.up[...]  = K_PSA_max_factor  * F_PSA_rated
    K_comp.up[...] = K_comp_max_factor * F_comp_rated
    if params.F_CO2_supply_max_kmolph is not None:
        F_supply.up[...] = params.F_CO2_supply_max_kmolph

    # ============= Schedule variables =============
    n_op   = gp.Variable(m, "n_op",   domain=t, type="Integer",  description="modules running at h")
    F_PSA  = gp.Variable(m, "F_PSA",  domain=t, type="Positive", description="PSA throughput [kmol/h]")
    F_comp = gp.Variable(m, "F_comp", domain=t, type="Positive", description="compressor flow [kmol/h]")
    Z_op   = gp.Variable(m, "Z_op",   domain=t, type="Binary",   description="PSA + compressor ON/OFF")
    S_CO2  = gp.Variable(m, "S_CO2",  domain=t, type="Positive", description="feed CO2 storage level")
    S_mix  = gp.Variable(m, "S_mix",  domain=t, type="Positive", description="intermediate storage level")
    R_h    = gp.Variable(m, "R_h",    domain=t, type="Positive", description="FR capacity [MW]")
    n_op.up[t] = N_R_max
    R_h.up[t]  = params.R_h_cap_factor * N_R_max * params.P_module_MW
    if not fr_active:
        R_h.up[t] = 0.0    # force FR off

    # =========================== Equations ===========================
    eq_n_le_NR = gp.Equation(m, "eq_n_le_NR", domain=t)
    eq_n_le_NR[t] = n_op[t] <= N_R

    # ---- FR-induced correction to nominal plasma operation ----
    # When FR is active the actual hourly-mean plasma power is
    #   P_actual_h = P_module * n_op[h] + R_h[h] * a_bar[h]
    # The effective "module-equivalent" load is therefore
    #   n_eff_h = n_op[h] + R_h[h] * a_bar[h] / P_module
    # Both the feed-CO2 draw (F_feed * n_eff) and the mix-gas production
    # ((1+alpha) * F_feed * n_eff) scale with n_eff, not n_op. Adding this
    # correction keeps the per-step storage balances internally consistent
    # with the DAM electricity term P_module*n_op + R_h*a_bar.
    R_over_P = 1.0 / params.P_module_MW                                # [1/MW]

    # Feed CO2 storage cyclic balance (no vent)
    eq_feed_balance = gp.Equation(m, "eq_feed_balance", domain=t)
    eq_feed_balance[t] = (
        gp.Sum(tt.where[succ[t, tt]], S_CO2[tt])
        == S_CO2[t] + (F_supply + alpha_CO2_p * F_PSA[t]
                       - F_feed_p * n_op[t]
                       - F_feed_p * R_over_P * R_h[t] * a_bar_p[t]) * dt_p
    )
    eq_feed_cap = gp.Equation(m, "eq_feed_cap", domain=t)
    eq_feed_cap[t] = S_CO2[t] <= S_bar_CO2

    # Intermediate storage cyclic balance (no vent)
    eq_mix_balance = gp.Equation(m, "eq_mix_balance", domain=t)
    eq_mix_balance[t] = (
        gp.Sum(tt.where[succ[t, tt]], S_mix[tt])
        == S_mix[t] + (one_plus_a * F_feed_p * n_op[t]
                       + one_plus_a * F_feed_p * R_over_P * R_h[t] * a_bar_p[t]
                       - F_PSA[t]) * dt_p
    )
    eq_mix_cap = gp.Equation(m, "eq_mix_cap", domain=t)
    eq_mix_cap[t] = S_mix[t] <= S_bar_mix

    # PSA + compressor ON/OFF + min load (same as Case 3)
    eq_PSA_cap = gp.Equation(m, "eq_PSA_cap", domain=t)
    eq_PSA_cap[t] = F_PSA[t] <= K_PSA
    eq_PSA_max = gp.Equation(m, "eq_PSA_max", domain=t)
    eq_PSA_max[t] = F_PSA[t] <= params.M_big * Z_op[t]
    eq_PSA_min = gp.Equation(m, "eq_PSA_min", domain=t)
    eq_PSA_min[t] = F_PSA[t] >= alpha_down_p * K_PSA - params.M_big * (1 - Z_op[t])

    eq_comp_link = gp.Equation(m, "eq_comp_link", domain=t)
    eq_comp_link[t] = F_comp[t] == alpha_CO2_p * F_PSA[t]
    eq_comp_cap = gp.Equation(m, "eq_comp_cap", domain=t)
    eq_comp_cap[t] = F_comp[t] <= K_comp
    eq_comp_max = gp.Equation(m, "eq_comp_max", domain=t)
    eq_comp_max[t] = F_comp[t] <= params.M_big * Z_op[t]
    eq_comp_min = gp.Equation(m, "eq_comp_min", domain=t)
    eq_comp_min[t] = F_comp[t] >= alpha_down_p * K_comp - params.M_big * (1 - Z_op[t])

    # Production target band (Case 3 style)
    tol = params.production_tolerance
    eq_production_lo = gp.Equation(m, "eq_production_lo")
    eq_production_lo[...] = (
        gp.Sum(t, alpha_CO_p * F_PSA[t] * dt_p) >= F_CO_target_kmol * (1.0 - tol)
    )
    eq_production_hi = gp.Equation(m, "eq_production_hi")
    eq_production_hi[...] = (
        gp.Sum(t, alpha_CO_p * F_PSA[t] * dt_p) <= F_CO_target_kmol * (1.0 + tol)
    )

    # ===================== FR-specific constraints =====================
    if fr_active:
        # Symmetric headroom: must be able to swing P by +-R_h around base.
        # Up: use idle modules (N_R - n_op) to scale up plasma.
        eq_FR_up = gp.Equation(m, "eq_FR_up", domain=t,
                               description="FR up-swing headroom from idle modules")
        eq_FR_up[t] = R_h[t] <= P_module_p * (N_R - n_op[t])

        # Down: reduce running modules to alpha_pl_down fraction of rated.
        eq_FR_down = gp.Equation(m, "eq_FR_down", domain=t,
                                 description="FR down-swing headroom from running modules")
        eq_FR_down[t] = R_h[t] <= P_module_p * n_op[t] * (1.0 - params.alpha_pl_down)

        # Storage budget. k_mix = mix kmol per MWh of plasma swing.
        # ΔS_mix from FR up-swing for duration max_drift_h = R_h * max_drift_h * k_mix
        eq_FR_S_up = gp.Equation(m, "eq_FR_S_up", domain=t,
                                 description="intermediate storage room above for FR up-swing")
        eq_FR_S_up[t] = R_h[t] * drift_p[t] * k_mix <= S_bar_mix - S_mix[t]

        eq_FR_S_down = gp.Equation(m, "eq_FR_S_down", domain=t,
                                   description="intermediate storage room below for FR down-swing")
        eq_FR_S_down[t] = R_h[t] * drift_p[t] * k_mix <= S_mix[t]

    # =========================== Objective ===========================
    # Electricity uses ACTUAL hourly average power (PDF eq 20):
    #   P_actual_h = P_module * n_op[h] + R_h * a_bar[h] + a_comp * F_comp[h]
    elec_horizon = gp.Sum(
        t, pi[t] * (P_module_p * n_op[t]
                    + R_h[t] * a_bar_p[t]
                    + a_comp_MW * F_comp[t]) * dt_p
    )

    co2_horizon = (
        params.c_CO2_USDpt * params.MW_CO2 * F_supply * (T_int * dt_h) / 1000.0
    )
    carbon_horizon = gp.Sum(
        t,
        params.c_C_USDpt * params.MW_C * params.alpha_plasma
        * params.F_feed_kmolph * n_op[t] * dt_p / 1000.0,
    )

    # FR revenue (subtracted): lambda_FR[h] * R_h * Δt
    fr_revenue_horizon = gp.Sum(t, lam_FR_p[t] * R_h[t] * dt_p)

    elec_annual       = w_year * elec_horizon
    co2_annual        = w_year * co2_horizon
    carbon_annual     = w_year * carbon_horizon
    fr_revenue_annual = w_year * fr_revenue_horizon

    # CAPEX (linearised PSA + compression around nominal)
    P_total_kW = params.P_module_MW * 1000.0 * N_R
    PCE_plasma = params.cap_plasma_USDpkW * P_total_kW
    a_psa, b_psa = params.linearise_psa(F_PSA_rated)
    a_c,   b_c   = params.linearise_compression(F_comp_rated)
    PCE_PSA  = a_psa + b_psa * K_PSA
    PCE_comp = a_c   + b_c   * K_comp
    PCE_storage = (params.cap_storage_USDpm3
                   * (S_bar_CO2 / params.rho_feed_kmolm3
                      + S_bar_mix / params.rho_mix_kmolm3))
    capex = params.capex_mult * (PCE_plasma + PCE_PSA + PCE_comp + PCE_storage)

    fixed_capex_pct = (params.f_maint + params.f_insurance + params.f_taxes
                       + params.f_overhead * params.f_maint)
    fixed_const = (params.operating_labor_USDpyr
                   * (1.0 + params.f_lab) * (1.0 + params.f_overhead))
    fixed_opex = fixed_capex_pct * capex + fixed_const

    annualised_capex = params.crf * capex
    total_annual_cost = (
        annualised_capex + fixed_opex
        + elec_annual + co2_annual + carbon_annual
        - fr_revenue_annual
    )

    model = gp.Model(
        m, name="plasma_co2_to_co_case4",
        equations=m.getEquations(),
        problem="MIP",
        sense=gp.Sense.MIN,
        objective=total_annual_cost,
    )

    # ============================ Warm start ============================
    if warm_start:
        n_init = max(1, math.ceil(params.N_R_nominal))
        if n_init > N_R_max:
            raise ValueError(f"N_R_max={N_R_max} below nominal {n_init}.")
        plasma_out_per_hr = (1.0 + params.alpha_plasma) * params.F_feed_kmolph * n_init
        F_supply_init_raw = (params.F_feed_kmolph * n_init
                             - params.alpha_PSA_CO2 * F_PSA_rated)
        F_supply_init = max(0.0, F_supply_init_raw)
        if params.F_CO2_supply_max_kmolph is not None:
            F_supply_init = min(F_supply_init, params.F_CO2_supply_max_kmolph)
        S_bar_CO2_init = 4.0 * params.F_feed_kmolph * n_init
        S_bar_mix_init = 4.0 * plasma_out_per_hr
        F_PSA_init  = plasma_out_per_hr
        F_comp_init = params.alpha_PSA_CO2 * F_PSA_init

        N_R.l[...]        = n_init
        K_PSA.l[...]      = F_PSA_rated
        K_comp.l[...]     = F_comp_rated
        F_supply.l[...]   = F_supply_init
        S_bar_CO2.l[...]  = S_bar_CO2_init
        S_bar_mix.l[...]  = S_bar_mix_init
        n_op.l[t]   = n_init
        F_PSA.l[t]  = F_PSA_init
        F_comp.l[t] = F_comp_init
        Z_op.l[t]   = 1
        S_CO2.l[t]  = 0.30 * S_bar_CO2_init
        S_mix.l[t]  = 0.50 * S_bar_mix_init    # mid-fill: room for FR both directions
        R_h.l[t]    = 0.0                       # let optimiser raise

    # ============================ Bundle =============================
    design_vars = {
        "N_R": N_R, "K_PSA": K_PSA, "K_comp": K_comp,
        "F_supply": F_supply, "S_bar_CO2": S_bar_CO2, "S_bar_mix": S_bar_mix,
    }
    schedule_vars = {
        "n_op": n_op, "F_PSA": F_PSA, "F_comp": F_comp, "Z_op": Z_op,
        "S_CO2": S_CO2, "S_mix": S_mix, "R_h": R_h,
    }

    symbols = {
        "container": m, "model": model, "case": 4,
        "design_vars": design_vars, "schedule_vars": schedule_vars,
        "constants": {
            "F_PSA_rated":  F_PSA_rated,
            "F_comp_rated": F_comp_rated,
            "k_mix":        k_mix,
            "fr_active":    fr_active,
        },
        "params": params, "data": data, "dt_h": dt_h,
        "annualization_factor": annualization_factor,
        "objective": total_annual_cost, "solver": solver,
    }
    return m, model, symbols

def solve(
    symbols: dict,
    time_limit_s: Optional[float] = None,
    mip_gap: float = 0.01,
    solver_options: Optional[dict] = None,
    log: bool = True,
) -> dict:
    """Solve the Case 4 MILP and return headline results."""
    model  = symbols["model"]
    solver = symbols["solver"]
    params: ProcessParameters = symbols["params"]
    data:   SchedulingData    = symbols["data"]
    dt_h:   float             = symbols["dt_h"]

    if solver_options is None:
        solver_options = {
            "mipemphasis": 1,
            "mipstart": 1,
            "heurfreq": 20,
            "rinsheur": 50,
            "repairtries": 5,
        }

    opt_kwargs = {"relative_optimality_gap": mip_gap}
    if time_limit_s is not None:
        opt_kwargs["time_limit"] = time_limit_s
    options = gp.Options(**opt_kwargs)
    model.solve(solver=solver, options=options, solver_options=solver_options,
                output=sys.stdout if log else None)

    def scalar(v):
        rec = v.records
        if rec is None or len(rec) == 0:
            return None
        return float(rec["level"].iloc[0])

    design = {k: scalar(v) for k, v in symbols["design_vars"].items()}

    M_CO_kmol_per_year = params.F_CO_product_rated_kmolph * params.operating_hours_per_year
    M_CO_t_per_year    = M_CO_kmol_per_year * params.MW_CO / 1000.0
    obj = model.objective_value
    has_solution = obj is not None and not (isinstance(obj, float) and obj != obj)
    if has_solution:
        obj = float(obj)
        lcoc_t = obj / M_CO_t_per_year
    else:
        obj, lcoc_t = float("nan"), float("nan")

    return {
        "case": symbols["case"],
        "model_status": str(model.status),
        "solve_status": str(model.solve_status),
        "objective_USD_per_year": obj,
        "M_CO_t_per_year":        M_CO_t_per_year,
        "LCOC_USD_per_t_CO":      lcoc_t,
        "design": design,
        "horizon_steps":          len(data.time_steps),
        "dt_h":                   dt_h,
    }


def export_solution(
    symbols: dict,
    csv_dir: Optional[str] = None,
    excel_path: Optional[str] = None,
) -> dict:
    """Pull variables + cost breakdown from a solved Case 4 model.

    Returns 'schedule' (per-t DataFrame including R_h and price signals),
    'design' (single-row DataFrame), 'costs' (Series of annual cost items, with
    FR revenue recorded as a positive number under 'fr_revenue' and subtracted
    from TOTAL).
    """
    from pathlib import Path

    params: ProcessParameters = symbols["params"]
    data:   SchedulingData    = symbols["data"]
    dt_h:   float             = symbols["dt_h"]
    w_year: float             = symbols["annualization_factor"]
    T_int  = len(data.time_steps)

    F_PSA_rated  = symbols["constants"]["F_PSA_rated"]
    F_comp_rated = symbols["constants"]["F_comp_rated"]
    k_mix        = symbols["constants"]["k_mix"]
    fr_active    = symbols["constants"]["fr_active"]

    # ---------- Schedule table ----------
    sched_cols = {}
    for name, var in symbols["schedule_vars"].items():
        rec = var.records
        if rec is None or len(rec) == 0:
            continue
        idx_col = rec.columns[0]
        sched_cols[name] = rec.set_index(idx_col)["level"]
    schedule_df = pd.DataFrame(sched_cols)
    schedule_df.index.name = "t"
    schedule_df = schedule_df.reindex(data.time_steps)

    schedule_df["price_USD_per_MWh"] = (
        data.electricity_price.set_index("t")["price_USD_per_MWh"]
    )
    if data.regd_stats is not None:
        regd_idx = data.regd_stats.set_index("t")
        schedule_df["a_bar"]       = regd_idx["a_bar"]
        schedule_df["max_drift_h"] = regd_idx["max_drift_h"].clip(lower=params.FR_drift_floor_h)
    else:
        schedule_df["a_bar"]       = 0.0
        schedule_df["max_drift_h"] = 0.0
    if data.fr_price is not None:
        schedule_df["fr_price_USDpMWh"] = (
            data.fr_price.set_index("t")["price_USDpMWh"]
        )
    else:
        schedule_df["fr_price_USDpMWh"] = (params.c_FR_USDpMWh if fr_active else 0.0)
    schedule_df["CO_out_kmolph"] = params.alpha_PSA_CO * schedule_df["F_PSA"]

    # ---------- Design table ----------
    design_row = {
        name: float(var.records["level"].iloc[0])
        for name, var in symbols["design_vars"].items()
    }
    design_df = pd.DataFrame([design_row])

    # ---------- Cost re-evaluation (sanity check vs objective) ----------
    n_op_v   = schedule_df["n_op"].to_numpy()
    F_PSA_v  = schedule_df["F_PSA"].to_numpy()
    F_comp_v = schedule_df["F_comp"].to_numpy()
    R_h_v    = schedule_df["R_h"].fillna(0).to_numpy()
    pi_v     = schedule_df["price_USD_per_MWh"].to_numpy()
    a_bar_v  = schedule_df["a_bar"].fillna(0).to_numpy()
    lam_FR_v = schedule_df["fr_price_USDpMWh"].fillna(0).to_numpy()
    F_supply_v = design_row["F_supply"]

    P_module_MW   = params.P_module_MW
    a_comp_MW     = params.a_comp_kWh_kmol / 1000.0

    elec_h = (pi_v * (P_module_MW * n_op_v + R_h_v * a_bar_v
                      + a_comp_MW * F_comp_v) * dt_h).sum()
    # Carbon consumption tracks effective module-load (n_op + R_h*a_bar/P_module),
    # matching the storage-balance treatment in build_case4.
    n_eff_v  = n_op_v + R_h_v * a_bar_v / params.P_module_MW
    co2_h    = params.c_CO2_USDpt * params.MW_CO2 * F_supply_v * (T_int * dt_h) / 1000.0
    carbon_h = (params.c_C_USDpt * params.MW_C * params.alpha_plasma
                * params.F_feed_kmolph * n_eff_v * dt_h / 1000.0).sum()
    fr_rev_h = (lam_FR_v * R_h_v * dt_h).sum()

    elec_annual       = w_year * elec_h
    co2_annual        = w_year * co2_h
    carbon_annual     = w_year * carbon_h
    fr_revenue_annual = w_year * fr_rev_h

    PCE_plasma = (params.cap_plasma_USDpkW * params.P_module_MW * 1000.0
                  * design_row["N_R"])
    PCE_PSA    = params.pce_psa_USD(design_row["K_PSA"])
    PCE_comp   = params.pce_compression_total_USD(design_row["K_comp"])
    PCE_storage = (params.cap_storage_USDpm3
                   * (design_row["S_bar_CO2"] / params.rho_feed_kmolm3
                      + design_row["S_bar_mix"] / params.rho_mix_kmolm3))
    capex = params.capex_mult * (PCE_plasma + PCE_PSA + PCE_comp + PCE_storage)

    fixed_capex_pct = (params.f_maint + params.f_insurance + params.f_taxes
                       + params.f_overhead * params.f_maint)
    fixed_const = (params.operating_labor_USDpyr
                   * (1.0 + params.f_lab) * (1.0 + params.f_overhead))
    fixed_opex = fixed_capex_pct * capex + fixed_const
    annualised_capex = params.crf * capex
    total = (annualised_capex + fixed_opex + elec_annual + co2_annual
             + carbon_annual - fr_revenue_annual)

    costs = pd.Series(
        {
            "annualised_capex": annualised_capex,
            "fixed_opex":       fixed_opex,
            "electricity":      elec_annual,
            "co2_supply":       co2_annual,
            "carbon_byproduct": carbon_annual,
            "fr_revenue":       fr_revenue_annual,   # positive number, subtracted in TOTAL
            "TOTAL":            total,
            "CAPEX_total":      capex,
            "PCE_plasma":       PCE_plasma,
            "PCE_PSA":          PCE_PSA,
            "PCE_compression":  PCE_comp,
            "PCE_storage":      PCE_storage,
        },
        name="USD_per_year",
    )

    # =================== Headline metrics (KPI Series) ====================
    import math as _math
    M_CO_kmol_yr = params.F_CO_product_rated_kmolph * params.operating_hours_per_year
    M_CO_t_yr    = M_CO_kmol_yr * params.MW_CO / 1000.0

    # Baseline: continuous nominal operation, no FR, at unweighted avg DAM price
    N_R_nom    = _math.ceil(params.N_R_nominal)
    pi_avg     = float(pi_v.mean())
    P_plasma_b = P_module_MW * N_R_nom
    P_comp_b   = a_comp_MW * params.F_comp_rated_kmolph    # MW
    elec_baseline = pi_avg * (P_plasma_b + P_comp_b) * params.operating_hours_per_year

    # Weighted-avg price actually paid (plasma base + FR signal + compressor)
    P_t_paid = P_module_MW * n_op_v + R_h_v * a_bar_v + a_comp_MW * F_comp_v
    pi_weighted = float((pi_v * P_t_paid).sum() / max(P_t_paid.sum(), 1e-9))

    # Production
    co_actual_kmol_horizon = (schedule_df["CO_out_kmolph"].astype(float) * dt_h).sum()
    co_actual_t_yr         = co_actual_kmol_horizon * w_year * params.MW_CO / 1000.0
    co_dev_pct             = 100.0 * (co_actual_t_yr - M_CO_t_yr) / M_CO_t_yr

    # FR utilisation
    fr_active = bool(R_h_v.max() > 1e-3)
    R_active_hours = float((R_h_v > 0.01).sum() * dt_h)

    metrics = pd.Series(
        {
            # --- LCOC ---
            "LCOC_USDpt":              total / M_CO_t_yr,
            "M_CO_t_per_yr_target":    M_CO_t_yr,
            "M_CO_t_per_yr_actual":    co_actual_t_yr,
            "production_dev_pct":      co_dev_pct,
            # --- Share of TOTAL ---
            "share_capex_pct":         100 * annualised_capex / max(total, 1e-9),
            "share_fixed_opex_pct":    100 * fixed_opex       / max(total, 1e-9),
            "share_electricity_pct":   100 * elec_annual      / max(total, 1e-9),
            "share_co2_pct":           100 * co2_annual       / max(total, 1e-9),
            "share_carbon_pct":        100 * carbon_annual    / max(total, 1e-9),
            "share_fr_revenue_pct":    -100 * fr_revenue_annual / max(total, 1e-9),
            # --- LCOC contribution (USD/t CO) ---
            "LCOC_capex_USDpt":        annualised_capex / M_CO_t_yr,
            "LCOC_fixed_opex_USDpt":   fixed_opex       / M_CO_t_yr,
            "LCOC_electricity_USDpt":  elec_annual      / M_CO_t_yr,
            "LCOC_co2_USDpt":          co2_annual       / M_CO_t_yr,
            "LCOC_carbon_USDpt":       carbon_annual    / M_CO_t_yr,
            "LCOC_fr_credit_USDpt":   -fr_revenue_annual / M_CO_t_yr,  # negative = saving
            # --- Electricity diagnostics ---
            "pi_unweighted_USDpMWh":   pi_avg,
            "pi_weighted_paid_USDpMWh": pi_weighted,
            "pi_weighting_gain_pct":   100 * (pi_avg - pi_weighted) / pi_avg,
            "elec_baseline_USDpyr":    elec_baseline,
            "elec_savings_USDpyr":     elec_baseline - elec_annual,
            "elec_reduction_pct":      100 * (elec_baseline - elec_annual) / elec_baseline,
            # --- FR diagnostics ---
            "FR_revenue_USDpyr":       fr_revenue_annual,
            "FR_revenue_per_elec_pct": 100 * fr_revenue_annual / max(elec_annual, 1e-9),
            "R_h_avg_MW":              float(R_h_v.mean()),
            "R_h_max_MW":              float(R_h_v.max()),
            "R_h_active_hours_per_yr": R_active_hours * w_year,
            "R_h_active_pct":          100 * R_active_hours / (len(R_h_v) * dt_h),
            # --- Plant operation ---
            "N_R_chosen":              float(design_row["N_R"]),
            "n_op_avg":                float(n_op_v.mean()),
            "n_op_std":                float(n_op_v.std()),
            "n_op_max":                float(n_op_v.max()),
            "hours_at_N_R":            float((n_op_v == design_row["N_R"]).sum() * dt_h),
            "hours_at_zero":           float((n_op_v < 0.5).sum() * dt_h),
            # --- Storage utilisation ---
            "S_mix_peak_kmol":         float(schedule_df["S_mix"].max()),
            "S_mix_util_peak_pct":     float(100 * schedule_df["S_mix"].max() / max(design_row["S_bar_mix"], 1e-9)),
            "S_CO2_peak_kmol":         float(schedule_df["S_CO2"].max()),
            "S_CO2_util_peak_pct":     float(100 * schedule_df["S_CO2"].max() / max(design_row["S_bar_CO2"], 1e-9)),
        },
        name="value",
    )

    if csv_dir is not None:
        out = Path(csv_dir); out.mkdir(parents=True, exist_ok=True)
        schedule_df.to_csv(out / "schedule.csv")
        design_df.to_csv(out / "design.csv", index=False)
        costs.to_csv(out / "costs.csv", header=True)
        metrics.to_csv(out / "metrics.csv", header=True)
        print(f"CSVs written to: {out.resolve()}  (schedule, design, costs, metrics)")

    if excel_path is not None:
        xlsx = Path(excel_path); xlsx.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
            schedule_df.to_excel(writer, sheet_name="schedule")
            design_df.to_excel(writer, sheet_name="design", index=False)
            costs.to_frame().to_excel(writer, sheet_name="costs")
            metrics.to_frame().to_excel(writer, sheet_name="metrics")
        print(f"Excel written to: {xlsx.resolve()}")

    return {"schedule": schedule_df, "design": design_df,
            "costs": costs, "metrics": metrics}
