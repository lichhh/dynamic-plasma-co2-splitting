"""Capacity-sizing and scheduling MILP for Cases 2 and 3 (Supplementary Note 4).

Case 2: plasma modules scheduled, PSA and recycle compression at constant throughput.
Case 3: plasma modules, PSA and recycle compression scheduled together; PSA and
compressor capacities are design variables (tangent-linearised cost, Supplementary Note 6).

Exported from plasma_grid_scheduling_gamspy.ipynb (cells 9, 11, 13) without changes
to the model, solver settings or post-processing.
"""
from __future__ import annotations

import math
import sys
from typing import Optional

import gamspy as gp
import pandas as pd

from .parameters import ProcessParameters, SchedulingData


def build_model(
    data: SchedulingData,
    params: ProcessParameters,
    case: int = 2,
    dt_h: float = 1.0,
    annualization_factor: Optional[float] = None,
    solver: str = "cplex",
    N_R_max: int = 60,
    K_PSA_max_factor: float = 1.5,
    K_comp_max_factor: float = 1.5,
    warm_start: bool = True,
):
    """Capacity-sizing + grid-scheduling MILP for the modular plasma CO2-to-CO plant.

    Cases (Table S4)
        case=2  plasma flexibility only. PSA and recycle compressor run at constant
                rated throughput. Schedules: n_op[t], S_CO2[t], S_mix[t].
        case=3  integrated flexibility. Adds K^PSA, K^comp as design variables and
                F_PSA[t], F_comp[t] as schedules, gated by one ON/OFF binary Z_op[t]
                with min-load alpha_down.
        K_PSA_max_factor, K_comp_max_factor (default 1.5): allowed over-sizing of
        the rated capacities. Set to 1.0 to lock at nominal.

    Cyclic balances (no vent streams):
        S_mix : (1+alpha) * F_feed * n_op_t  ->  F_PSA[t]
        S_CO2 : F_supply + F_recycle[t]      ->  F_feed * n_op_t
        with F_recycle[t] = alpha_PSA_CO2 * F_PSA[t]
        (case 2 uses F_PSA_var as the constant F_PSA value over t).

    Production tolerance band (both cases):
        (1 - tol) * target  <=  cumulative CO product  <=  (1 + tol) * target
        case 2 -> implicit, via F_PSA_var in [(1-tol), (1+tol)] * F_PSA_rated
        case 3 -> explicit lower/upper bound on sum alpha_PSA_CO * F_PSA[t] * dt
        The tolerance lets the model match the integer Sigma n_op without venting.

    Electricity    pi[t] * (P_module * n_op[t] + a_comp * F_comp[t]) * dt   (PSA excluded)
    Objective      CRF*CAPEX + fixed_OPEX + electricity + CO2_feed + carbon_bed
    """
    if case not in (2, 3):
        raise ValueError("case must be 2 (plasma-flex) or 3 (integrated-flex). "
                         "Use the case_1_continuous() helper for the reference case.")

    T_int = len(data.time_steps)
    if annualization_factor is None:
        annualization_factor = params.operating_hours_per_year / (T_int * dt_h)

    F_PSA_rated  = params.F_PSA_rated_kmolph
    F_comp_rated = params.F_comp_rated_kmolph
    F_CO_target_kmol = params.F_CO_product_rated_kmolph * T_int * dt_h

    m = gp.Container()

    # ============================== Sets ==============================
    t  = gp.Set(m, "t",  records=data.time_steps,
                description="Scheduling time steps")
    tt = gp.Alias(m, "tt", t)
    succ = gp.Set(
        m, "succ", domain=[t, tt],
        records=[(data.time_steps[i], data.time_steps[(i + 1) % T_int])
                 for i in range(T_int)],
        description="Cyclic successor map: succ[t, t+1]",
    )

    # ============================ Parameters ===========================
    dt_p           = gp.Parameter(m, "dt",            records=dt_h)
    w_year         = gp.Parameter(m, "w_year",        records=annualization_factor)
    F_feed_p       = gp.Parameter(m, "F_feed",        records=params.F_feed_kmolph)
    one_plus_a     = gp.Parameter(m, "one_plus_a",    records=1.0 + params.alpha_plasma)
    P_module_p     = gp.Parameter(m, "P_module_MW",   records=params.P_module_MW)
    a_comp_MW      = gp.Parameter(m, "a_comp_MW",     records=params.a_comp_kWh_kmol / 1000.0)
    alpha_down_p   = gp.Parameter(m, "alpha_down",    records=params.alpha_down)
    alpha_CO_p     = gp.Parameter(m, "alpha_PSA_CO",  records=params.alpha_PSA_CO)
    alpha_CO2_p    = gp.Parameter(m, "alpha_PSA_CO2", records=params.alpha_PSA_CO2)
    pi = gp.Parameter(
        m, "pi", domain=t,
        records=data.electricity_price.rename(columns={"price_USD_per_MWh": "value"}),
        description="Electricity price [USD/MWh]",
    )

    # ============= Common design variables (both cases) ==============
    N_R       = gp.Variable(m, "N_R",       type="Integer",  description="Plasma modules installed")
    F_supply  = gp.Variable(m, "F_supply",  type="Positive", description="External CO2 supply (constant) [kmol/h]")
    S_bar_CO2 = gp.Variable(m, "S_bar_CO2", type="Positive", description="Feed CO2 storage cap [kmol]")
    S_bar_mix = gp.Variable(m, "S_bar_mix", type="Positive", description="Intermediate storage cap [kmol]")
    N_R.up[...] = N_R_max
    if params.F_CO2_supply_max_kmolph is not None:
        F_supply.up[...] = params.F_CO2_supply_max_kmolph

    # ============ Common scheduling variables (both cases) ============
    n_op   = gp.Variable(m, "n_op",   domain=t, type="Integer",  description="Modules running at t")
    S_CO2  = gp.Variable(m, "S_CO2",  domain=t, type="Positive", description="Feed CO2 storage level [kmol]")
    S_mix  = gp.Variable(m, "S_mix",  domain=t, type="Positive", description="Intermediate storage level [kmol]")
    n_op.up[t] = N_R_max
    # No vent streams: the integer rounding of sum(n_op) is absorbed by F_PSA_var
    # (case 2) or by F_PSA[t] within the production-tolerance band (case 3). The
    # feed CO2 balance closes via the free design variable F_supply.

    # ============== Case-specific variables ==============
    if case == 2:
        # PSA throughput is constant over t but its chosen value can shift by
        # +-production_tolerance around the rated value. Cost (PCE_PSA, PCE_comp)
        # is held at the nominal F_PSA_rated since the band is tight (<= 0.5 %).
        F_PSA_var = gp.Variable(m, "F_PSA_var", type="Positive",
                                description="Case 2 PSA throughput (constant over t) [kmol/h]")
        F_PSA_var.lo[...] = F_PSA_rated * (1.0 - params.production_tolerance)
        F_PSA_var.up[...] = F_PSA_rated * (1.0 + params.production_tolerance)
    else:
        K_PSA  = gp.Variable(m, "K_PSA",  type="Positive", description="Rated PSA throughput [kmol/h]")
        K_comp = gp.Variable(m, "K_comp", type="Positive", description="Rated compressor capacity [kmol/h]")
        K_PSA.up[...]  = K_PSA_max_factor  * F_PSA_rated
        K_comp.up[...] = K_comp_max_factor * F_comp_rated

        F_PSA  = gp.Variable(m, "F_PSA",  domain=t, type="Positive", description="PSA throughput [kmol/h]")
        F_comp = gp.Variable(m, "F_comp", domain=t, type="Positive", description="Compressor flow [kmol/h]")
        Z_op   = gp.Variable(m, "Z_op",   domain=t, type="Binary",   description="PSA + compressor ON/OFF")

    # ============================ Equations ============================
    # ---- Reactor: operating <= installed ----
    eq_n_le_NR = gp.Equation(m, "eq_n_le_NR", domain=t,
                             description="Operating modules cannot exceed installed")
    eq_n_le_NR[t] = n_op[t] <= N_R

    # ---- Feed CO2 storage (cyclic) ----
    eq_feed_balance = gp.Equation(m, "eq_feed_balance", domain=t,
                                  description="Feed CO2 storage cyclic balance")
    if case == 2:
        eq_feed_balance[t] = (
            gp.Sum(tt.where[succ[t, tt]], S_CO2[tt])
            == S_CO2[t] + (F_supply + alpha_CO2_p * F_PSA_var
                           - F_feed_p * n_op[t]) * dt_p
        )
    else:
        eq_feed_balance[t] = (
            gp.Sum(tt.where[succ[t, tt]], S_CO2[tt])
            == S_CO2[t] + (F_supply + alpha_CO2_p * F_PSA[t]
                           - F_feed_p * n_op[t]) * dt_p
        )
    eq_feed_cap = gp.Equation(m, "eq_feed_cap", domain=t)
    eq_feed_cap[t] = S_CO2[t] <= S_bar_CO2

    # ---- Intermediate mixed-gas storage (cyclic) ----
    eq_mix_balance = gp.Equation(m, "eq_mix_balance", domain=t,
                                 description="Intermediate gas storage cyclic balance")
    if case == 2:
        eq_mix_balance[t] = (
            gp.Sum(tt.where[succ[t, tt]], S_mix[tt])
            == S_mix[t] + (one_plus_a * F_feed_p * n_op[t] - F_PSA_var) * dt_p
        )
    else:
        eq_mix_balance[t] = (
            gp.Sum(tt.where[succ[t, tt]], S_mix[tt])
            == S_mix[t] + (one_plus_a * F_feed_p * n_op[t] - F_PSA[t]) * dt_p
        )
    eq_mix_cap = gp.Equation(m, "eq_mix_cap", domain=t)
    eq_mix_cap[t] = S_mix[t] <= S_bar_mix

    # ---- Case-3 only: PSA + compressor capacity, Big-M, link, min load ----
    if case == 3:
        # PSA: F_PSA[t] in [alpha_down * K_PSA, K_PSA] when Z=1, =0 when Z=0
        eq_PSA_cap = gp.Equation(m, "eq_PSA_cap", domain=t)
        eq_PSA_cap[t] = F_PSA[t] <= K_PSA

        eq_PSA_max = gp.Equation(m, "eq_PSA_max", domain=t)
        eq_PSA_max[t] = F_PSA[t] <= params.M_big * Z_op[t]

        eq_PSA_min = gp.Equation(m, "eq_PSA_min", domain=t)
        eq_PSA_min[t] = F_PSA[t] >= alpha_down_p * K_PSA - params.M_big * (1 - Z_op[t])

        # Compressor: F_comp = alpha_PSA_CO2 * F_PSA; capacity & on/off same Z
        eq_comp_link = gp.Equation(m, "eq_comp_link", domain=t)
        eq_comp_link[t] = F_comp[t] == alpha_CO2_p * F_PSA[t]

        eq_comp_cap = gp.Equation(m, "eq_comp_cap", domain=t)
        eq_comp_cap[t] = F_comp[t] <= K_comp

        eq_comp_max = gp.Equation(m, "eq_comp_max", domain=t)
        eq_comp_max[t] = F_comp[t] <= params.M_big * Z_op[t]

        eq_comp_min = gp.Equation(m, "eq_comp_min", domain=t)
        eq_comp_min[t] = F_comp[t] >= alpha_down_p * K_comp - params.M_big * (1 - Z_op[t])

    # ---- Production target ----
    # Case 2: F_CO_product = alpha_PSA_CO * F_PSA_var; the bounds on F_PSA_var
    # already keep the cumulative target in [(1 - tol), (1 + tol)] * target.
    # Case 3: enforce the same band explicitly on sum(alpha_PSA_CO * F_PSA[t]).
    # The band lets F_PSA absorb the integer rounding of sum(n_op) without venting.
    if case == 3:
        tol = params.production_tolerance
        eq_production_lo = gp.Equation(m, "eq_production_lo",
                                       description="cumulative CO product >= (1 - tol) * target")
        eq_production_lo[...] = (
            gp.Sum(t, alpha_CO_p * F_PSA[t] * dt_p) >= F_CO_target_kmol * (1.0 - tol)
        )
        eq_production_hi = gp.Equation(m, "eq_production_hi",
                                       description="cumulative CO product <= (1 + tol) * target")
        eq_production_hi[...] = (
            gp.Sum(t, alpha_CO_p * F_PSA[t] * dt_p) <= F_CO_target_kmol * (1.0 + tol)
        )

    # =========================== Objective ===========================
    # ---- Electricity (plasma + compressor only, per PDF eq. on page 18) ----
    if case == 2:
        # F_comp = alpha_PSA_CO2 * F_PSA_var (constant over t but a design var)
        elec_horizon = gp.Sum(
            t, pi[t] * (P_module_p * n_op[t]
                        + a_comp_MW * alpha_CO2_p * F_PSA_var) * dt_p
        )
    else:
        elec_horizon = gp.Sum(
            t, pi[t] * (P_module_p * n_op[t] + a_comp_MW * F_comp[t]) * dt_p
        )

    # F_supply is constant: horizon cost = price * MW * F_supply * T_total_hours
    co2_horizon = (
        params.c_CO2_USDpt * params.MW_CO2 * F_supply * (T_int * dt_h) / 1000.0
    )
    carbon_horizon = gp.Sum(
        t,
        params.c_C_USDpt * params.MW_C * params.alpha_plasma
        * params.F_feed_kmolph * n_op[t] * dt_p / 1000.0,
    )

    elec_annual   = w_year * elec_horizon
    co2_annual    = w_year * co2_horizon
    carbon_annual = w_year * carbon_horizon

    # ---- CAPEX ----
    P_total_kW = params.P_module_MW * 1000.0 * N_R
    PCE_plasma = params.cap_plasma_USDpkW * P_total_kW

    # PSA & compression cost: constant in Case 2 (rated), linearised in Case 3
    if case == 2:
        PCE_PSA  = params.pce_psa_USD(F_PSA_rated)
        PCE_comp = params.pce_compression_total_USD(F_comp_rated)
    else:
        a_psa, b_psa = params.linearise_psa(F_PSA_rated)
        a_c,   b_c   = params.linearise_compression(F_comp_rated)
        PCE_PSA  = a_psa + b_psa * K_PSA
        PCE_comp = a_c   + b_c   * K_comp

    PCE_storage = (
        params.cap_storage_USDpm3 * (S_bar_CO2 / params.rho_feed_kmolm3)
        + params.cap_storage_USDpm3 * (S_bar_mix / params.rho_mix_kmolm3)
    )

    capex = params.capex_mult * (PCE_plasma + PCE_PSA + PCE_comp + PCE_storage)

    # ---- Fixed OPEX (Table S3) ----
    # = (f_maint + f_insur + f_taxes + f_overhead*f_maint) * CAPEX
    #   + labor * (1 + f_lab) * (1 + f_overhead)
    fixed_capex_pct = (
        params.f_maint + params.f_insurance + params.f_taxes
        + params.f_overhead * params.f_maint
    )
    fixed_const = (
        params.operating_labor_USDpyr
        * (1.0 + params.f_lab)
        * (1.0 + params.f_overhead)
    )
    fixed_opex = fixed_capex_pct * capex + fixed_const

    annualised_capex  = params.crf * capex
    total_annual_cost = (
        annualised_capex + fixed_opex
        + elec_annual + co2_annual + carbon_annual
    )

    model = gp.Model(
        m, name=f"plasma_co2_to_co_case{case}",
        equations=m.getEquations(),
        problem="MIP",
        sense=gp.Sense.MIN,
        objective=total_annual_cost,
    )

    # ========================= Warm start =========================
    # Constant schedule at n_op = ceil(N_R_nominal). The cyclic mix-balance slack
    # from integer rounding is absorbed by F_PSA_var (case 2) or F_PSA[t]
    # within the production band (case 3).
    if warm_start:
        n_init = max(1, math.ceil(params.N_R_nominal))
        if n_init > N_R_max:
            raise ValueError(f"N_R_max={N_R_max} below nominal {n_init}.")

        plasma_out_per_hr = (1.0 + params.alpha_plasma) * params.F_feed_kmolph * n_init
        F_supply_init_raw = (params.F_feed_kmolph * n_init
                             - params.alpha_PSA_CO2 * F_PSA_rated)
        F_supply_init     = max(0.0, F_supply_init_raw)
        if params.F_CO2_supply_max_kmolph is not None:
            F_supply_init = min(F_supply_init, params.F_CO2_supply_max_kmolph)

        S_bar_CO2_init = 4.0 * params.F_feed_kmolph * n_init
        S_bar_mix_init = 4.0 * plasma_out_per_hr

        N_R.l[...]        = n_init
        F_supply.l[...]   = F_supply_init
        S_bar_CO2.l[...]  = S_bar_CO2_init
        S_bar_mix.l[...]  = S_bar_mix_init
        n_op.l[t]    = n_init
        S_CO2.l[t]   = 0.30 * S_bar_CO2_init
        S_mix.l[t]   = 0.30 * S_bar_mix_init

        if case == 2:
            F_PSA_var.l[...] = F_PSA_rated
        else:
            F_PSA_init  = (1.0 + params.alpha_plasma) * params.F_feed_kmolph * n_init
            F_comp_init = params.alpha_PSA_CO2 * F_PSA_init
            K_PSA.l[...]  = F_PSA_rated
            K_comp.l[...] = F_comp_rated
            F_PSA.l[t]    = F_PSA_init
            F_comp.l[t]   = F_comp_init
            Z_op.l[t]     = 1

    # Bundle everything the post-processor needs
    design_vars = {
        "N_R": N_R, "F_supply": F_supply,
        "S_bar_CO2": S_bar_CO2, "S_bar_mix": S_bar_mix,
    }
    schedule_vars = {"n_op": n_op, "S_CO2": S_CO2, "S_mix": S_mix}
    if case == 2:
        design_vars["F_PSA_var"] = F_PSA_var
    else:
        design_vars.update({"K_PSA": K_PSA, "K_comp": K_comp})
        schedule_vars.update({"F_PSA": F_PSA, "F_comp": F_comp, "Z_op": Z_op})

    symbols = {
        "container": m,
        "model": model,
        "case": case,
        "design_vars": design_vars,
        "schedule_vars": schedule_vars,
        "constants": {
            "F_PSA_rated":  F_PSA_rated,
            "F_comp_rated": F_comp_rated,
            "F_recycle_const_case2": params.alpha_PSA_CO2 * F_PSA_rated if case == 2 else None,
        },
        "params": params,
        "data": data,
        "dt_h": dt_h,
        "annualization_factor": annualization_factor,
        "objective": total_annual_cost,
        "solver": solver,
    }
    return m, model, symbols

def solve(
    symbols: dict,
    time_limit_s: Optional[float] = None,
    mip_gap: float = 0.01,
    solver_options: Optional[dict] = None,
    log: bool = True,
) -> dict:
    """Solve the MILP and return headline results.

    `time_limit_s=None` (default) lets the solver run until optimal / mip_gap.
    Pass a number to cap wall-clock time (returns best incumbent on timeout).
    """
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
    model.solve(
        solver=solver,
        options=options,
        solver_options=solver_options,
        output=sys.stdout if log else None,
    )

    def scalar(v):
        rec = v.records
        if rec is None or len(rec) == 0:
            return None
        return float(rec["level"].iloc[0])

    design = {k: scalar(v) for k, v in symbols["design_vars"].items()}

    hrs_yr = params.operating_hours_per_year
    M_CO_kmol_per_year = params.F_CO_product_rated_kmolph * hrs_yr
    M_CO_t_per_year    = M_CO_kmol_per_year * params.MW_CO / 1000.0
    obj = model.objective_value
    has_solution = obj is not None and not (isinstance(obj, float) and obj != obj)
    if has_solution:
        obj = float(obj)
        lcoc_kmol = obj / M_CO_kmol_per_year
        lcoc_t    = obj / M_CO_t_per_year
    else:
        obj = float("nan"); lcoc_kmol = float("nan"); lcoc_t = float("nan")

    return {
        "case": symbols["case"],
        "model_status": str(model.status),
        "solve_status": str(model.solve_status),
        "objective_USD_per_year": obj,
        "M_CO_kmol_per_year": M_CO_kmol_per_year,
        "M_CO_t_per_year": M_CO_t_per_year,
        "LCOC_USD_per_kmol_CO": lcoc_kmol,
        "LCOC_USD_per_t_CO": lcoc_t,
        "design": design,
        "horizon_steps": len(data.time_steps),
        "dt_h": dt_h,
    }

def export_solution(
    symbols: dict,
    csv_dir: Optional[str] = None,
    excel_path: Optional[str] = None,
) -> dict:
    """Extract variable levels + cost breakdown from a solved model.

    Returns dict with:
      'schedule' : DataFrame indexed by t. Contains every scheduling variable plus
                   'price_USD_per_MWh' and 'CO_out_kmolph'.
      'design'   : single-row DataFrame of design variable levels.
      'costs'    : Series of annual cost components [USD/year], including TOTAL.

    Optional outputs:
      csv_dir    -- writes schedule.csv, design.csv, costs.csv into that folder.
      excel_path -- writes a single .xlsx with three sheets ('schedule','design','costs').
                    Requires `openpyxl`.
    """
    from pathlib import Path

    params: ProcessParameters = symbols["params"]
    data:   SchedulingData    = symbols["data"]
    dt_h:   float             = symbols["dt_h"]
    w_year: float             = symbols["annualization_factor"]
    case:   int               = symbols["case"]
    T_int                     = len(data.time_steps)

    F_PSA_rated  = symbols["constants"]["F_PSA_rated"]
    F_comp_rated = symbols["constants"]["F_comp_rated"]

    # ---------- Scheduling table ----------
    schedule_cols = {}
    for name, var in symbols["schedule_vars"].items():
        rec = var.records
        if rec is None or len(rec) == 0:
            continue
        idx_col = rec.columns[0]
        schedule_cols[name] = rec.set_index(idx_col)["level"]
    schedule_df = pd.DataFrame(schedule_cols)
    schedule_df.index.name = "t"
    schedule_df = schedule_df.reindex(data.time_steps)
    schedule_df["price_USD_per_MWh"] = (
        data.electricity_price.set_index("t")["price_USD_per_MWh"]
    )

    # ---------- Design table ----------
    design_row = {
        name: float(var.records["level"].iloc[0])
        for name, var in symbols["design_vars"].items()
    }

    # For Case 2, F_PSA is constant over t but a design variable F_PSA_var; mirror
    # it (and the derived F_comp) into the schedule for a unified output shape.
    if case == 2:
        F_PSA_solved  = design_row.pop("F_PSA_var")
        F_comp_solved = params.alpha_PSA_CO2 * F_PSA_solved
        schedule_df["F_PSA"]  = F_PSA_solved
        schedule_df["F_comp"] = F_comp_solved
        schedule_df["Z_op"]   = 1
        design_row["K_PSA"]   = F_PSA_solved
        design_row["K_comp"]  = F_comp_solved
    schedule_df["CO_out_kmolph"] = params.alpha_PSA_CO * schedule_df["F_PSA"]

    design_df = pd.DataFrame([design_row])

    # ---------- Cost breakdown (re-evaluated from levels) ----------
    n_op_v   = schedule_df["n_op"].to_numpy()
    F_PSA_v  = schedule_df["F_PSA"].to_numpy()
    F_comp_v = schedule_df["F_comp"].to_numpy()
    pi_v     = schedule_df["price_USD_per_MWh"].to_numpy()
    F_supply_v = design_row["F_supply"]

    P_module_MW = params.P_module_MW
    a_comp_MW   = params.a_comp_kWh_kmol / 1000.0

    elec_h = (pi_v * (P_module_MW * n_op_v + a_comp_MW * F_comp_v) * dt_h).sum()
    co2_h    = params.c_CO2_USDpt * params.MW_CO2 * F_supply_v * (T_int * dt_h) / 1000.0
    carbon_h = (params.c_C_USDpt * params.MW_C * params.alpha_plasma
                * params.F_feed_kmolph * n_op_v * dt_h / 1000.0).sum()

    elec_annual   = w_year * elec_h
    co2_annual    = w_year * co2_h
    carbon_annual = w_year * carbon_h

    PCE_plasma = (params.cap_plasma_USDpkW * params.P_module_MW * 1000.0
                  * design_row["N_R"])
    PCE_PSA    = params.pce_psa_USD(design_row["K_PSA"])
    PCE_comp   = params.pce_compression_total_USD(design_row["K_comp"])
    PCE_storage = (params.cap_storage_USDpm3
                   * (design_row["S_bar_CO2"] / params.rho_feed_kmolm3
                      + design_row["S_bar_mix"] / params.rho_mix_kmolm3))

    capex = params.capex_mult * (PCE_plasma + PCE_PSA + PCE_comp + PCE_storage)

    fixed_capex_pct = (
        params.f_maint + params.f_insurance + params.f_taxes
        + params.f_overhead * params.f_maint
    )
    fixed_const = (params.operating_labor_USDpyr
                   * (1.0 + params.f_lab) * (1.0 + params.f_overhead))
    fixed_opex = fixed_capex_pct * capex + fixed_const
    annualised_capex = params.crf * capex
    total = (annualised_capex + fixed_opex + elec_annual + co2_annual
             + carbon_annual)

    costs = pd.Series(
        {
            "annualised_capex": annualised_capex,
            "fixed_opex":       fixed_opex,
            "electricity":      elec_annual,
            "co2_supply":       co2_annual,
            "carbon_byproduct": carbon_annual,
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
    M_CO_kmol_yr = params.F_CO_product_rated_kmolph * params.operating_hours_per_year
    M_CO_t_yr    = M_CO_kmol_yr * params.MW_CO / 1000.0

    # Baseline: continuous nominal operation at unweighted avg DAM price
    import math as _math
    N_R_nom    = _math.ceil(params.N_R_nominal)
    pi_avg     = float(pi_v.mean())
    P_plasma_b = P_module_MW * N_R_nom
    P_comp_b   = a_comp_MW * params.F_comp_rated_kmolph    # MW
    elec_baseline = pi_avg * (P_plasma_b + P_comp_b) * params.operating_hours_per_year

    # Weighted-avg price actually paid by the case
    P_t = P_module_MW * n_op_v + a_comp_MW * F_comp_v
    pi_weighted = float((pi_v * P_t).sum() / max(P_t.sum(), 1e-9))

    # Plant operation stats
    co_actual_kmol_horizon = (schedule_df["CO_out_kmolph"].astype(float) * dt_h).sum()
    co_actual_t_yr         = co_actual_kmol_horizon * w_year * params.MW_CO / 1000.0
    co_target_t_yr         = M_CO_t_yr
    co_dev_pct             = 100.0 * (co_actual_t_yr - co_target_t_yr) / co_target_t_yr

    metrics = pd.Series(
        {
            # --- LCOC ---
            "LCOC_USDpt":              total / M_CO_t_yr,
            "M_CO_t_per_yr_target":    M_CO_t_yr,
            "M_CO_t_per_yr_actual":    co_actual_t_yr,
            "production_dev_pct":      co_dev_pct,
            # --- Share of TOTAL ---
            "share_capex_pct":         100 * annualised_capex / total,
            "share_fixed_opex_pct":    100 * fixed_opex       / total,
            "share_electricity_pct":   100 * elec_annual      / total,
            "share_co2_pct":           100 * co2_annual       / total,
            "share_carbon_pct":        100 * carbon_annual    / total,
            # --- LCOC contribution (USD/t CO) ---
            "LCOC_capex_USDpt":        annualised_capex / M_CO_t_yr,
            "LCOC_fixed_opex_USDpt":   fixed_opex       / M_CO_t_yr,
            "LCOC_electricity_USDpt":  elec_annual      / M_CO_t_yr,
            "LCOC_co2_USDpt":          co2_annual       / M_CO_t_yr,
            "LCOC_carbon_USDpt":       carbon_annual    / M_CO_t_yr,
            # --- Electricity diagnostics ---
            "pi_unweighted_USDpMWh":   pi_avg,
            "pi_weighted_paid_USDpMWh": pi_weighted,
            "pi_weighting_gain_pct":   100 * (pi_avg - pi_weighted) / pi_avg,
            "elec_baseline_USDpyr":    elec_baseline,
            "elec_savings_USDpyr":     elec_baseline - elec_annual,
            "elec_reduction_pct":      100 * (elec_baseline - elec_annual) / elec_baseline,
            # --- Plant operation ---
            "N_R_chosen":              float(design_row["N_R"]),
            "n_op_avg":                float(n_op_v.mean()),
            "n_op_std":                float(n_op_v.std()),
            "n_op_max":                float(n_op_v.max()),
            "n_op_min":                float(n_op_v.min()),
            "hours_at_N_R":            float((n_op_v == design_row["N_R"]).sum() * dt_h),
            "hours_at_zero":           float((n_op_v < 0.5).sum() * dt_h),
            # --- Storage utilisation ---
            "S_mix_peak_kmol":         float(schedule_df["S_mix"].max()),
            "S_mix_avg_kmol":          float(schedule_df["S_mix"].mean()),
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
        try:
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                schedule_df.to_excel(writer, sheet_name="schedule")
                design_df.to_excel(writer, sheet_name="design", index=False)
                costs.to_frame().to_excel(writer, sheet_name="costs")
                metrics.to_frame().to_excel(writer, sheet_name="metrics")
        except ImportError as exc:
            raise ImportError(
                "Writing .xlsx needs `openpyxl`. Install with: pip install openpyxl"
            ) from exc
        print(f"Excel written to: {xlsx.resolve()}")

    return {"schedule": schedule_df, "design": design_df,
            "costs": costs, "metrics": metrics}
