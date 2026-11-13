"""Case 1 (continuous operation) in closed form.

Case 1 runs at constant rated throughput with N_R = nominal module count, PSA and
compressor sized to the matching rated flows, and no buffer storage. Nothing varies
in time, so the cost is computed directly with the same PCE -> CAPEX ->
annualisation chain as the MILP (Supplementary Note 3).
"""
from __future__ import annotations

import math
from typing import Optional

from .parameters import ProcessParameters


def case1_lcoc(
    elec_baseline_USDpyr: float,
    pi_unweighted_USDpMWh: float,
    params: Optional[ProcessParameters] = None,
) -> dict:
    """Return Case 1 LCOC components (USD per t CO) and design variables.

    Parameters
    ----------
    elec_baseline_USDpyr : float
        Annual electricity cost when the plant runs always-on at the
        market unweighted mean price (written to metrics.csv by every Case 2-4 run).
    pi_unweighted_USDpMWh : float
        Market unweighted mean electricity price (only for traceability).
    params : ProcessParameters, optional
        Defaults to ProcessParameters().
    """
    p = params or ProcessParameters()

    # ----- Baseline design -----
    N_R = int(math.ceil(p.N_R_nominal))       # 35 with default parameters
    K_PSA = p.F_PSA_rated_kmolph              # ~439 kmol/h
    K_comp = p.F_comp_rated_kmolph            # ~290 kmol/h
    S_bar_CO2 = 0.0                           # no buffer
    S_bar_mix = 0.0

    # ----- Purchased equipment cost -----
    pce_plasma  = p.cap_plasma_USDpkW * N_R * p.P_module_MW * 1000.0
    pce_psa     = p.pce_psa_USD(K_PSA)
    pce_comp    = p.pce_compression_total_USD(K_comp)
    pce_storage = 0.0
    pce_total   = pce_plasma + pce_psa + pce_comp + pce_storage

    # ----- Total fixed capital (Table S3 factor chain) -----
    capex_total       = pce_total * p.capex_mult
    annualised_capex  = capex_total * p.crf

    # ----- Fixed OPEX (Table S3) -----
    maint     = p.f_maint     * capex_total
    labor     = p.operating_labor_USDpyr
    lab_burden = p.f_lab      * labor
    overhead  = p.f_overhead  * (labor + lab_burden + maint)
    insurance = p.f_insurance * capex_total
    taxes     = p.f_taxes     * capex_total
    fixed_opex_USDpyr = maint + labor + lab_burden + overhead + insurance + taxes

    # ----- Production basis: M_CO target (same denominator for ALL cases) -
    # All four cases are evaluated against the same production target so that
    # the LCOC values are directly comparable. The MILP solutions for Cases
    # 2-4 deliver M_CO_actual ~= 0.995 * target (lower edge of tolerance);
    # callers rescale those by actual/target so every bar is divided by the
    # same denominator.
    M_CO_t_per_yr = (
        p.F_CO_product_rated_kmolph * p.operating_hours_per_year
        * p.MW_CO / 1000.0
    )   # = 35,416.7 t/yr with defaults

    # ----- LCOC components computed from first principles ----------------
    capex_USDpt      = annualised_capex     / M_CO_t_per_yr
    fixed_opex_USDpt = fixed_opex_USDpyr    / M_CO_t_per_yr
    elec_USDpt       = elec_baseline_USDpyr / M_CO_t_per_yr

    return {
        # Design
        "N_R":           float(N_R),
        "K_PSA":         float(K_PSA),
        "K_comp":        float(K_comp),
        "S_bar_CO2":     S_bar_CO2,
        "S_bar_mix":     S_bar_mix,
        # Production
        "M_CO_t_per_yr": M_CO_t_per_yr,
        # LCOC components computed from baseline plant design.
        # CO2-feed and carbon components are NOT included here: they are set by
        # reaction stoichiometry and are independent of scheduling, so the caller
        # copies them from the corresponding Case 2 result.
        "LCOC_capex_USDpt":       capex_USDpt,
        "LCOC_fixed_opex_USDpt":  fixed_opex_USDpt,
        "LCOC_electricity_USDpt": elec_USDpt,
        # Quantities for traceability
        "elec_baseline_USDpyr":   elec_baseline_USDpyr,
        "pi_unweighted_USDpMWh":  pi_unweighted_USDpMWh,
    }
