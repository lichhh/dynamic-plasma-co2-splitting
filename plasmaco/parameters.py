"""Process and economic parameters shared by Cases 1-4, and the time-series container.

Values follow Supplementary Tables 2, 3 and 6. Entries marked * come from the
Aspen Plus flowsheet at the selected operating point (20 C, 8.4 A x 30 V = 0.25 kW
per tube, 2,000 mL/min CO2 per tube; Supplementary Fig. 17).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SchedulingData:
    """Time-indexed scheduling inputs.

    Cases 2 and 3 use only the electricity price. Case 4 also needs the RegD
    statistics and the FR capacity price.
    """

    time_steps: list[str]
    electricity_price: pd.DataFrame  # columns: t, price_USD_per_MWh
    regd_stats: Optional[pd.DataFrame] = None
    # If provided, regd_stats has columns: t, a_bar, max_drift_h
    # Otherwise FR = 0
    fr_price: Optional[pd.DataFrame] = None
    # If provided, fr_price has columns: t, price_USDpMWh
    # Otherwise use constant params.c_FR_USDpMWh


@dataclass
class ProcessParameters:
    """Plasma + downstream process parameters (Supplementary Tables 2, 3 and 6).

    Architecture (Supplementary Fig. 20):
        Feed CO2 storage -> Plasma reactor (+ carbon bed) -> Intermediate mixed-gas storage
                         -> PSA -> CO product   (CO2 recycle: vacuum pump + recycle compressor)

    Override any field per case. Aspen-derived values are flagged with `*`.
    """

    # --- Plasma reactor (single fixed operating point: 20 C, 8.4 A, 2000 ml/min) ---
    F_feed_kmolph: float = 10.438           # *CO2 feed per operating module [kmol/h]
    alpha_plasma: float = 0.21653           # *single-pass CO2 conversion
    P_module_MW:   float = 0.49             # Plasma electric power per module

    # --- PSA splits (Table S6) ---
    alpha_PSA_CO:  float = 0.338922         # PSA feed -> CO product fraction
    alpha_PSA_CO2: float = 0.660232         # PSA feed -> CO2 recycle fraction

    # --- Storage molar densities (informational; used in cost) ---
    rho_mix_kmolm3:  float = 0.0999996      # *intermediate gas storage
    rho_feed_kmolm3: float = 0.1004037      # *feed CO2 storage

    # --- Compression (combined vacuum pump + recycle compressor) ---
    a_comp_kWh_kmol: float = 3.992736       # *specific compression power
    alpha_down:      float = 0.6            # min load fraction of PSA/compressor when ON
    M_big:           float = 5000.0         # Big-M for ON/OFF indicators [kmol/h]
    vac_frac_W:      float = 0.770          # Aspen split: vacuum share of total comp. power
    vac_n_parallel:  int = 9                # parallel vacuum-pump units (Turton-valid range)

    # --- Plant size / production target (rate based on 4167 kg/h = 100 t/d) ---
    plant_rated_t_CO_per_day: float = 100.0
    F_CO2_supply_max_kmolph:  Optional[float] = None
    operating_hours_per_year: float = 8500.0   # annual operating hours (~97% availability)
    # Production tolerance: cumulative CO is constrained to
    #     [(1 - tol) * target, (1 + tol) * target]
    # so the integer rounding of Sigma n_op is absorbed by F_PSA[t] (case 3) or
    # by the design variable F_PSA_var (case 2). No vent streams needed.
    production_tolerance:     float = 0.005    # +- 0.5 %

    # ====== Case 4 / FR-specific (ignored by Cases 2 and 3) ======
    # Plasma is still operated at the SAME fixed point as Case 2 / 3 (each module
    # is either fully ON at P_module or OFF). FR capacity is therefore generated
    # by switching modules within the hour:
    #   - up   swing: ignite an idle module     (max = P_module * (N_R - n_op[h]))
    #   - down swing: shut a running module off (max = P_module * n_op[h])
    # That is alpha_pl_down = 0 below; raise it if your modules cannot fully
    # shut off within a 2-second RegD step (e.g. minimum hot-stand-by load).
    FR_enabled:       bool  = True
    alpha_pl_down:    float = 0.0       # min plasma module load fraction for FR down-swing
    # Safety FLOOR on the per-hour max RegD cumulative drift used in the
    # intermediate-storage budget. The constraint uses max(raw_drift_h, floor).
    FR_drift_floor_h: float = 0.0       # h of unit bandwidth (0 -> use raw data)
    c_FR_USDpMWh:     float = 13.55     # constant FR price, used only if no hourly price is given
    R_h_cap_factor:   float = 1.0       # R_h <= factor * (N_R_max * P_module) hard cap

    # --- Financial ---
    discount_rate:    float = 0.08
    plant_life_years: int   = 25
    c_CO2_USDpt:      float = 40.0          # captured CO2 cost
    c_C_USDpt:        float = 450.0         # solid carbon for the bed
    MW_CO2: float = 44.01
    MW_CO:  float = 28.01
    MW_C:   float = 12.011

    # --- CAPEX correlations (eqs. S7-S10 + Table S3) ---
    cap_plasma_USDpkW: float = 1422.34      # ~ 24.4 MUSD / (35 * 490 kW), incl. carbon bed
    cap_psa_base_MEUR: float = 1.75         # PSA reference, 1000 m^3/h feed, 2015
    cap_psa_basis_m3ph: float = 1000.0
    cap_psa_exp:        float = 0.67
    eur_to_usd:         float = 1.14
    cepci_2023:         float = 798.0
    cepci_2015:         float = 556.8
    cepci_comp_base:    float = 390.4       # Turton compressor base year
    cepci_vac_base:     float = 397.0       # Turton positive-displacement base year
    cap_comp_K1:  float = 6.77684           # Turton motor-driven centrifugal compressor
    cap_comp_exp: float = 0.9435
    cap_vac_K1:   float = 3.4471            # Turton positive-displacement vacuum pump
    cap_vac_K2:   float = 0.1350
    cap_vac_K3:   float = 0.1438
    cap_storage_USDpm3: float = 25.0        # low-pressure gas storage dome (eq. S13)

    # --- Direct & indirect cost factors (Table S3) ---
    f_install:     float = 0.25
    f_instr:       float = 0.06
    f_piping:      float = 0.10
    f_electrical:  float = 0.10
    f_buildings:   float = 0.10
    f_site:        float = 0.10
    f_eng:         float = 0.05
    f_construct:   float = 0.06
    f_contractor:  float = 0.02
    f_contingency: float = 0.05

    # --- Fixed OPEX (Table S3) ---
    f_maint:                float = 0.02
    operating_labor_USDpyr: float = 974_400.0
    f_lab:                  float = 0.10
    f_overhead:             float = 0.65
    f_insurance:            float = 0.004
    f_taxes:                float = 0.01

    # ===================== Derived / convenience =====================
    @property
    def y_CO(self) -> float:
        return 2.0 * self.alpha_plasma / (1.0 + self.alpha_plasma)

    @property
    def y_CO2(self) -> float:
        return (1.0 - self.alpha_plasma) / (1.0 + self.alpha_plasma)

    @property
    def F_CO_product_rated_kmolph(self) -> float:
        """Rated CO production rate, derived from plant_rated_t_CO_per_day."""
        return self.plant_rated_t_CO_per_day * 1000.0 / 24.0 / self.MW_CO

    @property
    def F_PSA_rated_kmolph(self) -> float:
        return self.F_CO_product_rated_kmolph / self.alpha_PSA_CO

    @property
    def F_comp_rated_kmolph(self) -> float:
        return self.alpha_PSA_CO2 * self.F_PSA_rated_kmolph

    @property
    def W_comp_total_rated_kW(self) -> float:
        return self.a_comp_kWh_kmol * self.F_comp_rated_kmolph

    @property
    def N_R_nominal(self) -> float:
        """Continuous-operation (Case 1) reference number of modules."""
        co_per_module = self.alpha_PSA_CO * (1.0 + self.alpha_plasma) * self.F_feed_kmolph
        return self.F_CO_product_rated_kmolph / co_per_module

    @property
    def crf(self) -> float:
        i, n = self.discount_rate, self.plant_life_years
        return i * (1 + i) ** n / ((1 + i) ** n - 1)

    @property
    def capex_mult(self) -> float:
        """PCE -> total fixed capital, from Table S3 factors (= 2.034 with defaults)."""
        direct = (1.0 + self.f_install + self.f_instr + self.f_piping
                  + self.f_electrical + self.f_buildings + self.f_site)
        indirect = 1.0 + self.f_eng + self.f_construct + self.f_contractor
        return direct * indirect / (1.0 - self.f_contingency)

    @property
    def mix_kmol_per_MWh(self) -> float:
        """kmol of mixed gas produced per MWh of plasma electricity (at fixed op pt)."""
        return (1.0 + self.alpha_plasma) * self.F_feed_kmolph / self.P_module_MW

    # ----------------------- PCE estimators (USD) -----------------------
    def pce_psa_USD(self, K_PSA_kmolph: float) -> float:
        """PSA PCE via eq. S8, sized for K^PSA at intermediate-gas density."""
        Q_m3ph = K_PSA_kmolph / self.rho_mix_kmolm3
        return (
            self.eur_to_usd * self.cap_psa_base_MEUR * 1e6
            * (self.cepci_2023 / self.cepci_2015)
            * (Q_m3ph / self.cap_psa_basis_m3ph) ** self.cap_psa_exp
        )

    def pce_recycle_comp_USD(self, W_kW: float) -> float:
        """Recycle compressor PCE via eq. S9 (motor-driven centrifugal)."""
        return (
            math.exp(self.cap_comp_K1)
            * (self.cepci_2023 / self.cepci_comp_base)
            * max(W_kW, 1e-3) ** self.cap_comp_exp
        )

    def pce_vacuum_unit_USD(self, W_per_unit_kW: float) -> float:
        """One Turton positive-displacement vacuum pump PCE (eq. S10)."""
        x = math.log10(max(W_per_unit_kW, 1.0))
        log_pce = self.cap_vac_K1 + self.cap_vac_K2 * x + self.cap_vac_K3 * x * x
        return 10 ** log_pce * (self.cepci_2023 / self.cepci_vac_base)

    def pce_vacuum_system_USD(self, W_total_kW: float) -> float:
        W_per_unit = W_total_kW / self.vac_n_parallel
        return self.vac_n_parallel * self.pce_vacuum_unit_USD(W_per_unit)

    def pce_compression_total_USD(self, K_comp_kmolph: float) -> float:
        """Combined vacuum + recycle-compressor PCE at rated flow K_comp."""
        W_total = self.a_comp_kWh_kmol * K_comp_kmolph
        W_vac = self.vac_frac_W * W_total
        W_rec = (1.0 - self.vac_frac_W) * W_total
        return self.pce_vacuum_system_USD(W_vac) + self.pce_recycle_comp_USD(W_rec)

    def pce_storage_USD(self, S_kmol: float, rho_kmolm3: float) -> float:
        """Low-pressure gas storage dome (eq. S13)."""
        return self.cap_storage_USDpm3 * S_kmol / rho_kmolm3

    # ----- Tangent-line linearisations (used by the MILP, Supplementary Note 6) -----
    def linearise_compression(self, K_nom_kmolph: Optional[float] = None) -> tuple:
        """Tangent line PCE_comp_total(K) ≈ a + b * K  around K_nom (avg exp ≈ 0.94)."""
        if K_nom_kmolph is None:
            K_nom_kmolph = self.F_comp_rated_kmolph
        pce_nom = self.pce_compression_total_USD(K_nom_kmolph)
        avg_exp = 0.94
        return pce_nom * (1.0 - avg_exp), pce_nom * avg_exp / K_nom_kmolph

    def linearise_psa(self, K_nom_kmolph: Optional[float] = None) -> tuple:
        """Tangent line PCE_PSA(K) ≈ a + b * K  around K_nom (exp 0.67)."""
        if K_nom_kmolph is None:
            K_nom_kmolph = self.F_PSA_rated_kmolph
        pce_nom = self.pce_psa_USD(K_nom_kmolph)
        return pce_nom * (1.0 - self.cap_psa_exp), pce_nom * self.cap_psa_exp / K_nom_kmolph
