"""2-s replay of the Case 4 commitments against the PJM RegD signal
(Supplementary Note 6 and Supplementary Fig. 30).

The hourly or 15-min MILP schedule (n_op, F_PSA, R_h) is expanded to 2-s steps.
The plasma target power P_base + R_h * a(tau) is clipped to [0, N_R * P_module],
and the CO2 and mixed-gas buffers are integrated with the 2-s mass balance while
PSA and recycle compression hold their scheduled flows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .parameters import ProcessParameters

DTAU_H = 2.0 / 3600.0                # 2 s in hours


def replay_2s(sch: pd.DataFrame, des: pd.Series, flat_regd: np.ndarray, dt_h: float,
              params: ProcessParameters | None = None):
    """Return (a, P_target, P_act, S_mix(tau), S_CO2(tau), stats) for one Case 4 solution."""
    p = params or ProcessParameters()
    F_FEED_KMOLPH = p.F_feed_kmolph
    ALPHA_PLASMA  = p.alpha_plasma
    P_MODULE_MW   = p.P_module_MW
    ALPHA_PSA_CO2 = p.alpha_PSA_CO2

    N_R          = float(des["N_R"])
    S_bar_mix    = float(des["S_bar_mix"])
    S_bar_CO2    = float(des["S_bar_CO2"])
    F_supply     = float(des["F_supply"])
    P_plant_max  = N_R * P_MODULE_MW

    n_steps       = len(sch)
    n_2s_per_step = int(round(dt_h / DTAU_H))
    n_h           = min(n_steps, len(flat_regd) // n_2s_per_step)
    total_2s      = n_h * n_2s_per_step

    n_op  = sch["n_op"].astype(float).to_numpy()[:n_h]
    F_PSA = sch["F_PSA"].astype(float).to_numpy()[:n_h]
    R_h   = sch["R_h"].astype(float).to_numpy()[:n_h]
    S_mix0 = float(sch["S_mix"].iloc[0])
    S_CO20 = float(sch["S_CO2"].iloc[0])
    F_rec  = ALPHA_PSA_CO2 * F_PSA

    a = flat_regd[:total_2s]

    # Expand step-wise MILP values to 2-s
    P_base = np.repeat(P_MODULE_MW * n_op,        n_2s_per_step)
    R_vec  = np.repeat(R_h,                       n_2s_per_step)
    F_PSA_v = np.repeat(F_PSA,                     n_2s_per_step)
    F_rec_v = np.repeat(F_rec,                     n_2s_per_step)

    P_target = P_base + R_vec * a
    P_act    = np.clip(P_target, 0.0, P_plant_max)
    n_clip_hi = int((P_target > P_plant_max).sum())
    n_clip_lo = int((P_target < 0).sum())

    # Storage traces via cumulative sum
    mix_in  = (1.0 + ALPHA_PLASMA) * F_FEED_KMOLPH * (P_act / P_MODULE_MW)
    d_mix   = (mix_in - F_PSA_v) * DTAU_H
    S_mix_t = S_mix0 + np.cumsum(d_mix)

    co2_out = F_FEED_KMOLPH * (P_act / P_MODULE_MW)
    d_co2   = (F_supply + F_rec_v - co2_out) * DTAU_H
    S_CO2_t = S_CO20 + np.cumsum(d_co2)

    stats = {
        "N_R": N_R,
        "P_max": P_plant_max,
        "total_2s": total_2s,
        "n_h": n_h,
        "S_bar_mix": S_bar_mix,
        "S_bar_CO2": S_bar_CO2,
        "n_clip_hi": n_clip_hi,
        "n_clip_lo": n_clip_lo,
        "pct_clip": 100 * (n_clip_hi + n_clip_lo) / total_2s,
        "n_mix_hi": int((S_mix_t > S_bar_mix + 1e-3).sum()),
        "n_mix_lo": int((S_mix_t < -1e-3).sum()),
        "n_co2_hi": int((S_CO2_t > S_bar_CO2 + 1e-3).sum()),
        "n_co2_lo": int((S_CO2_t < -1e-3).sum()),
        "S_mix_range": (float(S_mix_t.min()), float(S_mix_t.max())),
        "S_CO2_range": (float(S_CO2_t.min()), float(S_CO2_t.max())),
    }
    return a, P_target, P_act, S_mix_t, S_CO2_t, stats
