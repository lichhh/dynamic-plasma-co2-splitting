"""ALAMO surrogate models of single-pass CO2 conversion (Supplementary Note 1).

One algebraic model per reactor temperature, fitted with ALAMO to the 63 measured
operating points at that temperature (Supplementary Table 1). Inputs are the
applied current I (A) and the CO2 flow F (mL/min) per tube; the output is the
single-pass CO2 conversion in %. Power per tube is I x 30 V (eq. S3), so
2-10 A corresponds to 0.06-0.30 kW.

The same expressions are used in the Aspen Custom Modeler block
(process_model/aspen/CO2_Plasma.acm), where the 20 C model is active and the
5 C and 80 C models are swapped in for the corresponding flowsheet runs.
"""
from __future__ import annotations

import numpy as np

VOLTAGE_V = 30.0


def _conv_5C(I, F):
    return (-0.0918787122961595 * F - 168.927084150599 * np.log(F) + 15.8950257826925 * F ** 0.5
            + 0.69480500724112 * I ** 2 - 0.00273473305074747 * I ** 4
            - 0.00124325568436682 * I * F - 63998.1695561865 * (I / F) ** 2 + 762.262705654992)


def _conv_20C(I, F):
    return (-0.0873528150008443 * F - 174.612517924554 * np.log(F) + 15.7458358929333 * F ** 0.5
            + 1.40626942574087 * I ** 2 - 0.0796066014820672 * I ** 3
            - 0.00178828607559064 * I * F - 1832.3913605269 * I / F + 805.092190150748)


def _conv_80C(I, F):
    return (-56.2353113690814 * np.log(I) - 12.53144496654 * np.log(F) - 1.34101117564184e-06 * F ** 2
            + 2.04178020853152 * (I * F) ** 0.5 - 0.00718991384933179 * I * F
            + 4.44180524096855e-08 * (I * F) ** 2 + 1469.62760470153 * (I / F) ** 0.5
            - 4097.80320955799 * I / F)


MODELS = {5: _conv_5C, 20: _conv_20C, 80: _conv_80C}


def co2_conversion(T_C: int, current_A, flow_mL_min):
    """Single-pass CO2 conversion (%) at reactor temperature T_C in {5, 20, 80}."""
    I = np.asarray(current_A, dtype=float)
    F = np.asarray(flow_mL_min, dtype=float)
    return MODELS[int(T_C)](I, F)


def power_kW(current_A):
    """Plasma power per tube (kW) from the applied current (eq. S3)."""
    return np.asarray(current_A, dtype=float) * VOLTAGE_V / 1000.0


def current_A(power_kW_per_tube):
    """Applied current (A) for a given plasma power per tube (kW)."""
    return np.asarray(power_kW_per_tube, dtype=float) * 1000.0 / VOLTAGE_V
