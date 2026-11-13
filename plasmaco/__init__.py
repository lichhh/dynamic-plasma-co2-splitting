"""Models for 'Dynamic plasma catalysis for grid-responsive CO2 splitting enabled by
coordinatively unsaturated Mn sites'.

Modules
    parameters    process and cost parameters, time-series container
    data          electricity-price, RegD and FR-price loaders
    surrogate     ALAMO models of single-pass CO2 conversion at 5, 20 and 80 C
    case1         continuous-operation baseline in closed form
    model_case23  capacity-sizing and scheduling MILP, Cases 2 and 3 (GAMSPy)
    model_case4   capacity-sizing and scheduling MILP with frequency regulation, Case 4
    replay        2-s replay of the Case 4 commitments against the RegD signal
"""
from .parameters import ProcessParameters, SchedulingData

__all__ = ["ProcessParameters", "SchedulingData"]
