# Dynamic plasma catalysis for grid-responsive CO2 splitting

Code for the reactor surrogate models, the process model and the grid-responsive
capacity-sizing and scheduling optimization in

> H. Zhang, M. Qi, T. Xu, J. Cai, M. Wu, N. Tsubaki, W. Wu.
> *Dynamic plasma catalysis for grid-responsive CO2 splitting enabled by
> coordinatively unsaturated Mn sites* (submitted).

The repository covers the computational part of the work (Supplementary Notes 1-6).
Analysis of the experimental data and the plotting scripts are not included; the
values plotted in each figure are provided as Source Data with the article.

## Contents

| Path | What it does | Paper |
|---|---|---|
| `plasmaco/surrogate.py` | ALAMO models of single-pass CO2 conversion at 5, 20 and 80 °C | Suppl. Note 1, Suppl. Table 1, Suppl. Fig. 16 |
| `process_model/aspen/` | Aspen Plus flowsheet and Aspen Custom Modeler reactor block | Suppl. Note 2, Fig. 4d–f, Suppl. Figs. 17–18 |
| `plasmaco/parameters.py` | Process, cost and scheduling parameters | Suppl. Tables 2, 3, 6, 8 |
| `plasmaco/case1.py` | Continuous operation (Case 1) in closed form | Fig. 4g, Suppl. Fig. 19 |
| `plasmaco/model_case23.py` | MILP for Cases 2 and 3 (GAMSPy) | Suppl. Note 4, Fig. 5 |
| `plasmaco/model_case4.py` | MILP for Case 4 (Case 3 plus frequency regulation) | Suppl. Note 4, Fig. 5d |
| `plasmaco/replay.py` | 2-s replay of the Case 4 commitments against the RegD signal | Suppl. Note 6, Suppl. Fig. 30 |
| `plasmaco/data.py` | Loaders for electricity prices, RegD signal and FR price | Suppl. Note 5, Suppl. Table 7 |
| `scripts/` | Command-line entry points (below) | |
| `data/` | Input data (below) | |
| `results/` | Solutions reported in the paper | Suppl. Tables 9–10 |

## Requirements

* Python ≥ 3.10 with the packages in `requirements.txt`
* [GAMSPy](https://gamspy.readthedocs.io) with a GAMS licence that includes CPLEX.
  The MILPs were solved with CPLEX through GAMSPy; the repository was last tested
  with GAMSPy 1.23.0.
* Aspen Plus and Aspen Custom Modeler V14, only to open the process model.

```bash
pip install -r requirements.txt
```

## Reproducing the results

All commands are run from the repository root.

```bash
# Reactor surrogates: R2 and RMSE at 5, 20 and 80 C (Suppl. Fig. 16)
python scripts/verify_surrogates.py

# Cases 2 and 3, full year, six price series (Suppl. Table 9)
python scripts/run_cases23.py
# Cases 2 and 3, July 2020 slice (Suppl. Table 10)
python scripts/run_cases23.py --month 7
# Case 4, July 2020 (Suppl. Table 10)
python scripts/run_case4.py

# Case 1 and summary tables (Suppl. Tables 9 and 10)
python scripts/make_tables.py
# 2-s replay of Case 4 (Suppl. Fig. 30)
python scripts/replay_case4_2s.py
```

`run_cases23.py` and `run_case4.py` accept `--markets` (any of `de_dam_1h`,
`de_dam_15m`, `hou_dam`, `hou_rtm`, `pan_dam`, `pan_rtm`) and `--out`. Each run
writes `schedule.csv`, `design.csv`, `costs.csv` and `metrics.csv` to
`results/{market}/{case}/`. `results/` already holds the solutions reported in the
paper, so `make_tables.py` and `replay_case4_2s.py` can be run without a solver.

All models are solved to a 1% relative optimality gap with CPLEX default threading.
Typical solve times on a desktop workstation: hourly one-month models 2–8 min
(Cases 2–3) and 10–30 min (Case 4); hourly full-year models 20–60 min; 15-min
one-month models 30–90 min; 15-min full-year models more than 8 h.

### Market keys

| Key | Series | Steps |
|---|---|---|
| `de_dam_1h` | ENTSO-E Germany–Luxembourg day-ahead, 2025, hourly | 8,760 |
| `de_dam_15m` | ENTSO-E Germany–Luxembourg day-ahead, 2025, 15 min | 35,040 |
| `hou_dam` | ERCOT Houston day-ahead, 2020, hourly | 8,784 |
| `hou_rtm` | ERCOT Houston real-time, 2020, 15 min | 35,136 |
| `pan_dam` | ERCOT Panhandle day-ahead, 2020, hourly | 8,784 |
| `pan_rtm` | ERCOT Panhandle real-time, 2020, 15 min | 35,136 |

## Data

* `data/alamo_training_data.csv`: the 189 measured operating points (63 per
  temperature) with measured and ALAMO-predicted conversion (Suppl. Table 1).
  Power per tube is applied current × 30 V.
* `data/plasma_data_sources.xlsx`: all time series used by the scheduling models,
  assembled by `scripts/data_prep/build_data_workbook.py` from

  | Sheet | Source |
  |---|---|
  | `DE_LU_price` | ENTSO-E Transparency Platform, day-ahead prices, bidding zone DE-LU, 2025 |
  | `Hou_Pan_price` | Source Data of R. Wang et al., *Nat. Energy* **9**, 1064 (2024), sheet "Fig. 1a and Fig. 4a" (ERCOT, 2020) |
  | `PJM_RegD_signal_2s` | PJM, RTO regulation signal (RegD, "Dynamic"), July 2020 |
  | `PJM_FR_price_hourly` | PJM Data Miner 2, reserve market results, service REG, July 2020; `mcp` = capability + performance clearing price (`scripts/data_prep/convert_pjm_reserve_to_fr_price.py`) |

  EUR prices are converted at 1.14 USD per EUR inside the loaders.

## Process model

`process_model/aspen/CO2_plasma_process.bkp` is the Aspen Plus flowsheet of
Suppl. Fig. 17: plasma reactor, carbon bed, PSA, vacuum pump and recycle compressor.
Its sensitivity block `S-1` varies the applied current (2–10 A, 21 points) and the
CO2 feed per tube (about 500–3,000 mL/min, 21 points) and tabulates power and CO
output; these runs give the specific-electricity maps of Fig. 4e and Suppl. Fig. 18.

The plasma reactor is the Aspen Custom Modeler block in
`process_model/aspen/CO2_Plasma.acm`. It computes single-pass conversion from the
ALAMO model, closes the material balance with CO2 → CO + ½O2, and takes the electric
input as current × 30 V. The 20 °C model is active. For the 5 °C and 80 °C runs, the
corresponding commented expression replaces it. To use the block, open the `.acm`
file in Aspen Custom Modeler, export it as an Aspen Plus model package, and register
the package in Aspen Plus before opening the `.bkp` file.

The parameters passed from the flowsheet to the scheduling model are flagged with `*`
in `plasmaco/parameters.py`.

## Reactor surrogates

The ALAMO models were fitted with the basis functions listed in Suppl. Note 1
(constant, linear, square-root, quadratic and cubic terms, `multi2power`,
`multi3power` and `ratiopower` with exponents 0.5–4, `SOLVEMIP`). ALAMO is
commercial software. The fitted expressions are in `plasmaco/surrogate.py`, and
`data/alamo_training_data.csv` holds the data needed to refit them.
`scripts/verify_surrogates.py` reproduces the tabulated predictions to machine
precision.

## Licence

Code: MIT licence (`LICENSE`). Third-party data in `data/plasma_data_sources.xlsx`
remain subject to the terms of their providers (ENTSO-E, PJM and the authors of the
cited Source Data).

## Citation

See `CITATION.cff`.
