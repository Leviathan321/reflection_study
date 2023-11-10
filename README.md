# NSGAII-DT Replication Results

This repository holds the results of the NSGAII-DT replication study from the journal paper (ISTJ) ["Reflections on Surrogate-Assisted Search-Based Testing: A Taxonomy and Two Replication Studies based on Industrial ADAS and Simulink Models"](https://www.sciencedirect.com/science/article/abs/pii/S0950584923001404).

## Algorithm

The reflection study has been conducted with the support of the testing framework [OpenSBT](https://git.fortiss.org/opensbt/opensbt-core).
The implementation of NSGAII-DT is based on pymoo and an excerpt of the implementation is provided [here](algorithm/nsga2_dt_sim.py).

## Combined Results
The *Hypervolume/General Distance/Spread*analysis results after 10 runs are available here: 
<img src="subplots_combined.png" alt="results_10_runs" width="600"/>

The detailed *Hypervolume* analysis results are available here: [HV](/hv/)

NSGAII         |  NSGAII-DT
:-------------------------:|:-------------------------:
<img src="hv/hv_global_combined_NSGA-II.png" alt="HV" width="600"/>  |  <img src="hv/hv_global_combined_NSGA-II-DT.png" alt="HV" width="600"/>

The *Spread* analysis results are available here: [Spread](/sp/) 

NSGAII         |  NSGAII-DT
:-------------------------:|:-------------------------:
<img src="sp/sp_combined_NSGA-II.png" alt="SP" width="600"/> |  <img src="sp/sp_combined_NSGA-II-DT.png" alt="SP" width="600"/>

The *General Distance* analysis results are available here: [GD](/gd/) 

NSGAII         |  NSGAII-DT
:-------------------------:|:-------------------------:
<img src="gd/gd_combined_NSGA-II.png" alt="GD" width="600"/> | <img src="gd/gd_combined_NSGA-II-DT.png" alt="GD" width="600"/>


## Results Single Runs

The results of each single run of NSGAII are available here: [Single Runs NSGAII](NSGA-II/)

The results of each single run of NSGAII-DT are available here: [Single Runs NSGAII-DT](NSGA-II-DT/)

## Replication Study Authors

Lev Sorokin (sorokin@fortiss.org) \
Damir Safin (safin@fortiss.org)
