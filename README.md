# Stepwise Fine and Gray: Subject-Specific Variable Selection
Accompanying the paper
> Shen, X., Elmer, J. &amp; Chen, G.H.. (2025). Stepwise Fine and Gray: Subject-Specific Variable Selection Shows When Hemodynamic Data Improves Prognostication of Comatose Post-Cardiac Arrest Patients. <i>Proceedings of the 10th Machine Learning for Healthcare Conference</i>, in <i>Proceedings of Machine Learning Research</i> 298 Available from https://proceedings.mlr.press/v298/shen25a.html.


## Competing Risks Experiments

- Run the experiment results on competing risks
- Models: DeepSurv, DeepHit, Deep Survival Machine
- main py file of training Stepwise Fine and Gray: `/src/experiments/train_twoStage_nnFG.py`
- Script to run the experiment with Stepwise Fine and Gray: `scripts/run_2stage_nnfg.sh`
    - Go to the directory of `/src/experiments/`, use the command `bash ../../scripts/run_2stage_nnfg.sh`
- Script to run the experiment with Stepwise Fine and Gray excluding 2020-2021 (Covid period): `scripts/run_2stage_nnfg_exclude_covid.sh`
- Script to run the experiment with baseline competing risks models: `scripts/run_sota_cr.sh` and `scripts/run_sota_cr_exclude_covid.sh`

## Data
- `original-data/`
    - This directory keeps the raw data provided by Dr. Jonathan Elmer, without any changes. Be sure to not remove or modify data in this directory.
    - It also contains the feature dictionary for the metadata (registry data) - the pdf file.
    - `time first awake.csv` has the timestamp each patient was first documented to be conscious (if ever).
    - `death or discharge time.csv` has the timestamp each patient left the hospital (died or was discharged).
- `data-prep/`
    - This is the directory that keeps processed data, including the final data sets used to run the experiments.


    

## Experiment Log
- June 2, 2025: added experiment with Covid data (pid with 2020 and 2021) excluded

