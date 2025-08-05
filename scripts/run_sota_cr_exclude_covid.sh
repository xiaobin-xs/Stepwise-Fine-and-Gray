#!/bin/bash

base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

# results_path=$base_path/../results/competing_risks.csv
# if [ -f "$results_path" ]; then
#   rm $results_path
# fi

seeds=(1111 2222 3333 4444 1234 5555 6666)
dataset_names=('bps_cr')

for seed in "${seeds[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        echo "Running with seed=$seed, dataset_name=$dataset_name"
        python3 $base_path/train_sota_models_cr.py --seed "$seed" --dataset_name "$dataset_name" --exclude_covid True
    done
done
