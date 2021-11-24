#!/bin/bash

# Declare an array of string with type
declare -a runIds=("2lwgvbfm" "3f7fs0pc" "2uuhfqn6" "247r7xfl" "3ij6z4aq" "27z5khpa")
declare -a clips=("True" "False")

python --version

for runId in "${runIds[@]}"; do
  for clip in "${clips[@]}"; do
    python ./scripts/fid_score.py $runId $clip &
    echo $runId
    echo $clip
    sleep 5
  done
done
