#!/bin/bash

# Declare an array of string with type
declare -a runIds=("247r7xfl" "3ij6z4aq" "27z5khpa" "2lwgvbfm" "3f7fs0pc" "2uuhfqn6")
declare -a clips=("True" "False")
echo python $(python --version)

for runId in "${runIds[@]}"; do
  for clip in "${clips[@]}"; do
    sleep 3
    python ./scripts/fid_score.py $runId $clip &
    echo $runId
    echo $clip
  done
done
