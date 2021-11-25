#!/bin/bash

# Declare an array of string with type
declare -a runIds=("2d9vlo6q" "2lwgvbfm" "3ij6z4aq" "27z5khpa" "3f7fs0pc") # "1uk0nbqr" "3q7m01sq" "2uuhfqn6" "247r7xfl")
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
