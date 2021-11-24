#!/bin/bash

# Declare an array of string with type
declare -a runIds=("247r7xfl" "3ij6z4aq" "27z5khpa" "2lwgvbfm" "3f7fs0pc" "2uuhfqn6")
declare -a clips=("True" "False")

for runId in "${runIds[@]}"; do
  for clip in "${clips[@]}"; do
    sleep 1
    python ./scripts/fid_score.py $runId $clip
  done
done