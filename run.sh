#!/bin/bash

if [[ ! $# -eq 5 && ! $# -eq 5 ]]; then
  echo "Usage: run.sh <train_signal_file> <train_rel_file> <test_signal_file> <test_rel_file> <task>"
  exit
fi

train_signal_file=$1
train_rel_file=$2
test_signal_file=$3
test_rel_file=$4
task=$5

./l2r.sh $train_signal_file $train_rel_file $test_signal_file $task tmp.out.txt

# compute NDCG
echo ""
echo "# Executing: java -cp bin cs276.pa4.NdcgMain tmp.out.txt $test_rel_file"
java -cp bin cs276.pa4.NdcgMain tmp.out.txt $test_rel_file

rm -rf tmp.out.txt
