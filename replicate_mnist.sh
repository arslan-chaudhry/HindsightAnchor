# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#! /bin/bash

NUM_RUNS=5
BATCH_SIZE=10
EPS_MEM_BATCH_SIZE=10
MEM_SIZE=1
LOG_DIR='results/'

if [ ! -d $LOG_DIR ]; then
    mkdir -pv $LOG_DIR
fi
        
# Finetune
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --learning-rate 0.1 --imp-method VAN --synap-stgth 0.0 --log-dir results --mem-size 1 --examples-per-task 1000 --eps-mem-batch $EPS_MEM_BATCH_SIZE --anchor-eta 0.1
# EWC
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --learning-rate 0.1 --imp-method EWC --synap-stgth 10.0 --log-dir results --mem-size 1 --examples-per-task 1000 --eps-mem-batch $EPS_MEM_BATCH_SIZE --anchor-eta 0.1
# A-GEM
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --learning-rate 0.1 --imp-method A-GEM --synap-stgth 0.0 --log-dir results --mem-size 1 --examples-per-task 1000 --eps-mem-batch $EPS_MEM_BATCH_SIZE --anchor-eta 0.1
# MER
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --learning-rate 0.03 --imp-method MER --synap-stgth 0.0 --log-dir results --mem-size 1 --examples-per-task 1000 --eps-mem-batch $EPS_MEM_BATCH_SIZE --anchor-eta 0.1
# ER-Ringbuffer
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --learning-rate 0.1 --imp-method ER-Ring --synap-stgth 0.0 --log-dir results --mem-size 1 --examples-per-task 1000 --eps-mem-batch $EPS_MEM_BATCH_SIZE --anchor-eta 0.1
# HAL (Ours)
python fc_permute_mnist.py --train-single-epoch --cross-validate-mode --num-runs $NUM_RUNS --batch-size $BATCH_SIZE --learning-rate 0.1 --imp-method ER-Hindsight-Anchors --synap-stgth 0.1 --log-dir results --mem-size 1 --examples-per-task 1000 --eps-mem-batch $EPS_MEM_BATCH_SIZE --anchor-eta 0.1
