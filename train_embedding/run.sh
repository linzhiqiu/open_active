#!/bin/bash

export  WANDB_MODE=dryrun
rm -r temp
python classification.py \
--dir temp \
--data_path ~/data \
--data CIFAR100 \
--init_mode regular \
--bsize 64 \
--wd=1e-4 \
--fail_epoch=5