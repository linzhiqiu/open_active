#!/bin/bash

export  WANDB_MODE=dryrun
rm -r temp
python classification.py \
--dir temp_CUB \
--data_download_path /media/cheng/Samsung_T5/cub_200_2011/CUB_200_2011 \
--data CUB200 \
--data_config regular \
--bsize 64 \
--wd=3e-3 \
--fail_epoch=5 \
--model resnet18_high_res \