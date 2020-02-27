#!/bin/bash
python -u ../preprocess.py \
  -train_src ../data/epv7/ep_train.pt \
  -train_tgt ../data/epv7/ep_train.en \
  -train_xe_src ../data/os2018/os_train.pt \
  -train_xe_tgt ../data/os2018/os_train.en \
  -train_pg_src ../data/os2018/os_train.pt \
  -train_pg_tgt ../data/os2018/os_train.en \
  -valid_src ../data/os2018/os_valid.pt \
  -valid_tgt ../data/os2018/os_valid.en \
  -test_src ../data/os2018/os_test.pt \
  -test_tgt ../data/os2018/os_test.en \
  -save_data ../data/processed_all
