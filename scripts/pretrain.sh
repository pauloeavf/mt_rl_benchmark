#!/bin/bash
python -u ../train.py -data ../data/processed_all-train.pt \
                      -start_reinforce 11 \
                      -critic_pretrain_epochs 5 \
                      -end_epoch 15 \
                      -save_dir ../logs > ../logs/pre_train.log