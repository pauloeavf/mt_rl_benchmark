# mt_rl_benchmark
This repository is the implementation of my Minor Thesis named "Benchmarking Personalized Machine Translation algorithms under Bandit Feedback" (see PDF document in the root), as part of the Master of Data Science program at Monash University.

### Prepare datasets

Go to scripts folder
~~~~
> cd scripts
~~~~

Download and unzip datasets
~~~~
> bash get_datasets.sh
~~~~

Preprocess the datasets (using Moses Decoder libraries)
~~~~
> bash preprocess_datasets.sh
~~~~

Split between train / validation / test datasets
~~~~
> python split_datasets.py
~~~~

### Pretrain

Pretrain actor and critic with the out-of-domain datasets
~~~~
> bash pretrain.sh
~~~~

### Training

In the root directory

Unperturbed rewards
~~~~
> train.py -data data/processed_all-train.pt -load_from <pre_trained_model> -save_dir <save_dir> -start_reinforce -1 -end_epoch 20 -gpus 1
~~~~

Humanized perturbed rewards
~~~~
> train.py -data data/processed_all-train.pt -load_from <pre_trained_model> -save_dir <save_dir> -start_reinforce -1 -end_epoch 20 -pert_func <bin/var/skew> -pert_param <magnitude_param> -gpus 1
~~~~

### Evaluation

In the root directory
~~~~
> train.py -data data/processed_all-train.pt -load_from <final_model> -save_dir <save_dir> -eval
~~~~