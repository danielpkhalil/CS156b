#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH -J "first"   # job name
#SBATCH --mail-user=dkhalil@example.com   # email address
#SBATCH --mail-type=BEGIN

/home/dkhalil/miniconda3/envs/kaggle_cuties/bin/python -u /groups/CS156b/2024/kc/CS156b/train_specific_pathology.py --csv_path /groups/CS156b/data/student_labels/train2023.csv --data_path /groups/CS156b/data