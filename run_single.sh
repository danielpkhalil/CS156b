#!/bin/bash

#SBATCH --job-name=Process_Data
#SBATCH --output=/home/apenn2/repo/CS156b/output_files/%j.out
#SBATCH --error=/home/apenn2/repo/CS156b/output_files/%j.err
#SBATCH -A CS156b
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-user=apenn2@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /home/apenn2/repo/CS156b/

python train_specific_pathology.py --csv_path /groups/CS156b/data/student_labels/train2023.csv --data_path /groups/CS156b/data
