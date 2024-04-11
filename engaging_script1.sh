#!/bin/bash
#SBATCH -J testing-script           # Job Name 
#SBATCH -p sched_mit_hill
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16000
#SBATCH -t 00:15:00   
#SBATCH -o output/testing_script_result_%j.out

module load python/3.9.4
python 20240315_plot_degree_dist_L1.py filename6 0.03 42