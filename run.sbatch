#!/bin/bash

#SBATCH --job-name=s12ingle_gpu
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --account=pr_95_tandon_priority
#SBATCH --output=s12_gpu.log
#SBATCH --error=s12_gpu_err.log

module purge

singularity exec \
--nv --overlay /scratch/ar8692/filesystems/multi-view-fs.ext3:ro \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;
conda activate svd_mv;
cd /scratch/ar8692/svd-mv;
python -m accelerate.commands.launch --config_file accelerate_config/deepspeed_zero_2_one_gpu.yaml main.py train train_config/train_local_svd_mv.yaml
"