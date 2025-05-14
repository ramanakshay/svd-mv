#!/bin/bash

#SBATCH --job-name=rtx5_multi_gpu_f12
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=64GB
#SBATCH --account=pr_95_tandon_priority
#SBATCH --output=rm5_gpu.log
#SBATCH --error=rm5_gpu_err.log

module purge

singularity exec \
--nv --overlay /scratch/ar8692/filesystems/multi-view-fs.ext3:ro \
/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh;
conda activate svd_mv;
cd /scratch/ar8692/svd-mv;
python -m accelerate.commands.launch --config_file accelerate_config/deepspeed_zero_3_multi_node.yaml main.py train train_config/multi_gpu_train_svd_mv.yaml
"