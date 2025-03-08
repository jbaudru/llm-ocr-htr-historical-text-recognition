#!/usr/bin/bash
 
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=10gb
 
#SBATCH --job-name=test
#SBATCH --out=%x-%j.out
 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=seorin.kim@vub.be
 
echo Start Job
date
 
ml SciPy-bundle/2023.07-gfbf-2023a
ml PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
ml Pillow-SIMD/9.5.0-GCCcore-12.3.0
ml Transformers/4.48.2-gfbf-2023a-tokenizers-0.21.0
ml tqdm/4.66.1-GCCcore-12.3.0  
 
 
cd  ${VSC_DATA}
 
python -u LAM_Verification.py --end_epoch 1
             
echo End Job