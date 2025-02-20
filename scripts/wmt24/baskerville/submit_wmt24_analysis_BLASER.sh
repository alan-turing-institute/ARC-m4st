#!/bin/bash
#SBATCH --account vjgo8416-spchmetrics
#SBATCH --qos turing
#SBATCH --job-name WMT24_BLASER
#SBATCH --time 0-4:00:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --output /bask/homes/s/siql2253/vjgo8416-spchmetrics/kac/slurm_outputs/logs/%j.out
#SBATCH --tasks-per-node 1
#SBATCH --mem-per-cpu 16000
# Load required modules here
source deactivate
module purge
module load baskerville
module load bask-apps/live/live
module load Python
module load libsndfile

source /bask/homes/s/siql2253/vjgo8416-spchmetrics/bv/ARC-m4st/env/bin/activate

export PROJ="/bask/projects/v/vjgo8416-spchmetrics"
export TORCH_HOME="$PROJ/kac/torch-cache"
export HF_HOME="$PROJ/kac/hf-cache"
export HF_DATASETS_CACHE="$PROJ/kac/hf-cache"
export PYTORCH_FAIRSEQ_CACHE="$PROJ/kac/torch-cache/fairseq2"
export FAIRSEQ2_CACHE_DIR="$PROJ/kac/torch-cache/fairseq2"
export XDG_CACHE_HOME="$PROJ/kac/torch-cache/fairseq2"

python /bask/homes/s/siql2253/vjgo8416-spchmetrics/kac/ARC-m4st/scripts/wmt24/run_wmt24_analysis_blaser.py --source-audio-dir /bask/homes/s/siql2253/vjgo8416-spchmetrics/kac/mt-metrics-eval-data/WMT24_GeneralMT_audio --wmt-data-dir /bask/homes/s/siql2253/vjgo8416-spchmetrics/kac/mt-metrics-eval-data/mt-metrics-eval-v2/wmt24 --output-dir /bask/homes/s/siql2253/vjgo8416-spchmetrics/kac/ARC-m4st/outputs/wmt24/BLASER --lang-pair en-de
