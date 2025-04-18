#!/bin/bash
#SBATCH --account vjgo8416-spchmetrics
#SBATCH --qos turing
#SBATCH --job-name process_DEMETR_MetricX
#SBATCH --time 0-2:00:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --output /bask/projects/v/vjgo8416-spchmetrics/jr/slurm_outputs/logs/%j.out

# Load required modules here
module purge
module load baskerville
module load bask-apps/live/live
module load Python
module load libsndfile

# project sub-directory to use as root
export PROJ="/bask/projects/v/vjgo8416-spchmetrics/jr"

# uv setup
export PATH="${PROJ}/uv_install/bin:$PATH"
export UV_CACHE_DIR="${PROJ}/uv_install/cache"
export UV_PYTHON_INSTALL_DIR="${PROJ}/uv_install/pythons"
export UV_PYTHON_BIN_DIR="${PROJ}/uv_install/python_bins"
export UV_TOOL_DIR="${PROJ}/uv_install/tools"

source "${PROJ}/ARC-m4st/.venv/bin/activate"

export TORCH_HOME="$PROJ/torch-cache"
export HF_HOME="$PROJ/hf-cache"
export HF_DATASETS_CACHE="$PROJ/hf-cache"
export PYTORCH_FAIRSEQ_CACHE="$PROJ/torch-cache/fairseq2"
export FAIRSEQ2_CACHE_DIR="$PROJ/torch-cache/fairseq2"
export XDG_CACHE_HOME="$PROJ/torch-cache/fairseq2"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd "${PROJ}/ARC-m4st/scripts/demetr"
python process_demetr.py --metrics MetricX_ref --metricx-model "google/metricx-24-hybrid-large-v2p6"
python process_demetr.py --metrics MetricX_ref --metricx-model "google/metricx-24-hybrid-xl-v2p6"
python process_demetr.py --metrics MetricX_ref --metricx-model "google/metricx-24-hybrid-large-v2p6-bfloat16"
python process_demetr.py --metrics MetricX_ref --metricx-model "google/metricx-24-hybrid-xl-v2p6-bfloat16"

python process_demetr.py --metrics MetricX_qe --metricx-model "google/metricx-24-hybrid-large-v2p6"
python process_demetr.py --metrics MetricX_qe --metricx-model "google/metricx-24-hybrid-xl-v2p6"
python process_demetr.py --metrics MetricX_qe --metricx-model "google/metricx-24-hybrid-large-v2p6-bfloat16"
python process_demetr.py --metrics MetricX_qe --metricx-model "google/metricx-24-hybrid-xl-v2p6-bfloat16"

python process_demetr.py --metrics MetricX_ref --metricx-model "google/metricx-24-hybrid-xxl-v2p6"
python process_demetr.py --metrics MetricX_ref --metricx-model "google/metricx-24-hybrid-xxl-v2p6-bfloat16"
python process_demetr.py --metrics MetricX_qe --metricx-model "google/metricx-24-hybrid-xxl-v2p6"
python process_demetr.py --metrics MetricX_qe --metricx-model "google/metricx-24-hybrid-xxl-v2p6-bfloat16"
