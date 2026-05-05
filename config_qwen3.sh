#!/bin/bash
# GNN-as-Judge Configuration for Qwen3-4B-Instruct-2507 + LoRA
# Copy this file to config.sh or source it directly before running the pipeline

# ===== WORKSPACE CONFIGURATION =====
export WORKSPACE_DIR="/path/to/your/workspace"
export LF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/LLaMA-Factory"
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_NAME="GNNJudge"
export CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"

# ===== MODEL PATHS =====
# Qwen3-4B-Instruct-2507: lightweight 4B model, ideal for LoRA fine-tuning on a single GPU
export BASE_MODEL_PATH="/path/to/models/Qwen3-4B-Instruct-2507"
export TEMPLATE="qwen"  # Qwen series uses the "qwen" (ChatML) template in LLaMA-Factory

# ===== GNN CONFIGURATION =====
export GNN_TYPE="GCN"
export GNN_HIDDEN_DIM=64
export GNN_LAYERS=2

# ===== TRAINING HYPERPARAMETERS =====
# SFT (Supervised Fine-Tuning)
export LEARNING_RATE_SFT=1e-4
export BATCH_SIZE_SFT=4
export EPOCHS_SFT=5

# DPO (Direct Preference Optimization)
export LEARNING_RATE_DPO=5e-6
export BATCH_SIZE_DPO=2
export EPOCHS_DPO=3
export DPO_BETA=0.1

# LoRA — rank 16 gives a good capacity/efficiency trade-off for 4B models
export LORA_RANK=16
export LORA_ALPHA=32
export GRAD_ACCUM_STEPS=4

# ===== HARDWARE CONFIGURATION =====
export NUM_GPUS=1
export MAIN_PROCESS_PORT_BASE=29500
export VISIBLE_DEVICES="0"

# ===== NODE SELECTION PARAMETERS =====
export TOPK_INFLUENTIAL=1500
export CONFIDENCE_THRESHOLD=0.7
export MAX_SUBGRAPH_NODES=3000
