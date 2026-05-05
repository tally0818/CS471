#!/bin/bash
# =============================================================================
# GNN-as-Judge: Qwen3-4B-Instruct-2507 LoRA Fine-tuning Example
# =============================================================================
# This script demonstrates the full pipeline using Qwen3-4B-Instruct-2507
# with LoRA adapters on the Cora dataset (3-shot setting).
#
# Prerequisites:
#   1. conda env create -f environment.yml && conda activate GNNJudge
#   2. cd LLaMA-Factory && pip install -e ".[torch,metrics]" --no-build-isolation && cd ..
#   3. Download Qwen3-4B-Instruct-2507 weights (e.g., from HuggingFace)
#   4. Download graph datasets into datasets/ (see README.md)
#   5. Edit the paths below to match your environment
#
# Usage:
#   bash run_qwen3_example.sh [dataset] [shots] [seed]
#   bash run_qwen3_example.sh cora 3 42
# =============================================================================

set -euo pipefail

# ---------- Paths (EDIT THESE) ----------
WORKSPACE_DIR="/path/to/your/workspace"
MODEL_PATH="/path/to/models/Qwen3-4B-Instruct-2507"   # HuggingFace local path
# ----------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CLI args
DATASET=${1:-"cora"}
SHOTS=${2:-"3"}
SEED=${3:-"42"}

# Derived paths
LF_DIR="$SCRIPT_DIR/LLaMA-Factory"
PROJECT_DIR="$SCRIPT_DIR"
DATASET_DIR="$LF_DIR/data"
DATASET_INFO="$DATASET_DIR/dataset_info.json"
EXP_DIR="$WORKSPACE_DIR/results/gnn_as_judge"
RUN_ID="${DATASET}_${SHOTS}shot_seed${SEED}"
RUN_DIR="$EXP_DIR/$RUN_ID"

export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="$WORKSPACE_DIR/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p "$HF_HOME" "$DATASET_DIR" "$RUN_DIR"

# ===== Conda =====
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
source "$CONDA_SH"
conda activate GNNJudge

# =====================================================================
# STAGE 0: Train GNN (judge model)
# =====================================================================
echo "=== Stage 0: Train GNN ==="
cd "$PROJECT_DIR/GNN"
python main.py \
  --dataset "$DATASET" \
  --shots "$SHOTS" \
  --gnn_type GCN \
  --hidden_dim 64 \
  --n_layers 2 \
  --epochs 200 \
  --seed "$SEED"
cd "$PROJECT_DIR"

GNN_MODEL="$PROJECT_DIR/results/GNN/${DATASET}_${SHOTS}_shot_best_model_run0.pt"

# =====================================================================
# STAGE 1: Create SFT dataset
# =====================================================================
echo "=== Stage 1: Create SFT dataset ==="
SFT_PREFIX="${DATASET}_sft_${SHOTS}_shot"

python create_sft.py \
  --dataset "$DATASET" \
  --output "$DATASET_DIR/${DATASET}_sft.json" \
  --shots "$SHOTS" \
  --seed "$SEED" \
  --path_prefix "."

# Register splits in dataset_info.json
python - <<PYEOF
import json, os
path = "$DATASET_INFO"
os.makedirs(os.path.dirname(path), exist_ok=True)
info = json.load(open(path)) if os.path.exists(path) else {}
for split in ["train", "val", "test", "unlabeled"]:
    key = f"${SFT_PREFIX}_{split}"
    info[key] = {
        "file_name": f"{key}.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"}
    }
json.dump(info, open(path, 'w'), indent=2, ensure_ascii=False)
PYEOF

# =====================================================================
# STAGE 2: LoRA SFT on Qwen3-4B-Instruct-2507
# =====================================================================
echo "=== Stage 2: LoRA SFT on Qwen3-4B ==="
SFT_DIR="$RUN_DIR/sft"
SFT_MODEL="$SFT_DIR/model"
SFT_LOG="$SFT_DIR/logs"
mkdir -p "$SFT_MODEL" "$SFT_LOG"

cd "$LF_DIR"
python -m accelerate.commands.accelerate_cli launch \
  --config_file "$LF_DIR/examples/accelerate/single_config.yaml" \
  --num_processes 1 \
  --main_process_port 29500 \
  "$LF_DIR/src/train.py" \
  --stage sft \
  --do_train \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_dir "$DATASET_DIR" \
  --dataset "${SFT_PREFIX}_train" \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target all \
  --output_dir "$SFT_MODEL" \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len 2048 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --save_steps 500 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --plot_loss \
  --bf16 \
  --save_total_limit 3 \
  --logging_dir "$SFT_LOG" 2>&1 | tee "$SFT_LOG/train.log"

# Pick latest checkpoint
BEST_SFT="$SFT_MODEL"
LATEST=$(ls -dt "$SFT_MODEL"/checkpoint-* 2>/dev/null | head -n 1)
[ -n "$LATEST" ] && BEST_SFT="$LATEST"
echo "SFT checkpoint: $BEST_SFT"

# =====================================================================
# STAGE 3: Select influential nodes
# =====================================================================
echo "=== Stage 3: Select influential nodes ==="
cd "$PROJECT_DIR"
SELECTED="$RUN_DIR/${RUN_ID}_selected_nodes.json"

python select_influential_nodes.py \
  --dataset "$DATASET" \
  --k 1500 \
  --output_file "$SELECTED" \
  --shots "$SHOTS" \
  --seed "$SEED" \
  --method auto \
  --max_subgraph_nodes 3000 \
  --max_distance 3 \
  --path_prefix "."

# Filter unlabeled set to selected nodes
SELECTED_DS="${SFT_PREFIX}_selected"
SELECTED_FILE="$DATASET_DIR/${SELECTED_DS}.json"

python - <<PYEOF
import json, os
with open("$SELECTED") as f:
    sel = set(json.load(f)['selected_node_ids'])
nid_path = "$DATASET_DIR/${DATASET}_${SHOTS}_shot_unlabeled_node_ids.json"
with open(nid_path) as f:
    all_ids = json.load(f)['selected_node_ids']
with open("$DATASET_DIR/${SFT_PREFIX}_unlabeled.json") as f:
    unlabeled = json.load(f)
filtered, ordered = [], []
for i, nid in enumerate(all_ids):
    if nid in sel:
        filtered.append(unlabeled[i])
        ordered.append(nid)
with open("$SELECTED_FILE", 'w') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)
ordered_path = "${SELECTED}".replace('.json', '_ordered.json')
with open(ordered_path, 'w') as f:
    json.dump({"selected_node_ids": ordered}, f, indent=2)
info_path = "$DATASET_INFO"
info = json.load(open(info_path))
info["$SELECTED_DS"] = {
    "file_name": os.path.basename("$SELECTED_FILE"),
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}
json.dump(info, open(info_path, 'w'), indent=2, ensure_ascii=False)
print(f"Filtered to {len(filtered)} selected nodes")
PYEOF

# =====================================================================
# STAGE 4: Generate LLM predictions on selected nodes
# =====================================================================
echo "=== Stage 4: LLM inference ==="
LLM_PRED="$RUN_DIR/${RUN_ID}_llm_preds.jsonl"
cd "$LF_DIR"

python src/vllm_infer.py \
  --model_name_or_path "$MODEL_PATH" \
  --adapter_name_or_path "$BEST_SFT" \
  --dataset "$SELECTED_DS" \
  --template qwen \
  --dataset_dir "$DATASET_DIR" \
  --save_name "$LLM_PRED"

# =====================================================================
# STAGE 5: Create WSFT (DPO) dataset via GNN-as-Judge
# =====================================================================
echo "=== Stage 5: GNN-as-Judge creates DPO dataset ==="
cd "$PROJECT_DIR"
DPO_DS="${RUN_ID}_dpo"
DPO_FILE="$DATASET_DIR/${DPO_DS}.json"
SFT_DPO_FILE="$DATASET_DIR/${DPO_DS}_sft.json"

python create_wsft.py \
  --dataset "$DATASET" \
  --selected_nodes_path "${SELECTED%.json}_ordered.json" \
  --pretrained_model "$GNN_MODEL" \
  --llm_predictions "$LLM_PRED" \
  --dpo_output_path "$DPO_FILE" \
  --sft_output_path "$SFT_DPO_FILE" \
  --confidence_threshold 0.7 \
  --shots "$SHOTS" \
  --gnn_type GCN \
  --hidden_dim 64 \
  --n_layers 2 \
  --seed "$SEED" \
  --device "cuda:0"

# Register DPO dataset
python - <<PYEOF
import json
path = "$DATASET_INFO"
info = json.load(open(path))
info["$DPO_DS"] = {
    "file_name": "${DPO_DS}.json",
    "formatting": "sharegpt",
    "ranking": True,
    "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected"}
}
info["${DPO_DS}_sft"] = {
    "file_name": "${DPO_DS}_sft.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"}
}
json.dump(info, open(path, 'w'), indent=2, ensure_ascii=False)
PYEOF

# =====================================================================
# STAGE 6: LoRA DPO on Qwen3-4B-Instruct-2507
# =====================================================================
echo "=== Stage 6: LoRA DPO on Qwen3-4B ==="
DPO_DIR="$RUN_DIR/dpo"
DPO_MODEL="$DPO_DIR/model"
DPO_LOG="$DPO_DIR/logs"
DPO_PRED_DIR="$DPO_DIR/predictions"
mkdir -p "$DPO_MODEL" "$DPO_LOG" "$DPO_PRED_DIR"

cd "$LF_DIR"
python -m accelerate.commands.accelerate_cli launch \
  --config_file "$LF_DIR/examples/accelerate/single_config.yaml" \
  --num_processes 1 \
  --main_process_port 29501 \
  "$LF_DIR/src/train.py" \
  --stage dpo \
  --do_train \
  --model_name_or_path "$MODEL_PATH" \
  --adapter_name_or_path "$BEST_SFT" \
  --create_new_adapter \
  --dataset_dir "$DATASET_DIR" \
  --dataset "$DPO_DS" \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target all \
  --pref_beta 0.1 \
  --pref_loss orpo \
  --output_dir "$DPO_MODEL" \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len 2048 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --save_steps 100 \
  --learning_rate 5e-6 \
  --num_train_epochs 3 \
  --plot_loss \
  --bf16 \
  --save_total_limit 3 \
  --logging_dir "$DPO_LOG" 2>&1 | tee "$DPO_LOG/train.log"

BEST_DPO="$DPO_MODEL"
LATEST=$(ls -dt "$DPO_MODEL"/checkpoint-* 2>/dev/null | head -n 1)
[ -n "$LATEST" ] && BEST_DPO="$LATEST"

# =====================================================================
# STAGE 7: Final evaluation
# =====================================================================
echo "=== Stage 7: Final evaluation ==="
TEST_PRED="$DPO_PRED_DIR/dpo_test_predictions.jsonl"

cd "$LF_DIR"
python src/vllm_infer.py \
  --model_name_or_path "$MODEL_PATH" \
  --adapter_name_or_path "$BEST_DPO" \
  --dataset "${SFT_PREFIX}_test" \
  --template qwen \
  --dataset_dir "$DATASET_DIR" \
  --save_name "$TEST_PRED"

cd "$PROJECT_DIR"
python evaluate_predictions.py \
  --dataset "$DATASET" \
  --pred_file "$TEST_PRED" \
  --output_dir "$DPO_PRED_DIR/final_eval" \
  --model_name "qwen3_4b_dpo" \
  --path_prefix "."

echo "=== Done! ==="
echo "Results: $DPO_PRED_DIR/final_eval"

METRICS="$DPO_PRED_DIR/final_eval/qwen3_4b_dpo/metrics.json"
if [ -f "$METRICS" ]; then
  python -c "
import json
with open('$METRICS') as f:
    m = json.load(f)
print(f'Accuracy: {m.get(\"accuracy\", \"N/A\"):.4f}')
print(f'Macro-F1: {m.get(\"macro_f1\", \"N/A\"):.4f}')
"
fi
