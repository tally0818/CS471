import argparse
import json
import os
from typing import Dict, Any

import torch
import torch.nn.functional as F
import random
from common import (
    GNNEncoder,
    DIRECT_PROMPTS,
    create_few_shot_dataset,
    load_graph_dataset,
    set_seed,
)
from node_selection import (
    find_agreed_and_disagreed_nodes,
    load_llm_predictions_for_selected,
    filter_disagreed_by_preference,
    retrain_gnn_on_agreed,
)


def prepare_dpo_dataset(agreed_nodes, disagreed_nodes, gnn_predictions, graph_data, dataset_name, dpo_path, sft_path):
    prompt = DIRECT_PROMPTS.get(dataset_name.lower(), "")
    if not prompt:
        return

    dpo_dataset = []
    sft_dataset = []
    
    os.makedirs(os.path.dirname(dpo_path), exist_ok=True)
    os.makedirs(os.path.dirname(sft_path), exist_ok=True)
    
    gnn_preds_all = gnn_predictions.argmax(dim=1).cpu()

    for node_id, value in agreed_nodes.items():
        try:
            node_text = graph_data.raw_texts[node_id]
            pred_idx = int(gnn_preds_all[node_id].item())
            
            if pred_idx < 0 or pred_idx >= len(graph_data.label_name):
                continue
                
            label = graph_data.label_name[pred_idx]
            
            sft_dataset.append({
                "conversations": [
                    {"from": "human", "value": f"{node_text}\n{prompt}"},
                    {"from": "gpt", "value": label}
                ]
            })

            dpo_dataset.append({
                "conversations": [{"from": "human", "value": f"{node_text}\n{prompt}"}],
                "chosen": {"from": "gpt", "value": label},
                "rejected": {"from": "gpt", "value": label}
            })  
            
        except Exception:
            continue
    
    for node_id, triple in disagreed_nodes.items():
        try:
            node_text = graph_data.raw_texts[node_id]
            
            if isinstance(triple, (list, tuple)) and len(triple) >= 2:
                gnn_pred_idx, llm_pred_idx = int(triple[0]), int(triple[1])
            elif isinstance(triple, dict):
                gnn_pred_idx = triple.get('gnn_pred')
                llm_pred_idx = triple.get('llm_pred')
                if gnn_pred_idx is None or llm_pred_idx is None:
                    continue
                gnn_pred_idx, llm_pred_idx = int(gnn_pred_idx), int(llm_pred_idx)
            else:
                continue
            
            if (gnn_pred_idx < 0 or gnn_pred_idx >= len(graph_data.label_name) or 
                llm_pred_idx < 0 or llm_pred_idx >= len(graph_data.label_name)):
                continue
                
            if gnn_pred_idx == llm_pred_idx:
                continue
                
            chosen_label = graph_data.label_name[gnn_pred_idx]
            rejected_label = graph_data.label_name[llm_pred_idx]
            
            dpo_dataset.append({
                "conversations": [{"from": "human", "value": f"{node_text}\n{prompt}"}],
                "chosen": {"from": "gpt", "value": chosen_label},
                "rejected": {"from": "gpt", "value": rejected_label}
            })
            
            sft_dataset.append({
                "conversations": [
                    {"from": "human", "value": f"{node_text}\n{prompt}"},
                    {"from": "gpt", "value": chosen_label}
                ]
            })
            
        except Exception:
            continue

    with open(dpo_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_dataset, f, indent=2)
    
    with open(sft_path, 'w', encoding='utf-8') as f:
        json.dump(sft_dataset, f, indent=2)


def _load_ensemble_models(graph_data, args, model_specs):
    """Load ensemble GNN models and return (models list, soft-voted probs).

    model_specs: list of (gnn_type, checkpoint_path) tuples.
    """
    num_classes = 47 if args.dataset == "ogbn-products_subset" else graph_data.y.max().item() + 1
    models = []
    prob_sum = None
    for gtype, path in model_specs:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ensemble checkpoint not found: {path}")
        model = GNNEncoder(
            input_dim=graph_data.x.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            n_layers=args.n_layers,
            gnn_type=gtype,
        ).to(args.device)
        model.load_state_dict(torch.load(path, map_location=args.device))
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(graph_data.x, graph_data.edge_index), dim=1)
        if prob_sum is None:
            prob_sum = probs
        else:
            prob_sum = prob_sum + probs
        models.append(model)
        print(f"  Loaded {gtype}: {path}")
    ensemble_probs = prob_sum / len(models)
    return models, ensemble_probs


def _ensemble_vote(models, graph_data):
    """Run soft voting across already-loaded models, return averaged probs."""
    prob_sum = None
    for model in models:
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(graph_data.x, graph_data.edge_index), dim=1)
        if prob_sum is None:
            prob_sum = probs
        else:
            prob_sum = prob_sum + probs
    return prob_sum / len(models)


def load_ensemble_predictions(graph_data, args):
    """Load K GNN checkpoints (homo-ensemble) and return (models, soft-voted probs)."""
    ensemble_dir = args.ensemble_model_dir
    specs = []
    for i in range(args.ensemble_k):
        path = os.path.join(ensemble_dir, f"{args.dataset}_{args.shots}_shot_best_model_ensemble{i}.pt")
        specs.append((args.gnn_type, path))
    print(f"[Homo-Ensemble] Loading {args.ensemble_k} GNN models from {ensemble_dir}")
    models, probs = _load_ensemble_models(graph_data, args, specs)
    print(f"[Homo-Ensemble] Soft voting done — avg of {args.ensemble_k} models")
    return models, probs


def load_hetero_ensemble_predictions(graph_data, args):
    """Load GCN/GAT/SAGE checkpoints and return (models, soft-voted probs)."""
    hetero_types = args.hetero_models
    ensemble_dir = args.ensemble_model_dir
    specs = []
    for gtype in hetero_types:
        path = os.path.join(ensemble_dir, f"{args.dataset}_{args.shots}_shot_best_model_{gtype}.pt")
        specs.append((gtype, path))
    print(f"[Hetero-Ensemble] Loading {hetero_types} from {ensemble_dir}")
    models, probs = _load_ensemble_models(graph_data, args, specs)
    print(f"[Hetero-Ensemble] Soft voting done — avg of {len(models)} models ({hetero_types})")
    return models, probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--selected_nodes_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Single GNN checkpoint path (required unless using ensemble)")
    parser.add_argument("--llm_predictions", type=str, required=True)
    parser.add_argument("--dpo_output_path", type=str, required=True)
    parser.add_argument("--sft_output_path", type=str, required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--gnn_type", type=str, default="GCN", choices=["GCN", "GAT", "SAGE","SGConv"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--path_prefix", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    # Ensemble flags
    parser.add_argument("--use_ensemble", action="store_true", default=False,
                        help="Use multiple GNN ensemble instead of single model")
    parser.add_argument("--ensemble_k", type=int, default=3,
                        help="Number of GNN models in ensemble (3 or 5)")
    parser.add_argument("--ensemble_model_dir", type=str, default=None,
                        help="Directory containing ensemble checkpoints (default: results/GNN)")
    parser.add_argument("--voting_method", type=str, default="soft", choices=["soft"],
                        help="Ensemble voting method")
    # Hetero-Ensemble flags
    parser.add_argument("--hetero_models", type=str, nargs="+", default=None,
                        help="GNN types for hetero-ensemble, e.g. --hetero_models GCN GAT SAGE")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.ensemble_model_dir is None:
        args.ensemble_model_dir = os.path.join(args.path_prefix, "results", "GNN")

    if args.shots:
        graph_data = create_few_shot_dataset(args.dataset, shots=args.shots, seed=args.seed, device=args.device, path_prefix=args.path_prefix)
    else:
        graph_data, _, _ = load_graph_dataset(args.dataset, device=args.device, path_prefix=args.path_prefix)

    num_classes = graph_data.y.max().item() + 1

    with open(args.selected_nodes_path, 'r') as f:
        selected_nodes_ids = json.load(f)["selected_node_ids"]

    if args.dataset == "ogbn-products_subset":
        num_classes = 47
    else:
        num_classes = graph_data.y.max().item() + 1

    # --- Hetero-Ensemble vs Homo-Ensemble vs Single GNN ---
    if not args.hetero_models and not args.use_ensemble and args.pretrained_model is None:
        parser.error("--pretrained_model is required when not using ensemble mode")

    is_prob = False
    ensemble_models = None
    if args.hetero_models:
        print(f"[Hetero-Ensemble mode] models={args.hetero_models}, voting={args.voting_method}")
        ensemble_models, gnn_predictions = load_hetero_ensemble_predictions(graph_data, args)
        is_prob = True
        gnn_model = None
    elif args.use_ensemble:
        print(f"[Homo-Ensemble mode] K={args.ensemble_k}, voting={args.voting_method}")
        ensemble_models, gnn_predictions = load_ensemble_predictions(graph_data, args)
        is_prob = True
        gnn_model = None
    else:
        gnn_model = GNNEncoder(
            input_dim=graph_data.x.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            n_layers=args.n_layers,
            gnn_type=args.gnn_type,
        ).to(args.device)
        gnn_model.load_state_dict(torch.load(args.pretrained_model, map_location=args.device))
        gnn_model.eval()
        with torch.no_grad():
            gnn_predictions = gnn_model(graph_data.x, graph_data.edge_index)

    print(f"Loading LLM predictions from {args.llm_predictions}")
    llm_predictions = load_llm_predictions_for_selected(args.llm_predictions, selected_nodes_ids, graph_data)
    print(f"LLM predictions mapped: {len(llm_predictions)}/{len(selected_nodes_ids)}")

    covered_ids = [int(nid) for nid in selected_nodes_ids if int(nid) in llm_predictions]
    selected_nodes_ids = covered_ids

    agreed_nodes, disagreed_nodes = find_agreed_and_disagreed_nodes(
        gnn_predictions, llm_predictions, graph_data, is_probability=is_prob)
    agreed_nodes = {nid: agreed_nodes[nid] for nid in selected_nodes_ids if nid in agreed_nodes}
    disagreed_nodes = {nid: disagreed_nodes[nid] for nid in selected_nodes_ids if nid in disagreed_nodes}
    print(f"Initial: {len(agreed_nodes)} agreed, {len(disagreed_nodes)} disagreed")

    if gnn_model is not None:
        if len(agreed_nodes) > 0:
            print(f"Retraining GNN on {len(agreed_nodes)} agreed nodes...")
            retrain_gnn_on_agreed(gnn_model, graph_data, agreed_nodes, args.device, lr=1e-3, epochs=50)
            with torch.no_grad():
                gnn_predictions = gnn_model(graph_data.x, graph_data.edge_index)
            is_prob = False
            agreed_nodes_all, disagreed_nodes_all = find_agreed_and_disagreed_nodes(
                gnn_predictions, llm_predictions, graph_data, is_probability=is_prob)
            agreed_nodes = {nid: agreed_nodes_all[nid] for nid in selected_nodes_ids if nid in agreed_nodes_all}
            disagreed_nodes = {nid: disagreed_nodes_all[nid] for nid in selected_nodes_ids if nid in disagreed_nodes_all}
            print(f"After retrain: {len(agreed_nodes)} agreed, {len(disagreed_nodes)} disagreed")
    elif ensemble_models is not None:
        print(f"[Ensemble] Retraining {len(ensemble_models)} models on per-model agreed nodes...")
        retrained_any = False
        all_agreed_sets = []
        for i, model in enumerate(ensemble_models):
            model.eval()
            with torch.no_grad():
                individual_preds = F.softmax(model(graph_data.x, graph_data.edge_index), dim=1)
            individual_agreed, _ = find_agreed_and_disagreed_nodes(
                individual_preds, llm_predictions, graph_data, is_probability=True)
            individual_agreed = {nid: individual_agreed[nid] for nid in selected_nodes_ids if nid in individual_agreed}
            all_agreed_sets.append(set(individual_agreed.keys()))
            if len(individual_agreed) > 0:
                retrain_gnn_on_agreed(model, graph_data, individual_agreed, args.device, lr=1e-3, epochs=50)
                retrained_any = True
                print(f"  Model {i}: {len(individual_agreed)} agreed nodes → retrained")
            else:
                print(f"  Model {i}: 0 agreed nodes → skipped")
        # Log pairwise overlap between per-model agreed sets
        if len(all_agreed_sets) > 1:
            print("[Ensemble] Per-model agreed set overlap:")
            for i in range(len(all_agreed_sets)):
                for j in range(i + 1, len(all_agreed_sets)):
                    intersection = len(all_agreed_sets[i] & all_agreed_sets[j])
                    union = len(all_agreed_sets[i] | all_agreed_sets[j])
                    overlap = intersection / union if union > 0 else 0.0
                    print(f"  Model {i} vs {j}: {intersection}/{union} overlap (IoU={overlap:.2%})")
        if retrained_any:
            gnn_predictions = _ensemble_vote(ensemble_models, graph_data)
            is_prob = True
            agreed_nodes_all, disagreed_nodes_all = find_agreed_and_disagreed_nodes(
                gnn_predictions, llm_predictions, graph_data, is_probability=is_prob)
            agreed_nodes = {nid: agreed_nodes_all[nid] for nid in selected_nodes_ids if nid in agreed_nodes_all}
            disagreed_nodes = {nid: disagreed_nodes_all[nid] for nid in selected_nodes_ids if nid in disagreed_nodes_all}
            print(f"After ensemble retrain: {len(agreed_nodes)} agreed, {len(disagreed_nodes)} disagreed")

    final_disagreed_nodes = filter_disagreed_by_preference(
        disagreed_nodes, gnn_predictions, args.confidence_threshold, is_probability=is_prob)
    print(f"Disagreed after confidence filter (>={args.confidence_threshold}): {len(final_disagreed_nodes)}")

    prepare_dpo_dataset(agreed_nodes, final_disagreed_nodes, gnn_predictions, graph_data, args.dataset, args.dpo_output_path, args.sft_output_path)

    dpo_count = len(json.load(open(args.dpo_output_path)))
    print(f"DPO dataset: {dpo_count} pairs saved to {args.dpo_output_path}")
    print(f"SFT dataset saved to {args.sft_output_path}")

if __name__ == "__main__":
    main() 