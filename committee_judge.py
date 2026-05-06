"""
Heterogeneous Committee of GNN Judges.

Instead of a single GNN judge, this module trains multiple GNNs (GCN, GAT, SAGE)
with different inductive biases and uses their collective agreement/disagreement
to produce higher-quality pseudo labels for LLM fine-tuning.

Key ideas:
  1. Robust Agreement: ALL GNNs + LLM must unanimously agree → Agreement Set
  2. Uncertainty-aware Disagreement: majority GNNs agree on class X, LLM predicts
     class Y != X, AND ensemble variance is low → true Hard Sample
  3. Ensemble preference score: averaged across all GNN judges
"""

import copy
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from common import GNNEncoder, DIRECT_PROMPTS, create_few_shot_dataset, set_seed
from node_selection import load_llm_predictions_for_selected


# ---------------------------------------------------------------------------
# 1. Train / load a committee of heterogeneous GNNs
# ---------------------------------------------------------------------------

def train_single_gnn(
    graph_data,
    gnn_type: str,
    hidden_dim: int,
    n_layers: int,
    num_classes: int,
    device: str,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    epochs: int = 200,
    patience: int = 50,
    dropout: float = 0.5,
    save_path: Optional[str] = None,
) -> GNNEncoder:
    """Train one GNN and return the best model (by val accuracy)."""
    model = GNNEncoder(
        input_dim=graph_data.x.shape[1],
        hidden_dim=hidden_dim,
        output_dim=num_classes,
        n_layers=n_layers,
        gnn_type=gnn_type,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None
    counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = F.cross_entropy(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(graph_data.x, graph_data.edge_index).argmax(dim=1)
            val_acc = (pred[graph_data.val_mask] == graph_data.y[graph_data.val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(best_state, save_path)

    # Report test acc for reference
    with torch.no_grad():
        pred = model(graph_data.x, graph_data.edge_index).argmax(dim=1)
        test_acc = (pred[graph_data.test_mask] == graph_data.y[graph_data.test_mask]).float().mean().item()
    print(f"  [{gnn_type}] val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}")

    return model


def train_gnn_committee(
    graph_data,
    gnn_types: List[str],
    hidden_dim: int = 64,
    n_layers: int = 2,
    device: str = "cuda:0",
    epochs: int = 200,
    patience: int = 50,
    dropout: float = 0.5,
    save_dir: Optional[str] = None,
    dataset_name: str = "",
    shots: int = 0,
) -> Dict[str, GNNEncoder]:
    """Train a committee of heterogeneous GNNs and return {type: model}."""
    num_classes = graph_data.y.max().item() + 1
    committee = {}
    print(f"Training GNN committee: {gnn_types}")
    for gnn_type in gnn_types:
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f"{dataset_name}_{shots}_shot_{gnn_type}_judge.pt")
        committee[gnn_type] = train_single_gnn(
            graph_data, gnn_type, hidden_dim, n_layers, num_classes,
            device, epochs=epochs, patience=patience, dropout=dropout,
            save_path=save_path,
        )
    return committee


def load_gnn_committee(
    graph_data,
    gnn_types: List[str],
    model_paths: Dict[str, str],
    hidden_dim: int = 64,
    n_layers: int = 2,
    device: str = "cuda:0",
) -> Dict[str, GNNEncoder]:
    """Load pre-trained committee members from saved checkpoints."""
    num_classes = graph_data.y.max().item() + 1
    committee = {}
    for gnn_type in gnn_types:
        model = GNNEncoder(
            input_dim=graph_data.x.shape[1],
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            n_layers=n_layers,
            gnn_type=gnn_type,
        ).to(device)
        model.load_state_dict(torch.load(model_paths[gnn_type], map_location=device))
        model.eval()
        committee[gnn_type] = model
    return committee


# ---------------------------------------------------------------------------
# 2. Gather predictions from the committee
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_committee_predictions(
    committee: Dict[str, GNNEncoder],
    graph_data,
) -> Dict[str, torch.Tensor]:
    """Return {gnn_type: logits_tensor} for each committee member."""
    predictions = {}
    for gnn_type, model in committee.items():
        model.eval()
        predictions[gnn_type] = model(graph_data.x, graph_data.edge_index)
    return predictions


# ---------------------------------------------------------------------------
# 3. Ensemble agreement / disagreement with uncertainty
# ---------------------------------------------------------------------------

def ensemble_find_agreed_and_disagreed(
    committee_predictions: Dict[str, torch.Tensor],
    llm_predictions: Dict[int, int],
    min_gnn_agreement: Optional[int] = None,
) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
    """
    Identify agreement and disagreement sets using the GNN committee.

    Agreement:  ALL GNNs predict the same class AND LLM also predicts that class.
    Disagreement: >=min_gnn_agreement GNNs agree on class X, LLM predicts class Y != X.

    Returns:
        agreed_nodes: {node_id: {"label": int, "avg_conf": float}}
        disagreed_nodes: {node_id: {"gnn_pred": int, "llm_pred": int,
                                     "consensus": float, "variance": float,
                                     "avg_conf_gnn": float, "avg_conf_llm": float}}
    """
    gnn_types = list(committee_predictions.keys())
    n_judges = len(gnn_types)
    if min_gnn_agreement is None:
        min_gnn_agreement = n_judges  # default: unanimous

    # Stack softmax probs: (n_judges, n_nodes, n_classes)
    prob_stack = torch.stack([
        F.softmax(committee_predictions[t], dim=1) for t in gnn_types
    ], dim=0)
    pred_stack = torch.stack([
        committee_predictions[t].argmax(dim=1) for t in gnn_types
    ], dim=0)  # (n_judges, n_nodes)

    # Mean probs across judges
    mean_probs = prob_stack.mean(dim=0)  # (n_nodes, n_classes)
    # Variance of probs across judges for each class
    var_probs = prob_stack.var(dim=0)    # (n_nodes, n_classes)

    agreed_nodes = {}
    disagreed_nodes = {}

    num_nodes = pred_stack.shape[1]

    for node_id, llm_pred in llm_predictions.items():
        if node_id < 0 or node_id >= num_nodes:
            continue

        # Per-judge predictions for this node
        judge_preds = pred_stack[:, node_id].cpu().tolist()  # list of ints
        vote_counter = Counter(judge_preds)
        majority_class, majority_count = vote_counter.most_common(1)[0]

        # Consensus ratio
        consensus = majority_count / n_judges

        # Average confidence of each judge for the majority class
        avg_conf_majority = prob_stack[:, node_id, majority_class].mean().item()

        # Variance for the majority class across judges
        var_majority = var_probs[node_id, majority_class].item()

        if majority_count == n_judges and majority_class == llm_pred:
            # Unanimous agreement: all GNNs + LLM agree
            agreed_nodes[node_id] = {
                "label": majority_class,
                "avg_conf": avg_conf_majority,
            }
        elif majority_count >= min_gnn_agreement and majority_class != llm_pred:
            # Committee majority disagrees with LLM
            avg_conf_llm = prob_stack[:, node_id, llm_pred].mean().item()
            disagreed_nodes[node_id] = {
                "gnn_pred": majority_class,
                "llm_pred": llm_pred,
                "consensus": consensus,
                "variance": var_majority,
                "avg_conf_gnn": avg_conf_majority,
                "avg_conf_llm": avg_conf_llm,
            }

    return agreed_nodes, disagreed_nodes


# ---------------------------------------------------------------------------
# 4. Uncertainty-aware disagreement filtering
# ---------------------------------------------------------------------------

def filter_disagreed_by_committee_consensus(
    disagreed_nodes: Dict[int, Dict],
    confidence_threshold: float = 0.5,
    variance_threshold: float = 0.1,
) -> Dict[int, Dict]:
    """
    Keep only disagreed nodes where:
      - Ensemble preference score (avg_conf_gnn - avg_conf_llm) >= confidence_threshold
      - Prediction variance across judges <= variance_threshold (low uncertainty)
    """
    filtered = {}
    for node_id, info in disagreed_nodes.items():
        pref_score = info["avg_conf_gnn"] - info["avg_conf_llm"]
        if pref_score >= confidence_threshold and info["variance"] <= variance_threshold:
            filtered[node_id] = info
    return filtered


# ---------------------------------------------------------------------------
# 5. Retrain committee on agreed nodes
# ---------------------------------------------------------------------------

def retrain_committee_on_agreed(
    committee: Dict[str, GNNEncoder],
    graph_data,
    agreed_nodes: Dict[int, Dict],
    device: str,
    lr: float = 1e-3,
    epochs: int = 50,
):
    """Retrain each committee member using agreed pseudo labels."""
    if not agreed_nodes:
        return

    # Build mask and pseudo labels
    pseudo_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool, device=device)
    pseudo_labels = graph_data.y.clone()
    for nid, info in agreed_nodes.items():
        pseudo_mask[nid] = True
        pseudo_labels[nid] = info["label"]

    # Combine original training data with pseudo labels
    combined_mask = graph_data.train_mask | pseudo_mask

    for gnn_type, model in committee.items():
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(graph_data.x, graph_data.edge_index)
            loss = F.cross_entropy(logits[combined_mask], pseudo_labels[combined_mask])
            loss.backward()
            optimizer.step()
        model.eval()


# ---------------------------------------------------------------------------
# 6. Build DPO / WSFT datasets from committee results
# ---------------------------------------------------------------------------

def prepare_committee_dpo_dataset(
    agreed_nodes: Dict[int, Dict],
    disagreed_nodes: Dict[int, Dict],
    graph_data,
    dataset_name: str,
    dpo_path: str,
    sft_path: str,
):
    """Create DPO and WSFT JSON files from committee agreement/disagreement sets."""
    prompt = DIRECT_PROMPTS.get(dataset_name.lower(), "")
    if not prompt:
        print(f"Warning: no prompt template for {dataset_name}")
        return

    dpo_dataset = []
    sft_dataset = []

    os.makedirs(os.path.dirname(dpo_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(sft_path) or ".", exist_ok=True)

    # Agreed nodes → SFT + trivial DPO
    for node_id, info in agreed_nodes.items():
        try:
            node_text = graph_data.raw_texts[node_id]
            label_idx = info["label"]
            if label_idx < 0 or label_idx >= len(graph_data.label_name):
                continue
            label = graph_data.label_name[label_idx]

            sft_dataset.append({
                "conversations": [
                    {"from": "human", "value": f"{node_text}\n{prompt}"},
                    {"from": "gpt", "value": label},
                ]
            })
            dpo_dataset.append({
                "conversations": [{"from": "human", "value": f"{node_text}\n{prompt}"}],
                "chosen": {"from": "gpt", "value": label},
                "rejected": {"from": "gpt", "value": label},
            })
        except Exception:
            continue

    # Disagreed nodes → preference pair (GNN majority = chosen, LLM = rejected)
    for node_id, info in disagreed_nodes.items():
        try:
            node_text = graph_data.raw_texts[node_id]
            gnn_idx = info["gnn_pred"]
            llm_idx = info["llm_pred"]
            if gnn_idx == llm_idx:
                continue
            if gnn_idx < 0 or gnn_idx >= len(graph_data.label_name):
                continue
            if llm_idx < 0 or llm_idx >= len(graph_data.label_name):
                continue

            chosen_label = graph_data.label_name[gnn_idx]
            rejected_label = graph_data.label_name[llm_idx]

            dpo_dataset.append({
                "conversations": [{"from": "human", "value": f"{node_text}\n{prompt}"}],
                "chosen": {"from": "gpt", "value": chosen_label},
                "rejected": {"from": "gpt", "value": rejected_label},
            })
            sft_dataset.append({
                "conversations": [
                    {"from": "human", "value": f"{node_text}\n{prompt}"},
                    {"from": "gpt", "value": chosen_label},
                ]
            })
        except Exception:
            continue

    with open(dpo_path, "w", encoding="utf-8") as f:
        json.dump(dpo_dataset, f, indent=2, ensure_ascii=False)
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, indent=2, ensure_ascii=False)

    n_pref = sum(1 for r in dpo_dataset if r["chosen"]["value"] != r["rejected"]["value"])
    print(f"Committee DPO: {len(dpo_dataset)} total, {n_pref} non-trivial preference pairs")
    print(f"Committee WSFT: {len(sft_dataset)} examples")


# ---------------------------------------------------------------------------
# 7. End-to-end committee judge pipeline (for CLI or notebook use)
# ---------------------------------------------------------------------------

def run_committee_judge(
    dataset: str,
    selected_nodes_path: str,
    llm_predictions_file: str,
    dpo_output_path: str,
    sft_output_path: str,
    gnn_types: List[str] = None,
    hidden_dim: int = 64,
    n_layers: int = 2,
    confidence_threshold: float = 0.5,
    variance_threshold: float = 0.1,
    min_gnn_agreement: Optional[int] = None,
    shots: int = 3,
    seed: int = 42,
    device: str = "cuda:0",
    path_prefix: str = ".",
    pretrained_paths: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
    retrain_epochs: int = 50,
) -> Dict[str, Any]:
    """
    Full committee-of-judges pipeline:
      1. Train or load heterogeneous GNN committee
      2. Get ensemble predictions
      3. Find robust agreement / uncertainty-aware disagreement
      4. Retrain committee on agreed nodes and re-evaluate
      5. Filter disagreed by consensus + variance
      6. Produce DPO / WSFT datasets

    Returns a summary dict with statistics.
    """
    if gnn_types is None:
        gnn_types = ["GCN", "GAT", "SAGE"]

    set_seed(seed)

    # Load graph
    graph_data = create_few_shot_dataset(dataset, shots=shots, seed=seed, device=device, path_prefix=path_prefix)
    num_classes = graph_data.y.max().item() + 1

    # Load selected node IDs
    with open(selected_nodes_path, "r") as f:
        selected_node_ids = json.load(f)["selected_node_ids"]

    # Train or load committee
    if pretrained_paths:
        committee = load_gnn_committee(graph_data, gnn_types, pretrained_paths, hidden_dim, n_layers, device)
    else:
        committee = train_gnn_committee(
            graph_data, gnn_types, hidden_dim, n_layers, device,
            epochs=200, patience=50, save_dir=save_dir,
            dataset_name=dataset, shots=shots,
        )

    # Get ensemble predictions
    committee_preds = get_committee_predictions(committee, graph_data)

    # Load LLM predictions
    llm_predictions = load_llm_predictions_for_selected(llm_predictions_file, selected_node_ids, graph_data)
    covered_ids = [int(nid) for nid in selected_node_ids if int(nid) in llm_predictions]
    selected_node_ids = covered_ids
    print(f"LLM predictions mapped: {len(llm_predictions)}/{len(selected_node_ids)}")

    # Find agreement / disagreement
    agreed, disagreed = ensemble_find_agreed_and_disagreed(
        committee_preds, llm_predictions, min_gnn_agreement=min_gnn_agreement,
    )
    # Restrict to selected nodes
    agreed = {nid: agreed[nid] for nid in selected_node_ids if nid in agreed}
    disagreed = {nid: disagreed[nid] for nid in selected_node_ids if nid in disagreed}
    print(f"Initial committee: {len(agreed)} agreed, {len(disagreed)} disagreed")

    # Retrain committee on agreed nodes
    if agreed:
        print(f"Retraining committee on {len(agreed)} agreed nodes...")
        retrain_committee_on_agreed(committee, graph_data, agreed, device, epochs=retrain_epochs)
        committee_preds = get_committee_predictions(committee, graph_data)
        agreed, disagreed = ensemble_find_agreed_and_disagreed(
            committee_preds, llm_predictions, min_gnn_agreement=min_gnn_agreement,
        )
        agreed = {nid: agreed[nid] for nid in selected_node_ids if nid in agreed}
        disagreed = {nid: disagreed[nid] for nid in selected_node_ids if nid in disagreed}
        print(f"After retrain: {len(agreed)} agreed, {len(disagreed)} disagreed")

    # Filter disagreed by confidence + variance
    final_disagreed = filter_disagreed_by_committee_consensus(
        disagreed, confidence_threshold, variance_threshold,
    )
    print(f"Disagreed after committee filter (conf>={confidence_threshold}, var<={variance_threshold}): {len(final_disagreed)}")

    # Produce datasets
    prepare_committee_dpo_dataset(
        agreed, final_disagreed, graph_data, dataset, dpo_output_path, sft_output_path,
    )

    summary = {
        "gnn_types": gnn_types,
        "n_selected": len(selected_node_ids),
        "n_agreed": len(agreed),
        "n_disagreed_raw": len(disagreed),
        "n_disagreed_filtered": len(final_disagreed),
        "confidence_threshold": confidence_threshold,
        "variance_threshold": variance_threshold,
    }
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Committee of GNN Judges")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--selected_nodes_path", type=str, required=True)
    parser.add_argument("--llm_predictions", type=str, required=True)
    parser.add_argument("--dpo_output_path", type=str, required=True)
    parser.add_argument("--sft_output_path", type=str, required=True)
    parser.add_argument("--gnn_types", nargs="+", default=["GCN", "GAT", "SAGE"])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--variance_threshold", type=float, default=0.1)
    parser.add_argument("--min_gnn_agreement", type=int, default=None)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--path_prefix", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--retrain_epochs", type=int, default=50)
    args = parser.parse_args()

    summary = run_committee_judge(
        dataset=args.dataset,
        selected_nodes_path=args.selected_nodes_path,
        llm_predictions_file=args.llm_predictions,
        dpo_output_path=args.dpo_output_path,
        sft_output_path=args.sft_output_path,
        gnn_types=args.gnn_types,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        confidence_threshold=args.confidence_threshold,
        variance_threshold=args.variance_threshold,
        min_gnn_agreement=args.min_gnn_agreement,
        shots=args.shots,
        seed=args.seed,
        device=args.device,
        path_prefix=args.path_prefix,
        save_dir=args.save_dir,
        retrain_epochs=args.retrain_epochs,
    )
    print(json.dumps(summary, indent=2))
