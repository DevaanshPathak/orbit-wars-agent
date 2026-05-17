import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path


HF_REPO_ID = "devaanshpa/orbit-wars-agent"
HF_REPO_TYPE = "model"
HF_REMOTE_PREFIX = "v17"

METADATA_COLS = {
    "label",
    "selected",
    "outcome_weight",
    "game_result",
    "reward_margin",
    "agent_reward",
    "opponent_reward",
    "selected_heuristic_rank",
    "counterfactual_positive",
    "counterfactual_reason",
    "failure_overcommit",
    "failure_missed_tactical",
    "failure_missed_comet",
    "failure_slow_expansion",
    "turn_advantage",
    "future_advantage_delta_5",
    "future_advantage_delta_15",
    "future_advantage_delta_30",
    "future_production_delta_15",
    "future_planet_delta_15",
    "game_id",
    "candidate_id",
    "version",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the v17 Orbit Wars supervised ensemble ranker.")
    parser.add_argument("--csv", default=os.environ.get("CANDIDATES_CSV", ""))
    parser.add_argument(
        "--prefer-local-data",
        action="store_true",
        help="Use newest local data/*/candidates_v7.csv before Hugging Face. Default is to prefer Hugging Face.",
    )
    parser.add_argument("--export-dir", default="notebooks/v17/exports")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("V17_EPOCHS", "300")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("V17_BATCH_SIZE", "4096")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("V17_LR", "0.00045")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("V17_WEIGHT_DECAY", "0.00030")))
    parser.add_argument("--pair-loss-weight", type=float, default=float(os.environ.get("V17_PAIR_LOSS_WEIGHT", "1.10")))
    parser.add_argument("--max-pairs-per-turn", type=int, default=int(os.environ.get("V17_MAX_PAIRS_PER_TURN", "14")))
    parser.add_argument("--ensemble-size", type=int, default=int(os.environ.get("V17_ENSEMBLE_SIZE", "8")))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("V17_PATIENCE", "40")))
    parser.add_argument("--dropout", type=float, default=float(os.environ.get("V17_DROPOUT", "0.12")))
    parser.add_argument("--score-scale", type=float, default=float(os.environ.get("V17_SCORE_SCALE", "205.0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("V17_SEED", "1031")))
    parser.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("V17_CHECKPOINT_EVERY", "40")))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--hf-repo-type", default=HF_REPO_TYPE)
    return parser.parse_args()


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def find_training_csv(csv_arg, prefer_local=False):
    if csv_arg:
        path = Path(csv_arg)
        if not path.exists():
            raise FileNotFoundError(f"Training CSV does not exist: {path}")
        return path

    root = Path("data")
    candidates = (
        sorted(root.glob("*/candidates_v7.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if root.exists()
        else []
    )
    if prefer_local and candidates:
        return candidates[0]

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        if candidates:
            print("HF_TOKEN not set; falling back to newest local CSV.", flush=True)
            return candidates[0]
        raise FileNotFoundError("No candidates_v7.csv found locally and HF_TOKEN is not set for download.")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to download data: pip install huggingface_hub") from exc

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    data_csvs = sorted(
        [f for f in files if f.startswith("data/") and f.endswith("/candidates_v7.csv")],
        reverse=True,
    )
    if not data_csvs:
        raise FileNotFoundError("No data/*/candidates_v7.csv found on Hugging Face.")
    newest = data_csvs[0]
    print(f"Auto-discovered newest HF dataset: {newest}", flush=True)
    return Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=newest, token=token))


def row_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Training CSV has no rows.")
    feature_names = [name for name in rows[0] if name not in METADATA_COLS]
    x_raw = [[row_float(row, name, 0.0) for name in feature_names] for row in rows]
    y = [max(0.0, min(1.0, row_float(row, "label", 0.0))) for row in rows]
    sample_weights = [max(0.05, row_float(row, "outcome_weight", 1.0)) for row in rows]
    selected = [row_float(row, "selected", 0.0) >= 0.5 for row in rows]
    counterfactual = [row_float(row, "counterfactual_positive", 0.0) >= 0.5 for row in rows]
    return rows, feature_names, x_raw, y, sample_weights, selected, counterfactual


def split_by_game(rows, seed, validation_fraction=0.2):
    games = sorted({row.get("game_id", "") for row in rows})
    rng = random.Random(seed)
    rng.shuffle(games)
    valid_game_count = max(1, int(len(games) * validation_fraction)) if len(games) > 1 else 1
    valid_games = set(games[:valid_game_count])
    valid_indices = [i for i, row in enumerate(rows) if row.get("game_id", "") in valid_games]
    valid_set = set(valid_indices)
    train_indices = [i for i in range(len(rows)) if i not in valid_set] or valid_indices[:]
    return train_indices, valid_indices, games, valid_games


def normalize_from_train(x_raw, train_indices):
    train_raw = [x_raw[i] for i in train_indices]
    means = [sum(col) / len(col) for col in zip(*train_raw)]
    scales = []
    for j, mean in enumerate(means):
        var = sum((row[j] - mean) ** 2 for row in train_raw) / max(1, len(train_raw) - 1)
        scales.append(max(1e-6, math.sqrt(var)))

    def normalize(items):
        return [[(row[j] - means[j]) / scales[j] for j in range(len(means))] for row in items]

    return means, scales, normalize


def build_pair_indices(rows, labels, selected, counterfactual, indices, local_index, max_pairs_per_turn):
    grouped = defaultdict(list)
    for raw_index in indices:
        step = int(row_float(rows[raw_index], "step", 0.0))
        grouped[(rows[raw_index].get("game_id", ""), step)].append(raw_index)

    pairs = []
    for raw_indices in grouped.values():
        positives = [
            i
            for i in raw_indices
            if selected[i] or counterfactual[i] or labels[i] >= 0.55
        ]
        negatives = [
            i
            for i in raw_indices
            if labels[i] <= 0.30 and not counterfactual[i]
        ]
        if not positives or not negatives:
            continue
        positives.sort(key=lambda i: labels[i] + row_float(rows[i], "outcome_weight", 1.0) * 0.05, reverse=True)
        negatives.sort(key=lambda i: row_float(rows[i], "heuristic_score_scaled", 0.0), reverse=True)
        for pos in positives:
            for neg in negatives[:max_pairs_per_turn]:
                if labels[pos] <= labels[neg] + 0.12:
                    continue
                gap = max(0.05, labels[pos] - labels[neg])
                margin_weight = 1.0 + min(1.5, abs(row_float(rows[pos], "reward_margin", 0.0)) / 600.0)
                hard_pair_weight = (
                    1.35
                    if abs(
                        row_float(rows[pos], "heuristic_score_scaled", 0.0)
                        - row_float(rows[neg], "heuristic_score_scaled", 0.0)
                    )
                    <= 0.08
                    else 1.0
                )
                weight = gap * row_float(rows[pos], "outcome_weight", 1.0) * margin_weight * hard_pair_weight
                if counterfactual[pos]:
                    weight *= 1.45
                if row_float(rows[pos], "phase_opening", 0.0) >= 0.5:
                    weight *= 1.5
                pairs.append((local_index[pos], local_index[neg], max(0.05, weight)))
    return pairs


def sigmoid_prob(value):
    value = max(-50.0, min(50.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def grouped_top1_rate(rows, predictions, positive_mask, indices):
    grouped = defaultdict(list)
    for raw_index in indices:
        step = int(row_float(rows[raw_index], "step", 0.0))
        grouped[(rows[raw_index].get("game_id", ""), step)].append(raw_index)

    hits = 0
    total = 0
    rank_fractions = []
    for raw_indices in grouped.values():
        positives = [i for i in raw_indices if positive_mask[i]]
        if not positives:
            continue
        ordered = sorted(raw_indices, key=lambda i: predictions[i], reverse=True)
        total += 1
        if positive_mask[ordered[0]]:
            hits += 1
        best_positive_rank = min(ordered.index(i) + 1 for i in positives)
        rank_fractions.append(best_positive_rank / max(1, len(ordered)))
    mean_rank = sum(rank_fractions) / len(rank_fractions) if rank_fractions else 1.0
    return hits / total if total else 0.0, mean_rank, total


def grouped_top1_rate_subset(rows, predictions, positive_mask, indices):
    prediction_by_raw = {raw_index: predictions[local] for local, raw_index in enumerate(indices)}
    grouped = defaultdict(list)
    for raw_index in indices:
        step = int(row_float(rows[raw_index], "step", 0.0))
        grouped[(rows[raw_index].get("game_id", ""), step)].append(raw_index)

    hits = 0
    total = 0
    rank_fractions = []
    for raw_indices in grouped.values():
        positives = [i for i in raw_indices if positive_mask[i]]
        if not positives:
            continue
        ordered = sorted(raw_indices, key=lambda i: prediction_by_raw.get(i, 0.0), reverse=True)
        total += 1
        if positive_mask[ordered[0]]:
            hits += 1
        best_positive_rank = min(ordered.index(i) + 1 for i in positives)
        rank_fractions.append(best_positive_rank / max(1, len(ordered)))
    mean_rank = sum(rank_fractions) / len(rank_fractions) if rank_fractions else 1.0
    return hits / total if total else 0.0, mean_rank, total


def tune_blend(rows, probabilities, positive_mask, indices, score_scale):
    best = {"blend": 0.0, "top1": -1.0, "rank": 1.0}
    heuristic_scores = [row_float(row, "heuristic_score_scaled", 0.0) * 100.0 for row in rows]
    model_scores = [(probability - 0.5) * score_scale for probability in probabilities]
    for step in range(0, 61):
        blend = step / 100.0
        combined = [
            heuristic_scores[i] * (1.0 - blend) + (heuristic_scores[i] + model_scores[i]) * blend
            for i in range(len(rows))
        ]
        top1, rank, _ = grouped_top1_rate(rows, combined, positive_mask, indices)
        if top1 > best["top1"] or (abs(top1 - best["top1"]) < 1e-9 and rank < best["rank"]):
            best = {"blend": blend, "top1": top1, "rank": rank}
    return best


def save_member_checkpoint(model, best_state, nn, args, member_seed, epoch, best_valid_objective, checkpoint_context):
    saved_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    try:
        linear_layers = [module for module in model.net if isinstance(module, nn.Linear)]
        layers = []
        for index, layer in enumerate(linear_layers):
            layers.append(
                {
                    "weights": layer.weight.detach().cpu().tolist(),
                    "bias": layer.bias.detach().cpu().tolist(),
                    "activation": "relu" if index < len(linear_layers) - 1 else "linear",
                }
            )
    finally:
        model.load_state_dict(saved_state)
    payload = {
        "version": "v17",
        "model_type": "mlp_relu_candidate_ranker",
        "features": checkpoint_context["feature_names"],
        "mean": dict(zip(checkpoint_context["feature_names"], checkpoint_context["means"])),
        "scale": dict(zip(checkpoint_context["feature_names"], checkpoint_context["scales"])),
        "layers": layers,
        "activation": "relu",
        "score_scale": args.score_scale,
        "member_seed": member_seed,
        "epoch": epoch,
        "best_validation_objective": best_valid_objective,
    }
    checkpoint_path = checkpoint_context["checkpoint_dir"] / f"member_seed{member_seed}_epoch{epoch:03d}.json"
    checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    api = checkpoint_context.get("api")
    if api is None:
        print(f"Saved local v17 checkpoint {checkpoint_path}", flush=True)
        return checkpoint_path
    remote_path = f"{HF_REMOTE_PREFIX}/checkpoints/{checkpoint_path.name}"
    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=remote_path,
            repo_id=checkpoint_context["hf_repo_id"],
            repo_type=checkpoint_context["hf_repo_type"],
            commit_message=f"Upload v17 checkpoint member_seed={member_seed} epoch={epoch}",
        )
        print(f"Uploaded v17 checkpoint to {remote_path}", flush=True)
    except Exception as exc:
        print(f"Failed to upload v17 checkpoint {checkpoint_path.name}: {exc}", flush=True)
    return checkpoint_path


def train_member(torch, nn, functional, args, member_seed, tensors, pairs, feature_count, eval_context, checkpoint_context):
    torch.manual_seed(member_seed)
    x_train, y_train, w_train, x_valid, y_valid, w_valid = tensors
    device = x_train.device

    class CandidateRanker(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(max(0.05, args.dropout * 0.70)),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(max(0.03, args.dropout * 0.45)),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = CandidateRanker(feature_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    train_pairs, valid_pairs = pairs
    rows, valid_indices, positive_mask = eval_context
    history = []
    best_state = None
    best_valid_objective = float("inf")
    stale_epochs = 0
    started = time.time()

    def evaluate(x_tensor, y_tensor, w_tensor, pair_list):
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.sigmoid(logits)
            weighted_bce = float((bce_loss(logits, y_tensor) * w_tensor).mean().detach().cpu())
            mse = float(((probs - y_tensor) ** 2).mean().detach().cpu())
            accuracy = float(((probs >= 0.5) == (y_tensor >= 0.5)).float().mean().detach().cpu())
            pair_loss = 0.0
            pair_accuracy = 0.0
            if pair_list:
                pos_idx = torch.tensor([p[0] for p in pair_list], dtype=torch.long, device=device)
                neg_idx = torch.tensor([p[1] for p in pair_list], dtype=torch.long, device=device)
                pair_w = torch.tensor([p[2] for p in pair_list], dtype=torch.float32, device=device)
                margins = logits[pos_idx] - logits[neg_idx]
                pair_loss = float((functional.softplus(-margins.view(-1)) * pair_w).mean().detach().cpu())
                pair_accuracy = float((margins > 0).float().mean().detach().cpu())
        return weighted_bce, mse, accuracy, pair_loss, pair_accuracy

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], args.batch_size):
            batch_idx = permutation[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[batch_idx])
            loss = (bce_loss(logits, y_train[batch_idx]) * w_train[batch_idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if train_pairs and args.pair_loss_weight > 0.0:
            rng = random.Random(member_seed + epoch)
            shuffled_pairs = train_pairs[:]
            rng.shuffle(shuffled_pairs)
            for start in range(0, len(shuffled_pairs), args.batch_size):
                batch = shuffled_pairs[start:start + args.batch_size]
                pos_idx = torch.tensor([p[0] for p in batch], dtype=torch.long, device=device)
                neg_idx = torch.tensor([p[1] for p in batch], dtype=torch.long, device=device)
                pair_w = torch.tensor([p[2] for p in batch], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                margins = model(x_train[pos_idx]) - model(x_train[neg_idx])
                pair_loss = (functional.softplus(-margins.view(-1)) * pair_w).mean()
                loss = pair_loss * args.pair_loss_weight
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        train_bce, train_mse, train_acc, train_pair_loss, train_pair_acc = evaluate(x_train, y_train, w_train, train_pairs)
        valid_bce, valid_mse, valid_acc, valid_pair_loss, valid_pair_acc = evaluate(x_valid, y_valid, w_valid, valid_pairs)
        model.eval()
        with torch.no_grad():
            valid_probs = torch.sigmoid(model(x_valid)).view(-1).detach().cpu().tolist()
        valid_top1, valid_rank, _ = grouped_top1_rate_subset(rows, valid_probs, positive_mask, valid_indices)
        valid_objective = (
            valid_bce
            + args.pair_loss_weight * valid_pair_loss
            + (1.0 - valid_top1) * 1.15
            + valid_rank * 0.30
        )
        item = {
            "epoch": epoch,
            "train_logloss": train_bce,
            "valid_logloss": valid_bce,
            "train_pair_loss": train_pair_loss,
            "valid_pair_loss": valid_pair_loss,
            "train_pair_accuracy": train_pair_acc,
            "valid_pair_accuracy": valid_pair_acc,
            "valid_turn_top1_positive_rate": valid_top1,
            "valid_positive_mean_rank_fraction": valid_rank,
            "train_accuracy": train_acc,
            "valid_accuracy": valid_acc,
            "train_mse": train_mse,
            "valid_mse": valid_mse,
            "valid_objective": valid_objective,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        history.append(item)
        print(
            f"member_seed={member_seed} epoch={epoch:03d}/{args.epochs} "
            f"valid_bce={valid_bce:.5f} valid_pair_acc={valid_pair_acc:.4f} "
            f"valid_top1={valid_top1:.4f} valid_obj={valid_objective:.5f} "
            f"elapsed={item['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if valid_objective + 1e-5 < best_valid_objective:
            best_valid_objective = valid_objective
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"member_seed={member_seed} early_stop epoch={epoch} best_valid_objective={best_valid_objective:.5f}",
                    flush=True,
                )
                break

        if (
            checkpoint_context is not None
            and args.checkpoint_every > 0
            and epoch % args.checkpoint_every == 0
            and best_state is not None
        ):
            save_member_checkpoint(
                model, best_state, nn, args, member_seed, epoch, best_valid_objective, checkpoint_context
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    linear_layers = [module for module in model.net if isinstance(module, nn.Linear)]
    layers = []
    for index, layer in enumerate(linear_layers):
        layers.append(
            {
                "weights": layer.weight.detach().cpu().tolist(),
                "bias": layer.bias.detach().cpu().tolist(),
                "activation": "relu" if index < len(linear_layers) - 1 else "linear",
            }
        )
    return model, layers, history, best_valid_objective


def train(args):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for v17 training. Install torch or run on Kaggle/Colab.") from exc

    random.seed(args.seed)
    data_path = find_training_csv(args.csv, getattr(args, "prefer_local_data", False))
    rows, feature_names, x_raw, y, sample_weights, selected, counterfactual = read_rows(data_path)
    train_indices, valid_indices, games, valid_games = split_by_game(rows, args.seed)
    means, scales, normalize = normalize_from_train(x_raw, train_indices)
    all_x = normalize(x_raw)
    train_x = [all_x[i] for i in train_indices]
    valid_x = [all_x[i] for i in valid_indices]
    train_y = [y[i] for i in train_indices]
    valid_y = [y[i] for i in valid_indices]
    train_w = [sample_weights[i] for i in train_indices]
    valid_w = [sample_weights[i] for i in valid_indices]

    train_local = {raw_index: local for local, raw_index in enumerate(train_indices)}
    valid_local = {raw_index: local for local, raw_index in enumerate(valid_indices)}
    train_pairs = build_pair_indices(rows, y, selected, counterfactual, train_indices, train_local, args.max_pairs_per_turn)
    valid_pairs = build_pair_indices(rows, y, selected, counterfactual, valid_indices, valid_local, args.max_pairs_per_turn)
    positive_mask = [selected[i] or counterfactual[i] or y[i] >= 0.55 for i in range(len(rows))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_y, dtype=torch.float32, device=device).view(-1, 1)
    w_train = torch.tensor(train_w, dtype=torch.float32, device=device).view(-1, 1)
    x_valid = torch.tensor(valid_x, dtype=torch.float32, device=device)
    y_valid = torch.tensor(valid_y, dtype=torch.float32, device=device).view(-1, 1)
    w_valid = torch.tensor(valid_w, dtype=torch.float32, device=device).view(-1, 1)
    tensors = (x_train, y_train, w_train, x_valid, y_valid, w_valid)
    pairs = (train_pairs, valid_pairs)
    eval_context = (rows, valid_indices, positive_mask)

    print(
        json.dumps(
            {
                "csv": str(data_path),
                "rows": len(rows),
                "features": len(feature_names),
                "games": len(games),
                "validation_games": len(valid_games),
                "train_rows": len(train_x),
                "validation_rows": len(valid_x),
                "train_pairs": len(train_pairs),
                "validation_pairs": len(valid_pairs),
                "positive_rate": sum(1 for value in y if value >= 0.5) / len(y),
                "selected_rate": sum(1 for value in selected if value) / len(selected),
                "counterfactual_rate": sum(1 for value in counterfactual if value) / len(counterfactual),
                "ensemble_size": args.ensemble_size,
                "device": str(device),
            },
            indent=2,
        ),
        flush=True,
    )

    export_dir = Path(args.export_dir)
    graph_dir = export_dir / "graphs"
    checkpoint_dir = export_dir / "checkpoints"
    export_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hf_api = None
    if args.upload:
        load_dotenv()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --upload.")
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install huggingface_hub to upload: pip install huggingface_hub") from exc
        hf_api = HfApi(token=token)
        hf_api.create_repo(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type, exist_ok=True)

    checkpoint_context = {
        "feature_names": feature_names,
        "means": means,
        "scales": scales,
        "checkpoint_dir": checkpoint_dir,
        "api": hf_api,
        "hf_repo_id": args.hf_repo_id,
        "hf_repo_type": args.hf_repo_type,
    }

    members = []
    histories = []
    models = []
    for member_index in range(args.ensemble_size):
        member_seed = args.seed + member_index * 9973
        model, layers, history, best_objective = train_member(
            torch,
            nn,
            functional,
            args,
            member_seed,
            tensors,
            pairs,
            len(feature_names),
            eval_context,
            checkpoint_context,
        )
        models.append(model)
        histories.append({"seed": member_seed, "history": history, "best_validation_objective": best_objective})
        members.append(
            {
                "version": "v17",
                "model_type": "mlp_relu_candidate_ranker",
                "features": feature_names,
                "mean": dict(zip(feature_names, means)),
                "scale": dict(zip(feature_names, scales)),
                "layers": layers,
                "activation": "relu",
                "score_scale": args.score_scale,
            }
        )

    for model in models:
        model.eval()
    with torch.no_grad():
        all_tensor = torch.tensor(all_x, dtype=torch.float32, device=device)
        member_probs = []
        for model in models:
            logits = model(all_tensor).view(-1).detach().cpu().tolist()
            member_probs.append([sigmoid_prob(value) for value in logits])
    all_probs = [sum(member[i] for member in member_probs) / len(member_probs) for i in range(len(rows))]
    train_predictions = [all_probs[i] for i in train_indices]
    valid_predictions = [all_probs[i] for i in valid_indices]
    valid_top1, valid_rank, valid_turns = grouped_top1_rate(rows, all_probs, positive_mask, valid_indices)
    train_top1, train_rank, train_turns = grouped_top1_rate(rows, all_probs, positive_mask, train_indices)
    tuned_blend = tune_blend(rows, all_probs, positive_mask, valid_indices, args.score_scale)

    def log_loss(preds, labels):
        return -sum(
            label * math.log(max(1e-9, pred)) + (1.0 - label) * math.log(max(1e-9, 1.0 - pred))
            for pred, label in zip(preds, labels)
        ) / len(labels)

    def accuracy(preds, labels):
        return sum((pred >= 0.5) == (label >= 0.5) for pred, label in zip(preds, labels)) / len(labels)

    metrics = {
        "rows": len(rows),
        "features": len(feature_names),
        "games": len(games),
        "train_rows": len(train_x),
        "validation_rows": len(valid_x),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(valid_pairs),
        "positive_rate": sum(1 for value in y if value >= 0.5) / len(y),
        "selected_rate": sum(1 for value in selected if value) / len(selected),
        "counterfactual_rate": sum(1 for value in counterfactual if value) / len(counterfactual),
        "train_accuracy": accuracy(train_predictions, train_y),
        "validation_accuracy": accuracy(valid_predictions, valid_y),
        "train_logloss": log_loss(train_predictions, train_y),
        "validation_logloss": log_loss(valid_predictions, valid_y),
        "train_turn_top1_positive_rate": train_top1,
        "validation_turn_top1_positive_rate": valid_top1,
        "train_positive_mean_rank_fraction": train_rank,
        "validation_positive_mean_rank_fraction": valid_rank,
        "train_turns_ranked": train_turns,
        "validation_turns_ranked": valid_turns,
        "tuned_blend": tuned_blend["blend"],
        "tuned_blend_validation_top1": tuned_blend["top1"],
        "tuned_blend_validation_rank_fraction": tuned_blend["rank"],
        "ensemble_size": len(members),
        "device": str(device),
        "pair_loss_weight": args.pair_loss_weight,
        "dropout": args.dropout,
        "score_scale": args.score_scale,
    }

    artifact = {
        "version": "v17",
        "created_at": int(time.time()),
        "source_csv": str(data_path),
        "model_type": "ensemble_mlp_relu_candidate_ranker",
        "members": members,
        "blend": tuned_blend["blend"],
        "metrics": metrics,
    }

    (export_dir / "model_weights_v17.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "feature_schema_v17.json").write_text(
        json.dumps({"features": feature_names}, indent=2), encoding="utf-8"
    )
    (export_dir / "metrics_v17.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "training_history_v17.json").write_text(
        json.dumps(histories, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (export_dir / "predictions_v17.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "label", "prediction", "selected", "counterfactual_positive", "game_result", "split"])
        valid_set = set(valid_indices)
        for i, pred in enumerate(all_probs):
            writer.writerow(
                [
                    i,
                    y[i],
                    pred,
                    float(selected[i]),
                    float(counterfactual[i]),
                    rows[i].get("game_result", 0.0),
                    "validation" if i in valid_set else "train",
                ]
            )

    try:
        import matplotlib.pyplot as plt

        first_history = histories[0]["history"] if histories else []
        epochs = [item["epoch"] for item in first_history]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [item["valid_logloss"] for item in first_history], label="validation BCE")
        plt.plot(epochs, [item["valid_objective"] for item in first_history], label="validation objective")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("v17 first member validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "loss_curve_v17.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.hist([pred for pred, label in zip(all_probs, y) if label >= 0.5], bins=30, alpha=0.65, label="positive")
        plt.hist([pred for pred, label in zip(all_probs, y) if label < 0.5], bins=30, alpha=0.65, label="negative")
        plt.xlabel("prediction")
        plt.ylabel("rows")
        plt.title("v17 ensemble prediction distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "prediction_histogram_v17.png", dpi=150)
        plt.close()
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped graph generation.", flush=True)

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"Saved v17 model artifact: {export_dir / 'model_weights_v17.json'}", flush=True)

    if args.upload and hf_api is not None:
        hf_api.upload_folder(
            folder_path=str(export_dir),
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
            path_in_repo=HF_REMOTE_PREFIX,
            commit_message="Upload v17 Orbit Wars supervised ensemble ranker artifacts and graphs",
        )
        print(f"Uploaded {export_dir} to https://huggingface.co/{args.hf_repo_id}/tree/main/{HF_REMOTE_PREFIX}", flush=True)


def main():
    load_dotenv()
    train(parse_args())


if __name__ == "__main__":
    main()
