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
HF_REMOTE_PREFIX = "v6"
METADATA_COLS = {
    "label",
    "selected",
    "outcome_weight",
    "game_result",
    "reward_margin",
    "agent_reward",
    "opponent_reward",
    "game_id",
    "candidate_id",
    "version",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the v6 outcome-weighted Orbit Wars candidate ranker.")
    parser.add_argument("--csv", default=os.environ.get("CANDIDATES_CSV", ""))
    parser.add_argument("--export-dir", default="notebooks/v6/exports")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("V6_EPOCHS", "140")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("V6_BATCH_SIZE", "1024")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("V6_LR", "0.0008")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("V6_WEIGHT_DECAY", "0.00015")))
    parser.add_argument("--pair-loss-weight", type=float, default=float(os.environ.get("V6_PAIR_LOSS_WEIGHT", "0.55")))
    parser.add_argument("--max-pairs-per-turn", type=int, default=5)
    parser.add_argument("--patience", type=int, default=int(os.environ.get("V6_PATIENCE", "20")))
    parser.add_argument("--seed", type=int, default=42)
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


def find_training_csv(csv_arg):
    if csv_arg:
        path = Path(csv_arg)
        if not path.exists():
            raise FileNotFoundError(f"Training CSV does not exist: {path}")
        return path

    fixed = Path("notebooks/v6/data/candidates_v6.csv")
    if fixed.exists():
        return fixed

    root = Path("data")
    candidates = sorted(root.glob("*/candidates_v6.csv"), key=lambda path: path.stat().st_mtime, reverse=True) if root.exists() else []
    if candidates:
        return candidates[0]

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise FileNotFoundError("No candidates_v6.csv found locally and HF_TOKEN is not set for download.")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to download data: pip install huggingface_hub") from exc

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
    remote_csvs = sorted(
        [name for name in files if name.startswith("data/") and name.endswith("/candidates_v6.csv")],
        reverse=True,
    )
    if not remote_csvs:
        raise FileNotFoundError("No candidates_v6.csv found locally or in Hugging Face data folders.")
    return Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, filename=remote_csvs[0], token=token))


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Training CSV has no rows.")
    feature_names = [name for name in rows[0] if name not in METADATA_COLS]
    x_raw = [[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in rows]
    y = [max(0.0, min(1.0, float(row.get("label", 0.0) or 0.0))) for row in rows]
    sample_weights = [max(0.05, float(row.get("outcome_weight", 1.0) or 1.0)) for row in rows]
    selected = [float(row.get("selected", 0.0) or 0.0) >= 0.5 for row in rows]
    return rows, feature_names, x_raw, y, sample_weights, selected


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


def build_pair_indices(rows, labels, selected, indices, local_index, max_pairs_per_turn):
    grouped = defaultdict(list)
    for raw_index in indices:
        step = int(float(rows[raw_index].get("step", 0.0) or 0.0))
        grouped[(rows[raw_index].get("game_id", ""), step)].append(raw_index)

    pairs = []
    for raw_indices in grouped.values():
        positives = [i for i in raw_indices if selected[i] or labels[i] >= 0.55]
        negatives = [i for i in raw_indices if labels[i] <= 0.25]
        if not positives or not negatives:
            continue
        negatives.sort(
            key=lambda i: float(rows[i].get("heuristic_score_scaled", 0.0) or 0.0),
            reverse=True,
        )
        for pos in positives[:2]:
            for neg in negatives[:max_pairs_per_turn]:
                if labels[pos] <= labels[neg] + 0.15:
                    continue
                weight = max(
                    0.05,
                    abs(labels[pos] - labels[neg])
                    * float(rows[pos].get("outcome_weight", 1.0) or 1.0),
                )
                pairs.append((local_index[pos], local_index[neg], weight))
    return pairs


def sigmoid_prob(value):
    value = max(-50.0, min(50.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def grouped_top1_rate(rows, predictions, selected, indices):
    grouped = defaultdict(list)
    for raw_index in indices:
        step = int(float(rows[raw_index].get("step", 0.0) or 0.0))
        grouped[(rows[raw_index].get("game_id", ""), step)].append(raw_index)

    hits = 0
    total = 0
    selected_ranks = []
    for raw_indices in grouped.values():
        positives = [i for i in raw_indices if selected[i]]
        if not positives:
            continue
        ordered = sorted(raw_indices, key=lambda i: predictions[i], reverse=True)
        total += 1
        if selected[ordered[0]]:
            hits += 1
        best_positive_rank = min(ordered.index(i) + 1 for i in positives)
        selected_ranks.append(best_positive_rank / max(1, len(ordered)))
    mean_rank = sum(selected_ranks) / len(selected_ranks) if selected_ranks else 1.0
    return hits / total if total else 0.0, mean_rank, total


def train(args):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for the v6 pairwise MLP trainer. Install it in this "
            "kernel/venv with: python -m pip install torch"
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = find_training_csv(args.csv)
    rows, feature_names, x_raw, y, sample_weights, selected = read_rows(data_path)
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
    train_pairs = build_pair_indices(rows, y, selected, train_indices, train_local, args.max_pairs_per_turn)
    valid_pairs = build_pair_indices(rows, y, selected, valid_indices, valid_local, args.max_pairs_per_turn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.tensor(train_y, dtype=torch.float32, device=device).view(-1, 1)
    w_train = torch.tensor(train_w, dtype=torch.float32, device=device).view(-1, 1)
    x_valid = torch.tensor(valid_x, dtype=torch.float32, device=device)
    y_valid = torch.tensor(valid_y, dtype=torch.float32, device=device).view(-1, 1)
    w_valid = torch.tensor(valid_w, dtype=torch.float32, device=device).view(-1, 1)

    class CandidateRanker(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = CandidateRanker(len(feature_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    history = []
    best_state = None
    best_valid_objective = float("inf")
    stale_epochs = 0
    started = time.time()

    def evaluate(x_tensor, y_tensor, w_tensor, pairs):
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.sigmoid(logits)
            weighted_bce = float((bce_loss(logits, y_tensor) * w_tensor).mean().detach().cpu())
            mse = float(((probs - y_tensor) ** 2).mean().detach().cpu())
            accuracy = float(((probs >= 0.5) == (y_tensor >= 0.5)).float().mean().detach().cpu())
            pair_accuracy = 0.0
            pair_loss = 0.0
            if pairs:
                pos_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                neg_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
                pair_w = torch.tensor([p[2] for p in pairs], dtype=torch.float32, device=device)
                margins = logits[pos_idx] - logits[neg_idx]
                pair_loss = float((functional.softplus(-margins.view(-1)) * pair_w).mean().detach().cpu())
                pair_accuracy = float((margins > 0).float().mean().detach().cpu())
        return weighted_bce, mse, accuracy, pair_loss, pair_accuracy

    print(json.dumps({
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
        "device": str(device),
    }, indent=2), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(x_train.shape[0], device=device)
        epoch_bce = 0.0
        bce_batches = 0
        for start in range(0, x_train.shape[0], args.batch_size):
            batch_idx = permutation[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[batch_idx])
            loss = (bce_loss(logits, y_train[batch_idx]) * w_train[batch_idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_bce += float(loss.detach().cpu())
            bce_batches += 1

        epoch_pair_loss = 0.0
        pair_batches = 0
        if train_pairs and args.pair_loss_weight > 0.0:
            rng = random.Random(args.seed + epoch)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_pair_loss += float(pair_loss.detach().cpu())
                pair_batches += 1

        train_bce, train_mse, train_acc, train_pair_loss, train_pair_acc = evaluate(x_train, y_train, w_train, train_pairs)
        valid_bce, valid_mse, valid_acc, valid_pair_loss, valid_pair_acc = evaluate(x_valid, y_valid, w_valid, valid_pairs)
        valid_objective = valid_bce + args.pair_loss_weight * valid_pair_loss
        item = {
            "epoch": epoch,
            "batch_bce": epoch_bce / max(1, bce_batches),
            "batch_pair_loss": epoch_pair_loss / max(1, pair_batches),
            "train_logloss": train_bce,
            "valid_logloss": valid_bce,
            "train_pair_loss": train_pair_loss,
            "valid_pair_loss": valid_pair_loss,
            "train_pair_accuracy": train_pair_acc,
            "valid_pair_accuracy": valid_pair_acc,
            "train_mse": train_mse,
            "valid_mse": valid_mse,
            "train_accuracy": train_acc,
            "valid_accuracy": valid_acc,
            "valid_objective": valid_objective,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        history.append(item)
        print(
            f"epoch={epoch:03d}/{args.epochs} "
            f"train_bce={train_bce:.5f} valid_bce={valid_bce:.5f} "
            f"valid_pair_acc={valid_pair_acc:.4f} valid_obj={valid_objective:.5f} "
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
                print(f"early_stop epoch={epoch} best_valid_objective={best_valid_objective:.5f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        all_tensor = torch.tensor(all_x, dtype=torch.float32, device=device)
        all_logits = model(all_tensor).view(-1).detach().cpu().tolist()
    all_probs = [sigmoid_prob(value) for value in all_logits]
    train_predictions = [all_probs[i] for i in train_indices]
    valid_predictions = [all_probs[i] for i in valid_indices]
    valid_top1, valid_selected_rank, valid_turns = grouped_top1_rate(rows, all_probs, selected, valid_indices)
    train_top1, train_selected_rank, train_turns = grouped_top1_rate(rows, all_probs, selected, train_indices)

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
        "train_accuracy": accuracy(train_predictions, train_y),
        "validation_accuracy": accuracy(valid_predictions, valid_y),
        "train_logloss": log_loss(train_predictions, train_y),
        "validation_logloss": log_loss(valid_predictions, valid_y),
        "train_turn_top1_selected_rate": train_top1,
        "validation_turn_top1_selected_rate": valid_top1,
        "train_selected_mean_rank_fraction": train_selected_rank,
        "validation_selected_mean_rank_fraction": valid_selected_rank,
        "train_turns_ranked": train_turns,
        "validation_turns_ranked": valid_turns,
        "best_validation_objective": best_valid_objective,
        "epochs_run": len(history),
        "device": str(device),
        "pair_loss_weight": args.pair_loss_weight,
    }

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

    artifact = {
        "version": "v6",
        "created_at": int(time.time()),
        "source_csv": str(data_path),
        "model_type": "mlp_relu_candidate_ranker",
        "features": feature_names,
        "mean": dict(zip(feature_names, means)),
        "scale": dict(zip(feature_names, scales)),
        "layers": layers,
        "activation": "relu",
        "score_scale": 180.0,
        "blend": 0.30,
        "metrics": metrics,
    }

    export_dir = Path(args.export_dir)
    graph_dir = export_dir / "graphs"
    export_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "model_weights_v6.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "feature_schema_v6.json").write_text(json.dumps({"features": feature_names}, indent=2), encoding="utf-8")
    (export_dir / "metrics_v6.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "training_history_v6.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    with (export_dir / "predictions_v6.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "label", "prediction", "selected", "game_result", "split"])
        valid_set = set(valid_indices)
        for i, pred in enumerate(all_probs):
            writer.writerow([i, y[i], pred, float(selected[i]), rows[i].get("game_result", 0.0), "validation" if i in valid_set else "train"])

    try:
        import matplotlib.pyplot as plt

        epochs = [item["epoch"] for item in history]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [item["train_logloss"] for item in history], label="train BCE")
        plt.plot(epochs, [item["valid_logloss"] for item in history], label="validation BCE")
        plt.plot(epochs, [item["valid_objective"] for item in history], label="validation objective")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("v6 ranker loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "loss_curve_v6.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [item["valid_pair_accuracy"] for item in history], label="validation pair accuracy")
        plt.xlabel("epoch")
        plt.ylabel("pair accuracy")
        plt.title("v6 pairwise ranking accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "pair_accuracy_v6.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.hist([pred for pred, label in zip(all_probs, y) if label >= 0.5], bins=30, alpha=0.65, label="positive")
        plt.hist([pred for pred, label in zip(all_probs, y) if label < 0.5], bins=30, alpha=0.65, label="negative")
        plt.xlabel("prediction")
        plt.ylabel("rows")
        plt.title("v6 prediction distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "prediction_histogram_v6.png", dpi=150)
        plt.close()

        first_layer = linear_layers[0].weight.detach().cpu().abs().mean(dim=0).tolist()
        top = sorted(zip(feature_names, first_layer), key=lambda item: item[1], reverse=True)[:24]
        plt.figure(figsize=(8, 7))
        plt.barh([name for name, _ in reversed(top)], [value for _, value in reversed(top)])
        plt.xlabel("mean absolute first-layer weight")
        plt.title("v6 strongest input features")
        plt.tight_layout()
        plt.savefig(graph_dir / "feature_weights_v6.png", dpi=150)
        plt.close()
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped graph generation.", flush=True)

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"Saved v6 model artifact: {export_dir / 'model_weights_v6.json'}", flush=True)

    if args.upload:
        load_dotenv()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for --upload.")
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install huggingface_hub to upload: pip install huggingface_hub") from exc
        api = HfApi(token=token)
        api.create_repo(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type, exist_ok=True)
        api.upload_folder(
            folder_path=str(export_dir),
            repo_id=args.hf_repo_id,
            repo_type=args.hf_repo_type,
            path_in_repo=HF_REMOTE_PREFIX,
            commit_message="Upload v6 Orbit Wars pairwise ranker artifacts and graphs",
        )
        print(f"Uploaded {export_dir} to https://huggingface.co/{args.hf_repo_id}/tree/main/{HF_REMOTE_PREFIX}", flush=True)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
