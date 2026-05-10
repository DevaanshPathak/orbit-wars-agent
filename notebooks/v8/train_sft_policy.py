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
HF_REMOTE_PREFIX = "v8/sft"

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
    parser = argparse.ArgumentParser(
        description="Train the v8 SFT candidate policy from Orbit Wars candidate groups."
    )
    parser.add_argument("--csv", default=os.environ.get("CANDIDATES_CSV", ""))
    parser.add_argument(
        "--data-remote-path",
        default=os.environ.get("V8_SFT_DATA_REMOTE_PATH", ""),
        help="Optional exact Hugging Face repo path for candidates_v7.csv. If omitted, the newest data/*/candidates_v7.csv is used.",
    )
    parser.add_argument(
        "--prefer-local-data",
        action="store_true",
        help="Use a local candidates_v7.csv/candidates_v8.csv before trying Hugging Face. Default is Hugging Face.",
    )
    parser.add_argument("--export-dir", default="notebooks/v8/exports/sft")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("V8_SFT_EPOCHS", "180")))
    parser.add_argument("--batch-groups", type=int, default=int(os.environ.get("V8_SFT_BATCH_GROUPS", "192")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("V8_SFT_LR", "0.00065")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("V8_SFT_WEIGHT_DECAY", "0.00025")))
    parser.add_argument("--dropout", type=float, default=float(os.environ.get("V8_SFT_DROPOUT", "0.14")))
    parser.add_argument("--bce-weight", type=float, default=float(os.environ.get("V8_SFT_BCE_WEIGHT", "0.28")))
    parser.add_argument("--pair-weight", type=float, default=float(os.environ.get("V8_SFT_PAIR_WEIGHT", "0.25")))
    parser.add_argument("--rank-weight", type=float, default=float(os.environ.get("V8_SFT_RANK_WEIGHT", "0.42")))
    parser.add_argument("--ensemble-size", type=int, default=int(os.environ.get("V8_SFT_ENSEMBLE_SIZE", "3")))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("V8_SFT_PATIENCE", "28")))
    parser.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("V8_SFT_CHECKPOINT_EVERY", "30")))
    parser.add_argument(
        "--device",
        choices=("tpu", "cuda", "cpu", "auto"),
        default=os.environ.get("V8_DEVICE", "tpu"),
        help="Training device. Defaults to TPU for Kaggle TPU v5e-8 runs.",
    )
    parser.add_argument("--seed", type=int, default=811)
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


def download_training_csv(remote_path, repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE):
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to download candidates_v7.csv from Hugging Face.")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to download data: pip install huggingface_hub") from exc

    if not remote_path:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        remote_csvs = sorted(
            [
                name
                for name in files
                if name.startswith("data/") and name.endswith("/candidates_v7.csv")
            ],
            reverse=True,
        )
        if not remote_csvs:
            raise FileNotFoundError("No data/*/candidates_v7.csv found in Hugging Face repo.")
        remote_path = remote_csvs[0]

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=remote_path,
            token=token,
        )
    )


def find_training_csv(csv_arg, remote_path="", prefer_local=False):
    if csv_arg:
        path = Path(csv_arg)
        if not path.exists():
            raise FileNotFoundError(f"Training CSV does not exist: {path}")
        return path

    if not prefer_local:
        return download_training_csv(remote_path)

    local_candidates = []
    for pattern in (
        "notebooks/v8/data/candidates_v8.csv",
        "notebooks/v8/data/candidates_v7.csv",
    ):
        path = Path(pattern)
        if path.exists():
            return path

    root = Path("data")
    if root.exists():
        local_candidates.extend(root.glob("*/candidates_v8.csv"))
        local_candidates.extend(root.glob("*/candidates_v7.csv"))
    if local_candidates:
        return sorted(local_candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]

    return download_training_csv(remote_path)


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
    labels = [max(0.0, min(1.0, row_float(row, "label", 0.0))) for row in rows]
    selected = [row_float(row, "selected", 0.0) >= 0.5 for row in rows]
    counterfactual = [row_float(row, "counterfactual_positive", 0.0) >= 0.5 for row in rows]
    sample_weights = [max(0.05, row_float(row, "outcome_weight", 1.0)) for row in rows]
    return rows, feature_names, x_raw, labels, selected, counterfactual, sample_weights


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


def build_groups(rows, indices):
    grouped = defaultdict(list)
    for raw_index in indices:
        step = int(row_float(rows[raw_index], "step", 0.0))
        grouped[(rows[raw_index].get("game_id", ""), step)].append(raw_index)
    return [items for items in grouped.values() if len(items) >= 2]


def target_distribution(rows, labels, selected, counterfactual, group):
    weights = []
    for raw_index in group:
        label = labels[raw_index]
        score = row_float(rows[raw_index], "heuristic_score_scaled", 0.0)
        turn_delta = (
            row_float(rows[raw_index], "future_advantage_delta_15", 0.0) * 0.010
            + row_float(rows[raw_index], "future_production_delta_15", 0.0) * 0.040
            + row_float(rows[raw_index], "future_planet_delta_15", 0.0) * 0.120
        )
        value = max(0.01, label + 0.06 * score + turn_delta)
        if selected[raw_index]:
            value += 0.35
        if counterfactual[raw_index]:
            value += 0.25
        if row_float(rows[raw_index], "failure_overcommit", 0.0) >= 0.5:
            value *= 0.45
        weights.append(max(0.01, value))
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / len(group)] * len(group)
    return [weight / total for weight in weights]


def build_hard_pairs(rows, labels, selected, counterfactual, groups, local_index, max_pairs_per_group=8):
    pairs = []
    for group in groups:
        positives = [
            i
            for i in group
            if selected[i] or counterfactual[i] or labels[i] >= 0.55
        ]
        negatives = [
            i
            for i in group
            if labels[i] <= 0.35 and not counterfactual[i]
        ]
        if not positives or not negatives:
            continue
        positives.sort(
            key=lambda i: labels[i] + row_float(rows[i], "future_advantage_delta_15", 0.0) * 0.004,
            reverse=True,
        )
        negatives.sort(key=lambda i: row_float(rows[i], "heuristic_score_scaled", 0.0), reverse=True)
        emitted = 0
        for pos in positives:
            for neg in negatives:
                if emitted >= max_pairs_per_group:
                    break
                if labels[pos] <= labels[neg] + 0.10:
                    continue
                gap = max(0.05, labels[pos] - labels[neg])
                margin_weight = 1.0 + min(1.5, abs(row_float(rows[pos], "reward_margin", 0.0)) / 650.0)
                pairs.append((local_index[pos], local_index[neg], gap * margin_weight))
                emitted += 1
    return pairs


def sigmoid_prob(value):
    value = max(-50.0, min(50.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def grouped_metrics(rows, predictions, positive_mask, indices):
    groups = build_groups(rows, indices)
    hits = 0
    total = 0
    rank_fractions = []
    for group in groups:
        positives = [i for i in group if positive_mask[i]]
        if not positives:
            continue
        ordered = sorted(group, key=lambda i: predictions[i], reverse=True)
        total += 1
        if positive_mask[ordered[0]]:
            hits += 1
        best_rank = min(ordered.index(i) + 1 for i in positives)
        rank_fractions.append(best_rank / max(1, len(ordered)))
    return {
        "top1": hits / total if total else 0.0,
        "rank_fraction": sum(rank_fractions) / len(rank_fractions) if rank_fractions else 1.0,
        "turns": total,
    }


def tune_blend(rows, probabilities, positive_mask, indices, score_scale):
    best = {"blend": 0.0, "top1": -1.0, "rank": 1.0}
    heuristic_scores = [row_float(row, "heuristic_score_scaled", 0.0) * 100.0 for row in rows]
    model_scores = [(prob - 0.5) * score_scale for prob in probabilities]
    for step in range(0, 61):
        blend = step / 100.0
        mixed = [
            heuristic_scores[i] * (1.0 - blend) + (heuristic_scores[i] + model_scores[i]) * blend
            for i in range(len(rows))
        ]
        metrics = grouped_metrics(rows, mixed, positive_mask, indices)
        if metrics["top1"] > best["top1"] or (
            abs(metrics["top1"] - best["top1"]) <= 1e-9 and metrics["rank_fraction"] < best["rank"]
        ):
            best = {"blend": blend, "top1": metrics["top1"], "rank": metrics["rank_fraction"]}
    return best


def make_model_class(torch, nn):
    class CandidatePolicy(nn.Module):
        def __init__(self, feature_count, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_count, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, inputs):
            return self.net(inputs).view(-1)

    return CandidatePolicy


def choose_device(torch, args):
    requested = args.device
    if requested == "auto":
        requested = "tpu" if os.environ.get("TPU_NAME") or os.environ.get("PJRT_DEVICE") == "TPU" else "cuda"
    if requested == "tpu":
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla.core.xla_model as xm
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch_xla is required for V8_DEVICE=tpu. Use a Kaggle TPU v5e-8 runtime.") from exc
        return xm.xla_device(), xm
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), None
    return torch.device("cpu"), None


def optimizer_step(optimizer, xm):
    if xm is None:
        optimizer.step()
    else:
        xm.optimizer_step(optimizer, barrier=False)
        xm.mark_step()


def layers_from_model(model, nn):
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
    return layers


def maybe_upload_file(args, path, path_in_repo, commit_message):
    if not args.upload:
        return
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for checkpoint upload.")
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to upload checkpoints: pip install huggingface_hub") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        commit_message=commit_message,
    )
    print(f"Uploaded checkpoint to https://huggingface.co/{args.hf_repo_id}/blob/main/{path_in_repo}", flush=True)


def save_sft_checkpoint(args, model, nn, member_seed, epoch, item, history, feature_names, means, scales):
    checkpoint_dir = Path(args.export_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v8_sft_checkpoint",
        "created_at": int(time.time()),
        "seed": member_seed,
        "epoch": epoch,
        "latest_metrics": item,
        "history": history,
        "member": {
            "version": "v8_sft",
            "model_type": "mlp_relu_candidate_ranker",
            "features": feature_names,
            "mean": dict(zip(feature_names, means)),
            "scale": dict(zip(feature_names, scales)),
            "layers": layers_from_model(model, nn),
            "activation": "relu",
            "score_scale": 210.0,
        },
    }
    path = checkpoint_dir / f"sft_seed_{member_seed}_epoch_{epoch:03d}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved v8 SFT checkpoint: {path}", flush=True)
    maybe_upload_file(
        args,
        path,
        f"{HF_REMOTE_PREFIX}/checkpoints/{path.name}",
        f"Upload v8 SFT checkpoint seed {member_seed} epoch {epoch}",
    )


def train_member(torch, nn, functional, args, member_seed, context):
    (
        rows,
        feature_names,
        means,
        scales,
        feature_count,
        all_x,
        labels,
        selected,
        counterfactual,
        sample_weights,
        train_indices,
        valid_indices,
        train_groups,
        valid_groups,
        train_pairs,
        valid_pairs,
        device,
        xm,
    ) = context
    torch.manual_seed(member_seed)
    random.seed(member_seed)
    CandidatePolicy = make_model_class(torch, nn)
    model = CandidatePolicy(feature_count, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    x_all = torch.tensor(all_x, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
    weight_tensor = torch.tensor(sample_weights, dtype=torch.float32, device=device)
    raw_to_local = {raw: raw for raw in range(len(rows))}
    best_state = None
    best_objective = float("inf")
    stale = 0
    history = []

    def eval_groups(groups, pairs):
        model.eval()
        total_ce = 0.0
        total_bce = 0.0
        total_groups = 0
        with torch.no_grad():
            logits_all = model(x_all)
            probs_all = torch.sigmoid(logits_all)
            for group in groups:
                indices = torch.tensor(group, dtype=torch.long, device=device)
                logits = logits_all[indices]
                targets = torch.tensor(
                    target_distribution(rows, labels, selected, counterfactual, group),
                    dtype=torch.float32,
                    device=device,
                )
                total_ce += float(-(targets * functional.log_softmax(logits, dim=0)).sum().detach().cpu())
                group_labels = label_tensor[indices]
                group_weights = weight_tensor[indices]
                total_bce += float(
                    functional.binary_cross_entropy(probs_all[indices], group_labels, weight=group_weights).detach().cpu()
                )
                total_groups += 1
            pair_loss_value = 0.0
            if pairs:
                pos_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
                neg_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
                pair_w = torch.tensor([p[2] for p in pairs], dtype=torch.float32, device=device)
                pair_loss_value = float(
                    (functional.softplus(-(logits_all[pos_idx] - logits_all[neg_idx])) * pair_w).mean().detach().cpu()
                )
            predictions = [sigmoid_prob(value) for value in logits_all.detach().cpu().tolist()]
        ce = total_ce / max(1, total_groups)
        bce = total_bce / max(1, total_groups)
        metrics = grouped_metrics(rows, predictions, [selected[i] or counterfactual[i] or labels[i] >= 0.55 for i in range(len(rows))], valid_indices)
        objective = ce + args.bce_weight * bce + args.pair_weight * pair_loss_value + args.rank_weight * (1.0 - metrics["top1"])
        return ce, bce, pair_loss_value, objective, predictions

    for epoch in range(1, args.epochs + 1):
        model.train()
        groups = train_groups[:]
        random.shuffle(groups)
        total_loss = 0.0
        batches = 0
        for offset in range(0, len(groups), args.batch_groups):
            batch_groups = groups[offset : offset + args.batch_groups]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = None
            for group in batch_groups:
                indices = torch.tensor(group, dtype=torch.long, device=device)
                logits = model(x_all[indices])
                targets = torch.tensor(
                    target_distribution(rows, labels, selected, counterfactual, group),
                    dtype=torch.float32,
                    device=device,
                )
                listwise_loss = -(targets * functional.log_softmax(logits, dim=0)).sum()
                row_labels = label_tensor[indices]
                row_weights = weight_tensor[indices]
                bce_loss = functional.binary_cross_entropy_with_logits(logits, row_labels, weight=row_weights)
                loss = listwise_loss + args.bce_weight * bce_loss
                batch_loss = loss if batch_loss is None else batch_loss + loss
            if batch_loss is None:
                continue
            batch_loss = batch_loss / max(1, len(batch_groups))
            if train_pairs:
                pair_sample = random.sample(train_pairs, min(len(train_pairs), max(64, args.batch_groups * 4)))
                pos_idx = torch.tensor([p[0] for p in pair_sample], dtype=torch.long, device=device)
                neg_idx = torch.tensor([p[1] for p in pair_sample], dtype=torch.long, device=device)
                pair_w = torch.tensor([p[2] for p in pair_sample], dtype=torch.float32, device=device)
                pos_logits = model(x_all[pos_idx])
                neg_logits = model(x_all[neg_idx])
                pair_loss = (functional.softplus(-(pos_logits - neg_logits)) * pair_w).mean()
                batch_loss = batch_loss + args.pair_weight * pair_loss
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_step(optimizer, xm)
            total_loss += float(batch_loss.detach().cpu())
            batches += 1
        scheduler.step()
        if epoch >= 1:
            valid_ce, valid_bce, valid_pair, valid_objective, valid_preds = eval_groups(valid_groups, valid_pairs)
            train_metrics = grouped_metrics(
                rows,
                valid_preds,
                [selected[i] or counterfactual[i] or labels[i] >= 0.55 for i in range(len(rows))],
                train_indices,
            )
            valid_metrics = grouped_metrics(
                rows,
                valid_preds,
                [selected[i] or counterfactual[i] or labels[i] >= 0.55 for i in range(len(rows))],
                valid_indices,
            )
            item = {
                "epoch": epoch,
                "train_loss": total_loss / max(1, batches),
                "valid_ce": valid_ce,
                "valid_bce": valid_bce,
                "valid_pair_loss": valid_pair,
                "valid_objective": valid_objective,
                "train_turn_top1": train_metrics["top1"],
                "valid_turn_top1": valid_metrics["top1"],
                "valid_rank_fraction": valid_metrics["rank_fraction"],
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(item)
            print(
                f"sft seed={member_seed} epoch={epoch:03d} "
                f"loss={item['train_loss']:.5f} valid_obj={valid_objective:.5f} "
                f"valid_top1={valid_metrics['top1']:.4f} rank={valid_metrics['rank_fraction']:.4f} "
                f"lr={item['lr']:.6f}",
                flush=True,
            )
            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                save_sft_checkpoint(args, model, nn, member_seed, epoch, item, history, feature_names, means, scales)
            if valid_objective + 1e-5 < best_objective:
                best_objective = valid_objective
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"sft seed={member_seed} early_stop epoch={epoch} best={best_objective:.5f}", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, layers_from_model(model, nn), history, best_objective


def train(args):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for v8 SFT training. Install torch or use Kaggle/Colab.") from exc

    random.seed(args.seed)
    data_path = find_training_csv(
        args.csv,
        remote_path=args.data_remote_path,
        prefer_local=args.prefer_local_data,
    )
    rows, feature_names, x_raw, labels, selected, counterfactual, sample_weights = read_rows(data_path)
    train_indices, valid_indices, games, valid_games = split_by_game(rows, args.seed)
    means, scales, normalize = normalize_from_train(x_raw, train_indices)
    all_x = normalize(x_raw)
    train_groups = build_groups(rows, train_indices)
    valid_groups = build_groups(rows, valid_indices)
    train_local = {raw_index: raw_index for raw_index in train_indices}
    valid_local = {raw_index: raw_index for raw_index in valid_indices}
    train_pairs = build_hard_pairs(rows, labels, selected, counterfactual, train_groups, train_local)
    valid_pairs = build_hard_pairs(rows, labels, selected, counterfactual, valid_groups, valid_local)
    positive_mask = [selected[i] or counterfactual[i] or labels[i] >= 0.55 for i in range(len(rows))]
    device, xm = choose_device(torch, args)

    print(
        json.dumps(
            {
                "csv": str(data_path),
                "data_remote_path": args.data_remote_path or "latest data/*/candidates_v7.csv",
                "data_source": "local_override" if args.csv else ("local_preferred" if args.prefer_local_data else "hugging_face"),
                "rows": len(rows),
                "features": len(feature_names),
                "games": len(games),
                "validation_games": len(valid_games),
                "train_groups": len(train_groups),
                "validation_groups": len(valid_groups),
                "train_pairs": len(train_pairs),
                "validation_pairs": len(valid_pairs),
                "selected_rate": sum(1 for value in selected if value) / len(selected),
                "counterfactual_rate": sum(1 for value in counterfactual if value) / len(counterfactual),
                "device": str(device),
                "ensemble_size": args.ensemble_size,
                "checkpoint_every": args.checkpoint_every,
            },
            indent=2,
        ),
        flush=True,
    )

    context = (
        rows,
        feature_names,
        means,
        scales,
        len(feature_names),
        all_x,
        labels,
        selected,
        counterfactual,
        sample_weights,
        train_indices,
        valid_indices,
        train_groups,
        valid_groups,
        train_pairs,
        valid_pairs,
        device,
        xm,
    )
    members = []
    histories = []
    models = []
    for member_index in range(args.ensemble_size):
        member_seed = args.seed + member_index * 1009
        model, layers, history, best_objective = train_member(
            torch,
            nn,
            functional,
            args,
            member_seed,
            context,
        )
        models.append(model)
        histories.append({"seed": member_seed, "history": history, "best_validation_objective": best_objective})
        members.append(
            {
                "version": "v8_sft",
                "model_type": "mlp_relu_candidate_ranker",
                "features": feature_names,
                "mean": dict(zip(feature_names, means)),
                "scale": dict(zip(feature_names, scales)),
                "layers": layers,
                "activation": "relu",
                "score_scale": 210.0,
            }
        )

    with torch.no_grad():
        all_tensor = torch.tensor(all_x, dtype=torch.float32, device=device)
        member_probs = []
        for model in models:
            logits = model(all_tensor).view(-1).detach().cpu().tolist()
            member_probs.append([sigmoid_prob(value) for value in logits])
    all_probs = [sum(member[i] for member in member_probs) / len(member_probs) for i in range(len(rows))]
    train_metrics = grouped_metrics(rows, all_probs, positive_mask, train_indices)
    valid_metrics = grouped_metrics(rows, all_probs, positive_mask, valid_indices)
    tuned_blend = tune_blend(rows, all_probs, positive_mask, valid_indices, 210.0)

    metrics = {
        "rows": len(rows),
        "features": len(feature_names),
        "games": len(games),
        "train_groups": len(train_groups),
        "validation_groups": len(valid_groups),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(valid_pairs),
        "selected_rate": sum(1 for value in selected if value) / len(selected),
        "counterfactual_rate": sum(1 for value in counterfactual if value) / len(counterfactual),
        "train_turn_top1_positive_rate": train_metrics["top1"],
        "validation_turn_top1_positive_rate": valid_metrics["top1"],
        "train_positive_mean_rank_fraction": train_metrics["rank_fraction"],
        "validation_positive_mean_rank_fraction": valid_metrics["rank_fraction"],
        "tuned_blend": tuned_blend["blend"],
        "tuned_blend_validation_top1": tuned_blend["top1"],
        "tuned_blend_validation_rank_fraction": tuned_blend["rank"],
        "ensemble_size": len(members),
        "device": str(device),
    }
    artifact = {
        "version": "v8_sft",
        "created_at": int(time.time()),
        "source_csv": str(data_path),
        "model_type": "ensemble_mlp_relu_candidate_ranker",
        "members": members,
        "blend": tuned_blend["blend"],
        "metrics": metrics,
    }

    export_dir = Path(args.export_dir)
    graph_dir = export_dir / "graphs"
    export_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "model_weights_v8_sft.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "feature_schema_v8_sft.json").write_text(json.dumps({"features": feature_names}, indent=2), encoding="utf-8")
    (export_dir / "metrics_v8_sft.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "training_history_v8_sft.json").write_text(json.dumps(histories, indent=2, sort_keys=True), encoding="utf-8")
    with (export_dir / "predictions_v8_sft.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "label", "prediction", "selected", "counterfactual_positive", "game_result", "split"])
        valid_set = set(valid_indices)
        for i, pred in enumerate(all_probs):
            writer.writerow([
                i,
                labels[i],
                pred,
                float(selected[i]),
                float(counterfactual[i]),
                row_float(rows[i], "game_result", 0.0),
                "validation" if i in valid_set else "train",
            ])

    try:
        import matplotlib.pyplot as plt

        first_history = histories[0]["history"] if histories else []
        epochs = [item["epoch"] for item in first_history]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [item["valid_objective"] for item in first_history], label="validation objective")
        plt.plot(epochs, [item["valid_turn_top1"] for item in first_history], label="validation top1")
        plt.xlabel("epoch")
        plt.title("v8 SFT validation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "sft_validation_v8.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.hist([pred for pred, ok in zip(all_probs, positive_mask) if ok], bins=30, alpha=0.65, label="positive")
        plt.hist([pred for pred, ok in zip(all_probs, positive_mask) if not ok], bins=30, alpha=0.65, label="other")
        plt.xlabel("policy probability")
        plt.ylabel("rows")
        plt.title("v8 SFT prediction distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "sft_prediction_histogram_v8.png", dpi=150)
        plt.close()
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped graph generation.", flush=True)

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"Saved v8 SFT artifact: {export_dir / 'model_weights_v8_sft.json'}", flush=True)

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
            commit_message="Upload v8 SFT Orbit Wars policy artifacts and graphs",
        )
        print(f"Uploaded {export_dir} to https://huggingface.co/{args.hf_repo_id}/tree/main/{HF_REMOTE_PREFIX}", flush=True)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
