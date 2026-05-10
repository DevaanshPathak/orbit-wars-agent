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
SFT_REMOTE_PREFIX = "v9/sft"
GRPO_REMOTE_PREFIX = "v9/grpo"

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
        description="Train v9 Orbit Wars candidate policies on TPU v5e-8."
    )
    parser.add_argument("--mode", choices=("sft", "grpo"), required=True)
    parser.add_argument("--csv", default=os.environ.get("CANDIDATES_CSV", ""))
    parser.add_argument(
        "--data-remote-path",
        default=os.environ.get("V9_DATA_REMOTE_PATH", ""),
        help="Exact HF data path. If omitted, newest data/*/candidates_v9.csv or candidates_v7.csv is used.",
    )
    parser.add_argument("--sft-artifact", default=os.environ.get("V9_SFT_ARTIFACT", ""))
    parser.add_argument(
        "--sft-remote-path",
        default=os.environ.get("V9_SFT_REMOTE_PATH", f"{SFT_REMOTE_PREFIX}/model_weights_v9_sft.json"),
    )
    parser.add_argument("--export-dir", default=os.environ.get("V9_EXPORT_DIR", "notebooks/v9/exports"))
    parser.add_argument("--device", choices=("tpu", "cuda", "cpu", "auto"), default=os.environ.get("V9_DEVICE", "tpu"))
    parser.add_argument("--tpu-cores", type=int, default=int(os.environ.get("V9_TPU_CORES", "8")))
    parser.add_argument("--members", type=int, default=int(os.environ.get("V9_MEMBERS", "8")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("V9_EPOCHS", "0")))
    parser.add_argument("--row-batch-size", type=int, default=int(os.environ.get("V9_ROW_BATCH_SIZE", "4096")))
    parser.add_argument("--pair-batch-size", type=int, default=int(os.environ.get("V9_PAIR_BATCH_SIZE", "4096")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("V9_LR", "0.00055")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("V9_WEIGHT_DECAY", "0.00022")))
    parser.add_argument("--dropout", type=float, default=float(os.environ.get("V9_DROPOUT", "0.16")))
    parser.add_argument("--bce-weight", type=float, default=float(os.environ.get("V9_BCE_WEIGHT", "0.70")))
    parser.add_argument("--pair-weight", type=float, default=float(os.environ.get("V9_PAIR_WEIGHT", "0.42")))
    parser.add_argument("--reward-weight", type=float, default=float(os.environ.get("V9_REWARD_WEIGHT", "0.65")))
    parser.add_argument("--kl-weight", type=float, default=float(os.environ.get("V9_KL_WEIGHT", "0.060")))
    parser.add_argument("--anchor-weight", type=float, default=float(os.environ.get("V9_ANCHOR_WEIGHT", "0.12")))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("V9_PATIENCE", "30")))
    parser.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("V9_CHECKPOINT_EVERY", "30")))
    parser.add_argument("--eval-every", type=int, default=int(os.environ.get("V9_EVAL_EVERY", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("V9_SEED", "1909")))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo-id", default=os.environ.get("HF_REPO_ID", HF_REPO_ID))
    parser.add_argument("--hf-repo-type", default=os.environ.get("HF_REPO_TYPE", HF_REPO_TYPE))
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


def hf_token():
    load_dotenv()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def download_training_csv(remote_path, repo_id, repo_type):
    token = hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required to download training data from Hugging Face.")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to download data.") from exc

    if not remote_path:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        candidates_v9 = sorted(
            name for name in files if name.startswith("data/") and name.endswith("/candidates_v9.csv")
        )
        candidates_v7 = sorted(
            name for name in files if name.startswith("data/") and name.endswith("/candidates_v7.csv")
        )
        candidates = candidates_v9 or candidates_v7
        if not candidates:
            raise FileNotFoundError("No data/*/candidates_v9.csv or candidates_v7.csv found in Hugging Face repo.")
        remote_path = candidates[-1]

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=remote_path,
            token=token,
        )
    )


def download_sft_artifact(remote_path, repo_id, repo_type):
    token = hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required to download the SFT artifact from Hugging Face.")
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to download SFT artifacts.") from exc
    return Path(hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=remote_path, token=token))


def upload_file(args, path, path_in_repo, commit_message):
    if not args.upload:
        return
    token = hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required for upload.")
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to upload artifacts.") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path_in_repo,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        commit_message=commit_message,
    )
    print(f"uploaded {path_in_repo}", flush=True)


def upload_folder(args, folder, path_in_repo, commit_message):
    if not args.upload:
        return
    token = hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN is required for upload.")
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to upload artifacts.") from exc
    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo_id, repo_type=args.hf_repo_type, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    print(f"uploaded folder to {path_in_repo}", flush=True)


def resolve_local_inputs(args):
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Training CSV does not exist: {csv_path}")
    else:
        csv_path = download_training_csv(args.data_remote_path, args.hf_repo_id, args.hf_repo_type)
    args.csv = str(csv_path)

    if args.mode == "grpo":
        if args.sft_artifact:
            sft_path = Path(args.sft_artifact)
            if not sft_path.exists():
                raise FileNotFoundError(f"SFT artifact does not exist: {sft_path}")
        else:
            sft_path = download_sft_artifact(args.sft_remote_path, args.hf_repo_id, args.hf_repo_type)
        args.sft_artifact = str(sft_path)
    return args


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


def normalize_with_artifact(x_raw, feature_names, artifact):
    member = artifact["members"][0] if artifact.get("members") else artifact
    means = member.get("mean", {})
    scales = member.get("scale", {})
    normalized = []
    for row in x_raw:
        values = []
        for name, value in zip(feature_names, row):
            scale = float(scales.get(name, 1.0) or 1.0)
            values.append((value - float(means.get(name, 0.0))) / scale)
        normalized.append(values)
    return normalized, means, scales


def build_groups(rows, indices):
    grouped = defaultdict(list)
    for raw_index in indices:
        step = int(row_float(rows[raw_index], "step", 0.0))
        grouped[(rows[raw_index].get("game_id", ""), step)].append(raw_index)
    return [items for items in grouped.values() if len(items) >= 2]


def candidate_reward(row, label, selected, counterfactual):
    reward = (label - 0.5) * 2.0
    reward += max(-1.5, min(1.5, row_float(row, "future_advantage_delta_15", 0.0) / 80.0))
    reward += max(-1.0, min(1.0, row_float(row, "future_advantage_delta_30", 0.0) / 120.0)) * 0.55
    reward += max(-0.8, min(0.8, row_float(row, "future_production_delta_15", 0.0) / 8.0)) * 0.55
    reward += max(-0.8, min(0.8, row_float(row, "future_planet_delta_15", 0.0) / 3.0)) * 0.45
    reward += row_float(row, "game_result", 0.0) * 0.22
    reward += max(-1.0, min(1.0, row_float(row, "reward_margin", 0.0) / 700.0)) * 0.18
    if selected:
        reward += 0.18
    if counterfactual:
        reward += 0.32
    reward -= row_float(row, "failure_overcommit", 0.0) * 0.75
    reward -= row_float(row, "failure_missed_tactical", 0.0) * 0.30
    reward -= row_float(row, "failure_missed_comet", 0.0) * 0.30
    reward -= row_float(row, "failure_slow_expansion", 0.0) * 0.25
    return reward


def build_pairs(rows, labels, selected, counterfactual, groups, mode):
    pairs = []
    for group in groups:
        if mode == "sft":
            positives = [i for i in group if selected[i] or counterfactual[i] or labels[i] >= 0.55]
            negatives = [i for i in group if labels[i] <= 0.35 and not counterfactual[i]]
            positives.sort(key=lambda i: labels[i] + row_float(rows[i], "future_advantage_delta_15", 0.0) * 0.004, reverse=True)
            negatives.sort(key=lambda i: row_float(rows[i], "heuristic_score_scaled", 0.0), reverse=True)
        else:
            rewards = [(i, candidate_reward(rows[i], labels[i], selected[i], counterfactual[i])) for i in group]
            rewards.sort(key=lambda item: item[1], reverse=True)
            positives = [i for i, _ in rewards[: max(1, min(3, len(rewards) // 3))]]
            negatives = [i for i, _ in rewards[-max(1, min(5, len(rewards) // 2)) :]]
        emitted = 0
        for pos in positives:
            for neg in negatives:
                if pos == neg or emitted >= 10:
                    break
                gap = max(0.05, labels[pos] - labels[neg]) if mode == "sft" else 1.0
                pairs.append((pos, neg, gap))
                emitted += 1
    return pairs


def grouped_metrics(rows, scores, positive_mask, indices):
    groups = build_groups(rows, indices)
    hits = 0
    total = 0
    ranks = []
    for group in groups:
        positives = [i for i in group if positive_mask[i]]
        if not positives:
            continue
        ordered = sorted(group, key=lambda i: scores[i], reverse=True)
        total += 1
        if positive_mask[ordered[0]]:
            hits += 1
        ranks.append(min(ordered.index(i) + 1 for i in positives) / max(1, len(ordered)))
    return {
        "top1": hits / total if total else 0.0,
        "rank_fraction": sum(ranks) / len(ranks) if ranks else 1.0,
        "turns": total,
    }


def sigmoid_prob(value):
    value = max(-50.0, min(50.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def make_model_class(torch, nn):
    class CandidatePolicy(nn.Module):
        def __init__(self, feature_count, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_count, 192),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 1),
            )

        def forward(self, inputs):
            return self.net(inputs).view(-1)

    return CandidatePolicy


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


def load_member_into_model(torch, model, member):
    linear_layers = [module for module in model.net if module.__class__.__name__ == "Linear"]
    for layer_module, layer_data in zip(linear_layers, member.get("layers", [])):
        if len(layer_data.get("weights", [])) != layer_module.weight.shape[0]:
            continue
        layer_module.weight.data = torch.tensor(
            layer_data["weights"], dtype=layer_module.weight.dtype, device=layer_module.weight.device
        )
        layer_module.bias.data = torch.tensor(
            layer_data["bias"], dtype=layer_module.bias.dtype, device=layer_module.bias.device
        )


def prepare_device(args, ordinal=0):
    import torch

    requested = args.device
    if requested == "auto":
        requested = "tpu" if os.environ.get("TPU_NAME") or os.environ.get("PJRT_DEVICE") == "TPU" else "cuda"
    if requested == "tpu":
        try:
            import torch_xla.core.xla_model as xm
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch_xla is required for --device tpu. Use a Kaggle TPU v5e-8 runtime.") from exc
        return torch, xm.xla_device(), xm, True
    if requested == "cuda" and torch.cuda.is_available():
        return torch, torch.device(f"cuda:{ordinal % max(1, torch.cuda.device_count())}"), None, False
    return torch, torch.device("cpu"), None, False


def optimizer_step(optimizer, xm):
    if xm is None:
        optimizer.step()
    else:
        xm.optimizer_step(optimizer, barrier=False)
        xm.mark_step()


def metric_objective(metrics):
    return (1.0 - metrics["top1"]) + 0.35 * metrics["rank_fraction"]


def save_member(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def train_one_sft_member(args, member_index, ordinal):
    torch, device, xm, is_xla = prepare_device(args, ordinal)
    from torch import nn
    import torch.nn.functional as functional

    rows, feature_names, x_raw, labels, selected, counterfactual, sample_weights = read_rows(args.csv)
    train_indices, valid_indices, games, valid_games = split_by_game(rows, args.seed)
    means, scales, normalize = normalize_from_train(x_raw, train_indices)
    all_x = normalize(x_raw)
    train_groups = build_groups(rows, train_indices)
    valid_groups = build_groups(rows, valid_indices)
    train_pairs = build_pairs(rows, labels, selected, counterfactual, train_groups, "sft")
    positive_mask = [selected[i] or counterfactual[i] or labels[i] >= 0.55 for i in range(len(rows))]

    member_seed = args.seed + member_index * 1009
    random.seed(member_seed)
    torch.manual_seed(member_seed)
    CandidatePolicy = make_model_class(torch, nn)
    model = CandidatePolicy(len(feature_names), args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = args.epochs or 220
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    x_all = torch.tensor(all_x, dtype=torch.float32, device=device)
    y_all = torch.tensor(labels, dtype=torch.float32, device=device)
    w_all = torch.tensor(sample_weights, dtype=torch.float32, device=device)
    best_state = None
    best_objective = float("inf")
    stale = 0
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_indices)
        random.shuffle(train_pairs)
        total_loss = 0.0
        batches = 0
        pair_cursor = 0
        for offset in range(0, len(train_indices), args.row_batch_size):
            batch = train_indices[offset : offset + args.row_batch_size]
            if len(batch) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            idx = torch.tensor(batch, dtype=torch.long, device=device)
            logits = model(x_all[idx])
            bce = functional.binary_cross_entropy_with_logits(logits, y_all[idx], weight=w_all[idx])
            loss = args.bce_weight * bce
            if train_pairs:
                pair_batch = train_pairs[pair_cursor : pair_cursor + args.pair_batch_size]
                pair_cursor = (pair_cursor + args.pair_batch_size) % max(1, len(train_pairs))
                if pair_batch:
                    pos_idx = torch.tensor([p[0] for p in pair_batch], dtype=torch.long, device=device)
                    neg_idx = torch.tensor([p[1] for p in pair_batch], dtype=torch.long, device=device)
                    pair_w = torch.tensor([p[2] for p in pair_batch], dtype=torch.float32, device=device)
                    pair_loss = (functional.softplus(-(model(x_all[pos_idx]) - model(x_all[neg_idx]))) * pair_w).mean()
                    loss = loss + args.pair_weight * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_step(optimizer, xm)
            total_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits_all = model(x_all).detach().cpu().tolist()
            scores = [sigmoid_prob(value) for value in logits_all]
            valid_metrics = grouped_metrics(rows, scores, positive_mask, valid_indices)
            objective = metric_objective(valid_metrics)
            item = {
                "epoch": epoch,
                "train_loss": total_loss / max(1, batches),
                "valid_turn_top1": valid_metrics["top1"],
                "valid_rank_fraction": valid_metrics["rank_fraction"],
                "objective": objective,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": round(time.time() - start, 2),
            }
            history.append(item)
            print(
                f"v9 sft member={member_index} epoch={epoch:03d} "
                f"loss={item['train_loss']:.5f} top1={item['valid_turn_top1']:.4f} "
                f"rank={item['valid_rank_fraction']:.4f} device={device}",
                flush=True,
            )
            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                checkpoint = {
                    "version": "v9_sft_checkpoint",
                    "member_index": member_index,
                    "epoch": epoch,
                    "latest_metrics": item,
                    "history": history,
                    "member": {
                        "version": "v9_sft",
                        "model_type": "mlp_relu_candidate_ranker",
                        "features": feature_names,
                        "mean": dict(zip(feature_names, means)),
                        "scale": dict(zip(feature_names, scales)),
                        "layers": layers_from_model(model, nn),
                        "activation": "relu",
                        "score_scale": 235.0,
                    },
                }
                path = save_member(
                    Path(args.export_dir) / "sft" / "checkpoints" / f"sft_member_{member_index:02d}_epoch_{epoch:03d}.json",
                    checkpoint,
                )
                upload_file(args, path, f"{SFT_REMOTE_PREFIX}/checkpoints/{path.name}", f"Upload v9 SFT member {member_index} epoch {epoch}")
            if objective + 1e-5 < best_objective:
                best_objective = objective
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"v9 sft member={member_index} early_stop epoch={epoch}", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    payload = {
        "version": "v9_sft",
        "model_type": "mlp_relu_candidate_ranker",
        "features": feature_names,
        "mean": dict(zip(feature_names, means)),
        "scale": dict(zip(feature_names, scales)),
        "layers": layers_from_model(model, nn),
        "activation": "relu",
        "score_scale": 235.0,
        "member_index": member_index,
        "history": history,
        "best_validation_objective": best_objective,
    }
    path = save_member(Path(args.export_dir) / "sft" / "members" / f"member_{member_index:02d}.json", payload)
    print(f"saved sft member {member_index}: {path}", flush=True)


def train_one_grpo_member(args, member_index, ordinal):
    torch, device, xm, is_xla = prepare_device(args, ordinal)
    from torch import nn
    import torch.nn.functional as functional

    sft_artifact = json.loads(Path(args.sft_artifact).read_text(encoding="utf-8"))
    rows, feature_names, x_raw, labels, selected, counterfactual, sample_weights = read_rows(args.csv)
    all_x, means, scales = normalize_with_artifact(x_raw, feature_names, sft_artifact)
    train_indices, valid_indices, games, valid_games = split_by_game(rows, args.seed)
    train_groups = build_groups(rows, train_indices)
    train_pairs = build_pairs(rows, labels, selected, counterfactual, train_groups, "grpo")
    rewards = [candidate_reward(row, labels[i], selected[i], counterfactual[i]) for i, row in enumerate(rows)]
    reward_targets = [sigmoid_prob(reward) for reward in rewards]
    positive_mask = [selected[i] or counterfactual[i] or labels[i] >= 0.55 for i in range(len(rows))]

    member_seed = args.seed + 5000 + member_index * 997
    random.seed(member_seed)
    torch.manual_seed(member_seed)
    CandidatePolicy = make_model_class(torch, nn)
    model = CandidatePolicy(len(feature_names), args.dropout).to(device)
    ref_model = CandidatePolicy(len(feature_names), args.dropout).to(device)
    base_members = sft_artifact.get("members", []) or [sft_artifact]
    load_member_into_model(torch, model, base_members[member_index % len(base_members)])
    load_member_into_model(torch, ref_model, base_members[member_index % len(base_members)])
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    epochs = args.epochs or 140
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    x_all = torch.tensor(all_x, dtype=torch.float32, device=device)
    reward_all = torch.tensor(reward_targets, dtype=torch.float32, device=device)
    label_all = torch.tensor(labels, dtype=torch.float32, device=device)
    best_state = None
    best_objective = float("inf")
    stale = 0
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_indices)
        random.shuffle(train_pairs)
        total_loss = 0.0
        batches = 0
        pair_cursor = 0
        for offset in range(0, len(train_indices), args.row_batch_size):
            batch = train_indices[offset : offset + args.row_batch_size]
            if len(batch) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            idx = torch.tensor(batch, dtype=torch.long, device=device)
            logits = model(x_all[idx])
            with torch.no_grad():
                ref_logits = ref_model(x_all[idx])
            reward_loss = functional.binary_cross_entropy_with_logits(logits, reward_all[idx])
            anchor_loss = functional.binary_cross_entropy_with_logits(logits, label_all[idx])
            kl_proxy = ((logits - ref_logits) ** 2).mean()
            loss = args.reward_weight * reward_loss + args.anchor_weight * anchor_loss + args.kl_weight * kl_proxy
            if train_pairs:
                pair_batch = train_pairs[pair_cursor : pair_cursor + args.pair_batch_size]
                pair_cursor = (pair_cursor + args.pair_batch_size) % max(1, len(train_pairs))
                if pair_batch:
                    pos_idx = torch.tensor([p[0] for p in pair_batch], dtype=torch.long, device=device)
                    neg_idx = torch.tensor([p[1] for p in pair_batch], dtype=torch.long, device=device)
                    pair_loss = functional.softplus(-(model(x_all[pos_idx]) - model(x_all[neg_idx]))).mean()
                    loss = loss + args.pair_weight * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer_step(optimizer, xm)
            total_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits_all = model(x_all).detach().cpu().tolist()
            scores = [sigmoid_prob(value) for value in logits_all]
            valid_metrics = grouped_metrics(rows, scores, positive_mask, valid_indices)
            objective = metric_objective(valid_metrics)
            item = {
                "epoch": epoch,
                "train_loss": total_loss / max(1, batches),
                "valid_turn_top1": valid_metrics["top1"],
                "valid_rank_fraction": valid_metrics["rank_fraction"],
                "objective": objective,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": round(time.time() - start, 2),
            }
            history.append(item)
            print(
                f"v9 grpo member={member_index} epoch={epoch:03d} "
                f"loss={item['train_loss']:.5f} top1={item['valid_turn_top1']:.4f} "
                f"rank={item['valid_rank_fraction']:.4f} device={device}",
                flush=True,
            )
            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                checkpoint = {
                    "version": "v9_grpo_checkpoint",
                    "member_index": member_index,
                    "epoch": epoch,
                    "latest_metrics": item,
                    "history": history,
                    "member": {
                        "version": "v9_grpo",
                        "model_type": "mlp_relu_candidate_ranker",
                        "features": feature_names,
                        "mean": means,
                        "scale": scales,
                        "layers": layers_from_model(model, nn),
                        "activation": "relu",
                        "score_scale": 260.0,
                    },
                }
                path = save_member(
                    Path(args.export_dir) / "grpo" / "checkpoints" / f"grpo_member_{member_index:02d}_epoch_{epoch:03d}.json",
                    checkpoint,
                )
                upload_file(args, path, f"{GRPO_REMOTE_PREFIX}/checkpoints/{path.name}", f"Upload v9 GRPO member {member_index} epoch {epoch}")
            if objective + 1e-5 < best_objective:
                best_objective = objective
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"v9 grpo member={member_index} early_stop epoch={epoch}", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "version": "v9_grpo",
        "model_type": "mlp_relu_candidate_ranker",
        "features": feature_names,
        "mean": means,
        "scale": scales,
        "layers": layers_from_model(model, nn),
        "activation": "relu",
        "score_scale": 260.0,
        "member_index": member_index,
        "history": history,
        "best_validation_objective": best_objective,
    }
    path = save_member(Path(args.export_dir) / "grpo" / "members" / f"member_{member_index:02d}.json", payload)
    print(f"saved grpo member {member_index}: {path}", flush=True)


def worker_main(ordinal, args_dict):
    args = argparse.Namespace(**args_dict)
    world_size = max(1, int(args.world_size))
    for member_index in range(ordinal, args.members, world_size):
        if args.mode == "sft":
            train_one_sft_member(args, member_index, ordinal)
        else:
            train_one_grpo_member(args, member_index, ordinal)


def aggregate_artifact(args):
    mode_dir = Path(args.export_dir) / args.mode
    member_files = sorted((mode_dir / "members").glob("member_*.json"))
    if not member_files:
        raise RuntimeError(f"No member files found in {mode_dir / 'members'}")
    members = [json.loads(path.read_text(encoding="utf-8")) for path in member_files]
    histories = [{"member_index": member.get("member_index"), "history": member.pop("history", [])} for member in members]
    bests = [member.pop("best_validation_objective", None) for member in members]
    version = f"v9_{args.mode}"
    score_scale = 235.0 if args.mode == "sft" else 260.0
    blend = 0.30 if args.mode == "sft" else 0.42
    artifact = {
        "version": version,
        "created_at": int(time.time()),
        "source_csv": args.csv,
        "source_sft_artifact": args.sft_artifact if args.mode == "grpo" else "",
        "model_type": "ensemble_mlp_relu_candidate_ranker",
        "members": members,
        "blend": blend,
        "score_scale": score_scale,
        "metrics": {
            "members": len(members),
            "best_validation_objectives": bests,
            "device": args.device,
            "tpu_cores": args.tpu_cores,
            "row_batch_size": args.row_batch_size,
            "pair_batch_size": args.pair_batch_size,
        },
    }
    mode_dir.mkdir(parents=True, exist_ok=True)
    weights_name = f"model_weights_v9_{args.mode}.json"
    metrics_name = f"metrics_v9_{args.mode}.json"
    history_name = f"training_history_v9_{args.mode}.json"
    (mode_dir / weights_name).write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    (mode_dir / metrics_name).write_text(json.dumps(artifact["metrics"], indent=2, sort_keys=True), encoding="utf-8")
    (mode_dir / history_name).write_text(json.dumps(histories, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(artifact["metrics"], indent=2, sort_keys=True), flush=True)
    print(f"saved v9 {args.mode} artifact: {mode_dir / weights_name}", flush=True)
    remote_prefix = SFT_REMOTE_PREFIX if args.mode == "sft" else GRPO_REMOTE_PREFIX
    upload_folder(args, mode_dir, remote_prefix, f"Upload v9 {args.mode.upper()} TPU artifacts")


def run_training(args):
    args = resolve_local_inputs(args)
    default_epochs = 220 if args.mode == "sft" else 140
    if args.epochs <= 0:
        args.epochs = default_epochs
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto" and (os.environ.get("TPU_NAME") or os.environ.get("PJRT_DEVICE") == "TPU"):
        args.device = "tpu"
    args.world_size = min(max(1, args.members), max(1, args.tpu_cores if args.device == "tpu" else 1))

    print(
        json.dumps(
            {
                "mode": args.mode,
                "csv": args.csv,
                "sft_artifact": args.sft_artifact,
                "device": args.device,
                "world_size": args.world_size,
                "members": args.members,
                "epochs": args.epochs,
                "row_batch_size": args.row_batch_size,
                "pair_batch_size": args.pair_batch_size,
                "export_dir": args.export_dir,
                "upload": args.upload,
            },
            indent=2,
        ),
        flush=True,
    )

    if args.device == "tpu":
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla.distributed.xla_multiprocessing as xmp
        except ModuleNotFoundError as exc:
            raise RuntimeError("torch_xla is required for TPU training. Enable a Kaggle TPU v5e-8 runtime.") from exc
        xmp.spawn(worker_main, args=(vars(args),), nprocs=args.world_size, start_method="fork")
    else:
        worker_main(0, vars(args))
    aggregate_artifact(args)


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
