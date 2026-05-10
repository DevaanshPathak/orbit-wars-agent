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
SFT_REMOTE_PREFIX = "v7/sft"
GRPO_REMOTE_PREFIX = "v7/grpo"

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
        description="Run v8 constrained GRPO-style policy improvement over Orbit Wars candidate groups."
    )
    parser.add_argument("--csv", default=os.environ.get("CANDIDATES_CSV", ""))
    parser.add_argument(
        "--data-remote-path",
        default=os.environ.get("V8_GRPO_DATA_REMOTE_PATH", ""),
        help="Optional exact Hugging Face repo path for candidates_v7.csv. If omitted, the newest data/*/candidates_v7.csv is used.",
    )
    parser.add_argument(
        "--prefer-local-data",
        action="store_true",
        help="Use a local candidates_v7.csv/candidates_v8.csv before trying Hugging Face. Default is Hugging Face.",
    )
    parser.add_argument(
        "--sft-artifact",
        default=os.environ.get("V8_SFT_ARTIFACT", ""),
        help="Optional local SFT JSON override. By default GRPO downloads the SFT artifact from Hugging Face.",
    )
    parser.add_argument(
        "--sft-remote-path",
        default=os.environ.get("V8_SFT_REMOTE_PATH", f"{SFT_REMOTE_PREFIX}/model_weights_v8_sft.json"),
        help="Hugging Face repo path for the SFT artifact used by GRPO.",
    )
    parser.add_argument(
        "--prefer-local-sft",
        action="store_true",
        help="Use notebooks/v8/exports/sft/model_weights_v8_sft.json before trying Hugging Face.",
    )
    parser.add_argument("--export-dir", default="notebooks/v8/exports/grpo")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("V8_GRPO_EPOCHS", "120")))
    parser.add_argument("--batch-groups", type=int, default=int(os.environ.get("V8_GRPO_BATCH_GROUPS", "160")))
    parser.add_argument("--samples-per-group", type=int, default=int(os.environ.get("V8_GRPO_SAMPLES", "10")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("V8_GRPO_TEMPERATURE", "0.90")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("V8_GRPO_LR", "0.00025")))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("V8_GRPO_WEIGHT_DECAY", "0.00018")))
    parser.add_argument("--kl-weight", type=float, default=float(os.environ.get("V8_GRPO_KL_WEIGHT", "0.065")))
    parser.add_argument("--entropy-weight", type=float, default=float(os.environ.get("V8_GRPO_ENTROPY_WEIGHT", "0.012")))
    parser.add_argument("--supervised-anchor", type=float, default=float(os.environ.get("V8_GRPO_SUPERVISED_ANCHOR", "0.14")))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("V8_GRPO_PATIENCE", "24")))
    parser.add_argument("--seed", type=int, default=1701)
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


def find_training_csv(csv_arg, remote_path="", prefer_local=False, repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE):
    if csv_arg:
        path = Path(csv_arg)
        if not path.exists():
            raise FileNotFoundError(f"Training CSV does not exist: {path}")
        return path

    if not prefer_local:
        return download_training_csv(remote_path, repo_id=repo_id, repo_type=repo_type)

    root = Path("data")
    candidates = []
    if root.exists():
        candidates.extend(root.glob("*/candidates_v8.csv"))
        candidates.extend(root.glob("*/candidates_v7.csv"))
    if candidates:
        return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]

    return download_training_csv(remote_path, repo_id=repo_id, repo_type=repo_type)


def download_sft_artifact(remote_path, repo_id, repo_type):
    load_dotenv()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to download the SFT artifact from Hugging Face.")
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install huggingface_hub to download SFT artifacts.") from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=remote_path,
            token=token,
        )
    )


def find_sft_artifact(path_arg, repo_id, repo_type, remote_path, prefer_local=False):
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"SFT artifact does not exist: {path}")
        return path

    local = Path("notebooks/v8/exports/sft/model_weights_v8_sft.json")
    if prefer_local and local.exists():
        return local

    return download_sft_artifact(remote_path, repo_id, repo_type)


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
    return rows, feature_names, x_raw, labels, selected, counterfactual


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


def target_distribution(rows, rewards, group):
    values = [math.exp(max(-6.0, min(6.0, rewards[i]))) for i in group]
    total = sum(values)
    return [value / total for value in values] if total > 0.0 else [1.0 / len(group)] * len(group)


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


def make_model_class(torch, nn):
    class CandidatePolicy(nn.Module):
        def __init__(self, feature_count):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feature_count, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, inputs):
            return self.net(inputs).view(-1)

    return CandidatePolicy


def load_member_into_model(torch, model, member):
    linear_layers = [module for module in model.net if module.__class__.__name__ == "Linear"]
    for layer_module, layer_data in zip(linear_layers, member.get("layers", [])):
        layer_module.weight.data = torch.tensor(layer_data["weights"], dtype=layer_module.weight.dtype, device=layer_module.weight.device)
        layer_module.bias.data = torch.tensor(layer_data["bias"], dtype=layer_module.bias.dtype, device=layer_module.bias.device)


def sigmoid_prob(value):
    value = max(-50.0, min(50.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def train(args):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for v8 GRPO training. Install torch or use Kaggle/Colab.") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_path = find_training_csv(
        args.csv,
        remote_path=args.data_remote_path,
        prefer_local=args.prefer_local_data,
        repo_id=args.hf_repo_id,
        repo_type=args.hf_repo_type,
    )
    sft_path = find_sft_artifact(
        args.sft_artifact,
        args.hf_repo_id,
        args.hf_repo_type,
        args.sft_remote_path,
        prefer_local=args.prefer_local_sft,
    )
    sft_artifact = json.loads(Path(sft_path).read_text(encoding="utf-8"))
    rows, feature_names, x_raw, labels, selected, counterfactual = read_rows(data_path)
    all_x, means, scales = normalize_with_artifact(x_raw, feature_names, sft_artifact)
    train_indices, valid_indices, games, valid_games = split_by_game(rows, args.seed)
    train_groups = build_groups(rows, train_indices)
    valid_groups = build_groups(rows, valid_indices)
    rewards = [
        candidate_reward(row, labels[i], selected[i], counterfactual[i])
        for i, row in enumerate(rows)
    ]
    positive_mask = [selected[i] or counterfactual[i] or labels[i] >= 0.55 for i in range(len(rows))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CandidatePolicy = make_model_class(torch, nn)
    base_members = sft_artifact.get("members", []) or [sft_artifact]
    model = CandidatePolicy(len(feature_names)).to(device)
    ref_model = CandidatePolicy(len(feature_names)).to(device)
    load_member_into_model(torch, model, base_members[0])
    load_member_into_model(torch, ref_model, base_members[0])
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    x_all = torch.tensor(all_x, dtype=torch.float32, device=device)

    print(
        json.dumps(
            {
                "csv": str(data_path),
                "data_remote_path": args.data_remote_path or "latest data/*/candidates_v7.csv",
                "data_source": "local_override" if args.csv else ("local_preferred" if args.prefer_local_data else "hugging_face"),
                "sft_artifact": str(sft_path),
                "sft_remote_path": args.sft_remote_path,
                "sft_source": "local_override" if args.sft_artifact else ("local_preferred" if args.prefer_local_sft else "hugging_face"),
                "rows": len(rows),
                "features": len(feature_names),
                "games": len(games),
                "validation_games": len(valid_games),
                "train_groups": len(train_groups),
                "validation_groups": len(valid_groups),
                "samples_per_group": args.samples_per_group,
                "device": str(device),
                "remote_upload_path": GRPO_REMOTE_PREFIX,
            },
            indent=2,
        ),
        flush=True,
    )

    best_state = None
    best_objective = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        groups = train_groups[:]
        random.shuffle(groups)
        total_loss = 0.0
        total_policy = 0.0
        total_kl = 0.0
        total_entropy = 0.0
        batches = 0
        for offset in range(0, len(groups), args.batch_groups):
            batch_groups = groups[offset : offset + args.batch_groups]
            optimizer.zero_grad(set_to_none=True)
            loss_acc = None
            policy_acc = 0.0
            kl_acc = 0.0
            entropy_acc = 0.0
            for group in batch_groups:
                indices = torch.tensor(group, dtype=torch.long, device=device)
                logits = model(x_all[indices]) / max(0.20, args.temperature)
                with torch.no_grad():
                    ref_logits = ref_model(x_all[indices]) / max(0.20, args.temperature)
                probs = torch.softmax(logits, dim=0)
                ref_probs = torch.softmax(ref_logits, dim=0)
                log_probs = torch.log(probs.clamp_min(1e-8))
                group_rewards = torch.tensor([rewards[i] for i in group], dtype=torch.float32, device=device)
                sampled = torch.multinomial(probs.detach(), num_samples=min(args.samples_per_group, len(group)), replacement=True)
                sampled_rewards = group_rewards[sampled]
                if sampled_rewards.numel() > 1:
                    advantages = (sampled_rewards - sampled_rewards.mean()) / sampled_rewards.std(unbiased=False).clamp_min(1e-4)
                else:
                    advantages = sampled_rewards - sampled_rewards.mean()
                policy_loss = -(advantages.detach() * log_probs[sampled]).mean()
                kl = (probs * (torch.log(probs.clamp_min(1e-8)) - torch.log(ref_probs.clamp_min(1e-8)))).sum()
                entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
                target = torch.tensor(target_distribution(rows, rewards, group), dtype=torch.float32, device=device)
                anchor = -(target * functional.log_softmax(logits, dim=0)).sum()
                loss = policy_loss + args.kl_weight * kl - args.entropy_weight * entropy + args.supervised_anchor * anchor
                loss_acc = loss if loss_acc is None else loss_acc + loss
                policy_acc += float(policy_loss.detach().cpu())
                kl_acc += float(kl.detach().cpu())
                entropy_acc += float(entropy.detach().cpu())
            if loss_acc is None:
                continue
            loss_acc = loss_acc / max(1, len(batch_groups))
            loss_acc.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            total_loss += float(loss_acc.detach().cpu())
            total_policy += policy_acc / max(1, len(batch_groups))
            total_kl += kl_acc / max(1, len(batch_groups))
            total_entropy += entropy_acc / max(1, len(batch_groups))
            batches += 1
        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                logits_all = model(x_all).detach().cpu().tolist()
            scores = [sigmoid_prob(value) for value in logits_all]
            valid_metrics = grouped_metrics(rows, scores, positive_mask, valid_indices)
            objective = (1.0 - valid_metrics["top1"]) + 0.35 * valid_metrics["rank_fraction"] + 0.02 * (total_kl / max(1, batches))
            item = {
                "epoch": epoch,
                "loss": total_loss / max(1, batches),
                "policy_loss": total_policy / max(1, batches),
                "kl": total_kl / max(1, batches),
                "entropy": total_entropy / max(1, batches),
                "valid_turn_top1": valid_metrics["top1"],
                "valid_rank_fraction": valid_metrics["rank_fraction"],
                "objective": objective,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(item)
            print(
                f"grpo epoch={epoch:03d} loss={item['loss']:.5f} "
                f"policy={item['policy_loss']:.5f} kl={item['kl']:.5f} entropy={item['entropy']:.5f} "
                f"valid_top1={item['valid_turn_top1']:.4f} rank={item['valid_rank_fraction']:.4f}",
                flush=True,
            )
            if objective + 1e-5 < best_objective:
                best_objective = objective
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"grpo early_stop epoch={epoch} best_objective={best_objective:.5f}", flush=True)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_all = model(x_all).detach().cpu().tolist()
    all_probs = [sigmoid_prob(value) for value in logits_all]
    train_metrics = grouped_metrics(rows, all_probs, positive_mask, train_indices)
    valid_metrics = grouped_metrics(rows, all_probs, positive_mask, valid_indices)

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
    score_scale = 230.0
    blend = max(0.18, min(0.58, float(sft_artifact.get("blend", 0.32)) + 0.08))
    member = {
        "version": "v8_grpo",
        "model_type": "mlp_relu_candidate_ranker",
        "features": feature_names,
        "mean": means,
        "scale": scales,
        "layers": layers,
        "activation": "relu",
        "score_scale": score_scale,
    }
    metrics = {
        "rows": len(rows),
        "features": len(feature_names),
        "games": len(games),
        "train_groups": len(train_groups),
        "validation_groups": len(valid_groups),
        "train_turn_top1_positive_rate": train_metrics["top1"],
        "validation_turn_top1_positive_rate": valid_metrics["top1"],
        "train_positive_mean_rank_fraction": train_metrics["rank_fraction"],
        "validation_positive_mean_rank_fraction": valid_metrics["rank_fraction"],
        "blend": blend,
        "score_scale": score_scale,
        "device": str(device),
        "kl_weight": args.kl_weight,
        "entropy_weight": args.entropy_weight,
        "supervised_anchor": args.supervised_anchor,
    }
    artifact = {
        "version": "v8_grpo",
        "created_at": int(time.time()),
        "source_csv": str(data_path),
        "source_sft_artifact": str(sft_path),
        "model_type": "ensemble_mlp_relu_candidate_ranker",
        "members": [member],
        "blend": blend,
        "metrics": metrics,
    }

    export_dir = Path(args.export_dir)
    graph_dir = export_dir / "graphs"
    export_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "model_weights_v8_grpo.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "metrics_v8_grpo.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "training_history_v8_grpo.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    with (export_dir / "predictions_v8_grpo.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "reward", "label", "prediction", "selected", "counterfactual_positive", "split"])
        valid_set = set(valid_indices)
        for i, pred in enumerate(all_probs):
            writer.writerow([
                i,
                rewards[i],
                labels[i],
                pred,
                float(selected[i]),
                float(counterfactual[i]),
                "validation" if i in valid_set else "train",
            ])

    try:
        import matplotlib.pyplot as plt

        epochs = [item["epoch"] for item in history]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [item["valid_turn_top1"] for item in history], label="validation top1")
        plt.plot(epochs, [item["kl"] for item in history], label="KL")
        plt.plot(epochs, [item["entropy"] for item in history], label="entropy")
        plt.xlabel("epoch")
        plt.title("v8 GRPO diagnostics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "grpo_diagnostics_v8.png", dpi=150)
        plt.close()
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped graph generation.", flush=True)

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"Saved v8 GRPO artifact: {export_dir / 'model_weights_v8_grpo.json'}", flush=True)

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
            path_in_repo=GRPO_REMOTE_PREFIX,
            commit_message="Upload v8 GRPO Orbit Wars policy artifacts and graphs",
        )
        print(f"Uploaded {export_dir} to https://huggingface.co/{args.hf_repo_id}/tree/main/{GRPO_REMOTE_PREFIX}", flush=True)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
