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
SFT_REMOTE_PREFIX = "v19/sft"
GRPO_REMOTE_PREFIX = "v19/grpo"
PINNED_DATASET = "data/20260520_061012/candidates_v19.csv"

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
    "cf_margin_delta",
    "cf_prod_delta",
    "cf_planet_delta",
    "cf_survival",
    "cf_crash",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run v19 constrained GRPO-style policy improvement over Orbit Wars candidate groups."
    )
    parser.add_argument("--csv", default=os.environ.get("CANDIDATES_CSV", ""))
    parser.add_argument(
        "--data-remote-path",
        default=os.environ.get("V19_GRPO_DATA_REMOTE_PATH", os.environ.get("v19_GRPO_DATA_REMOTE_PATH", PINNED_DATASET)),
        help="Optional exact Hugging Face repo path for candidates_v19.csv. Defaults to the pinned v19 dataset.",
    )
    parser.add_argument(
        "--prefer-local-data",
        action="store_true",
        help="Use a local candidates_v19.csv before trying Hugging Face. Default is Hugging Face.",
    )
    parser.add_argument(
        "--sft-artifact",
        default=os.environ.get("V19_SFT_ARTIFACT", os.environ.get("v19_SFT_ARTIFACT", "")),
        help="Optional local SFT JSON override. By default GRPO downloads the SFT artifact from Hugging Face.",
    )
    parser.add_argument(
        "--sft-remote-path",
        default=os.environ.get("V19_SFT_REMOTE_PATH", os.environ.get("v19_SFT_REMOTE_PATH", f"{SFT_REMOTE_PREFIX}/model_weights_v19_sft.json")),
        help="Hugging Face repo path for the SFT artifact used by GRPO.",
    )
    parser.add_argument(
        "--prefer-local-sft",
        action="store_true",
        help="Use notebooks/v19/exports/sft/model_weights_v19_sft.json before trying Hugging Face.",
    )
    parser.add_argument("--export-dir", default="notebooks/v19/exports/grpo")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("V19_GRPO_EPOCHS", os.environ.get("v19_GRPO_EPOCHS", "50"))))
    parser.add_argument("--batch-groups", type=int, default=int(os.environ.get("V19_GRPO_BATCH_GROUPS", os.environ.get("v19_GRPO_BATCH_GROUPS", "192"))))
    parser.add_argument("--samples-per-group", type=int, default=int(os.environ.get("V19_GRPO_SAMPLES", os.environ.get("v19_GRPO_SAMPLES", "12"))))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("V19_GRPO_TEMPERATURE", os.environ.get("v19_GRPO_TEMPERATURE", "0.85"))))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("V19_GRPO_LR", os.environ.get("v19_GRPO_LR", "0.00035"))))
    parser.add_argument("--weight-decay", type=float, default=float(os.environ.get("V19_GRPO_WEIGHT_DECAY", os.environ.get("v19_GRPO_WEIGHT_DECAY", "0.00018"))))
    parser.add_argument("--kl-weight", type=float, default=float(os.environ.get("V19_GRPO_KL_WEIGHT", os.environ.get("v19_GRPO_KL_WEIGHT", "0.15"))))
    parser.add_argument("--entropy-weight", type=float, default=float(os.environ.get("V19_GRPO_ENTROPY_WEIGHT", os.environ.get("v19_GRPO_ENTROPY_WEIGHT", "0.018"))))
    parser.add_argument("--supervised-anchor", type=float, default=float(os.environ.get("V19_GRPO_SUPERVISED_ANCHOR", os.environ.get("v19_GRPO_SUPERVISED_ANCHOR", "0.18"))))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("V19_GRPO_PATIENCE", os.environ.get("v19_GRPO_PATIENCE", "36"))))
    parser.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("V19_GRPO_CHECKPOINT_EVERY", os.environ.get("v19_GRPO_CHECKPOINT_EVERY", "30"))))
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default=os.environ.get("V19_DEVICE", os.environ.get("v19_DEVICE", "cuda")),
        help="Training device. Defaults to CUDA for Kaggle 2*T4 runs.",
    )
    parser.add_argument("--seed", type=int, default=1819)
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
        raise RuntimeError("HF_TOKEN is required to download candidates_v19.csv from Hugging Face.")
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
                if name.startswith("data/") and name.endswith("/candidates_v19.csv")
            ],
            reverse=True,
        )
        if not remote_csvs:
            raise FileNotFoundError("No data/*/candidates_v19.csv found in Hugging Face repo.")
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
        candidates.extend(root.glob("*/candidates_v19.csv"))
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

    local = Path("notebooks/v19/exports/sft/model_weights_v19_sft.json")
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


def clamp(value, low, high):
    return max(low, min(high, value))


def failure_exposure(row):
    return (
        row_float(row, "failure_overcommit", 0.0) * 1.25
        + row_float(row, "failure_missed_tactical", 0.0) * 0.45
        + row_float(row, "failure_missed_comet", 0.0) * 0.45
        + row_float(row, "failure_slow_expansion", 0.0) * 0.35
        + row_float(row, "kind_crash", 0.0) * 1.50
        + max(0.0, row_float(row, "ship_cost_fraction", 0.0) - 0.62) * 0.55
    )


def candidate_reward_components(row, label, selected, counterfactual):
    """Reward-resistant offline proxy.

    This reward relies heavily on causal counterfactual rollout deltas.
    The selected bonus is kept small.
    """
    components = {
        "label_signal": (label - 0.5) * 0.55,
        "advantage_delta_15": clamp(row_float(row, "future_advantage_delta_15", 0.0) / 85.0, -1.25, 1.25) * 0.45,
        "advantage_delta_30": clamp(row_float(row, "future_advantage_delta_30", 0.0) / 140.0, -0.90, 0.90) * 0.25,
        "production_delta": clamp(row_float(row, "future_production_delta_15", 0.0) / 9.0, -0.75, 0.75) * 0.25,
        "planet_delta": clamp(row_float(row, "future_planet_delta_15", 0.0) / 3.0, -0.75, 0.75) * 0.25,
        "final_result": clamp(row_float(row, "game_result", 0.0), -1.0, 1.0) * 0.16,
        "final_margin": clamp(row_float(row, "reward_margin", 0.0) / 800.0, -1.0, 1.0) * 0.12,
        "selected_anchor": 0.05 if selected else 0.0,
        "counterfactual_bonus": 0.22 if counterfactual else 0.0,
        
        "cf_margin": clamp(row_float(row, "cf_margin_delta", 0.0) / 85.0, -1.5, 1.5) * 1.5,
        "cf_prod": clamp(row_float(row, "cf_prod_delta", 0.0) / 9.0, -1.0, 1.0) * 0.75,
        "cf_planet": clamp(row_float(row, "cf_planet_delta", 0.0) / 3.0, -1.0, 1.0) * 0.5,
        "cf_survival_bonus": (row_float(row, "cf_survival", 1.0) - 1.0) * 2.0,
        "cf_crash_penalty": -row_float(row, "cf_crash", 0.0) * 2.0,

        "overcommit_penalty": -row_float(row, "failure_overcommit", 0.0) * 1.15,
        "missed_tactical_penalty": -row_float(row, "failure_missed_tactical", 0.0) * 0.45,
        "missed_comet_penalty": -row_float(row, "failure_missed_comet", 0.0) * 0.45,
        "slow_expansion_penalty": -row_float(row, "failure_slow_expansion", 0.0) * 0.35,
        "crash_penalty": -row_float(row, "kind_crash", 0.0) * 1.60,
        "cost_penalty": -max(0.0, row_float(row, "ship_cost_fraction", 0.0) - 0.62) * 0.65,
        "eta_penalty": -max(0.0, row_float(row, "eta_fraction_remaining", 0.0) - 0.45) * 0.22,
    }
    if row_float(row, "kind_defend", 0.0) >= 0.5 and row_float(row, "enemy_pressure", 0.0) > 0.0:
        components["pressure_defense_bonus"] = clamp(row_float(row, "enemy_pressure", 0.0) / 65.0, 0.0, 0.35)
    return components


def candidate_reward(row, label, selected, counterfactual):
    return clamp(sum(candidate_reward_components(row, label, selected, counterfactual).values()), -4.0, 4.0)


def target_distribution(rows, rewards, group):
    values = [math.exp(max(-6.0, min(6.0, rewards[i]))) for i in group]
    total = sum(values)
    return [value / total for value in values] if total > 0.0 else [1.0 / len(group)] * len(group)


def grouped_metrics(rows, scores, positive_mask, indices, rewards=None):
    groups = build_groups(rows, indices)
    hits = 0
    total = 0
    ranks = []
    reward_gaps = []
    top_rewards = []
    failure_values = []
    for group in groups:
        positives = [i for i in group if positive_mask[i]]
        if not positives:
            continue
        ordered = sorted(group, key=lambda i: scores[i], reverse=True)
        total += 1
        if positive_mask[ordered[0]]:
            hits += 1
        ranks.append(min(ordered.index(i) + 1 for i in positives) / max(1, len(ordered)))
        top = ordered[0]
        failure_values.append(failure_exposure(rows[top]))
        if rewards is not None:
            ordered_rewards = sorted(rewards[i] for i in group)
            median_reward = ordered_rewards[len(ordered_rewards) // 2]
            top_rewards.append(rewards[top])
            reward_gaps.append(rewards[top] - median_reward)
    return {
        "top1": hits / total if total else 0.0,
        "rank_fraction": sum(ranks) / len(ranks) if ranks else 1.0,
        "turns": total,
        "top_reward": sum(top_rewards) / len(top_rewards) if top_rewards else 0.0,
        "reward_gap": sum(reward_gaps) / len(reward_gaps) if reward_gaps else 0.0,
        "failure_exposure": sum(failure_values) / len(failure_values) if failure_values else 0.0,
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


def choose_device(torch, args):
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), None
    if requested == "cuda":
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
    return torch.device("cpu"), None


def optimizer_step(optimizer, xm):
    if xm is None:
        optimizer.step()
    else:
        xm.optimizer_step(optimizer, barrier=False)
        xm.mark_step()


def layers_from_model(torch, nn, model):
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


def save_grpo_checkpoint(args, torch, nn, model, epoch, item, history, feature_names, means, scales, sft_path, blend):
    checkpoint_dir = Path(args.export_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    member = {
        "version": "v19_grpo",
        "model_type": "mlp_relu_candidate_ranker",
        "features": feature_names,
        "mean": means,
        "scale": scales,
        "layers": layers_from_model(torch, nn, model),
        "activation": "relu",
        "score_scale": 235.0,
    }
    payload = {
        "version": "v19_grpo_checkpoint",
        "created_at": int(time.time()),
        "epoch": epoch,
        "latest_metrics": item,
        "history": history,
        "source_sft_artifact": str(sft_path),
        "model_type": "ensemble_mlp_relu_candidate_ranker",
        "members": [member],
        "blend": blend,
    }
    path = checkpoint_dir / f"grpo_epoch_{epoch:03d}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved v19 GRPO checkpoint: {path}", flush=True)
    maybe_upload_file(
        args,
        path,
        f"{GRPO_REMOTE_PREFIX}/checkpoints/{path.name}",
        f"Upload v19 GRPO checkpoint epoch {epoch}",
    )


def sigmoid_prob(value):
    value = max(-50.0, min(50.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def train(args):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for v19 GRPO training. Install torch or use Kaggle/Colab.") from exc

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
    device, xm = choose_device(torch, args)
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
                "data_remote_path": args.data_remote_path or "latest data/*/candidates_v19.csv",
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
                "checkpoint_every": args.checkpoint_every,
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
    checkpoint_blend = max(0.16, min(0.54, float(sft_artifact.get("blend", 0.32)) + 0.06))
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
            optimizer_step(optimizer, xm)
            total_loss += float(loss_acc.detach().cpu())
            total_policy += policy_acc / max(1, len(batch_groups))
            total_kl += kl_acc / max(1, len(batch_groups))
            total_entropy += entropy_acc / max(1, len(batch_groups))
            batches += 1
        scheduler.step()

        if epoch >= 1:
            model.eval()
            with torch.no_grad():
                logits_all = model(x_all).detach().cpu().tolist()
            scores = [sigmoid_prob(value) for value in logits_all]
            valid_metrics = grouped_metrics(rows, scores, positive_mask, valid_indices, rewards)
            reward_hack_risk = (
                valid_metrics["failure_exposure"] * 0.16
                + max(0.0, -valid_metrics["reward_gap"]) * 0.06
                + max(0.0, (total_kl / max(1, batches)) - 0.18) * 0.04
            )
            objective = (
                (1.0 - valid_metrics["top1"])
                + 0.35 * valid_metrics["rank_fraction"]
                + 0.02 * (total_kl / max(1, batches))
                + reward_hack_risk
            )
            item = {
                "epoch": epoch,
                "loss": total_loss / max(1, batches),
                "policy_loss": total_policy / max(1, batches),
                "kl": total_kl / max(1, batches),
                "entropy": total_entropy / max(1, batches),
                "valid_turn_top1": valid_metrics["top1"],
                "valid_rank_fraction": valid_metrics["rank_fraction"],
                "valid_reward_gap": valid_metrics["reward_gap"],
                "valid_top_reward": valid_metrics["top_reward"],
                "valid_failure_exposure": valid_metrics["failure_exposure"],
                "reward_hack_risk": reward_hack_risk,
                "objective": objective,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(item)
            print(
                f"grpo epoch={epoch:03d} loss={item['loss']:.5f} "
                f"policy={item['policy_loss']:.5f} kl={item['kl']:.5f} entropy={item['entropy']:.5f} "
                f"valid_top1={item['valid_turn_top1']:.4f} rank={item['valid_rank_fraction']:.4f} "
                f"reward_gap={item['valid_reward_gap']:.4f} fail={item['valid_failure_exposure']:.4f} "
                f"hack_risk={item['reward_hack_risk']:.4f}",
                flush=True,
            )
            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                save_grpo_checkpoint(
                    args,
                    torch,
                    nn,
                    model,
                    epoch,
                    item,
                    history,
                    feature_names,
                    means,
                    scales,
                    sft_path,
                    checkpoint_blend,
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
    train_metrics = grouped_metrics(rows, all_probs, positive_mask, train_indices, rewards)
    valid_metrics = grouped_metrics(rows, all_probs, positive_mask, valid_indices, rewards)

    score_scale = 235.0
    blend = checkpoint_blend
    member = {
        "version": "v19_grpo",
        "model_type": "mlp_relu_candidate_ranker",
        "features": feature_names,
        "mean": means,
        "scale": scales,
        "layers": layers_from_model(torch, nn, model),
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
        "train_reward_gap": train_metrics["reward_gap"],
        "validation_reward_gap": valid_metrics["reward_gap"],
        "train_top_reward": train_metrics["top_reward"],
        "validation_top_reward": valid_metrics["top_reward"],
        "train_failure_exposure": train_metrics["failure_exposure"],
        "validation_failure_exposure": valid_metrics["failure_exposure"],
        "blend": blend,
        "score_scale": score_scale,
        "device": str(device),
        "kl_weight": args.kl_weight,
        "entropy_weight": args.entropy_weight,
        "supervised_anchor": args.supervised_anchor,
        "reward_design": "bounded_multi_component_with_failure_penalties_and_kl_gate",
    }
    artifact = {
        "version": "v19_grpo",
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
    (export_dir / "model_weights_v19_grpo.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "metrics_v19_grpo.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (export_dir / "training_history_v19_grpo.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
    with (export_dir / "predictions_v19_grpo.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_index", "reward", "failure_exposure", "label", "prediction", "selected", "counterfactual_positive", "split"])
        valid_set = set(valid_indices)
        for i, pred in enumerate(all_probs):
            writer.writerow([
                i,
                rewards[i],
                failure_exposure(rows[i]),
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
        plt.title("v19 GRPO diagnostics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(graph_dir / "grpo_diagnostics_v19.png", dpi=150)
        plt.close()
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipped graph generation.", flush=True)

    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"Saved v19 GRPO artifact: {export_dir / 'model_weights_v19_grpo.json'}", flush=True)

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
            commit_message="Upload v19 GRPO Orbit Wars policy artifacts and graphs",
        )
        print(f"Uploaded {export_dir} to https://huggingface.co/{args.hf_repo_id}/tree/main/{GRPO_REMOTE_PREFIX}", flush=True)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
