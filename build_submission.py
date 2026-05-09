import argparse
import json
import pprint
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a single-file Orbit Wars Kaggle submission with embedded model weights."
    )
    parser.add_argument(
        "--source",
        default="main.py",
        help="Source agent file. Defaults to main.py.",
    )
    parser.add_argument(
        "--weights",
        default="notebooks/v6/exports/model_weights_v6.json",
        help="Model JSON to embed. Defaults to the v6 notebook export.",
    )
    parser.add_argument(
        "--output",
        default="models/v6_kaggle/main.py",
        help="Generated submission file. Keep this under gitignored models/.",
    )
    return parser.parse_args()


def build_submission(source_path, weights_path, output_path):
    source = Path(source_path).read_text(encoding="utf-8")
    model = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    embedded = pprint.pformat(model, width=100, sort_dicts=True)

    if "USE_MODEL_SCORER = False" not in source:
        raise RuntimeError("Could not find USE_MODEL_SCORER = False in source agent.")
    if "MODEL_WEIGHTS = None" not in source:
        raise RuntimeError("Could not find MODEL_WEIGHTS = None in source agent.")

    built = source.replace("USE_MODEL_SCORER = False", "USE_MODEL_SCORER = True", 1)
    built = built.replace("MODEL_WEIGHTS = None", f"MODEL_WEIGHTS = {embedded}", 1)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(built, encoding="utf-8")
    return output


def main():
    args = parse_args()
    output = build_submission(args.source, args.weights, args.output)
    print(f"Built submission agent: {output}")


if __name__ == "__main__":
    main()
