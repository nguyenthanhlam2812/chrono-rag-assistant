import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


logger = setup_logger("train_ml_classifier")


def ensure_supported_runtime() -> None:
    version = sys.version_info
    if version.major != 3 or version.minor < 10 or version.minor > 12:
        executable = Path(sys.executable)
        message = (
            "Unsupported Python runtime for ML training: "
            f"{version.major}.{version.minor}.{version.micro} at {executable}\n"
            "Use Python 3.10-3.12. Activate the project venv first:\n"
            "  .venv\\Scripts\\Activate.ps1   (Windows PowerShell)\n"
            "  source .venv/bin/activate     (Linux / macOS)\n"
            "Then re-run: python scripts/04_train_ml_classifier.py"
        )
        raise SystemExit(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ChronoRAG TF-IDF ML baseline classifiers."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/labeled/labeled_sentences.csv",
        help="Path to final labeled CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "saved_models",
        help="Directory for saved model .pkl files.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=PROJECT_ROOT / "data/eval",
        help="Directory for metrics JSON and split CSV.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/figures",
        help="Directory for confusion matrix plots.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=120000,
        help="Total TF-IDF feature budget across word and char features.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "linearsvm", "sgd_log"],
        choices=["logreg", "linearsvm", "sgd_log"],
        help="Baseline models to train.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override models.random_seed from configs/config.yaml.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Override train split ratio. Must be combined with --val-ratio and --test-ratio and sum to 1.0.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Override val split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Override test split ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_supported_runtime()

    # Import heavy scientific libraries only after runtime validation. This
    # avoids native SciPy/sklearn crashes when Windows resolves `python` to an
    # incompatible global interpreter.
    from src.models.ml_baseline import SplitConfig, train_ml_baselines
    from src.utils.config import load_config

    config = load_config()
    split_config = SplitConfig.from_config_dict(config)

    # CLI flags override config file values when supplied.
    cli_ratios = {
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    if any(v is not None for v in cli_ratios.values()):
        merged = {
            name: (cli_ratios[name] if cli_ratios[name] is not None else getattr(split_config, name))
            for name in ("train_ratio", "val_ratio", "test_ratio")
        }
        split_config = SplitConfig(
            random_seed=split_config.random_seed,
            **merged,
        )

    if args.random_seed is not None:
        split_config = SplitConfig(
            random_seed=args.random_seed,
            train_ratio=split_config.train_ratio,
            val_ratio=split_config.val_ratio,
            test_ratio=split_config.test_ratio,
        )

    logger.info("Starting Sprint 4 ML baseline training")
    logger.info("Input CSV: %s", args.input)
    logger.info("Models: %s", ", ".join(args.models))
    logger.info("Max TF-IDF features: %s", args.max_features)
    logger.info(
        "Split ratios: train=%.2f val=%.2f test=%.2f (seed=%d)",
        split_config.train_ratio,
        split_config.val_ratio,
        split_config.test_ratio,
        split_config.random_seed,
    )

    metrics = train_ml_baselines(
        input_csv=args.input,
        output_dir=args.output_dir,
        eval_dir=args.eval_dir,
        figures_dir=args.figures_dir,
        random_seed=split_config.random_seed,
        max_features=args.max_features,
        model_names=args.models,
        split_config=split_config,
    )

    logger.info("Saved metrics: %s", metrics["metrics_path"])
    logger.info("Saved split file: %s", metrics["split_path"])
    for model_name, model_metrics in metrics["models"].items():
        binary_test = model_metrics["binary"]["test"]
        type_test = model_metrics["event_type"]["test"]
        logger.info(
            "%s binary test macro-F1: %.4f | weighted-F1: %.4f",
            model_name,
            binary_test["f1_macro"],
            binary_test["f1_weighted"],
        )
        if "f1_macro" in type_test:
            logger.info(
                "%s event-type test macro-F1: %.4f | weighted-F1: %.4f",
                model_name,
                type_test["f1_macro"],
                type_test["f1_weighted"],
            )


if __name__ == "__main__":
    main()
