import argparse
import time
from pathlib import Path

from src.utils import get_logger, load_config, set_seed, ensure_dirs
from src.ingestion import CMAPSSIngestion
from src.transformation import CMAPSSTransformer
from src.trainer import RULTrainer
from src.evaluator import RULEvaluator
from src.predictor import RULPredictor

logger = get_logger(__name__)


def run_pipeline(subset: str, model_type: str, skip_train: bool = False):
    config = load_config()
    set_seed(config["model"]["seed"])
    ensure_dirs()

    logger.info("="*50)
    logger.info("Starting RUL Pipeline | subset=%s | model=%s", subset, model_type)
    logger.info("="*50)

    # ── Step 1: Ingestion ──────────────────────────
    t0 = time.time()
    logger.info("Step 1: Ingestion")
    ing = CMAPSSIngestion()
    data = ing.load_subset(subset)
    ing.validate(data, subset)
    logger.info("Ingestion done in %.1fs", time.time() - t0)

    # ── Step 2: Transformation ─────────────────────
    t0 = time.time()
    logger.info("Step 2: Transformation")
    trans = CMAPSSTransformer()
    X_train, y_train, X_test, y_test = trans.run(
        data["train_df"], data["test_df"], data["rul_df"], subset
    )
    logger.info("Transformation done in %.1fs", time.time() - t0)
    logger.info("X_train%s y_train%s X_test%s y_test%s",
                X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # ── Step 3: Training ───────────────────────────
    if not skip_train:
        t0 = time.time()
        logger.info("Step 3: Training (%s)", model_type)
        trainer = RULTrainer()

        if model_type == "lstm":
            model, train_losses, val_losses = trainer.train(X_train, y_train, subset)
            trainer.plot_losses(train_losses, val_losses, subset)
        elif model_type == "xgboost":
            model = trainer.train_xgboost(X_train, y_train, subset)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        logger.info("Training done in %.1fs", time.time() - t0)
    else:
        logger.info("Step 3: Skipping training (--skip-train flag set)")

    # ── Step 4: Prediction ─────────────────────────
    t0 = time.time()
    logger.info("Step 4: Prediction")
    predictor = RULPredictor()
    y_pred = predictor.predict(X_test, subset, model_type=model_type)
    df_results = predictor.get_results_df(y_pred, y_test, subset)
    logger.info("Prediction done in %.1fs", time.time() - t0)

    # ── Step 5: Evaluation ─────────────────────────
    t0 = time.time()
    logger.info("Step 5: Evaluation")
    evaluator = RULEvaluator()
    results = evaluator.evaluate(y_test, y_pred, subset)
    evaluator.plot_predictions(y_test, y_pred, subset)
    evaluator.print_report(results, subset)
    logger.info("Evaluation done in %.1fs", time.time() - t0)

    logger.info("="*50)
    logger.info("Pipeline complete for subset=%s", subset)
    logger.info("="*50)

    return results


def run_all_subsets(model_type: str, skip_train: bool = False):
    config = load_config()
    subsets = config["dataset"]["subsets"]

    all_results = {}
    for subset in subsets:
        logger.info("\n")
        results = run_pipeline(subset, model_type, skip_train)
        all_results[subset] = results

    # print comparison table
    print("\n")
    print("="*60)
    print("FINAL COMPARISON — ALL SUBSETS")
    print("="*60)
    print(f"{'Subset':<10} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'NASA':>12}")
    print("-"*60)
    for subset, res in all_results.items():
        print(
            f"{subset:<10}"
            f"{res['rmse']:>10.2f}"
            f"{res['mae']:>10.2f}"
            f"{res['r2']:>10.4f}"
            f"{res['nasa_score']:>12.2f}"
        )
    print("="*60)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RUL Prediction on C-MAPSS")
    parser.add_argument(
        "--subset",
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004", "ALL"],
        help="Which subset to run"
    )
    parser.add_argument(
        "--model",
        default="lstm",
        choices=["lstm", "xgboost"],
        help="Model type to use"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use saved model"
    )
    args = parser.parse_args()

    if args.subset == "ALL":
        run_all_subsets(args.model, args.skip_train)
    else:
        run_pipeline(args.subset, args.model, args.skip_train)