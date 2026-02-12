from __future__ import annotations

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.ml.io import load_df_from_cfg
from src.ml.pipeline import run_training

# Decorator that tells Hydra where to find and apply the config
@hydra.main(
    version_base=None, 
    config_path="../conf",  # directory with structured configs
    config_name="config"    # configuration file to run
    )   
def main(cfg: DictConfig) -> None:
    # Hydra changes the working directory to a run dir by default.
    # Use  original CWD as project root.
    project_root = Path(hydra.utils.get_original_cwd())

    print(OmegaConf.to_yaml(cfg))

    df = load_df_from_cfg(cfg.data, project_root)

    outdir = project_root / str(cfg.outdir)
    results = run_training(cfg, df = df, outdir = outdir)


    print("Best CV AUC:", results["best_cv_score"])
    print("Best hyperparameters:", results["best_params"])
    print(f"Optimal threshold (max F1): {results['threshold_best']:.3f}")
    print("Test ROC-AUC:", results["test_roc_auc"])
    print("\nConfusion matrix:\n", results["eval_opt"]["confusion_matrix"])
    print("\nClassification report:\n", results["eval_opt"]["classification_report"])

if __name__ == "__main__":
    main()