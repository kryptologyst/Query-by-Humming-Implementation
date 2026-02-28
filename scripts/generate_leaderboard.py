"""
Generate leaderboard for Query by Humming evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.models.query_matcher import QueryByHummingMatcher
from src.metrics.evaluation import QueryByHummingMetrics
from src.metrics.fairness import FairnessEvaluator, run_fairness_evaluation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    return OmegaConf.load(config_path)


def evaluate_model_variants(
    configs: List[DictConfig],
    output_dir: Path
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple model variants.
    
    Args:
        configs: List of model configurations.
        output_dir: Output directory for results.
        
    Returns:
        Dictionary with results for each model.
    """
    results = {}
    
    for i, config in enumerate(configs):
        model_name = config.get("experiment", {}).get("name", f"model_{i}")
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Initialize matcher
            matcher = QueryByHummingMatcher(config)
            
            # Load reference database
            synthetic_dir = Path(config.paths.data_dir) / "synthetic"
            meta_path = synthetic_dir / "meta.csv"
            
            if not meta_path.exists():
                logger.warning(f"No synthetic dataset found for {model_name}")
                continue
            
            import pandas as pd
            meta_df = pd.read_csv(meta_path)
            reference_df = meta_df[meta_df["split"] == "reference"]
            
            # Load reference songs
            for _, row in reference_df.iterrows():
                matcher.add_reference_song(
                    audio_path=row["audio_path"],
                    title=row["title"],
                    artist=row["artist"],
                    metadata={"genre": row.get("genre", "unknown")}
                )
            
            # Evaluate model
            query_df = meta_df[meta_df["split"] == "query"]
            metrics_calculator = QueryByHummingMetrics(k_values=config.eval.k_values)
            
            query_results = []
            
            for _, query_row in query_df.iterrows():
                try:
                    # Search for matches
                    matches = matcher.search(
                        query_audio=query_row["audio_path"],
                        top_k=config.eval.k_values[-1],
                        threshold=0.0
                    )
                    
                    # Find ground truth
                    ground_truth_indices = []
                    for j, ref_song in enumerate(matcher.reference_database):
                        if ref_song["title"] == query_row["reference_title"]:
                            ground_truth_indices.append(j)
                            break
                    
                    query_results.append((matches, ground_truth_indices))
                    
                except Exception as e:
                    logger.error(f"Error processing query {query_row['query_id']}: {e}")
                    continue
            
            # Compute average metrics
            if query_results:
                avg_metrics = metrics_calculator.evaluate_batch(query_results)
                results[model_name] = avg_metrics
                
                # Run fairness evaluation
                try:
                    fairness_report = run_fairness_evaluation(matcher, str(meta_path))
                    results[f"{model_name}_fairness"] = fairness_report
                except Exception as e:
                    logger.error(f"Error in fairness evaluation for {model_name}: {e}")
                
                logger.info(f"Completed evaluation for {model_name}")
            else:
                logger.warning(f"No valid results for {model_name}")
                
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            continue
    
    return results


def create_model_variants(base_config: DictConfig) -> List[DictConfig]:
    """
    Create model variants for evaluation.
    
    Args:
        base_config: Base configuration.
        
    Returns:
        List of model configurations.
    """
    variants = []
    
    # Baseline DTW model
    baseline_config = base_config.copy()
    baseline_config.experiment.name = "dtw_baseline"
    variants.append(baseline_config)
    
    # DTW with different features
    feature_types = ["mfcc", "chroma", "spectral", "pitch"]
    for feature_type in feature_types:
        config = base_config.copy()
        config.experiment.name = f"dtw_{feature_type}"
        config.model.features.type = feature_type
        variants.append(config)
    
    # DTW with different distance metrics
    distance_metrics = ["euclidean", "cosine", "manhattan"]
    for metric in distance_metrics:
        config = base_config.copy()
        config.experiment.name = f"dtw_{metric}"
        config.model.dtw.distance_metric = metric
        variants.append(config)
    
    # DTW with different window sizes
    window_sizes = [5, 10, 20, 50]
    for window_size in window_sizes:
        config = base_config.copy()
        config.experiment.name = f"dtw_window_{window_size}"
        config.model.dtw.window_size = window_size
        variants.append(config)
    
    return variants


def generate_leaderboard_table(
    results: Dict[str, Dict[str, float]],
    output_dir: Path
) -> None:
    """
    Generate leaderboard table.
    
    Args:
        results: Evaluation results.
        output_dir: Output directory.
    """
    # Filter out fairness results for main leaderboard
    main_results = {k: v for k, v in results.items() if not k.endswith("_fairness")}
    
    if not main_results:
        logger.warning("No main results to generate leaderboard")
        return
    
    # Create DataFrame
    df = pd.DataFrame(main_results).T
    
    # Sort by MAP@1 (primary metric)
    if "map_at_1" in df.columns:
        df = df.sort_values("map_at_1", ascending=False)
    
    # Select key metrics for display
    key_metrics = [
        "map_at_1", "map_at_5", "map_at_10",
        "recall_at_1", "recall_at_5", "recall_at_10",
        "precision_at_1", "precision_at_5", "precision_at_10",
        "mean_distance", "mean_confidence"
    ]
    
    display_df = df[[col for col in key_metrics if col in df.columns]]
    
    # Save CSV
    display_df.to_csv(output_dir / "leaderboard.csv")
    
    # Create markdown table
    markdown_table = display_df.to_markdown(floatfmt=".4f")
    
    with open(output_dir / "leaderboard.md", "w") as f:
        f.write("# Query by Humming Leaderboard\n\n")
        f.write("## Main Results\n\n")
        f.write(markdown_table)
        f.write("\n\n")
        
        # Add fairness results
        fairness_results = {k: v for k, v in results.items() if k.endswith("_fairness")}
        if fairness_results:
            f.write("## Fairness Analysis\n\n")
            for model_name, fairness_data in fairness_results.items():
                base_name = model_name.replace("_fairness", "")
                f.write(f"### {base_name}\n\n")
                
                if "overall_fairness" in fairness_data:
                    overall = fairness_data["overall_fairness"]
                    f.write(f"- **Overall Fairness Score**: {overall.get('fairness_score', 0):.3f}\n")
                    f.write(f"- **Average Fairness Gap**: {overall.get('average_fairness_gap', 0):.3f}\n")
                    f.write(f"- **Max Fairness Gap**: {overall.get('max_fairness_gap', 0):.3f}\n\n")
                
                # Genre fairness
                if "genre_fairness" in fairness_data and "fairness_analysis" in fairness_data["genre_fairness"]:
                    genre_fairness = fairness_data["genre_fairness"]["fairness_analysis"]
                    f.write(f"- **Genre Fairness Gap**: {genre_fairness.get('fairness_gap', 0):.3f}\n")
                
                # Artist fairness
                if "artist_fairness" in fairness_data and "fairness_analysis" in fairness_data["artist_fairness"]:
                    artist_fairness = fairness_data["artist_fairness"]["fairness_analysis"]
                    f.write(f"- **Artist Fairness Gap**: {artist_fairness.get('fairness_gap', 0):.3f}\n")
                
                f.write("\n")
    
    logger.info(f"Leaderboard saved to {output_dir}")


def generate_ablation_study(
    results: Dict[str, Dict[str, float]],
    output_dir: Path
) -> None:
    """
    Generate ablation study results.
    
    Args:
        results: Evaluation results.
        output_dir: Output directory.
    """
    # Filter main results
    main_results = {k: v for k, v in results.items() if not k.endswith("_fairness")}
    
    if not main_results:
        return
    
    df = pd.DataFrame(main_results).T
    
    # Feature ablation
    feature_results = {}
    for model_name in df.index:
        if model_name.startswith("dtw_"):
            parts = model_name.split("_")
            if len(parts) >= 3:
                feature = parts[1]
                if feature in ["mfcc", "chroma", "spectral", "pitch"]:
                    feature_results[feature] = df.loc[model_name, "map_at_1"]
    
    if feature_results:
        feature_df = pd.DataFrame(list(feature_results.items()), columns=["Feature", "MAP@1"])
        feature_df = feature_df.sort_values("MAP@1", ascending=False)
        feature_df.to_csv(output_dir / "feature_ablation.csv", index=False)
    
    # Distance metric ablation
    metric_results = {}
    for model_name in df.index:
        if model_name.startswith("dtw_"):
            parts = model_name.split("_")
            if len(parts) >= 2:
                metric = parts[1]
                if metric in ["euclidean", "cosine", "manhattan"]:
                    metric_results[metric] = df.loc[model_name, "map_at_1"]
    
    if metric_results:
        metric_df = pd.DataFrame(list(metric_results.items()), columns=["Distance_Metric", "MAP@1"])
        metric_df = metric_df.sort_values("MAP@1", ascending=False)
        metric_df.to_csv(output_dir / "distance_metric_ablation.csv", index=False)
    
    # Window size ablation
    window_results = {}
    for model_name in df.index:
        if model_name.startswith("dtw_window_"):
            window_size = model_name.split("_")[-1]
            window_results[int(window_size)] = df.loc[model_name, "map_at_1"]
    
    if window_results:
        window_df = pd.DataFrame(list(window_results.items()), columns=["Window_Size", "MAP@1"])
        window_df = window_df.sort_values("Window_Size")
        window_df.to_csv(output_dir / "window_size_ablation.csv", index=False)
    
    logger.info("Ablation study results saved")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate Query by Humming leaderboard")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets",
        help="Output directory for results"
    )
    parser.add_argument(
        "--variants",
        action="store_true",
        help="Generate model variants for comprehensive evaluation"
    )
    
    args = parser.parse_args()
    
    # Load base configuration
    base_config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.variants:
        # Generate model variants
        logger.info("Generating model variants...")
        variants = create_model_variants(base_config)
        
        # Evaluate all variants
        results = evaluate_model_variants(variants, output_dir)
        
        # Generate ablation study
        generate_ablation_study(results, output_dir)
    else:
        # Single model evaluation
        logger.info("Evaluating single model...")
        matcher = QueryByHummingMatcher(base_config)
        
        # Load reference database
        synthetic_dir = Path(base_config.paths.data_dir) / "synthetic"
        meta_path = synthetic_dir / "meta.csv"
        
        if meta_path.exists():
            import pandas as pd
            meta_df = pd.read_csv(meta_path)
            reference_df = meta_df[meta_df["split"] == "reference"]
            
            for _, row in reference_df.iterrows():
                matcher.add_reference_song(
                    audio_path=row["audio_path"],
                    title=row["title"],
                    artist=row["artist"],
                    metadata={"genre": row.get("genre", "unknown")}
                )
            
            # Evaluate model
            query_df = meta_df[meta_df["split"] == "query"]
            metrics_calculator = QueryByHummingMetrics(k_values=base_config.eval.k_values)
            
            query_results = []
            
            for _, query_row in query_df.iterrows():
                try:
                    matches = matcher.search(
                        query_audio=query_row["audio_path"],
                        top_k=base_config.eval.k_values[-1],
                        threshold=0.0
                    )
                    
                    ground_truth_indices = []
                    for j, ref_song in enumerate(matcher.reference_database):
                        if ref_song["title"] == query_row["reference_title"]:
                            ground_truth_indices.append(j)
                            break
                    
                    query_results.append((matches, ground_truth_indices))
                    
                except Exception as e:
                    logger.error(f"Error processing query {query_row['query_id']}: {e}")
                    continue
            
            if query_results:
                avg_metrics = metrics_calculator.evaluate_batch(query_results)
                results = {"baseline": avg_metrics}
                
                # Run fairness evaluation
                try:
                    fairness_report = run_fairness_evaluation(matcher, str(meta_path))
                    results["baseline_fairness"] = fairness_report
                except Exception as e:
                    logger.error(f"Error in fairness evaluation: {e}")
            else:
                logger.warning("No valid results")
                results = {}
        else:
            logger.error("No synthetic dataset found")
            results = {}
    
    # Generate leaderboard
    generate_leaderboard_table(results, output_dir)
    
    # Save raw results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Leaderboard generation completed")


if __name__ == "__main__":
    main()
