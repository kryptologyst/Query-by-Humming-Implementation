"""
Evaluation script for Query by Humming models.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.models.query_matcher import QueryByHummingMatcher
from src.metrics.evaluation import QueryByHummingMetrics
from src.utils.device import get_device, set_seed

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


def load_test_queries(config: DictConfig) -> pd.DataFrame:
    """
    Load test queries from dataset.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        DataFrame with test queries.
    """
    synthetic_dir = Path(config.paths.data_dir) / "synthetic"
    meta_path = synthetic_dir / "meta.csv"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    meta_df = pd.read_csv(meta_path)
    query_df = meta_df[meta_df["split"] == "query"]
    
    logger.info(f"Loaded {len(query_df)} test queries")
    return query_df


def evaluate_single_query(
    matcher: QueryByHummingMatcher,
    query_row: pd.Series,
    k_values: List[int]
) -> Dict[str, float]:
    """
    Evaluate a single query.
    
    Args:
        matcher: Query by Humming matcher.
        query_row: Query data row.
        k_values: List of k values for evaluation.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    try:
        # Load query audio
        query_audio, _ = matcher.load_audio(query_row["audio_path"])
        
        # Search for matches
        matches = matcher.search(
            query_audio=query_audio,
            top_k=max(k_values),
            threshold=0.0  # No threshold for evaluation
        )
        
        # Ground truth (reference song index)
        ground_truth_indices = []
        
        # Find ground truth index in reference database
        for i, ref_song in enumerate(matcher.reference_database):
            if ref_song["title"] == query_row["reference_title"]:
                ground_truth_indices.append(i)
                break
        
        # Compute metrics
        metrics_calculator = QueryByHummingMetrics(k_values=k_values)
        metrics = metrics_calculator.evaluate_query(matches, ground_truth_indices)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error processing query {query_row['query_id']}: {e}")
        return {}


def run_evaluation(
    config: DictConfig,
    checkpoint_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Run evaluation on test set.
    
    Args:
        config: Configuration dictionary.
        checkpoint_path: Path to model checkpoint.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")
    
    # Initialize matcher
    matcher = QueryByHummingMatcher(config)
    
    # Load reference database
    if checkpoint_path:
        matcher.load_database(checkpoint_path)
        logger.info(f"Loaded database from {checkpoint_path}")
    else:
        # Load from synthetic dataset
        synthetic_dir = Path(config.paths.data_dir) / "synthetic"
        meta_path = synthetic_dir / "meta.csv"
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        meta_df = pd.read_csv(meta_path)
        reference_df = meta_df[meta_df["split"] == "reference"]
        
        logger.info(f"Loading {len(reference_df)} reference songs...")
        
        for _, row in reference_df.iterrows():
            matcher.add_reference_song(
                audio_path=row["audio_path"],
                title=row["title"],
                artist=row["artist"],
                metadata={"genre": row.get("genre", "unknown")}
            )
        
        logger.info("Reference database loaded successfully")
    
    # Load test queries
    test_queries = load_test_queries(config)
    
    # Initialize metrics calculator
    metrics_calculator = QueryByHummingMetrics(
        k_values=config.eval.k_values
    )
    
    # Evaluate each query
    all_metrics = []
    
    for _, query_row in test_queries.iterrows():
        metrics = evaluate_single_query(
            matcher, query_row, config.eval.k_values
        )
        if metrics:
            all_metrics.append(metrics)
    
    # Compute average metrics
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = float(sum(values) / len(values)) if values else 0.0
        
        return avg_metrics
    else:
        logger.warning("No valid queries processed")
        return {}


def generate_leaderboard(
    results: Dict[str, Dict[str, float]],
    output_path: Path
) -> None:
    """
    Generate a leaderboard from evaluation results.
    
    Args:
        results: Dictionary with model results.
        output_path: Path to save leaderboard.
    """
    if not results:
        logger.warning("No results to generate leaderboard")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results).T
    
    # Sort by MAP@1 (primary metric)
    if "map_at_1" in df.columns:
        df = df.sort_values("map_at_1", ascending=False)
    
    # Save leaderboard
    df.to_csv(output_path / "leaderboard.csv")
    
    # Create markdown table
    markdown_table = df.to_markdown(floatfmt=".4f")
    
    with open(output_path / "leaderboard.md", "w") as f:
        f.write("# Query by Humming Leaderboard\n\n")
        f.write(markdown_table)
    
    logger.info(f"Leaderboard saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Query by Humming model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override seed if provided
    if args.seed != 42:
        config.seed = args.seed
    
    # Run evaluation
    logger.info("Starting evaluation...")
    metrics = run_evaluation(config, args.checkpoint)
    
    # Log results
    logger.info("Evaluation Results:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Generate leaderboard
    generate_leaderboard({"baseline": metrics}, output_dir)


if __name__ == "__main__":
    main()
