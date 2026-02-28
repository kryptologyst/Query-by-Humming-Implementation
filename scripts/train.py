"""
Training script for Query by Humming models.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.synthetic import SyntheticDatasetGenerator
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


def generate_dataset(config: DictConfig) -> None:
    """
    Generate synthetic dataset if needed.
    
    Args:
        config: Configuration dictionary.
    """
    data_dir = Path(config.paths.data_dir)
    synthetic_dir = data_dir / "synthetic"
    
    if not synthetic_dir.exists():
        logger.info("Generating synthetic dataset...")
        
        generator = SyntheticDatasetGenerator(
            sample_rate=config.data.audio.sample_rate,
            duration_range=(5.0, 15.0),
            tempo_range=(60.0, 180.0)
        )
        
        generator.generate_dataset(
            n_reference_songs=config.data.get("n_reference_songs", 100),
            n_humming_queries=config.data.get("n_humming_queries", 50),
            output_dir=str(synthetic_dir)
        )
        
        logger.info(f"Synthetic dataset generated at {synthetic_dir}")
    else:
        logger.info(f"Synthetic dataset already exists at {synthetic_dir}")


def load_reference_database(
    matcher: QueryByHummingMatcher,
    config: DictConfig
) -> None:
    """
    Load reference songs into the matcher database.
    
    Args:
        matcher: Query by Humming matcher.
        config: Configuration dictionary.
    """
    import pandas as pd
    
    synthetic_dir = Path(config.paths.data_dir) / "synthetic"
    meta_path = synthetic_dir / "meta.csv"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    # Load metadata
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


def evaluate_model(
    matcher: QueryByHummingMatcher,
    config: DictConfig
) -> Dict[str, float]:
    """
    Evaluate the model on test queries.
    
    Args:
        matcher: Query by Humming matcher.
        config: Configuration dictionary.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    import pandas as pd
    
    synthetic_dir = Path(config.paths.data_dir) / "synthetic"
    meta_path = synthetic_dir / "meta.csv"
    
    # Load test queries
    meta_df = pd.read_csv(meta_path)
    query_df = meta_df[meta_df["split"] == "query"]
    
    logger.info(f"Evaluating on {len(query_df)} test queries...")
    
    # Initialize metrics
    metrics_calculator = QueryByHummingMetrics(
        k_values=config.eval.k_values
    )
    
    query_results = []
    
    for _, row in query_df.iterrows():
        try:
            # Load query audio
            query_audio, _ = matcher.feature_extractor.extract(
                matcher.load_audio(row["audio_path"])[0]
            )
            
            # Search for matches
            matches = matcher.search(
                query_audio=query_audio,
                top_k=config.eval.k_values[-1],  # Use largest k
                threshold=0.0  # No threshold for evaluation
            )
            
            # Ground truth (reference song index)
            ref_song_id = row["reference_song_id"]
            ground_truth_indices = []
            
            # Find ground truth index in reference database
            for i, ref_song in enumerate(matcher.reference_database):
                if ref_song["title"] == row["reference_title"]:
                    ground_truth_indices.append(i)
                    break
            
            query_results.append((matches, ground_truth_indices))
            
        except Exception as e:
            logger.error(f"Error processing query {row['query_id']}: {e}")
            continue
    
    # Compute average metrics
    avg_metrics = metrics_calculator.evaluate_batch(query_results)
    
    return avg_metrics


def train_model(config: DictConfig) -> None:
    """
    Train the Query by Humming model.
    
    Args:
        config: Configuration dictionary.
    """
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")
    
    # Generate dataset if needed
    generate_dataset(config)
    
    # Initialize matcher
    matcher = QueryByHummingMatcher(config)
    
    # Load reference database
    load_reference_database(matcher, config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = evaluate_model(matcher, config)
    
    # Log results
    logger.info("Evaluation Results:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    # Save results
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "evaluation_results.json"
    import json
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Save model database
    db_path = output_dir / "reference_database.pkl"
    matcher.save_database(db_path)
    logger.info(f"Reference database saved to {db_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Query by Humming model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file"
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
    
    # Train model
    train_model(config)


if __name__ == "__main__":
    main()
