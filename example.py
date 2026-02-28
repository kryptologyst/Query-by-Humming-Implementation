"""
Example usage of Query by Humming system.
"""

import logging
from pathlib import Path

from src.models.query_matcher import QueryByHummingMatcher
from src.data.synthetic import SyntheticDatasetGenerator
from src.metrics.evaluation import QueryByHummingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Query by Humming Example")
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset...")
    generator = SyntheticDatasetGenerator()
    generator.generate_dataset(
        n_reference_songs=20,
        n_humming_queries=10,
        output_dir="data/synthetic"
    )
    
    # Initialize matcher
    logger.info("Initializing Query by Humming matcher...")
    matcher = QueryByHummingMatcher()
    
    # Load reference songs
    logger.info("Loading reference songs...")
    import pandas as pd
    
    meta_df = pd.read_csv("data/synthetic/meta.csv")
    reference_df = meta_df[meta_df["split"] == "reference"]
    
    for _, row in reference_df.iterrows():
        matcher.add_reference_song(
            audio_path=row["audio_path"],
            title=row["title"],
            artist=row["artist"],
            metadata={"genre": row.get("genre", "unknown")}
        )
    
    logger.info(f"Loaded {len(reference_df)} reference songs")
    
    # Test queries
    query_df = meta_df[meta_df["split"] == "query"]
    
    logger.info("Testing queries...")
    metrics_calculator = QueryByHummingMetrics(k_values=[1, 5, 10])
    
    for i, (_, query_row) in enumerate(query_df.iterrows()):
        logger.info(f"Processing query {i+1}/{len(query_df)}: {query_row['query_id']}")
        
        # Search for matches
        matches = matcher.search(
            query_audio=query_row["audio_path"],
            top_k=10,
            threshold=0.0
        )
        
        # Find ground truth
        ground_truth_indices = []
        for j, ref_song in enumerate(matcher.reference_database):
            if ref_song["title"] == query_row["reference_title"]:
                ground_truth_indices.append(j)
                break
        
        # Compute metrics
        metrics = metrics_calculator.evaluate_query(matches, ground_truth_indices)
        
        logger.info(f"  MAP@1: {metrics['map_at_1']:.3f}")
        logger.info(f"  MAP@5: {metrics['map_at_5']:.3f}")
        logger.info(f"  MAP@10: {metrics['map_at_10']:.3f}")
        
        # Show top 3 results
        logger.info("  Top 3 results:")
        for j, match in enumerate(matches[:3]):
            logger.info(f"    {j+1}. {match.title} - {match.artist} (conf: {match.confidence:.3f})")
        
        logger.info("")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
