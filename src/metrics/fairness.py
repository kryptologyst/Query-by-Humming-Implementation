"""
Fairness evaluation for Query by Humming.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.models.query_matcher import QueryByHummingMatcher, QueryMatch
from src.metrics.evaluation import QueryByHummingMetrics

logger = logging.getLogger(__name__)


class FairnessEvaluator:
    """Fairness evaluation for Query by Humming."""
    
    def __init__(self, matcher: QueryByHummingMatcher):
        """
        Initialize fairness evaluator.
        
        Args:
            matcher: Query by Humming matcher.
        """
        self.matcher = matcher
        self.metrics_calculator = QueryByHummingMetrics()
    
    def evaluate_by_genre(
        self,
        test_queries: List[Tuple[str, str, str]],  # (query_path, reference_title, genre)
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by genre.
        
        Args:
            test_queries: List of test queries with genre information.
            k_values: List of k values for evaluation.
            
        Returns:
            Dictionary with metrics by genre.
        """
        genre_metrics = {}
        
        # Group queries by genre
        genre_groups = {}
        for query_path, ref_title, genre in test_queries:
            if genre not in genre_groups:
                genre_groups[genre] = []
            genre_groups[genre].append((query_path, ref_title))
        
        # Evaluate each genre
        for genre, queries in genre_groups.items():
            logger.info(f"Evaluating genre: {genre} ({len(queries)} queries)")
            
            genre_results = []
            
            for query_path, ref_title in queries:
                try:
                    # Search for matches
                    matches = self.matcher.search(
                        query_audio=query_path,
                        top_k=max(k_values),
                        threshold=0.0
                    )
                    
                    # Find ground truth
                    ground_truth_indices = []
                    for i, ref_song in enumerate(self.matcher.reference_database):
                        if ref_song["title"] == ref_title:
                            ground_truth_indices.append(i)
                            break
                    
                    genre_results.append((matches, ground_truth_indices))
                    
                except Exception as e:
                    logger.error(f"Error processing query {query_path}: {e}")
                    continue
            
            # Compute average metrics for this genre
            if genre_results:
                avg_metrics = self.metrics_calculator.evaluate_batch(genre_results)
                genre_metrics[genre] = avg_metrics
            else:
                logger.warning(f"No valid results for genre {genre}")
                genre_metrics[genre] = {}
        
        return genre_metrics
    
    def evaluate_by_artist(
        self,
        test_queries: List[Tuple[str, str, str]],  # (query_path, reference_title, artist)
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by artist.
        
        Args:
            test_queries: List of test queries with artist information.
            k_values: List of k values for evaluation.
            
        Returns:
            Dictionary with metrics by artist.
        """
        artist_metrics = {}
        
        # Group queries by artist
        artist_groups = {}
        for query_path, ref_title, artist in test_queries:
            if artist not in artist_groups:
                artist_groups[artist] = []
            artist_groups[artist].append((query_path, ref_title))
        
        # Evaluate each artist
        for artist, queries in artist_groups.items():
            logger.info(f"Evaluating artist: {artist} ({len(queries)} queries)")
            
            artist_results = []
            
            for query_path, ref_title in queries:
                try:
                    # Search for matches
                    matches = self.matcher.search(
                        query_audio=query_path,
                        top_k=max(k_values),
                        threshold=0.0
                    )
                    
                    # Find ground truth
                    ground_truth_indices = []
                    for i, ref_song in enumerate(self.matcher.reference_database):
                        if ref_song["title"] == ref_title:
                            ground_truth_indices.append(i)
                            break
                    
                    artist_results.append((matches, ground_truth_indices))
                    
                except Exception as e:
                    logger.error(f"Error processing query {query_path}: {e}")
                    continue
            
            # Compute average metrics for this artist
            if artist_results:
                avg_metrics = self.metrics_calculator.evaluate_batch(artist_results)
                artist_metrics[artist] = avg_metrics
            else:
                logger.warning(f"No valid results for artist {artist}")
                artist_metrics[artist] = {}
        
        return artist_metrics
    
    def evaluate_by_duration(
        self,
        test_queries: List[Tuple[str, str, float]],  # (query_path, reference_title, duration)
        duration_bins: List[Tuple[float, float]] = [(0, 5), (5, 10), (10, 15), (15, 30)],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by duration bins.
        
        Args:
            test_queries: List of test queries with duration information.
            duration_bins: List of duration bin ranges.
            k_values: List of k values for evaluation.
            
        Returns:
            Dictionary with metrics by duration bin.
        """
        duration_metrics = {}
        
        # Group queries by duration bins
        bin_groups = {}
        for query_path, ref_title, duration in test_queries:
            bin_name = None
            for min_dur, max_dur in duration_bins:
                if min_dur <= duration < max_dur:
                    bin_name = f"{min_dur}-{max_dur}s"
                    break
            
            if bin_name is None:
                bin_name = f">{duration_bins[-1][1]}s"
            
            if bin_name not in bin_groups:
                bin_groups[bin_name] = []
            bin_groups[bin_name].append((query_path, ref_title))
        
        # Evaluate each duration bin
        for bin_name, queries in bin_groups.items():
            logger.info(f"Evaluating duration bin: {bin_name} ({len(queries)} queries)")
            
            bin_results = []
            
            for query_path, ref_title in queries:
                try:
                    # Search for matches
                    matches = self.matcher.search(
                        query_audio=query_path,
                        top_k=max(k_values),
                        threshold=0.0
                    )
                    
                    # Find ground truth
                    ground_truth_indices = []
                    for i, ref_song in enumerate(self.matcher.reference_database):
                        if ref_song["title"] == ref_title:
                            ground_truth_indices.append(i)
                            break
                    
                    bin_results.append((matches, ground_truth_indices))
                    
                except Exception as e:
                    logger.error(f"Error processing query {query_path}: {e}")
                    continue
            
            # Compute average metrics for this bin
            if bin_results:
                avg_metrics = self.metrics_calculator.evaluate_batch(bin_results)
                duration_metrics[bin_name] = avg_metrics
            else:
                logger.warning(f"No valid results for duration bin {bin_name}")
                duration_metrics[bin_name] = {}
        
        return duration_metrics
    
    def compute_fairness_metrics(
        self,
        group_metrics: Dict[str, Dict[str, float]],
        primary_metric: str = "map_at_1"
    ) -> Dict[str, float]:
        """
        Compute fairness metrics across groups.
        
        Args:
            group_metrics: Metrics by group.
            primary_metric: Primary metric to analyze.
            
        Returns:
            Dictionary with fairness metrics.
        """
        if not group_metrics:
            return {}
        
        # Extract primary metric values
        metric_values = []
        for group, metrics in group_metrics.items():
            if primary_metric in metrics:
                metric_values.append(metrics[primary_metric])
        
        if not metric_values:
            return {}
        
        metric_values = np.array(metric_values)
        
        # Compute fairness metrics
        fairness_metrics = {
            "mean_performance": float(np.mean(metric_values)),
            "std_performance": float(np.std(metric_values)),
            "min_performance": float(np.min(metric_values)),
            "max_performance": float(np.max(metric_values)),
            "performance_range": float(np.max(metric_values) - np.min(metric_values)),
            "coefficient_of_variation": float(np.std(metric_values) / np.mean(metric_values)) if np.mean(metric_values) > 0 else 0.0,
            "fairness_gap": float(np.max(metric_values) - np.min(metric_values))
        }
        
        return fairness_metrics
    
    def generate_fairness_report(
        self,
        genre_metrics: Dict[str, Dict[str, float]],
        artist_metrics: Dict[str, Dict[str, float]],
        duration_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict]:
        """
        Generate comprehensive fairness report.
        
        Args:
            genre_metrics: Metrics by genre.
            artist_metrics: Metrics by artist.
            duration_metrics: Metrics by duration.
            
        Returns:
            Comprehensive fairness report.
        """
        report = {
            "genre_fairness": {
                "metrics": genre_metrics,
                "fairness_analysis": self.compute_fairness_metrics(genre_metrics)
            },
            "artist_fairness": {
                "metrics": artist_metrics,
                "fairness_analysis": self.compute_fairness_metrics(artist_metrics)
            },
            "duration_fairness": {
                "metrics": duration_metrics,
                "fairness_analysis": self.compute_fairness_metrics(duration_metrics)
            }
        }
        
        # Overall fairness assessment
        all_fairness_gaps = []
        for category in ["genre_fairness", "artist_fairness", "duration_fairness"]:
            if "fairness_analysis" in report[category] and "fairness_gap" in report[category]["fairness_analysis"]:
                all_fairness_gaps.append(report[category]["fairness_analysis"]["fairness_gap"])
        
        if all_fairness_gaps:
            report["overall_fairness"] = {
                "average_fairness_gap": float(np.mean(all_fairness_gaps)),
                "max_fairness_gap": float(np.max(all_fairness_gaps)),
                "fairness_score": float(1.0 - np.mean(all_fairness_gaps))  # Higher is better
            }
        
        return report


def load_test_queries_with_metadata(metadata_path: str) -> Tuple[List, List, List]:
    """
    Load test queries with metadata for fairness evaluation.
    
    Args:
        metadata_path: Path to metadata CSV file.
        
    Returns:
        Tuple of (genre_queries, artist_queries, duration_queries).
    """
    df = pd.read_csv(metadata_path)
    query_df = df[df["split"] == "query"]
    
    # Prepare queries for different fairness evaluations
    genre_queries = []
    artist_queries = []
    duration_queries = []
    
    for _, row in query_df.iterrows():
        genre_queries.append((row["audio_path"], row["reference_title"], row.get("genre", "unknown")))
        artist_queries.append((row["audio_path"], row["reference_title"], row["reference_artist"]))
        
        # Get duration from audio file
        try:
            import librosa
            audio, sr = librosa.load(row["audio_path"])
            duration = len(audio) / sr
            duration_queries.append((row["audio_path"], row["reference_title"], duration))
        except Exception as e:
            logger.warning(f"Could not get duration for {row['audio_path']}: {e}")
            duration_queries.append((row["audio_path"], row["reference_title"], 0.0))
    
    return genre_queries, artist_queries, duration_queries


def run_fairness_evaluation(
    matcher: QueryByHummingMatcher,
    metadata_path: str = "data/synthetic/meta.csv"
) -> Dict[str, Dict]:
    """
    Run comprehensive fairness evaluation.
    
    Args:
        matcher: Query by Humming matcher.
        metadata_path: Path to metadata file.
        
    Returns:
        Comprehensive fairness report.
    """
    logger.info("Starting fairness evaluation...")
    
    # Initialize evaluator
    evaluator = FairnessEvaluator(matcher)
    
    # Load test queries
    genre_queries, artist_queries, duration_queries = load_test_queries_with_metadata(metadata_path)
    
    logger.info(f"Loaded {len(genre_queries)} queries for fairness evaluation")
    
    # Evaluate by different criteria
    genre_metrics = evaluator.evaluate_by_genre(genre_queries)
    artist_metrics = evaluator.evaluate_by_artist(artist_queries)
    duration_metrics = evaluator.evaluate_by_duration(duration_queries)
    
    # Generate comprehensive report
    report = evaluator.generate_fairness_report(
        genre_metrics, artist_metrics, duration_metrics
    )
    
    logger.info("Fairness evaluation completed")
    
    return report
