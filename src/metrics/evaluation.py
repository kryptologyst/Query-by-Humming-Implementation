"""
Evaluation metrics for Query by Humming.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

from src.models.query_matcher import QueryMatch

logger = logging.getLogger(__name__)


class QueryByHummingMetrics:
    """Evaluation metrics for Query by Humming."""
    
    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        Initialize metrics calculator.
        
        Args:
            k_values: List of k values for top-k metrics.
        """
        self.k_values = k_values
    
    def compute_map_at_k(
        self,
        matches: List[QueryMatch],
        ground_truth_indices: List[int],
        k: int
    ) -> float:
        """
        Compute Mean Average Precision at k.
        
        Args:
            matches: List of query matches.
            ground_truth_indices: Indices of ground truth matches.
            k: Number of top results to consider.
            
        Returns:
            MAP@k score.
        """
        if not ground_truth_indices:
            return 0.0
        
        # Take top k matches
        top_k_matches = matches[:k]
        
        # Create binary relevance labels
        relevance = []
        for i, match in enumerate(top_k_matches):
            # Check if this match is in ground truth
            # For simplicity, we'll use the index in the original database
            is_relevant = i in ground_truth_indices
            relevance.append(1 if is_relevant else 0)
        
        if not any(relevance):
            return 0.0
        
        # Compute average precision
        return average_precision_score(relevance, relevance)
    
    def compute_recall_at_k(
        self,
        matches: List[QueryMatch],
        ground_truth_indices: List[int],
        k: int
    ) -> float:
        """
        Compute Recall at k.
        
        Args:
            matches: List of query matches.
            ground_truth_indices: Indices of ground truth matches.
            k: Number of top results to consider.
            
        Returns:
            Recall@k score.
        """
        if not ground_truth_indices:
            return 0.0
        
        # Take top k matches
        top_k_matches = matches[:k]
        
        # Count relevant items in top k
        relevant_found = 0
        for i, match in enumerate(top_k_matches):
            if i in ground_truth_indices:
                relevant_found += 1
        
        return relevant_found / len(ground_truth_indices)
    
    def compute_precision_at_k(
        self,
        matches: List[QueryMatch],
        ground_truth_indices: List[int],
        k: int
    ) -> float:
        """
        Compute Precision at k.
        
        Args:
            matches: List of query matches.
            ground_truth_indices: Indices of ground truth matches.
            k: Number of top results to consider.
            
        Returns:
            Precision@k score.
        """
        if k == 0:
            return 0.0
        
        # Take top k matches
        top_k_matches = matches[:k]
        
        # Count relevant items in top k
        relevant_found = 0
        for i, match in enumerate(top_k_matches):
            if i in ground_truth_indices:
                relevant_found += 1
        
        return relevant_found / k
    
    def compute_dtw_distance_stats(
        self,
        distances: List[float]
    ) -> Dict[str, float]:
        """
        Compute DTW distance statistics.
        
        Args:
            distances: List of DTW distances.
            
        Returns:
            Dictionary with distance statistics.
        """
        if not distances:
            return {}
        
        distances = np.array(distances)
        
        return {
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "median_distance": float(np.median(distances))
        }
    
    def compute_confidence_stats(
        self,
        confidences: List[float]
    ) -> Dict[str, float]:
        """
        Compute confidence statistics.
        
        Args:
            confidences: List of confidence scores.
            
        Returns:
            Dictionary with confidence statistics.
        """
        if not confidences:
            return {}
        
        confidences = np.array(confidences)
        
        return {
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "median_confidence": float(np.median(confidences))
        }
    
    def evaluate_query(
        self,
        matches: List[QueryMatch],
        ground_truth_indices: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate a single query.
        
        Args:
            matches: List of query matches.
            ground_truth_indices: Indices of ground truth matches.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        metrics = {}
        
        # Compute metrics for each k value
        for k in self.k_values:
            metrics[f"map_at_{k}"] = self.compute_map_at_k(matches, ground_truth_indices, k)
            metrics[f"recall_at_{k}"] = self.compute_recall_at_k(matches, ground_truth_indices, k)
            metrics[f"precision_at_{k}"] = self.compute_precision_at_k(matches, ground_truth_indices, k)
        
        # Compute distance and confidence statistics
        distances = [match.distance for match in matches]
        confidences = [match.confidence for match in matches]
        
        distance_stats = self.compute_dtw_distance_stats(distances)
        confidence_stats = self.compute_confidence_stats(confidences)
        
        metrics.update(distance_stats)
        metrics.update(confidence_stats)
        
        return metrics
    
    def evaluate_batch(
        self,
        query_results: List[Tuple[List[QueryMatch], List[int]]]
    ) -> Dict[str, float]:
        """
        Evaluate a batch of queries.
        
        Args:
            query_results: List of (matches, ground_truth_indices) tuples.
            
        Returns:
            Dictionary with average evaluation metrics.
        """
        if not query_results:
            return {}
        
        all_metrics = []
        
        for matches, ground_truth_indices in query_results:
            metrics = self.evaluate_query(matches, ground_truth_indices)
            all_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = float(np.mean(values)) if values else 0.0
        
        return avg_metrics
    
    def generate_leaderboard(
        self,
        query_results: List[Tuple[List[QueryMatch], List[int]]],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate a leaderboard comparing different models.
        
        Args:
            query_results: List of (matches, ground_truth_indices) tuples.
            model_names: Names of models being compared.
            
        Returns:
            Dictionary with leaderboard results.
        """
        if model_names is None:
            model_names = [f"Model_{i}" for i in range(len(query_results))]
        
        leaderboard = {}
        
        for i, (matches, ground_truth_indices) in enumerate(query_results):
            model_name = model_names[i]
            metrics = self.evaluate_query(matches, ground_truth_indices)
            leaderboard[model_name] = metrics
        
        return leaderboard
