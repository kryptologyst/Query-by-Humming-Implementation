"""
Dynamic Time Warping implementation for Query by Humming.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class DTW:
    """Dynamic Time Warping implementation."""
    
    def __init__(
        self,
        distance_metric: str = "euclidean",
        window_size: Optional[int] = None,
        step_pattern: str = "symmetric2"
    ):
        """
        Initialize DTW.
        
        Args:
            distance_metric: Distance metric for local cost.
            window_size: Sakoe-Chiba band width.
            step_pattern: Step pattern for DTW.
        """
        self.distance_metric = distance_metric
        self.window_size = window_size
        self.step_pattern = step_pattern
        
    def compute_distance_matrix(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray
    ) -> np.ndarray:
        """
        Compute distance matrix between two sequences.
        
        Args:
            seq1: First sequence.
            seq2: Second sequence.
            
        Returns:
            Distance matrix.
        """
        return cdist(seq1, seq2, metric=self.distance_metric)
    
    def compute_dtw_path(
        self, 
        cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute DTW path and distance.
        
        Args:
            cost_matrix: Local cost matrix.
            
        Returns:
            Tuple of (path, distance).
        """
        n, m = cost_matrix.shape
        
        # Initialize accumulated cost matrix
        acc_cost = np.full((n + 1, m + 1), np.inf)
        acc_cost[0, 0] = 0
        
        # Apply window constraint
        if self.window_size is not None:
            for i in range(n + 1):
                for j in range(m + 1):
                    if abs(i - j) > self.window_size:
                        acc_cost[i, j] = np.inf
        
        # Fill accumulated cost matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if acc_cost[i, j] != np.inf:
                    if self.step_pattern == "symmetric1":
                        # Symmetric1: (1,0), (0,1), (1,1)
                        acc_cost[i, j] = cost_matrix[i-1, j-1] + min(
                            acc_cost[i-1, j],
                            acc_cost[i, j-1],
                            acc_cost[i-1, j-1]
                        )
                    elif self.step_pattern == "symmetric2":
                        # Symmetric2: (1,0), (0,1), (1,1), (2,1), (1,2)
                        acc_cost[i, j] = cost_matrix[i-1, j-1] + min(
                            acc_cost[i-1, j],
                            acc_cost[i, j-1],
                            acc_cost[i-1, j-1],
                            acc_cost[i-2, j-1] if i > 1 else np.inf,
                            acc_cost[i-1, j-2] if j > 1 else np.inf
                        )
                    else:
                        # Default: only diagonal
                        acc_cost[i, j] = cost_matrix[i-1, j-1] + acc_cost[i-1, j-1]
        
        # Backtrack to find path
        path = []
        i, j = n, m
        
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            if self.step_pattern == "symmetric1":
                if acc_cost[i-1, j] <= acc_cost[i, j-1] and acc_cost[i-1, j] <= acc_cost[i-1, j-1]:
                    i -= 1
                elif acc_cost[i, j-1] <= acc_cost[i-1, j-1]:
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            elif self.step_pattern == "symmetric2":
                costs = [
                    acc_cost[i-1, j],
                    acc_cost[i, j-1],
                    acc_cost[i-1, j-1],
                    acc_cost[i-2, j-1] if i > 1 else np.inf,
                    acc_cost[i-1, j-2] if j > 1 else np.inf
                ]
                min_idx = np.argmin(costs)
                
                if min_idx == 0:
                    i -= 1
                elif min_idx == 1:
                    j -= 1
                elif min_idx == 2:
                    i -= 1
                    j -= 1
                elif min_idx == 3:
                    i -= 2
                    j -= 1
                else:  # min_idx == 4
                    i -= 1
                    j -= 2
            else:
                i -= 1
                j -= 1
        
        path.reverse()
        distance = acc_cost[n, m]
        
        return np.array(path), distance
    
    def compute_dtw_distance(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray
    ) -> float:
        """
        Compute DTW distance between two sequences.
        
        Args:
            seq1: First sequence.
            seq2: Second sequence.
            
        Returns:
            DTW distance.
        """
        # Ensure sequences are 2D
        if seq1.ndim == 1:
            seq1 = seq1.reshape(-1, 1)
        if seq2.ndim == 1:
            seq2 = seq2.reshape(-1, 1)
            
        # Compute distance matrix
        cost_matrix = self.compute_distance_matrix(seq1, seq2)
        
        # Compute DTW path and distance
        _, distance = self.compute_dtw_path(cost_matrix)
        
        return distance
    
    def compute_dtw_path_and_distance(
        self, 
        seq1: np.ndarray, 
        seq2: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute DTW path and distance between two sequences.
        
        Args:
            seq1: First sequence.
            seq2: Second sequence.
            
        Returns:
            Tuple of (path, distance).
        """
        # Ensure sequences are 2D
        if seq1.ndim == 1:
            seq1 = seq1.reshape(-1, 1)
        if seq2.ndim == 1:
            seq2 = seq2.reshape(-1, 1)
            
        # Compute distance matrix
        cost_matrix = self.compute_distance_matrix(seq1, seq2)
        
        # Compute DTW path and distance
        path, distance = self.compute_dtw_path(cost_matrix)
        
        return path, distance


def fast_dtw_distance(
    seq1: np.ndarray, 
    seq2: np.ndarray,
    radius: int = 1
) -> float:
    """
    Fast DTW implementation using radius constraint.
    
    Args:
        seq1: First sequence.
        seq2: Second sequence.
        radius: Radius for constraint.
        
    Returns:
        DTW distance.
    """
    n, m = len(seq1), len(seq2)
    
    # Initialize cost matrix
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0
    
    # Fill cost matrix with radius constraint
    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(m + 1, i + radius + 1)):
            cost[i, j] = np.linalg.norm(seq1[i-1] - seq2[j-1]) + min(
                cost[i-1, j],
                cost[i, j-1],
                cost[i-1, j-1]
            )
    
    return cost[n, m]
