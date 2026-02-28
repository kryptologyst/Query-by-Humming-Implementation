"""
Query by Humming - Music Information Retrieval Package
"""

__version__ = "1.0.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from .models.query_matcher import QueryByHummingMatcher, QueryMatch
from .features.extractor import FeatureExtractor, create_feature_extractor
from .decoding.dtw import DTW, fast_dtw_distance
from .metrics.evaluation import QueryByHummingMetrics
from .data.synthetic import SyntheticDatasetGenerator

__all__ = [
    "QueryByHummingMatcher",
    "QueryMatch", 
    "FeatureExtractor",
    "create_feature_extractor",
    "DTW",
    "fast_dtw_distance",
    "QueryByHummingMetrics",
    "SyntheticDatasetGenerator"
]
