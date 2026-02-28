"""
Query by Humming model implementation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

from src.decoding.dtw import DTW, fast_dtw_distance
from src.features.extractor import FeatureExtractor, create_feature_extractor
from src.utils.audio import load_audio, normalize_audio, resample_audio
from src.utils.device import get_device

logger = logging.getLogger(__name__)


class QueryMatch:
    """Represents a query match result."""
    
    def __init__(
        self,
        title: str,
        artist: str,
        distance: float,
        confidence: float,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize query match.
        
        Args:
            title: Song title.
            artist: Artist name.
            distance: DTW distance.
            confidence: Match confidence.
            metadata: Additional metadata.
        """
        self.title = title
        self.artist = artist
        self.distance = distance
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"QueryMatch(title='{self.title}', artist='{self.artist}', distance={self.distance:.3f}, confidence={self.confidence:.3f})"


class QueryByHummingMatcher:
    """Query by Humming matcher using DTW."""
    
    def __init__(self, config: Optional[DictConfig] = None):
        """
        Initialize Query by Humming matcher.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or self._get_default_config()
        self.device = get_device(self.config.get("device", "auto"))
        
        # Initialize feature extractor
        self.feature_extractor = create_feature_extractor(
            feature_type=self.config.model.features.type,
            sample_rate=self.config.data.audio.sample_rate,
            n_mfcc=self.config.model.features.n_mfcc,
            delta=self.config.model.features.delta,
            delta_delta=self.config.model.features.delta_delta
        )
        
        # Initialize DTW
        self.dtw = DTW(
            distance_metric=self.config.model.dtw.distance_metric,
            window_size=self.config.model.dtw.window_size,
            step_pattern=self.config.model.dtw.step_pattern
        )
        
        # Reference database
        self.reference_database: List[Dict] = []
        self.reference_features: List[np.ndarray] = []
        
        logger.info(f"Initialized QueryByHummingMatcher with device: {self.device}")
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Tuple of (audio_array, sample_rate).
        """
        return load_audio(file_path, sr=self.config.data.audio.sample_rate)
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration."""
        return DictConfig({
            "model": {
                "features": {
                    "type": "mfcc",
                    "n_mfcc": 13,
                    "delta": True,
                    "delta_delta": True
                },
                "dtw": {
                    "distance_metric": "euclidean",
                    "window_size": 10,
                    "step_pattern": "symmetric2"
                },
                "matching": {
                    "top_k": 10,
                    "threshold": 0.5,
                    "normalize_distances": True
                }
            },
            "data": {
                "audio": {
                    "sample_rate": 16000
                }
            },
            "device": "auto"
        })
    
    def add_reference_song(
        self,
        audio_path: Union[str, Path],
        title: str,
        artist: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a reference song to the database.
        
        Args:
            audio_path: Path to audio file.
            title: Song title.
            artist: Artist name.
            metadata: Additional metadata.
        """
        try:
            # Load audio
            audio, sr = load_audio(audio_path)
            
            # Resample if necessary
            if sr != self.config.data.audio.sample_rate:
                from src.utils.audio import resample_audio
                audio = resample_audio(audio, sr, self.config.data.audio.sample_rate)
            
            # Normalize audio
            audio = normalize_audio(audio)
            
            # Extract features
            features = self.feature_extractor.extract(audio)
            
            # Add to database
            song_info = {
                "title": title,
                "artist": artist,
                "audio_path": str(audio_path),
                "metadata": metadata or {}
            }
            
            self.reference_database.append(song_info)
            self.reference_features.append(features)
            
            logger.info(f"Added reference song: {title} by {artist}")
            
        except Exception as e:
            logger.error(f"Error adding reference song {audio_path}: {e}")
            raise
    
    def search(
        self,
        query_audio: Union[np.ndarray, str, Path],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[QueryMatch]:
        """
        Search for matches to the query audio.
        
        Args:
            query_audio: Query audio array or path to audio file.
            top_k: Number of top matches to return.
            threshold: Distance threshold for filtering.
            
        Returns:
            List of QueryMatch objects.
        """
        if len(self.reference_database) == 0:
            logger.warning("No reference songs in database")
            return []
        
        # Load query audio if path provided
        if isinstance(query_audio, (str, Path)):
            query_audio, sr = load_audio(query_audio)
            if sr != self.config.data.audio.sample_rate:
                from src.utils.audio import resample_audio
                query_audio = resample_audio(query_audio, sr, self.config.data.audio.sample_rate)
        
        # Normalize query audio
        query_audio = normalize_audio(query_audio)
        
        # Extract query features
        query_features = self.feature_extractor.extract(query_audio)
        
        # Compute distances
        distances = []
        for ref_features in self.reference_features:
            distance = self.dtw.compute_dtw_distance(query_features.T, ref_features.T)
            distances.append(distance)
        
        distances = np.array(distances)
        
        # Normalize distances if configured
        if self.config.model.matching.normalize_distances:
            distances = distances / np.max(distances)
        
        # Create matches
        matches = []
        for i, distance in enumerate(distances):
            # Compute confidence (inverse of normalized distance)
            confidence = 1.0 / (1.0 + distance)
            
            match = QueryMatch(
                title=self.reference_database[i]["title"],
                artist=self.reference_database[i]["artist"],
                distance=distance,
                confidence=confidence,
                metadata=self.reference_database[i]["metadata"]
            )
            matches.append(match)
        
        # Sort by distance (ascending)
        matches.sort(key=lambda x: x.distance)
        
        # Apply threshold filter
        threshold = threshold or self.config.model.matching.threshold
        matches = [m for m in matches if m.confidence >= threshold]
        
        # Apply top_k filter
        top_k = top_k or self.config.model.matching.top_k
        matches = matches[:top_k]
        
        logger.info(f"Found {len(matches)} matches for query")
        
        return matches
    
    def clear_database(self) -> None:
        """Clear the reference database."""
        self.reference_database.clear()
        self.reference_features.clear()
        logger.info("Cleared reference database")
    
    def get_database_info(self) -> Dict:
        """
        Get information about the reference database.
        
        Returns:
            Dictionary with database information.
        """
        return {
            "num_songs": len(self.reference_database),
            "songs": [
                {
                    "title": song["title"],
                    "artist": song["artist"],
                    "metadata": song["metadata"]
                }
                for song in self.reference_database
            ]
        }
    
    def save_database(self, file_path: Union[str, Path]) -> None:
        """
        Save the reference database to file.
        
        Args:
            file_path: Path to save database.
        """
        import pickle
        
        database_data = {
            "reference_database": self.reference_database,
            "reference_features": self.reference_features,
            "config": self.config
        }
        
        with open(file_path, "wb") as f:
            pickle.dump(database_data, f)
        
        logger.info(f"Saved database to {file_path}")
    
    def load_database(self, file_path: Union[str, Path]) -> None:
        """
        Load the reference database from file.
        
        Args:
            file_path: Path to load database from.
        """
        import pickle
        
        with open(file_path, "rb") as f:
            database_data = pickle.load(f)
        
        self.reference_database = database_data["reference_database"]
        self.reference_features = database_data["reference_features"]
        self.config = database_data["config"]
        
        logger.info(f"Loaded database from {file_path}")
