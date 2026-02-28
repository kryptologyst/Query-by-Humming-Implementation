"""
Test suite for Query by Humming.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.models.query_matcher import QueryByHummingMatcher, QueryMatch
from src.features.extractor import MFCCExtractor, ChromaExtractor
from src.decoding.dtw import DTW, fast_dtw_distance
from src.metrics.evaluation import QueryByHummingMetrics
from src.data.synthetic import SyntheticDatasetGenerator
from src.utils.device import get_device, set_seed
from src.utils.audio import normalize_audio, resample_audio


class TestQueryMatch:
    """Test QueryMatch class."""
    
    def test_query_match_creation(self):
        """Test QueryMatch creation."""
        match = QueryMatch(
            title="Test Song",
            artist="Test Artist",
            distance=0.5,
            confidence=0.8
        )
        
        assert match.title == "Test Song"
        assert match.artist == "Test Artist"
        assert match.distance == 0.5
        assert match.confidence == 0.8
        assert isinstance(match.metadata, dict)
    
    def test_query_match_repr(self):
        """Test QueryMatch string representation."""
        match = QueryMatch("Song", "Artist", 0.3, 0.9)
        repr_str = repr(match)
        
        assert "Song" in repr_str
        assert "Artist" in repr_str
        assert "0.300" in repr_str
        assert "0.900" in repr_str


class TestFeatureExtractor:
    """Test feature extraction."""
    
    def test_mfcc_extractor(self):
        """Test MFCC feature extraction."""
        extractor = MFCCExtractor(n_mfcc=13, delta=True, delta_delta=True)
        
        # Generate test audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        
        features = extractor.extract(audio)
        
        assert features.ndim == 2
        assert features.shape[0] == 39  # 13 + 13 + 13 (mfcc + delta + delta-delta)
        assert features.shape[1] > 0
    
    def test_chroma_extractor(self):
        """Test chroma feature extraction."""
        extractor = ChromaExtractor(n_chroma=12)
        
        # Generate test audio
        audio = np.random.randn(16000)
        
        features = extractor.extract(audio)
        
        assert features.ndim == 2
        assert features.shape[0] == 12
        assert features.shape[1] > 0


class TestDTW:
    """Test Dynamic Time Warping."""
    
    def test_dtw_distance(self):
        """Test DTW distance computation."""
        dtw = DTW()
        
        # Test with simple sequences
        seq1 = np.array([[1], [2], [3]])
        seq2 = np.array([[1], [2], [3]])
        
        distance = dtw.compute_dtw_distance(seq1, seq2)
        
        assert distance == 0.0  # Identical sequences
    
    def test_dtw_path(self):
        """Test DTW path computation."""
        dtw = DTW()
        
        seq1 = np.array([[1], [2], [3]])
        seq2 = np.array([[1], [2], [3]])
        
        path, distance = dtw.compute_dtw_path_and_distance(seq1, seq2)
        
        assert len(path) > 0
        assert distance == 0.0
    
    def test_fast_dtw(self):
        """Test fast DTW implementation."""
        seq1 = np.array([1, 2, 3])
        seq2 = np.array([1, 2, 3])
        
        distance = fast_dtw_distance(seq1, seq2)
        
        assert distance == 0.0


class TestQueryByHummingMatcher:
    """Test QueryByHummingMatcher class."""
    
    def test_matcher_initialization(self):
        """Test matcher initialization."""
        matcher = QueryByHummingMatcher()
        
        assert matcher.device is not None
        assert matcher.feature_extractor is not None
        assert matcher.dtw is not None
        assert len(matcher.reference_database) == 0
        assert len(matcher.reference_features) == 0
    
    def test_add_reference_song(self):
        """Test adding reference songs."""
        matcher = QueryByHummingMatcher()
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Generate test audio
            audio = np.random.randn(16000)
            import soundfile as sf
            sf.write(tmp_file.name, audio, 16000)
            
            try:
                matcher.add_reference_song(
                    audio_path=tmp_file.name,
                    title="Test Song",
                    artist="Test Artist"
                )
                
                assert len(matcher.reference_database) == 1
                assert len(matcher.reference_features) == 1
                assert matcher.reference_database[0]["title"] == "Test Song"
                
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)
    
    def test_search_empty_database(self):
        """Test search with empty database."""
        matcher = QueryByHummingMatcher()
        
        audio = np.random.randn(16000)
        matches = matcher.search(audio)
        
        assert len(matches) == 0
    
    def test_clear_database(self):
        """Test clearing database."""
        matcher = QueryByHummingMatcher()
        
        # Add a song
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            audio = np.random.randn(16000)
            import soundfile as sf
            sf.write(tmp_file.name, audio, 16000)
            
            try:
                matcher.add_reference_song(
                    audio_path=tmp_file.name,
                    title="Test Song",
                    artist="Test Artist"
                )
                
                assert len(matcher.reference_database) > 0
                
                matcher.clear_database()
                
                assert len(matcher.reference_database) == 0
                assert len(matcher.reference_features) == 0
                
            finally:
                Path(tmp_file.name).unlink(missing_ok=True)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics calculator initialization."""
        metrics = QueryByHummingMetrics(k_values=[1, 5, 10])
        
        assert metrics.k_values == [1, 5, 10]
    
    def test_map_at_k(self):
        """Test MAP@k computation."""
        metrics = QueryByHummingMetrics()
        
        # Create mock matches
        matches = [
            QueryMatch("Song1", "Artist1", 0.1, 0.9),
            QueryMatch("Song2", "Artist2", 0.2, 0.8),
            QueryMatch("Song3", "Artist3", 0.3, 0.7)
        ]
        
        ground_truth_indices = [0, 2]  # First and third are relevant
        
        map_score = metrics.compute_map_at_k(matches, ground_truth_indices, k=3)
        
        assert 0.0 <= map_score <= 1.0
    
    def test_recall_at_k(self):
        """Test Recall@k computation."""
        metrics = QueryByHummingMetrics()
        
        matches = [
            QueryMatch("Song1", "Artist1", 0.1, 0.9),
            QueryMatch("Song2", "Artist2", 0.2, 0.8),
            QueryMatch("Song3", "Artist3", 0.3, 0.7)
        ]
        
        ground_truth_indices = [0, 2]
        
        recall = metrics.compute_recall_at_k(matches, ground_truth_indices, k=3)
        
        assert 0.0 <= recall <= 1.0


class TestSyntheticDataset:
    """Test synthetic dataset generation."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SyntheticDatasetGenerator()
        
        assert generator.sample_rate == 16000
        assert generator.duration_range == (5.0, 15.0)
        assert generator.tempo_range == (60.0, 180.0)
    
    def test_tone_generation(self):
        """Test tone generation."""
        generator = SyntheticDatasetGenerator()
        
        tone = generator.generate_tone(440.0, 1.0, 0.5)
        
        assert len(tone) == 16000  # 1 second at 16kHz
        assert np.max(np.abs(tone)) <= 0.5
    
    def test_melody_generation(self):
        """Test melody generation."""
        generator = SyntheticDatasetGenerator()
        
        melody = generator.generate_melody(
            generator.major_scale,
            generator.melodic_patterns[0],
            120.0,  # 120 BPM
            440.0   # A4
        )
        
        assert len(melody) > 0
        assert isinstance(melody, np.ndarray)


class TestUtils:
    """Test utility functions."""
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device("auto")
        
        assert device is not None
        assert hasattr(device, 'type')
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        rand1 = np.random.rand()
        rand2 = np.random.rand()
        
        # Set seed again
        set_seed(42)
        
        # Generate same random numbers
        rand3 = np.random.rand()
        rand4 = np.random.rand()
        
        assert rand1 == rand3
        assert rand2 == rand4
    
    def test_audio_normalization(self):
        """Test audio normalization."""
        audio = np.array([2.0, -3.0, 1.0, -1.0])
        
        normalized = normalize_audio(audio)
        
        assert np.max(np.abs(normalized)) <= 1.0
        assert np.max(np.abs(normalized)) > 0.0
    
    def test_audio_resampling(self):
        """Test audio resampling."""
        audio = np.random.randn(8000)  # 0.5 seconds at 16kHz
        
        resampled = resample_audio(audio, 16000, 8000)
        
        assert len(resampled) == 4000  # 0.5 seconds at 8kHz


if __name__ == "__main__":
    pytest.main([__file__])
