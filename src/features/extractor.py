"""
Feature extraction for Query by Humming.
"""

import logging
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Base class for audio feature extraction."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            n_mels: Number of mel bins.
            fmin: Minimum frequency.
            fmax: Maximum frequency.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Extracted features.
        """
        raise NotImplementedError


class MFCCExtractor(FeatureExtractor):
    """MFCC feature extractor."""
    
    def __init__(
        self,
        n_mfcc: int = 13,
        delta: bool = True,
        delta_delta: bool = True,
        **kwargs
    ):
        """
        Initialize MFCC extractor.
        
        Args:
            n_mfcc: Number of MFCC coefficients.
            delta: Include delta features.
            delta_delta: Include delta-delta features.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(**kwargs)
        self.n_mfcc = n_mfcc
        self.delta = delta
        self.delta_delta = delta_delta
        
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features.
        
        Args:
            audio: Input audio array.
            
        Returns:
            MFCC features.
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        features = [mfcc]
        
        # Add delta features
        if self.delta:
            delta = librosa.feature.delta(mfcc)
            features.append(delta)
            
        # Add delta-delta features
        if self.delta_delta:
            delta_delta = librosa.feature.delta(mfcc, order=2)
            features.append(delta_delta)
            
        return np.concatenate(features, axis=0)


class ChromaExtractor(FeatureExtractor):
    """Chroma feature extractor."""
    
    def __init__(
        self,
        n_chroma: int = 12,
        **kwargs
    ):
        """
        Initialize chroma extractor.
        
        Args:
            n_chroma: Number of chroma bins.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(**kwargs)
        self.n_chroma = n_chroma
        
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chroma features.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Chroma features.
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )
        
        return chroma


class SpectralExtractor(FeatureExtractor):
    """Spectral feature extractor."""
    
    def __init__(
        self,
        include_spectral_centroid: bool = True,
        include_spectral_rolloff: bool = True,
        include_zero_crossing_rate: bool = True,
        **kwargs
    ):
        """
        Initialize spectral extractor.
        
        Args:
            include_spectral_centroid: Include spectral centroid.
            include_spectral_rolloff: Include spectral rolloff.
            include_zero_crossing_rate: Include zero crossing rate.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(**kwargs)
        self.include_spectral_centroid = include_spectral_centroid
        self.include_spectral_rolloff = include_spectral_rolloff
        self.include_zero_crossing_rate = include_zero_crossing_rate
        
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral features.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Spectral features.
        """
        features = []
        
        if self.include_spectral_centroid:
            centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features.append(centroid)
            
        if self.include_spectral_rolloff:
            rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features.append(rolloff)
            
        if self.include_zero_crossing_rate:
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )
            features.append(zcr)
            
        return np.concatenate(features, axis=0) if features else np.array([])


class PitchExtractor(FeatureExtractor):
    """Pitch feature extractor."""
    
    def __init__(
        self,
        fmin: float = 80.0,
        fmax: float = 400.0,
        **kwargs
    ):
        """
        Initialize pitch extractor.
        
        Args:
            fmin: Minimum frequency for pitch detection.
            fmax: Maximum frequency for pitch detection.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(fmin=fmin, fmax=fmax, **kwargs)
        
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch features.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Pitch features.
        """
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            fmin=self.fmin,
            fmax=self.fmax,
            threshold=0.1
        )
        
        # Extract pitch contour
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch)
            
        return np.array(pitch_contour)


def create_feature_extractor(
    feature_type: str,
    **kwargs
) -> FeatureExtractor:
    """
    Create feature extractor based on type.
    
    Args:
        feature_type: Type of features to extract.
        **kwargs: Additional arguments for extractor.
        
    Returns:
        Feature extractor instance.
    """
    extractors = {
        "mfcc": MFCCExtractor,
        "chroma": ChromaExtractor,
        "spectral": SpectralExtractor,
        "pitch": PitchExtractor
    }
    
    if feature_type not in extractors:
        raise ValueError(f"Unknown feature type: {feature_type}")
        
    return extractors[feature_type](**kwargs)
