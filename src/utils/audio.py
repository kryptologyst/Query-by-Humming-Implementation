"""
Audio processing utilities for Query by Humming.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio(
    file_path: Union[str, Path], 
    sr: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    
    Args:
        file_path: Path to audio file.
        sr: Target sample rate. If None, use original.
        mono: Convert to mono if True.
        
    Returns:
        Tuple of (audio_array, sample_rate).
    """
    try:
        audio, sr = librosa.load(file_path, sr=sr, mono=mono)
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sr: int,
    format: str = "WAV"
) -> None:
    """
    Save audio array to file.
    
    Args:
        audio: Audio array to save.
        file_path: Output file path.
        sr: Sample rate.
        format: Audio format.
    """
    try:
        sf.write(file_path, audio, sr, format=format)
    except Exception as e:
        logger.error(f"Error saving audio file {file_path}: {e}")
        raise


def resample_audio(
    audio: np.ndarray, 
    orig_sr: int, 
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Input audio array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.
        
    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio: Input audio array.
        
    Returns:
        Normalized audio array.
    """
    if np.max(np.abs(audio)) > 0:
        return audio / np.max(np.abs(audio))
    return audio


def apply_preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to audio.
    
    Args:
        audio: Input audio array.
        coeff: Pre-emphasis coefficient.
        
    Returns:
        Pre-emphasized audio array.
    """
    return librosa.effects.preemphasis(audio, coef=coeff)


def trim_silence(
    audio: np.ndarray, 
    sr: int,
    top_db: float = 20
) -> np.ndarray:
    """
    Trim silence from audio.
    
    Args:
        audio: Input audio array.
        sr: Sample rate.
        top_db: Silence threshold in dB.
        
    Returns:
        Trimmed audio array.
    """
    return librosa.effects.trim(audio, top_db=top_db)[0]


def pad_or_truncate(
    audio: np.ndarray, 
    target_length: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad or truncate audio to target length.
    
    Args:
        audio: Input audio array.
        target_length: Target length in samples.
        pad_value: Value to use for padding.
        
    Returns:
        Padded or truncated audio array.
    """
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        padding = np.full(target_length - len(audio), pad_value)
        return np.concatenate([audio, padding])
    else:
        return audio


def get_audio_info(file_path: Union[str, Path]) -> dict:
    """
    Get audio file information.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        Dictionary with audio information.
    """
    try:
        info = sf.info(file_path)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype
        }
    except Exception as e:
        logger.error(f"Error getting audio info for {file_path}: {e}")
        raise


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if file is a valid audio file.
    
    Args:
        file_path: Path to audio file.
        
    Returns:
        True if valid audio file, False otherwise.
    """
    try:
        sf.info(file_path)
        return True
    except Exception:
        return False
