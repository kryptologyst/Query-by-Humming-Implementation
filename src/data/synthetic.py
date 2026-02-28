"""
Synthetic dataset generation for Query by Humming.
"""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """Generate synthetic datasets for Query by Humming."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        duration_range: Tuple[float, float] = (5.0, 15.0),
        tempo_range: Tuple[float, float] = (60.0, 180.0)
    ):
        """
        Initialize synthetic dataset generator.
        
        Args:
            sample_rate: Audio sample rate.
            duration_range: Range of song durations in seconds.
            tempo_range: Range of tempos in BPM.
        """
        self.sample_rate = sample_rate
        self.duration_range = duration_range
        self.tempo_range = tempo_range
        
        # Musical scales and patterns
        self.major_scale = [0, 2, 4, 5, 7, 9, 11]  # C major scale
        self.minor_scale = [0, 2, 3, 5, 7, 8, 10]  # A minor scale
        self.pentatonic_scale = [0, 2, 4, 7, 9]  # Pentatonic scale
        
        # Common melodic patterns
        self.melodic_patterns = [
            [0, 2, 4, 2, 0],  # Ascending and descending
            [0, 4, 7, 4, 0],  # Arpeggio
            [0, 2, 4, 5, 4, 2, 0],  # Scale up and down
            [0, 7, 5, 4, 2, 0],  # Descending pattern
            [0, 2, 0, 4, 0, 2, 0],  # Ornamented
        ]
    
    def generate_tone(
        self,
        frequency: float,
        duration: float,
        amplitude: float = 0.3
    ) -> np.ndarray:
        """
        Generate a pure tone.
        
        Args:
            frequency: Frequency in Hz.
            duration: Duration in seconds.
            amplitude: Amplitude (0-1).
            
        Returns:
            Audio array.
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    def generate_melody(
        self,
        scale: List[int],
        pattern: List[int],
        tempo: float,
        key_frequency: float = 440.0
    ) -> np.ndarray:
        """
        Generate a melody based on scale and pattern.
        
        Args:
            scale: Musical scale intervals.
            pattern: Melodic pattern.
            tempo: Tempo in BPM.
            key_frequency: Base frequency for the key.
            
        Returns:
            Audio array of the melody.
        """
        # Calculate note duration
        beat_duration = 60.0 / tempo
        note_duration = beat_duration * 0.5  # Quarter notes
        
        melody = []
        
        for note_interval in pattern:
            # Get frequency for this note
            frequency = key_frequency * (2 ** (note_interval / 12.0))
            
            # Generate note with slight decay
            note = self.generate_tone(frequency, note_duration)
            
            # Apply envelope (attack, decay, sustain, release)
            envelope = self._apply_envelope(note)
            melody.append(envelope)
            
            # Add short silence between notes
            silence = np.zeros(int(self.sample_rate * 0.05))
            melody.append(silence)
        
        return np.concatenate(melody)
    
    def _apply_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply ADSR envelope to audio.
        
        Args:
            audio: Input audio array.
            
        Returns:
            Audio with envelope applied.
        """
        length = len(audio)
        
        # Attack (10% of note)
        attack_length = int(length * 0.1)
        attack = np.linspace(0, 1, attack_length)
        
        # Decay (20% of note)
        decay_length = int(length * 0.2)
        decay = np.linspace(1, 0.7, decay_length)
        
        # Sustain (50% of note)
        sustain_length = length - attack_length - decay_length
        sustain = np.full(sustain_length, 0.7)
        
        # Release (20% of note)
        release_length = int(length * 0.2)
        release = np.linspace(0.7, 0, release_length)
        
        envelope = np.concatenate([attack, decay, sustain, release])
        
        # Ensure envelope matches audio length
        if len(envelope) > length:
            envelope = envelope[:length]
        elif len(envelope) < length:
            envelope = np.pad(envelope, (0, length - len(envelope)), 'constant')
        
        return audio * envelope
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
        """
        Add noise to audio.
        
        Args:
            audio: Input audio array.
            noise_factor: Noise level (0-1).
            
        Returns:
            Audio with noise added.
        """
        noise = np.random.normal(0, noise_factor, len(audio))
        return audio + noise
    
    def add_reverb(self, audio: np.ndarray, reverb_factor: float = 0.3) -> np.ndarray:
        """
        Add simple reverb to audio.
        
        Args:
            audio: Input audio array.
            reverb_factor: Reverb level (0-1).
            
        Returns:
            Audio with reverb added.
        """
        # Simple reverb using delayed and attenuated copies
        reverb = np.zeros_like(audio)
        
        # Add delayed copies
        delays = [int(self.sample_rate * d) for d in [0.1, 0.2, 0.3]]
        attenuations = [0.5, 0.3, 0.2]
        
        for delay, attenuation in zip(delays, attenuations):
            if delay < len(audio):
                reverb[delay:] += audio[:-delay] * attenuation * reverb_factor
        
        return audio + reverb
    
    def generate_reference_song(
        self,
        song_id: str,
        title: str,
        artist: str,
        genre: str = "synthetic"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a reference song.
        
        Args:
            song_id: Unique song identifier.
            title: Song title.
            artist: Artist name.
            genre: Genre.
            
        Returns:
            Tuple of (audio_array, metadata).
        """
        # Random parameters
        duration = random.uniform(*self.duration_range)
        tempo = random.uniform(*self.tempo_range)
        scale = random.choice([self.major_scale, self.minor_scale, self.pentatonic_scale])
        pattern = random.choice(self.melodic_patterns)
        key_frequency = random.uniform(220, 880)  # A3 to A5
        
        # Generate melody
        melody = self.generate_melody(scale, pattern, tempo, key_frequency)
        
        # Trim or pad to target duration
        target_length = int(self.sample_rate * duration)
        if len(melody) > target_length:
            melody = melody[:target_length]
        else:
            padding = np.zeros(target_length - len(melody))
            melody = np.concatenate([melody, padding])
        
        # Add effects
        melody = self.add_noise(melody, noise_factor=0.005)
        melody = self.add_reverb(melody, reverb_factor=0.2)
        
        # Normalize
        melody = melody / np.max(np.abs(melody)) * 0.8
        
        metadata = {
            "song_id": song_id,
            "title": title,
            "artist": artist,
            "genre": genre,
            "duration": duration,
            "tempo": tempo,
            "scale": scale,
            "pattern": pattern,
            "key_frequency": key_frequency
        }
        
        return melody, metadata
    
    def generate_humming_query(
        self,
        reference_audio: np.ndarray,
        reference_metadata: Dict,
        query_id: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a humming query based on a reference song.
        
        Args:
            reference_audio: Reference song audio.
            reference_metadata: Reference song metadata.
            query_id: Query identifier.
            
        Returns:
            Tuple of (audio_array, metadata).
        """
        # Extract melody parameters from reference
        tempo = reference_metadata["tempo"]
        scale = reference_metadata["scale"]
        pattern = reference_metadata["pattern"]
        key_frequency = reference_metadata["key_frequency"]
        
        # Add variations to make it sound like humming
        tempo_variation = random.uniform(0.8, 1.2)
        tempo *= tempo_variation
        
        # Slight pitch variations
        pitch_variation = random.uniform(-0.5, 0.5)  # Semitones
        key_frequency *= (2 ** (pitch_variation / 12.0))
        
        # Generate humming version
        humming = self.generate_melody(scale, pattern, tempo, key_frequency)
        
        # Make it shorter (humming is usually shorter than full song)
        humming_duration = random.uniform(3.0, 8.0)
        humming_length = int(self.sample_rate * humming_duration)
        
        if len(humming) > humming_length:
            # Take a random segment
            start = random.randint(0, len(humming) - humming_length)
            humming = humming[start:start + humming_length]
        else:
            # Pad if too short
            padding = np.zeros(humming_length - len(humming))
            humming = np.concatenate([humming, padding])
        
        # Add humming characteristics
        humming = self.add_noise(humming, noise_factor=0.01)
        
        # Normalize
        humming = humming / np.max(np.abs(humming)) * 0.6
        
        metadata = {
            "query_id": query_id,
            "reference_song_id": reference_metadata["song_id"],
            "reference_title": reference_metadata["title"],
            "reference_artist": reference_metadata["artist"],
            "tempo_variation": tempo_variation,
            "pitch_variation": pitch_variation,
            "duration": humming_duration
        }
        
        return humming, metadata
    
    def generate_dataset(
        self,
        n_reference_songs: int = 100,
        n_humming_queries: int = 50,
        output_dir: str = "data/synthetic"
    ) -> None:
        """
        Generate a complete synthetic dataset.
        
        Args:
            n_reference_songs: Number of reference songs to generate.
            n_humming_queries: Number of humming queries to generate.
            output_dir: Output directory for the dataset.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_path / "wav").mkdir(exist_ok=True)
        (output_path / "annotations").mkdir(exist_ok=True)
        
        logger.info(f"Generating {n_reference_songs} reference songs and {n_humming_queries} humming queries")
        
        # Generate reference songs
        reference_metadata = []
        for i in range(n_reference_songs):
            song_id = f"ref_{i:03d}"
            title = f"Synthetic Song {i+1}"
            artist = f"AI Artist {(i % 10) + 1}"
            genre = random.choice(["pop", "classical", "jazz", "folk", "electronic"])
            
            audio, metadata = self.generate_reference_song(song_id, title, artist, genre)
            
            # Save audio file
            audio_path = output_path / "wav" / f"{song_id}.wav"
            sf.write(audio_path, audio, self.sample_rate)
            
            # Store metadata
            metadata["audio_path"] = str(audio_path)
            reference_metadata.append(metadata)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{n_reference_songs} reference songs")
        
        # Generate humming queries
        query_metadata = []
        for i in range(n_humming_queries):
            query_id = f"query_{i:03d}"
            
            # Select a random reference song
            ref_idx = random.randint(0, n_reference_songs - 1)
            ref_metadata = reference_metadata[ref_idx]
            
            # Load reference audio
            ref_audio, _ = sf.read(ref_metadata["audio_path"])
            
            # Generate humming query
            humming, metadata = self.generate_humming_query(ref_audio, ref_metadata, query_id)
            
            # Save audio file
            audio_path = output_path / "wav" / f"{query_id}.wav"
            sf.write(audio_path, humming, self.sample_rate)
            
            # Store metadata
            metadata["audio_path"] = str(audio_path)
            query_metadata.append(metadata)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{n_humming_queries} humming queries")
        
        # Create metadata CSV
        self._create_metadata_csv(output_path, reference_metadata, query_metadata)
        
        logger.info(f"Dataset generation complete. Saved to {output_path}")
    
    def _create_metadata_csv(
        self,
        output_path: Path,
        reference_metadata: List[Dict],
        query_metadata: List[Dict]
    ) -> None:
        """
        Create metadata CSV files.
        
        Args:
            output_path: Output directory path.
            reference_metadata: Reference song metadata.
            query_metadata: Query metadata.
        """
        # Reference songs metadata
        ref_df = pd.DataFrame(reference_metadata)
        ref_df["split"] = "reference"
        ref_df.to_csv(output_path / "reference_songs.csv", index=False)
        
        # Queries metadata
        query_df = pd.DataFrame(query_metadata)
        query_df["split"] = "query"
        query_df.to_csv(output_path / "humming_queries.csv", index=False)
        
        # Combined metadata
        combined_df = pd.concat([ref_df, query_df], ignore_index=True)
        combined_df.to_csv(output_path / "meta.csv", index=False)
        
        logger.info("Created metadata CSV files")
