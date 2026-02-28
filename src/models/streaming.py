"""
Streaming Query by Humming implementation.
"""

import logging
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig

from src.models.query_matcher import QueryByHummingMatcher
from src.utils.device import get_device

logger = logging.getLogger(__name__)


class StreamingQueryByHumming:
    """Streaming Query by Humming implementation."""
    
    def __init__(
        self,
        matcher: QueryByHummingMatcher,
        chunk_size: int = 16000,  # 1 second at 16kHz
        overlap_size: int = 8000,  # 0.5 second overlap
        min_chunks: int = 3,  # Minimum chunks before processing
        max_chunks: int = 30  # Maximum chunks to keep in buffer
    ):
        """
        Initialize streaming query by humming.
        
        Args:
            matcher: Query by Humming matcher.
            chunk_size: Size of audio chunks in samples.
            overlap_size: Overlap between chunks in samples.
            min_chunks: Minimum chunks before processing.
            max_chunks: Maximum chunks to keep in buffer.
        """
        self.matcher = matcher
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        
        # Audio buffer
        self.audio_buffer: List[np.ndarray] = []
        self.total_samples = 0
        
        # Processing state
        self.is_processing = False
        self.last_results: List = []
        
        logger.info(f"Initialized streaming query with chunk_size={chunk_size}, overlap_size={overlap_size}")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Add audio chunk to buffer.
        
        Args:
            audio_chunk: Audio chunk to add.
        """
        # Normalize audio chunk
        if np.max(np.abs(audio_chunk)) > 0:
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
        
        self.audio_buffer.append(audio_chunk)
        self.total_samples += len(audio_chunk)
        
        # Remove old chunks if buffer is too large
        while len(self.audio_buffer) > self.max_chunks:
            removed_chunk = self.audio_buffer.pop(0)
            self.total_samples -= len(removed_chunk)
        
        logger.debug(f"Added chunk. Buffer size: {len(self.audio_buffer)} chunks, {self.total_samples} samples")
    
    def get_current_audio(self) -> np.ndarray:
        """
        Get current audio from buffer.
        
        Returns:
            Concatenated audio from buffer.
        """
        if not self.audio_buffer:
            return np.array([])
        
        return np.concatenate(self.audio_buffer)
    
    def should_process(self) -> bool:
        """
        Check if we should process the current buffer.
        
        Returns:
            True if we should process.
        """
        return len(self.audio_buffer) >= self.min_chunks and not self.is_processing
    
    async def process_streaming_query(
        self,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List:
        """
        Process streaming query.
        
        Args:
            top_k: Number of top results.
            threshold: Confidence threshold.
            
        Returns:
            List of matches.
        """
        if not self.should_process():
            return self.last_results
        
        self.is_processing = True
        
        try:
            # Get current audio
            current_audio = self.get_current_audio()
            
            if len(current_audio) == 0:
                return []
            
            # Search for matches
            matches = self.matcher.search(
                query_audio=current_audio,
                top_k=top_k,
                threshold=threshold
            )
            
            self.last_results = matches
            
            logger.info(f"Processed streaming query. Found {len(matches)} matches")
            
            return matches
            
        except Exception as e:
            logger.error(f"Error processing streaming query: {e}")
            return []
        
        finally:
            self.is_processing = False
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        self.audio_buffer.clear()
        self.total_samples = 0
        self.last_results = []
        logger.info("Cleared audio buffer")
    
    def get_buffer_info(self) -> dict:
        """
        Get buffer information.
        
        Returns:
            Dictionary with buffer information.
        """
        return {
            "num_chunks": len(self.audio_buffer),
            "total_samples": self.total_samples,
            "total_duration": self.total_samples / self.matcher.config.data.audio.sample_rate,
            "is_processing": self.is_processing,
            "last_results_count": len(self.last_results)
        }


class RealTimeAudioProcessor:
    """Real-time audio processor for streaming."""
    
    def __init__(
        self,
        streaming_query: StreamingQueryByHumming,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0  # seconds
    ):
        """
        Initialize real-time audio processor.
        
        Args:
            streaming_query: Streaming query instance.
            sample_rate: Audio sample rate.
            chunk_duration: Duration of each chunk in seconds.
        """
        self.streaming_query = streaming_query
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        
        logger.info(f"Initialized real-time processor with chunk_duration={chunk_duration}s")
    
    async def process_audio_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        top_k: int = 10,
        threshold: float = 0.5
    ) -> AsyncGenerator[List, None]:
        """
        Process audio stream in real-time.
        
        Args:
            audio_stream: Async generator of audio chunks.
            top_k: Number of top results.
            threshold: Confidence threshold.
            
        Yields:
            List of matches for each processing cycle.
        """
        async for audio_chunk in audio_stream:
            # Add chunk to buffer
            self.streaming_query.add_audio_chunk(audio_chunk)
            
            # Process if ready
            if self.streaming_query.should_process():
                matches = await self.streaming_query.process_streaming_query(
                    top_k=top_k,
                    threshold=threshold
                )
                yield matches
    
    def get_processing_stats(self) -> dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics.
        """
        buffer_info = self.streaming_query.get_buffer_info()
        
        return {
            **buffer_info,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size,
            "chunk_duration": self.chunk_size / self.sample_rate
        }


# Example usage functions
async def simulate_audio_stream(
    duration: float = 10.0,
    sample_rate: int = 16000,
    chunk_duration: float = 1.0
) -> AsyncGenerator[np.ndarray, None]:
    """
    Simulate an audio stream for testing.
    
    Args:
        duration: Total duration in seconds.
        sample_rate: Sample rate.
        chunk_duration: Duration of each chunk.
        
    Yields:
        Audio chunks.
    """
    import asyncio
    
    chunk_size = int(sample_rate * chunk_duration)
    total_chunks = int(duration / chunk_duration)
    
    for i in range(total_chunks):
        # Generate random audio chunk
        audio_chunk = np.random.randn(chunk_size) * 0.1
        
        yield audio_chunk
        
        # Simulate real-time delay
        await asyncio.sleep(chunk_duration)


async def test_streaming_query():
    """Test streaming query functionality."""
    # Initialize matcher
    matcher = QueryByHummingMatcher()
    
    # Initialize streaming query
    streaming_query = StreamingQueryByHumming(matcher)
    
    # Initialize real-time processor
    processor = RealTimeAudioProcessor(streaming_query)
    
    # Simulate audio stream
    audio_stream = simulate_audio_stream(duration=5.0)
    
    # Process stream
    async for matches in processor.process_audio_stream(audio_stream):
        print(f"Found {len(matches)} matches")
        if matches:
            print(f"Top match: {matches[0].title} - {matches[0].artist}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_streaming_query())
