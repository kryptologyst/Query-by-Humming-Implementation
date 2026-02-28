# Query by Humming Implementation

A Privacy-preserving implementation of Query by Humming for Music Information Retrieval (MIR). This project allows users to search for songs by humming melodies using advanced audio processing and machine learning techniques.

## Privacy Disclaimer

**IMPORTANT**: This is a research and educational demonstration project. This software is NOT intended for biometric identification or voice cloning in production environments. Any misuse of this technology for unauthorized voice cloning, impersonation, or biometric surveillance is strictly prohibited and may violate privacy laws and ethical guidelines.

## Features

- **Advanced Melody Matching**: Dynamic Time Warping (DTW), embedding-based retrieval, and neural network approaches
- **Multiple Feature Extraction**: MFCC, chroma, spectral features, and learned embeddings
- **Comprehensive Evaluation**: mAP@k, Recall@k, DTW distance statistics
- **Interactive Demo**: Streamlit/Gradio interface for audio upload and real-time matching
- **Privacy-Preserving**: No raw audio storage, optional metadata anonymization
- **Modern Architecture**: PyTorch 2.x, type hints, comprehensive testing

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Query-by-Humming-Implementation.git
cd Query-by-Humming-Implementation

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.query_matcher import QueryByHummingMatcher
from src.data.dataset import HummingDataset

# Initialize the matcher
matcher = QueryByHummingMatcher()

# Load your humming query
query_audio, sr = librosa.load("path/to/humming.wav")

# Find matches
matches = matcher.search(query_audio, top_k=5)
print(f"Best match: {matches[0].title} (confidence: {matches[0].confidence:.3f})")
```

### Demo Interface

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py

# Or launch Gradio demo
python demo/gradio_app.py
```

## Project Structure

```
query-by-humming/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── data/              # Data loading and processing
│   ├── features/          # Feature extraction
│   ├── losses/            # Loss functions
│   ├── metrics/           # Evaluation metrics
│   ├── decoding/          # Decoding algorithms
│   ├── train/             # Training scripts
│   ├── eval/              # Evaluation scripts
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   ├── wav/              # Audio files
│   ├── meta.csv          # Metadata
│   └── annotations/      # Optional annotations
├── configs/              # Configuration files
├── scripts/              # Training/evaluation scripts
├── notebooks/            # Jupyter notebooks
├── tests/                # Test suite
├── assets/               # Generated artifacts
├── demo/                 # Demo applications
└── docs/                 # Documentation
```

## Dataset Schema

The project expects audio data in the following format:

- **Audio files**: WAV format, 16kHz sampling rate
- **Metadata CSV**: Contains columns for `id`, `path`, `title`, `artist`, `genre`, `split`
- **Optional annotations**: JSON files with pitch contours, note sequences

### Synthetic Dataset Generation

If no dataset is available, the system can generate a synthetic corpus:

```python
from src.data.synthetic import generate_synthetic_dataset

# Generate synthetic humming queries and reference songs
dataset = generate_synthetic_dataset(
    n_reference_songs=100,
    n_humming_queries=50,
    output_dir="data/synthetic"
)
```

## Training and Evaluation

### Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/advanced_model.yaml
```

### Evaluation

```bash
# Run evaluation
python scripts/eval.py --checkpoint checkpoints/best_model.pt

# Generate leaderboard
python scripts/generate_leaderboard.py
```

## Models

### 1. Dynamic Time Warping (DTW)
- Classic approach for sequence alignment
- Robust to tempo variations
- Good baseline for comparison

### 2. Embedding-Based Retrieval
- Learned embeddings using triplet loss
- Faster inference than DTW
- Better generalization

### 3. Neural Network Matching
- End-to-end trainable models
- Attention mechanisms for alignment
- State-of-the-art performance

## Evaluation Metrics

- **mAP@k**: Mean Average Precision at k
- **Recall@k**: Recall at k
- **DTW Distance**: Dynamic Time Warping distance statistics
- **Latency**: Inference time per query
- **Throughput**: Queries per second

## Configuration

The project uses OmegaConf for configuration management. Key configuration files:

- `configs/base.yaml`: Base configuration
- `configs/models/`: Model-specific configurations
- `configs/data/`: Dataset configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{query_by_humming,
  title={Query by Humming Implementation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Query-by-Humming-Implementation}
}
```

## Acknowledgments

- Librosa for audio processing
- PyTorch for deep learning
- The MIR community for datasets and benchmarks
# Query-by-Humming-Implementation
