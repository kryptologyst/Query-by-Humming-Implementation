"""
Streamlit demo for Query by Humming.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.query_matcher import QueryByHummingMatcher, QueryMatch
from src.data.synthetic import SyntheticDatasetGenerator
from src.utils.device import get_device_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Query by Humming Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Privacy Disclaimer</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <p>This software is NOT intended for biometric identification or voice cloning in production environments. 
    Any misuse of this technology for unauthorized voice cloning, impersonation, or biometric surveillance 
    is strictly prohibited and may violate privacy laws and ethical guidelines.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎵 Query by Humming Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Initialize session state
if "matcher" not in st.session_state:
    st.session_state.matcher = None
if "database_loaded" not in st.session_state:
    st.session_state.database_loaded = False
if "reference_songs" not in st.session_state:
    st.session_state.reference_songs = []

# Device information
device_info = get_device_info()
st.sidebar.markdown("### Device Information")
st.sidebar.write(f"CUDA Available: {device_info['cuda_available']}")
st.sidebar.write(f"MPS Available: {device_info['mps_available']}")

# Model configuration
st.sidebar.markdown("### Model Configuration")
feature_type = st.sidebar.selectbox(
    "Feature Type",
    ["mfcc", "chroma", "spectral", "pitch"],
    index=0
)

n_mfcc = st.sidebar.slider("Number of MFCC coefficients", 5, 20, 13)
include_delta = st.sidebar.checkbox("Include Delta Features", True)
include_delta_delta = st.sidebar.checkbox("Include Delta-Delta Features", True)

dtw_window = st.sidebar.slider("DTW Window Size", 5, 50, 10)
distance_metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean", "cosine", "manhattan"],
    index=0
)

top_k = st.sidebar.slider("Number of Results", 1, 20, 10)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Initialize matcher
if st.sidebar.button("Initialize Model") or st.session_state.matcher is None:
    try:
        from omegaconf import DictConfig
        
        config = DictConfig({
            "model": {
                "features": {
                    "type": feature_type,
                    "n_mfcc": n_mfcc,
                    "delta": include_delta,
                    "delta_delta": include_delta_delta
                },
                "dtw": {
                    "distance_metric": distance_metric,
                    "window_size": dtw_window,
                    "step_pattern": "symmetric2"
                },
                "matching": {
                    "top_k": top_k,
                    "threshold": confidence_threshold,
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
        
        st.session_state.matcher = QueryByHummingMatcher(config)
        st.sidebar.success("Model initialized successfully!")
        
    except Exception as e:
        st.sidebar.error(f"Error initializing model: {e}")

# Database management
st.sidebar.markdown("### Database Management")

# Generate synthetic dataset
if st.sidebar.button("Generate Synthetic Dataset"):
    with st.spinner("Generating synthetic dataset..."):
        try:
            generator = SyntheticDatasetGenerator()
            generator.generate_dataset(
                n_reference_songs=50,
                n_humming_queries=20,
                output_dir="data/synthetic"
            )
            st.sidebar.success("Synthetic dataset generated!")
        except Exception as e:
            st.sidebar.error(f"Error generating dataset: {e}")

# Load reference songs
if st.sidebar.button("Load Reference Songs"):
    if st.session_state.matcher is None:
        st.sidebar.error("Please initialize the model first!")
    else:
        try:
            # Load synthetic dataset if available
            synthetic_path = Path("data/synthetic")
            if synthetic_path.exists():
                meta_df = pd.read_csv(synthetic_path / "meta.csv")
                reference_df = meta_df[meta_df["split"] == "reference"]
                
                for _, row in reference_df.iterrows():
                    st.session_state.matcher.add_reference_song(
                        audio_path=row["audio_path"],
                        title=row["title"],
                        artist=row["artist"],
                        metadata={"genre": row.get("genre", "unknown")}
                    )
                
                st.session_state.database_loaded = True
                st.session_state.reference_songs = reference_df.to_dict("records")
                st.sidebar.success(f"Loaded {len(reference_df)} reference songs!")
            else:
                st.sidebar.warning("No synthetic dataset found. Please generate one first.")
                
        except Exception as e:
            st.sidebar.error(f"Error loading reference songs: {e}")

# Clear database
if st.sidebar.button("Clear Database"):
    if st.session_state.matcher:
        st.session_state.matcher.clear_database()
        st.session_state.database_loaded = False
        st.session_state.reference_songs = []
        st.sidebar.success("Database cleared!")

# Database info
if st.session_state.database_loaded:
    st.sidebar.markdown("### Database Info")
    db_info = st.session_state.matcher.get_database_info()
    st.sidebar.write(f"Number of songs: {db_info['num_songs']}")

# Main content
if not st.session_state.database_loaded:
    st.warning("Please load reference songs from the sidebar to start using the demo.")
    st.stop()

# Query section
st.markdown("## 🎤 Query by Humming")

# File upload
uploaded_file = st.file_uploader(
    "Upload an audio file (WAV, MP3, etc.)",
    type=["wav", "mp3", "flac", "m4a"],
    help="Upload a humming query or any audio file to search for matches"
)

# Audio recording (if available)
if st.checkbox("Record Audio"):
    st.markdown("**Note:** Audio recording requires microphone access and may not work in all browsers.")
    
    # Simple audio recording placeholder
    st.markdown("""
    <div style="text-align: center; padding: 2rem; border: 2px dashed #ccc; border-radius: 0.5rem;">
        <p>🎤 Audio recording would be implemented here</p>
        <p><small>This requires additional JavaScript integration</small></p>
    </div>
    """, unsafe_allow_html=True)

# Process query
if uploaded_file is not None:
    st.markdown("### Audio Analysis")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load and analyze audio
        audio, sr = librosa.load(tmp_path, sr=16000)
        
        # Display audio info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{len(audio) / sr:.2f}s")
        with col2:
            st.metric("Sample Rate", f"{sr} Hz")
        with col3:
            st.metric("Samples", f"{len(audio):,}")
        
        # Display waveform
        st.markdown("#### Waveform")
        fig_waveform = go.Figure()
        fig_waveform.add_trace(go.Scatter(
            y=audio,
            mode='lines',
            name='Audio Waveform',
            line=dict(color='blue', width=1)
        ))
        fig_waveform.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude",
            height=300
        )
        st.plotly_chart(fig_waveform, use_container_width=True)
        
        # Display spectrogram
        st.markdown("#### Spectrogram")
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        fig_spec = go.Figure(data=go.Heatmap(
            z=librosa.amplitude_to_db(magnitude),
            colorscale='Viridis',
            x=librosa.frames_to_time(np.arange(magnitude.shape[1])),
            y=librosa.fft_frequencies(sr=sr),
            hoverongaps=False
        ))
        fig_spec.update_layout(
            title="Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            height=400
        )
        st.plotly_chart(fig_spec, use_container_width=True)
        
        # Search for matches
        if st.button("🔍 Search for Matches"):
            with st.spinner("Searching for matches..."):
                try:
                    matches = st.session_state.matcher.search(
                        query_audio=audio,
                        top_k=top_k,
                        threshold=confidence_threshold
                    )
                    
                    if matches:
                        st.markdown("### 🎯 Search Results")
                        
                        # Display results
                        for i, match in enumerate(matches):
                            with st.container():
                                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                                
                                with col1:
                                    st.markdown(f"**#{i+1}**")
                                
                                with col2:
                                    st.markdown(f"**{match.title}**")
                                    st.markdown(f"by {match.artist}")
                                
                                with col3:
                                    st.metric("Confidence", f"{match.confidence:.3f}")
                                
                                with col4:
                                    st.metric("Distance", f"{match.distance:.3f}")
                                
                                # Progress bar for confidence
                                st.progress(match.confidence)
                                st.markdown("---")
                        
                        # Results visualization
                        st.markdown("#### Results Visualization")
                        
                        # Confidence scores
                        confidences = [match.confidence for match in matches]
                        distances = [match.distance for match in matches]
                        titles = [f"{match.title} - {match.artist}" for match in matches]
                        
                        fig_results = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Confidence Scores", "DTW Distances"),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        fig_results.add_trace(
                            go.Bar(x=titles, y=confidences, name="Confidence"),
                            row=1, col=1
                        )
                        
                        fig_results.add_trace(
                            go.Bar(x=titles, y=distances, name="Distance"),
                            row=1, col=2
                        )
                        
                        fig_results.update_layout(
                            height=400,
                            showlegend=False,
                            xaxis_tickangle=-45
                        )
                        
                        st.plotly_chart(fig_results, use_container_width=True)
                        
                    else:
                        st.warning("No matches found above the confidence threshold.")
                        
                except Exception as e:
                    st.error(f"Error during search: {e}")
                    logger.error(f"Search error: {e}")
    
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        logger.error(f"Audio processing error: {e}")
    
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)

# Reference songs display
if st.session_state.reference_songs:
    st.markdown("## 📚 Reference Songs Database")
    
    # Create DataFrame for display
    ref_df = pd.DataFrame(st.session_state.reference_songs)
    
    # Display table
    st.dataframe(
        ref_df[["title", "artist", "genre"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Statistics
    st.markdown("#### Database Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Songs", len(ref_df))
    
    with col2:
        genres = ref_df["genre"].value_counts()
        st.metric("Most Common Genre", genres.index[0] if len(genres) > 0 else "N/A")
    
    with col3:
        artists = ref_df["artist"].nunique()
        st.metric("Unique Artists", artists)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Query by Humming Demo - Research and Educational Use Only</p>
    <p>Built with Streamlit, PyTorch, and Librosa</p>
</div>
""", unsafe_allow_html=True)
