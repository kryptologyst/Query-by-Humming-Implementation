"""
FastAPI server for Query by Humming.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from src.models.query_matcher import QueryByHummingMatcher, QueryMatch
from src.utils.audio import load_audio, normalize_audio
from src.utils.device import get_device_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Query by Humming API",
    description="Music Information Retrieval API for Query by Humming",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global matcher instance
matcher: Optional[QueryByHummingMatcher] = None


class QueryRequest(BaseModel):
    """Request model for query."""
    top_k: int = 10
    threshold: float = 0.5


class QueryResponse(BaseModel):
    """Response model for query."""
    matches: List[Dict]
    processing_time: float
    device_info: Dict


class DatabaseInfo(BaseModel):
    """Database information model."""
    num_songs: int
    songs: List[Dict]


@app.on_event("startup")
async def startup_event():
    """Initialize the matcher on startup."""
    global matcher
    
    try:
        # Load configuration
        config_path = Path("configs/base.yaml")
        if config_path.exists():
            config = OmegaConf.load(config_path)
        else:
            # Use default config
            config = DictConfig({
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
        
        matcher = QueryByHummingMatcher(config)
        
        # Try to load existing database
        db_path = Path("assets/reference_database.pkl")
        if db_path.exists():
            matcher.load_database(db_path)
            logger.info(f"Loaded existing database from {db_path}")
        else:
            logger.info("No existing database found. Use /load-database endpoint to load songs.")
        
        logger.info("Query by Humming API started successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Query by Humming API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    device_info = get_device_info()
    
    return {
        "status": "healthy",
        "device_info": device_info,
        "database_loaded": matcher is not None and len(matcher.reference_database) > 0
    }


@app.get("/database-info", response_model=DatabaseInfo)
async def get_database_info():
    """Get database information."""
    if matcher is None:
        raise HTTPException(status_code=500, detail="Matcher not initialized")
    
    db_info = matcher.get_database_info()
    return DatabaseInfo(**db_info)


@app.post("/query", response_model=QueryResponse)
async def query_by_humming(
    file: UploadFile = File(...),
    top_k: int = 10,
    threshold: float = 0.5
):
    """
    Query by humming endpoint.
    
    Args:
        file: Audio file upload.
        top_k: Number of top results to return.
        threshold: Confidence threshold.
        
    Returns:
        Query results with matches.
    """
    if matcher is None:
        raise HTTPException(status_code=500, detail="Matcher not initialized")
    
    if len(matcher.reference_database) == 0:
        raise HTTPException(status_code=400, detail="No reference songs in database")
    
    import time
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load and process audio
            audio, sr = load_audio(tmp_path)
            
            # Search for matches
            matches = matcher.search(
                query_audio=audio,
                top_k=top_k,
                threshold=threshold
            )
            
            # Convert matches to dictionaries
            matches_dict = []
            for match in matches:
                matches_dict.append({
                    "title": match.title,
                    "artist": match.artist,
                    "distance": match.distance,
                    "confidence": match.confidence,
                    "metadata": match.metadata
                })
            
            processing_time = time.time() - start_time
            device_info = get_device_info()
            
            return QueryResponse(
                matches=matches_dict,
                processing_time=processing_time,
                device_info=device_info
            )
            
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/load-database")
async def load_database():
    """Load reference songs from synthetic dataset."""
    if matcher is None:
        raise HTTPException(status_code=500, detail="Matcher not initialized")
    
    try:
        import pandas as pd
        
        # Check if synthetic dataset exists
        synthetic_path = Path("data/synthetic")
        if not synthetic_path.exists():
            raise HTTPException(status_code=404, detail="Synthetic dataset not found")
        
        meta_path = synthetic_path / "meta.csv"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Metadata file not found")
        
        # Load metadata
        meta_df = pd.read_csv(meta_path)
        reference_df = meta_df[meta_df["split"] == "reference"]
        
        # Clear existing database
        matcher.clear_database()
        
        # Load reference songs
        for _, row in reference_df.iterrows():
            matcher.add_reference_song(
                audio_path=row["audio_path"],
                title=row["title"],
                artist=row["artist"],
                metadata={"genre": row.get("genre", "unknown")}
            )
        
        return {
            "message": f"Loaded {len(reference_df)} reference songs",
            "num_songs": len(reference_df)
        }
        
    except Exception as e:
        logger.error(f"Error loading database: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading database: {str(e)}")


@app.post("/clear-database")
async def clear_database():
    """Clear the reference database."""
    if matcher is None:
        raise HTTPException(status_code=500, detail="Matcher not initialized")
    
    matcher.clear_database()
    
    return {"message": "Database cleared successfully"}


@app.post("/save-database")
async def save_database():
    """Save the reference database."""
    if matcher is None:
        raise HTTPException(status_code=500, detail="Matcher not initialized")
    
    try:
        output_dir = Path("assets")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = output_dir / "reference_database.pkl"
        matcher.save_database(db_path)
        
        return {"message": f"Database saved to {db_path}"}
        
    except Exception as e:
        logger.error(f"Error saving database: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving database: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "demo.fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
