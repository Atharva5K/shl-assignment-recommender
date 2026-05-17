"""
Retrieval module: FAISS-based semantic search with metadata filtering.

Per Strategy doc Sections 8-9:
- Hybrid retrieval: semantic search + metadata filtering
- FAISS stores vectors only, metadata stored separately
- Index prebuilt offline and loaded at startup
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    FAISS_TOP_K,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    FAISS-based semantic retrieval system with metadata filtering.
    Implements hybrid retrieval as described in Strategy Section 8.
    """
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []
        self._loaded = False
    
    def load(self):
        """Load embedding model, FAISS index, and metadata."""
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            logger.info("Loading prebuilt FAISS index...")
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            logger.info("No prebuilt index found. Building from catalog...")
            self._build_index()
        
        self._loaded = True
    
    def _build_index(self):
        """Build FAISS index from catalog (fallback if prebuilt not found)."""
        from app.catalog import prepare_catalog
        
        documents, metadata_list, _ = prepare_catalog()
        self.metadata = metadata_list
        
        # Encode documents
        logger.info(f"Encoding {len(documents)} documents...")
        embeddings = self.model.encode(
            documents, 
            show_progress_bar=True, 
            normalize_embeddings=True,  # Normalize for cosine similarity via inner product
            batch_size=64,
        )
        
        # Build FAISS index (Inner Product on normalized vectors = cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        # Save
        Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Built and saved FAISS index with {self.index.ntotal} vectors")
    
    def search(
        self, 
        query: str, 
        top_k: int = FAISS_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with optional metadata filtering.
        
        Args:
            query: Natural language search query
            top_k: Number of candidates to retrieve
            filters: Optional metadata filters (job_levels, keys, remote, etc.)
        
        Returns:
            List of results with metadata + similarity scores
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")
        
        # Encode query
        query_embedding = self.model.encode(
            [query], 
            normalize_embeddings=True,
        ).astype(np.float32)
        
        # Search FAISS - retrieve more candidates than needed for filtering
        search_k = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            meta = self.metadata[idx].copy()
            meta["similarity_score"] = float(score)
            
            # Apply metadata filters
            if filters and not self._passes_filters(meta, filters):
                continue
            
            results.append(meta)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Look up an assessment by name (for comparison queries).
        Handles exact matches, substrings, abbreviations (OPQ → OPQ32r),
        and word-prefix matching (GSA → Global Skills Assessment).
        """
        name_lower = name.lower().strip()
        best_match = None
        best_score = 0.0
        
        for meta in self.metadata:
            meta_name = meta.get("name", "")
            meta_name_lower = meta_name.lower()
            
            # Priority 1: Exact match
            if meta_name_lower == name_lower:
                return meta.copy()
            
            # Priority 2: Assessment name starts with the search term
            # e.g., "OPQ" matches "OPQ32r", "OPQ Manager Plus Report"
            if meta_name_lower.startswith(name_lower):
                # Shorter names = better match (OPQ32r over OPQ Leadership Report)
                score = 100 - len(meta_name)
                if score > best_score:
                    best_score = score
                    best_match = meta.copy()
                continue
            
            # Priority 3: Search term is a substring
            if name_lower in meta_name_lower:
                score = 50 - abs(len(meta_name) - len(name))
                if score > best_score:
                    best_score = score
                    best_match = meta.copy()
                continue
            
            # Priority 4: Abbreviation matching
            # Check if search term matches initials of words in the name
            # e.g., "GSA" → "Global Skills Assessment"
            words = meta_name.split()
            if len(words) >= 2:
                initials = "".join(w[0] for w in words if w[0].isupper())
                if initials.lower() == name_lower or name_lower == initials.lower():
                    score = 80
                    if score > best_score:
                        best_score = score
                        best_match = meta.copy()
                    continue
            
            # Priority 5: Any word in the name starts with the search term
            for word in words:
                if word.lower().startswith(name_lower) and len(name_lower) >= 2:
                    score = 30
                    if score > best_score:
                        best_score = score
                        best_match = meta.copy()
                    break
        
        return best_match
    
    def get_retrieval_confidence(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate retrieval confidence.
        Per Strategy Section 12: Retrieval Confidence Threshold.
        
        Returns confidence metrics:
        - top_score: highest similarity score
        - avg_top5: average of top 5 scores
        - spread: difference between top and 5th score
        """
        if not results:
            return {"top_score": 0.0, "avg_top5": 0.0, "spread": 0.0}
        
        scores = [r["similarity_score"] for r in results[:10]]
        top_score = scores[0] if scores else 0.0
        avg_top5 = np.mean(scores[:5]) if len(scores) >= 5 else np.mean(scores)
        spread = (scores[0] - scores[min(4, len(scores)-1)]) if len(scores) > 1 else 0.0
        
        return {
            "top_score": float(top_score),
            "avg_top5": float(avg_top5),
            "spread": float(spread),
        }
    
    def _passes_filters(self, meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a result passes metadata filters."""
        # Job level filter
        if "job_levels" in filters and filters["job_levels"]:
            required_levels = set(l.lower() for l in filters["job_levels"])
            item_levels = set(l.lower() for l in meta.get("job_levels", []))
            if not required_levels.intersection(item_levels):
                return False
        
        # Assessment category filter
        if "keys" in filters and filters["keys"]:
            required_keys = set(k.lower() for k in filters["keys"])
            item_keys = set(k.lower() for k in meta.get("keys", []))
            if not required_keys.intersection(item_keys):
                return False
        
        # Remote filter
        if "remote" in filters and filters["remote"] is not None:
            if meta.get("remote") != filters["remote"]:
                return False
        
        # Language filter
        if "languages" in filters and filters["languages"]:
            required_langs = set(l.lower() for l in filters["languages"])
            item_langs = set(l.lower() for l in meta.get("languages", []))
            if required_langs and item_langs and not required_langs.intersection(item_langs):
                return False
        
        # Duration filter (max minutes)
        if "max_duration" in filters and filters["max_duration"] is not None:
            item_duration = meta.get("duration_minutes")
            if item_duration and item_duration > filters["max_duration"]:
                return False
        
        return True


# Singleton instance
retriever = SemanticRetriever()
