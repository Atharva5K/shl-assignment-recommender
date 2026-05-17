"""
Script to prebuilt the FAISS index offline.
Per Strategy Section 9: "The index is prebuilt offline and loaded during FastAPI startup."

Run this script before deployment:
    python -m scripts.build_index
"""
import sys
import os
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Build and save the FAISS index from the catalog."""
    from app.catalog import prepare_catalog
    from app.config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_MODEL_NAME, DATA_DIR
    
    start = time.time()
    
    # Step 1: Load and process catalog
    logger.info("Loading catalog...")
    documents, metadata_list, catalog = prepare_catalog()
    logger.info(f"Processed {len(documents)} assessments")
    
    # Step 2: Print some sample documents for verification
    logger.info("\n--- Sample Semantic Document ---")
    logger.info(documents[0][:500])
    logger.info("--- End Sample ---\n")
    
    # Step 3: Load embedding model and encode
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    logger.info(f"Encoding {len(documents)} documents...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64,
    )
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    # Step 4: Build FAISS index
    logger.info("Building FAISS index...")
    import faiss
    import pickle
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product on normalized = cosine similarity
    index.add(embeddings.astype(np.float32))
    logger.info(f"FAISS index built with {index.ntotal} vectors, dimension {dimension}")
    
    # Step 5: Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"Saved FAISS index to {FAISS_INDEX_PATH}")
    
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_list, f)
    logger.info(f"Saved metadata to {METADATA_PATH}")
    
    # Step 6: Verify
    logger.info("\n--- Verification ---")
    test_queries = [
        "Java developer technical assessment",
        "leadership personality assessment for senior manager",
        "entry-level cognitive aptitude test",
        "customer service representative assessment",
        "data science machine learning skills test",
    ]
    
    for query in test_queries:
        q_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(q_embedding, 5)
        logger.info(f"\nQuery: '{query}'")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:
                logger.info(f"  {i+1}. [{score:.3f}] {metadata_list[idx]['name']}")
    
    elapsed = time.time() - start
    logger.info(f"\nIndex build completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
