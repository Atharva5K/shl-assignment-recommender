"""
Catalog module: loads, processes, and transforms the SHL product catalog 
into semantic text documents for embedding.

Per Strategy doc Section 7: "Raw JSON is NOT directly embedded. Instead,
each assessment is transformed into a semantic text representation."
"""
import json
import logging
from typing import Dict, List, Any

from app.config import CATALOG_PATH

logger = logging.getLogger(__name__)


def load_catalog() -> List[Dict[str, Any]]:
    """Load the raw SHL product catalog from JSON."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        # strict=False to handle potential control characters in descriptions
        catalog = json.loads(f.read(), strict=False)
    
    # Filter out items with bad status
    valid = [item for item in catalog if item.get("status") == "ok"]
    logger.info(f"Loaded {len(valid)} valid assessments from catalog (of {len(catalog)} total)")
    return valid


def build_semantic_document(item: Dict[str, Any]) -> str:
    """
    Transform a catalog item into a natural-language semantic document
    for embedding. This follows Strategy doc Section 7.
    
    Embedding models perform better on natural language than raw JSON.
    """
    parts = []
    
    # Assessment name
    name = item.get("name", "Unknown")
    parts.append(f"Assessment Name: {name}")
    
    # Description (most semantically rich field)
    desc = item.get("description", "").strip()
    if desc:
        parts.append(f"\nDescription:\n{desc}")
    
    # Job levels
    levels = item.get("job_levels", [])
    if levels:
        parts.append(f"\nJob Levels:\n{', '.join(levels)}")
    
    # Assessment categories (keys)
    keys = item.get("keys", [])
    if keys:
        parts.append(f"\nAssessment Categories:\n{', '.join(keys)}")
    
    # Duration
    duration = item.get("duration", "").strip()
    if duration:
        parts.append(f"\nDuration: {duration}")
    
    # Remote support
    remote = item.get("remote", "")
    if remote:
        parts.append(f"\nRemote Testing: {'Yes' if remote == 'yes' else 'No'}")
    
    # Adaptive
    adaptive = item.get("adaptive", "")
    if adaptive:
        parts.append(f"\nAdaptive: {'Yes' if adaptive == 'yes' else 'No'}")
    
    # Languages
    languages = item.get("languages", [])
    if languages:
        parts.append(f"\nLanguages: {', '.join(languages)}")
    
    return "\n".join(parts)


def build_metadata(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured metadata for filtering and reranking.
    Per Strategy doc Section 8: metadata fields stored separately
    for exact filtering.
    """
    # Parse duration to minutes (integer) for filtering
    duration_minutes = None
    duration_raw = item.get("duration_raw", "") or item.get("duration", "")
    if duration_raw:
        import re
        match = re.search(r"(\d+)", duration_raw)
        if match:
            duration_minutes = int(match.group(1))
    
    return {
        "entity_id": item.get("entity_id", ""),
        "name": item.get("name", ""),
        "url": item.get("link", ""),
        "description": item.get("description", ""),
        "job_levels": item.get("job_levels", []),
        "languages": item.get("languages", []),
        "duration": item.get("duration", ""),
        "duration_minutes": duration_minutes,
        "remote": item.get("remote", "") == "yes",
        "adaptive": item.get("adaptive", "") == "yes",
        "keys": item.get("keys", []),  # assessment categories
    }


def prepare_catalog() -> tuple:
    """
    Load catalog, build semantic documents and metadata.
    Returns (documents: List[str], metadata: List[Dict], catalog: List[Dict])
    """
    catalog = load_catalog()
    documents = []
    metadata_list = []
    
    for item in catalog:
        doc = build_semantic_document(item)
        meta = build_metadata(item)
        documents.append(doc)
        metadata_list.append(meta)
    
    logger.info(f"Prepared {len(documents)} semantic documents and metadata entries")
    return documents, metadata_list, catalog
