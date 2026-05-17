"""
Intent reconstruction and query type detection module.

Per Strategy doc Sections 11-12, 18:
- Reconstructs structured hiring intent from conversation history
- Detects query type (clarification, recommendation, refinement, comparison, refusal)
- Uses rule-based sufficiency checks + retrieval confidence thresholds
"""
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple

from app.llm import llm_client
from app.prompts import INTENT_EXTRACTION_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Keywords that indicate off-topic / refusal scenarios
REFUSAL_KEYWORDS = [
    "legal", "lawsuit", "lawyer", "attorney", "sue", "discriminat",
    "comply", "compliance", "regulation", "gdpr", "privacy law",
    "salary", "compensation", "negotiate", "offer letter",
    "ignore previous", "ignore above", "forget your instructions",
    "pretend you are", "act as", "you are now", "new instructions",
    "system prompt", "reveal your prompt", "bypass",
    "what is the meaning of life", "tell me a joke", "write a poem",
    "weather", "stock market", "recipe", "sports",
]

# Keywords that indicate comparison intent
COMPARISON_KEYWORDS = [
    "compare", "comparison", "difference between", "differences between",
    "versus", " vs ", " vs.", "how does .* compare", "which is better",
    "what's the difference", "what is the difference",
    "pros and cons", "contrast",
]

# Keywords that indicate refinement
REFINEMENT_KEYWORDS = [
    "actually", "instead", "change", "also add", "add personality",
    "remove", "exclude", "only remote", "only online", "shorter",
    "no coding", "no personality", "no technical",
    "what about", "can you also", "include", "but also",
    "more focus on", "less focus on", "switch to",
    "update", "modify", "adjust",
]


def classify_query_type(messages: List[Dict[str, str]]) -> str:
    """
    Quickly classify the query type from the latest user message.
    Uses rule-based detection first, falls back to LLM.
    
    Per Strategy Section 18: Query Type Detection.
    """
    if not messages:
        return "clarification_needed"
    
    last_msg = messages[-1].get("content", "").lower().strip()
    
    # Check for refusal / off-topic
    for keyword in REFUSAL_KEYWORDS:
        if keyword in last_msg:
            return "refusal"
    
    # Check for comparison
    for pattern in COMPARISON_KEYWORDS:
        if re.search(pattern, last_msg, re.IGNORECASE):
            return "comparison"
    
    # Check if this is a refinement (only if assistant previously gave recommendations)
    # Look for signals that the assistant already provided a shortlist
    has_prior_recommendations = any(
        msg.get("role") == "assistant" and (
            "recommend" in msg.get("content", "").lower() or
            "assessment" in msg.get("content", "").lower() or
            "here are" in msg.get("content", "").lower() or
            "shortlist" in msg.get("content", "").lower() or
            "shl.com" in msg.get("content", "").lower()
        )
        for msg in messages[:-1]
    )
    
    if has_prior_recommendations:
        for keyword in REFINEMENT_KEYWORDS:
            if keyword in last_msg:
                return "refinement"
    
    # Check if first message or very vague
    user_messages = [m for m in messages if m.get("role") == "user"]
    if len(user_messages) == 1:
        # First user message - check if sufficient
        if _is_too_vague(last_msg):
            return "clarification_needed"
        return "recommendation"
    
    # Multiple user messages - likely enough context has been gathered
    return "recommendation"


def _is_too_vague(message: str) -> bool:
    """
    Check if a message is too vague for recommendations.
    Per Strategy Section 12: Rule-Based Sufficiency.
    """
    vague_patterns = [
        r"^i need (?:an |some )?assess",
        r"^(?:help|find|show|get|give) (?:me )?(?:an |some )?assess",
        r"^what assess",
        r"^recommend",
        r"^suggest",
        r"^i('m| am) (?:looking|searching)",
        r"^hi$", r"^hello$", r"^hey$",
        r"^help$",
    ]
    
    message_lower = message.lower().strip()
    
    # Very short messages are likely vague
    if len(message_lower.split()) <= 4:
        for pattern in vague_patterns:
            if re.match(pattern, message_lower):
                return True
    
    # Check if message contains any role/skill/domain indicators
    has_role = bool(re.search(
        r'(developer|engineer|manager|analyst|designer|accountant|sales|'
        r'marketing|nurse|doctor|teacher|driver|operator|technician|'
        r'administrator|executive|director|supervisor|consultant|'
        r'scientist|researcher|writer|editor|recruiter|hr |'
        r'customer service|support|leadership|data|software|'
        r'mechanical|electrical|civil|chemical|industrial|'
        r'finance|banking|accounting|audit|compliance|'
        r'java|python|javascript|\.net|c\+\+|react|angular|'
        r'sql|database|cloud|devops|agile|machine learning|ai |'
        r'project manage|product manage|business analyst|'
        r'full.?stack|front.?end|back.?end|qa |testing|'
        r'hiring .{3,}|looking for .{3,}|job description|'
        r'entry.?level|junior|senior|mid.?level|intern|graduate)',
        message_lower
    ))
    
    has_skill = bool(re.search(
        r'(communication|leadership|problem.?solving|analytical|'
        r'teamwork|collaboration|presentation|negotiation|'
        r'critical thinking|decision.?making|creativity|'
        r'attention to detail|time management|'
        r'cognitive|personality|behavioral|aptitude|'
        r'verbal|numerical|spatial|mechanical|'
        r'stakeholder|client.?facing|customer)',
        message_lower
    ))
    
    # If message has job description text (longer text with relevant keywords)
    if len(message_lower.split()) > 15:
        return False  # Long messages usually have enough context
    
    return not (has_role or has_skill)


def extract_intent(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Extract structured hiring intent from conversation history.
    Per Strategy Section 11: Intent Reconstruction.
    
    Uses LLM for complex extraction, with rule-based fallback.
    """
    # Format conversation for LLM
    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in messages
    )
    
    try:
        prompt = INTENT_EXTRACTION_PROMPT.format(conversation=conversation_text)
        intent = llm_client.generate_json(prompt, temperature=0.0)
        
        # Normalize fields: LLM may return null for list/dict fields.
        # setdefault doesn't help when the key exists with value None,
        # so we explicitly coalesce nulls.
        intent["role"] = intent.get("role") or None
        intent["domain"] = intent.get("domain") or None
        intent["seniority"] = intent.get("seniority") or None
        intent["skills"] = intent.get("skills") or []
        intent["assessment_preferences"] = intent.get("assessment_preferences") or []
        intent["constraints"] = intent.get("constraints") or {}
        intent["query_type"] = intent.get("query_type") or "recommendation"
        intent["comparison_items"] = intent.get("comparison_items") or []
        intent.setdefault("is_sufficient", True)
        
        return intent
        
    except Exception as e:
        logger.error(f"Intent extraction failed: {e}")
        return _fallback_intent_extraction(messages)


def _fallback_intent_extraction(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Rule-based fallback intent extraction when LLM fails.
    """
    all_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    
    skills = []
    # Extract technology skills
    tech_patterns = [
        "java", "python", "javascript", ".net", "c#", "c++", "react", "angular",
        "sql", "database", "cloud", "aws", "azure", "devops", "agile",
        "machine learning", "ai", "data science", "hadoop", "spark",
    ]
    for tech in tech_patterns:
        if tech in all_text:
            skills.append(tech)
    
    # Extract soft skills
    soft_patterns = [
        "communication", "leadership", "problem solving", "analytical",
        "teamwork", "collaboration", "presentation", "negotiation",
        "critical thinking", "decision making",
    ]
    for skill in soft_patterns:
        if skill in all_text:
            skills.append(skill)
    
    return {
        "role": None,
        "domain": None,
        "seniority": None,
        "skills": skills,
        "assessment_preferences": [],
        "constraints": {},
        "query_type": classify_query_type(messages),
        "comparison_items": [],
        "is_sufficient": len(skills) > 0 or len(all_text.split()) > 15,
    }


def build_search_query(intent: Dict[str, Any]) -> str:
    """
    Build a natural language search query from extracted intent.
    Per Strategy Section 10: Semantic Query Construction.
    """
    parts = []
    
    if intent.get("role"):
        parts.append(intent["role"])
    
    if intent.get("domain"):
        parts.append(f"in {intent['domain']}")
    
    if intent.get("seniority"):
        parts.append(f"{intent['seniority']} level")
    
    if intent.get("skills"):
        parts.append(f"skills: {', '.join(intent['skills'])}")
    
    if intent.get("assessment_preferences"):
        pref_text = ", ".join(intent["assessment_preferences"])
        parts.append(f"assessment types: {pref_text}")
    
    if not parts:
        # If we have no structured info, use the raw user messages
        return "general assessment"
    
    return " ".join(parts)


def build_metadata_filters(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build metadata filters from intent constraints.
    Per Strategy Section 8: Metadata filtering.
    """
    filters = {}
    constraints = intent.get("constraints", {})
    
    if constraints.get("remote_only"):
        filters["remote"] = True
    
    if constraints.get("max_duration_minutes"):
        filters["max_duration"] = constraints["max_duration_minutes"]
    
    if constraints.get("languages"):
        filters["languages"] = constraints["languages"]
    
    # Map assessment preferences to catalog keys
    pref_to_keys = {
        "technical": ["Knowledge & Skills"],
        "personality": ["Personality & Behavior"],
        "cognitive": ["Ability & Aptitude"],
        "behavioral": ["Personality & Behavior", "Competencies"],
        "situational": ["Biodata & Situational Judgment"],
        "simulation": ["Simulations"],
        "aptitude": ["Ability & Aptitude"],
        "competency": ["Competencies"],
    }
    
    prefs = intent.get("assessment_preferences") or []
    if prefs:
        key_set = set()
        for pref in prefs:
            pref_lower = pref.lower()
            for keyword, keys in pref_to_keys.items():
                if keyword in pref_lower:
                    key_set.update(keys)
        if key_set:
            filters["keys"] = list(key_set)
    
    # Map seniority to job levels
    seniority_map = {
        "entry": ["Entry-Level", "Graduate"],
        "junior": ["Entry-Level", "Graduate"],
        "intern": ["Entry-Level", "Graduate"],
        "graduate": ["Graduate"],
        "mid": ["Mid-Professional", "Professional Individual Contributor"],
        "senior": ["Mid-Professional", "Professional Individual Contributor", "Manager"],
        "lead": ["Manager", "Front Line Manager"],
        "manager": ["Manager", "Front Line Manager"],
        "director": ["Director"],
        "executive": ["Executive", "Director"],
        "supervisor": ["Supervisor", "Front Line Manager"],
    }
    
    seniority = (intent.get("seniority") or "").lower()
    for key, levels in seniority_map.items():
        if key in seniority:
            filters["job_levels"] = levels
            break
    
    return filters


def extract_comparison_items(messages: List[Dict[str, str]]) -> List[str]:
    """Extract assessment names from a comparison request."""
    last_msg = messages[-1].get("content", "")
    
    # Common patterns: "Compare X and Y", "X vs Y", "difference between X and Y"
    patterns = [
        r"(?:compare|contrast)\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+?)(?:\?|$|\.)",
        r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$|\.)",
        r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$|\.)",
        # Looser patterns for short names like "OPQ and GSA"
        r"(?:compare|difference|between).*?([A-Z][A-Za-z0-9+.-]{1,30})\s+(?:and|vs\.?|versus|with)\s+([A-Z][A-Za-z0-9+.-]{1,30})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, last_msg, re.IGNORECASE)
        if match:
            item1 = match.group(1).strip().strip('"').strip("'")
            item2 = match.group(2).strip().strip('"').strip("'").rstrip('?.')
            if item1 and item2:
                return [item1, item2]
    
    return []
