"""
Recommendation pipeline: orchestrates intent extraction, retrieval,
reranking, and grounded LLM generation.

Per Strategy doc Sections 10, 13-17:
- Full recommendation pipeline
- Reranking strategy
- Refinement handling
- Comparison handling
- Grounded generation
"""
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple

from app.config import FINAL_TOP_K, SIMILARITY_THRESHOLD, CONFIDENCE_HIGH
from app.retrieval import retriever
from app.intent import (
    classify_query_type,
    extract_intent,
    build_search_query,
    build_metadata_filters,
    extract_comparison_items,
)
from app.llm import llm_client
from app.prompts import (
    SYSTEM_PROMPT,
    RECOMMENDATION_PROMPT,
    CLARIFICATION_PROMPT,
    COMPARISON_PROMPT,
    REFUSAL_PROMPT,
    get_test_type,
)

logger = logging.getLogger(__name__)


# ─── Conversational Sufficiency Check ───

def _needs_clarification(intent: Dict[str, Any], messages: List[Dict[str, str]]) -> Tuple[bool, str]:
    """
    Determine whether we should ask a clarifying question instead of recommending.
    
    This is the KEY function for making the agent conversational.
    We check what information we have and decide if it's enough
    to make a good recommendation.
    
    CRITICAL CONSTRAINT: The evaluator caps at 8 turns total (user + assistant).
    We must leave room for: recommendation + possible refinement + response.
    So we can spend AT MOST 2 turns on clarification.
    
    Turn budget (worst case):
      T1: User (vague)          T2: Assistant (clarify)
      T3: User (answers)        T4: Assistant (clarify again)
      T5: User (answers)        T6: Assistant (RECOMMEND)
      T7: User (refine)         T8: Assistant (update)
    
    Returns:
        (needs_clarification: bool, missing_field: str)
    """
    user_messages = [m for m in messages if m.get("role") == "user"]
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    num_user_msgs = len(user_messages)
    num_assistant_msgs = len(assistant_messages)
    total_turns = len(messages)  # All messages = all turns used so far
    
    # ── HARD LIMIT: If 5+ turns already used, we MUST recommend now.
    # Remaining budget: need at least 1 turn for our recommendation,
    # ideally 2-3 for refinement cycle. Never waste turns clarifying.
    if total_turns >= 5:
        return False, ""
    
    # ── Safety: don't over-ask. After 2 assistant messages (clarifications
    # or otherwise), stop asking and recommend.
    if num_assistant_msgs >= 2:
        return False, ""
    
    # ── What do we know?
    has_role = bool(intent.get("role"))
    has_seniority = bool(intent.get("seniority"))
    has_skills = len(intent.get("skills") or []) > 0
    has_prefs = len(intent.get("assessment_preferences") or []) > 0
    
    last_user_msg = user_messages[-1]["content"] if user_messages else ""
    word_count = len(last_user_msg.split())
    
    # ── Long messages (e.g., pasted job descriptions) have enough context
    if word_count > 30:
        return False, ""
    
    # ── First user message: be conversational, ask follow-ups
    if num_user_msgs == 1:
        # No role at all → must ask
        if not has_role:
            return True, "role"
        
        # Has role but missing BOTH seniority and skills → ask
        if not has_seniority and not has_skills:
            return True, "seniority_and_skills"
        
        # Has role + seniority but no skills, and message is brief → ask
        if has_seniority and not has_skills and word_count < 15:
            return True, "skills"
    
    # ── Second user message: only clarify if still missing role
    if num_user_msgs == 2:
        if not has_role:
            return True, "role"
        # By the second message we should have enough — proceed to recommend
        return False, ""
    
    # ── 3+ messages: always proceed
    return False, ""


def _generate_targeted_clarification(
    messages: List[Dict[str, str]], 
    intent: Dict[str, Any],
    missing_field: str,
) -> Dict[str, Any]:
    """
    Generate a context-appropriate clarification question based on what's missing.
    Falls back to LLM-generated clarification if needed.
    """
    has_role = bool(intent.get("role"))
    role = intent.get("role") or "this position"
    
    # Fast, deterministic clarification for common cases
    if missing_field == "role":
        reply = "I'd love to help you find the right assessments! Could you tell me what role or job function you're hiring for?"
    elif missing_field == "seniority_and_skills":
        reply = (
            f"Great, you're looking for assessments for a {role}. "
            f"To narrow down the best options, could you tell me:\n"
            f"1. What seniority level is this position? (e.g., entry-level, mid-level, senior)\n"
            f"2. Are there specific skills or competencies that are most important?"
        )
    elif missing_field == "skills":
        seniority = intent.get("seniority") or ""
        reply = (
            f"Got it — a {seniority} {role}. "
            f"Are there particular skills or competencies you'd like to assess? "
            f"For example, technical skills, leadership, communication, problem-solving, etc."
        )
    else:
        # Fallback to LLM-generated clarification
        return _handle_clarification_llm(messages, intent)
    
    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


# ─── Main Pipeline ───

def process_chat(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Main recommendation pipeline entry point.
    
    Per Strategy Section 10: Full pipeline from conversation to response.
    
    Args:
        messages: Full conversation history
    
    Returns:
        Response dict with reply, recommendations, and end_of_conversation
    """
    if not messages:
        return {
            "reply": "Hello! I'm the SHL Assessment Recommendation Agent. I can help you find the right SHL assessments for your hiring needs. What role or position are you looking to assess candidates for?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    
    # Step 1: Quick query type classification (rule-based)
    query_type = classify_query_type(messages)
    logger.info(f"Query type (rule-based): {query_type}")
    
    # Step 2: Handle refusal immediately without LLM intent extraction
    if query_type == "refusal":
        return _handle_refusal(messages)
    
    # Step 3: Handle comparison
    if query_type == "comparison":
        return _handle_comparison(messages)
    
    # Step 4: Extract full intent using LLM
    intent = extract_intent(messages)
    logger.info(f"Extracted intent: {json.dumps(intent, default=str)[:300]}")
    
    # Override query type from LLM intent if it detects refusal/comparison
    llm_query_type = intent.get("query_type", query_type)
    if llm_query_type == "refusal":
        return _handle_refusal(messages)
    if llm_query_type == "comparison":
        return _handle_comparison(messages)
    
    # Step 5: Conversational sufficiency check
    # This is what makes the agent ASK questions instead of just recommending
    needs_clarify, missing_field = _needs_clarification(intent, messages)
    if needs_clarify:
        logger.info(f"Needs clarification: missing={missing_field}")
        return _generate_targeted_clarification(messages, intent, missing_field)
    
    # Also respect LLM's judgment if it says clarification is needed
    if not intent.get("is_sufficient", True) or llm_query_type == "clarification_needed":
        return _handle_clarification_llm(messages, intent)
    
    # Step 6: Build search query and retrieve
    search_query = build_search_query(intent)
    filters = build_metadata_filters(intent)
    logger.info(f"Search query: {search_query}")
    logger.info(f"Metadata filters: {filters}")
    
    # Step 7: Semantic retrieval
    results = retriever.search(search_query, filters=filters if filters else None)
    
    # Step 8: Check retrieval confidence
    confidence = retriever.get_retrieval_confidence(results)
    logger.info(f"Retrieval confidence: {confidence}")
    
    if confidence["top_score"] < SIMILARITY_THRESHOLD and not filters:
        # Very low confidence - ask for clarification
        return _handle_clarification_llm(messages, intent)
    
    # Step 9: Rerank results
    reranked = _rerank_results(results, intent)
    
    # Step 10: Generate grounded response
    return _generate_recommendation_response(messages, intent, reranked)


# ─── Clarification ───

def _handle_clarification_llm(
    messages: List[Dict[str, str]], 
    intent: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a clarification question using LLM.
    Per Strategy Section 12.
    """
    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    )
    
    try:
        prompt = CLARIFICATION_PROMPT.format(
            conversation=conversation_text,
            intent=json.dumps(intent, default=str),
        )
        response = llm_client.generate_json(prompt, system_prompt=SYSTEM_PROMPT)
        
        # Ensure schema compliance
        return {
            "reply": response.get("reply", "Could you tell me more about the role you're hiring for? What specific skills or competencies are important?"),
            "recommendations": [],
            "end_of_conversation": False,
        }
    except Exception as e:
        logger.error(f"Clarification generation failed: {e}")
        return {
            "reply": "Could you tell me more about the role you're hiring for? What specific skills or competencies are you looking to assess?",
            "recommendations": [],
            "end_of_conversation": False,
        }


# ─── Comparison ───

def _handle_comparison(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Handle assessment comparison requests.
    Per Strategy Section 14: Grounded comparison.
    """
    # Extract assessment names to compare
    items = extract_comparison_items(messages)
    
    if not items or len(items) < 2:
        return {
            "reply": "I'd be happy to compare assessments. Could you specify which two SHL assessments you'd like me to compare?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    
    # Look up assessments in catalog
    assessment_details = []
    for name in items:
        meta = retriever.search_by_name(name)
        if not meta:
            # Try semantic search as fallback for abbreviations
            search_results = retriever.search(name, top_k=3)
            if search_results:
                meta = search_results[0]
        
        if meta:
            detail = (
                f"Assessment: {meta['name']}\n"
                f"Description: {meta.get('description', 'N/A')}\n"
                f"Categories: {', '.join(meta.get('keys', []))}\n"
                f"Duration: {meta.get('duration', 'N/A')}\n"
                f"Job Levels: {', '.join(meta.get('job_levels', []))}\n"
                f"Remote: {'Yes' if meta.get('remote') else 'No'}\n"
                f"Adaptive: {'Yes' if meta.get('adaptive') else 'No'}\n"
                f"URL: {meta.get('url', 'N/A')}"
            )
            assessment_details.append(detail)
        else:
            assessment_details.append(f"Assessment '{name}' was not found in the SHL catalog.")
    
    question = messages[-1].get("content", "Compare these assessments")
    
    try:
        prompt = COMPARISON_PROMPT.format(
            assessment_details="\n\n---\n\n".join(assessment_details),
            question=question,
        )
        response = llm_client.generate_json(prompt, system_prompt=SYSTEM_PROMPT)
        
        return {
            "reply": response.get("reply", "Here is a comparison based on catalog data."),
            "recommendations": [],
            "end_of_conversation": False,
        }
    except Exception as e:
        logger.error(f"Comparison generation failed: {e}")
        return {
            "reply": f"I found information about the assessments but encountered an issue generating the comparison. Here are their details:\n\n" + "\n\n".join(assessment_details),
            "recommendations": [],
            "end_of_conversation": False,
        }


# ─── Refusal ───

def _handle_refusal(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Handle off-topic or out-of-scope requests.
    Per Strategy Section 19.
    """
    last_msg = messages[-1].get("content", "")
    
    try:
        prompt = REFUSAL_PROMPT.format(message=last_msg)
        response = llm_client.generate_json(prompt, system_prompt=SYSTEM_PROMPT)
        
        return {
            "reply": response.get("reply", "I'm designed to help with SHL assessment recommendations only. Could you tell me about a role you'd like to find assessments for?"),
            "recommendations": [],
            "end_of_conversation": False,
        }
    except Exception as e:
        logger.error(f"Refusal generation failed: {e}")
        return {
            "reply": "I appreciate your question, but I can only help with SHL assessment recommendations. I'm not able to provide advice on that topic. Would you like help finding the right SHL assessments for a specific role?",
            "recommendations": [],
            "end_of_conversation": False,
        }


# ─── Reranking ───

def _rerank_results(
    results: List[Dict[str, Any]], 
    intent: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Rerank retrieval results using multiple signals.
    Per Strategy Section 15: Reranking Strategy.
    
    final_score = semantic_similarity + skill_overlap + metadata_match
    """
    if not results:
        return []
    
    skills = set(s.lower() for s in (intent.get("skills") or []))
    role = (intent.get("role") or "").lower()
    domain = (intent.get("domain") or "").lower()
    prefs = set(p.lower() for p in (intent.get("assessment_preferences") or []))
    
    scored_results = []
    for result in results:
        score = result.get("similarity_score", 0.0)
        
        # Skill overlap bonus
        desc_lower = (result.get("description") or "").lower()
        name_lower = (result.get("name") or "").lower()
        combined_text = f"{desc_lower} {name_lower}"
        
        skill_matches = sum(1 for s in skills if s in combined_text)
        if skills:
            skill_bonus = 0.15 * (skill_matches / len(skills))
            score += skill_bonus
        
        # Role relevance bonus
        if role and role in combined_text:
            score += 0.1
        
        # Domain relevance bonus
        if domain and domain in combined_text:
            score += 0.05
        
        # Assessment type preference bonus
        result_keys = set(k.lower() for k in (result.get("keys") or []))
        pref_to_keys_map = {
            "technical": "knowledge & skills",
            "personality": "personality & behavior",
            "cognitive": "ability & aptitude",
            "behavioral": "personality & behavior",
            "situational": "biodata & situational judgment",
            "simulation": "simulations",
            "aptitude": "ability & aptitude",
        }
        
        for pref in prefs:
            mapped_key = pref_to_keys_map.get(pref, pref)
            if mapped_key in result_keys:
                score += 0.1
        
        result["final_score"] = score
        scored_results.append(result)
    
    # Sort by final score descending
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    return scored_results[:FINAL_TOP_K]


# ─── Recommendation Generation ───

def _generate_recommendation_response(
    messages: List[Dict[str, str]],
    intent: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a grounded recommendation response.
    Per Strategy Section 17: Grounded Generation.
    """
    if not results:
        return {
            "reply": "I couldn't find assessments that closely match your requirements. Could you provide more details about the role, skills, or competencies you're looking to assess?",
            "recommendations": [],
            "end_of_conversation": False,
        }
    
    # Format assessments for the prompt
    assessment_text = ""
    for i, r in enumerate(results, 1):
        assessment_text += (
            f"\n{i}. {r['name']}\n"
            f"   URL: {r['url']}\n"
            f"   Description: {r.get('description', 'N/A')}\n"
            f"   Categories: {', '.join(r.get('keys') or [])}\n"
            f"   Duration: {r.get('duration', 'N/A')}\n"
            f"   Job Levels: {', '.join(r.get('job_levels') or [])}\n"
            f"   Remote: {'Yes' if r.get('remote') else 'No'}\n"
            f"   Relevance Score: {r.get('final_score', r.get('similarity_score', 0)):.3f}\n"
        )
    
    conversation_text = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages[-6:]  # Last 6 messages for context
    )
    
    # Check if user indicated end of conversation
    last_msg = messages[-1].get("content", "").lower()
    end_indicators = ["thank", "thanks", "perfect", "great", "that's all", "that is all", "done", "no more", "goodbye", "bye"]
    is_end = any(indicator in last_msg for indicator in end_indicators)
    
    try:
        prompt = RECOMMENDATION_PROMPT.format(
            intent=json.dumps(intent, default=str),
            assessments=assessment_text,
            conversation=conversation_text,
        )
        response = llm_client.generate_json(prompt, system_prompt=SYSTEM_PROMPT)
        
        # Validate and fix recommendations
        recommendations = response.get("recommendations", [])
        validated_recs = _validate_recommendations(recommendations, results)
        
        # If LLM returned no recommendations but we have results, build from results
        if not validated_recs and results:
            validated_recs = _build_recommendations_from_results(results)
        
        return {
            "reply": response.get("reply", "Here are my recommended SHL assessments based on your requirements."),
            "recommendations": validated_recs,
            "end_of_conversation": is_end or response.get("end_of_conversation", False),
        }
    except Exception as e:
        logger.error(f"Recommendation generation failed: {e}")
        # Fallback: build response directly from results
        recs = _build_recommendations_from_results(results)
        return {
            "reply": f"Based on your requirements, here are {len(recs)} recommended SHL assessments.",
            "recommendations": recs,
            "end_of_conversation": False,
        }


# ─── Validation ───

def _validate_recommendations(
    llm_recs: List[Dict[str, Any]], 
    retrieved: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Validate LLM recommendations against retrieved results.
    Ensures we NEVER return URLs/names not in the catalog.
    """
    # Build lookup from retrieved results
    name_lookup = {r["name"].lower(): r for r in retrieved}
    url_lookup = {r["url"]: r for r in retrieved}
    
    validated = []
    for rec in llm_recs:
        rec_name = rec.get("name", "").lower()
        rec_url = rec.get("url", "")
        
        # Try to match by name
        matched = None
        if rec_name in name_lookup:
            matched = name_lookup[rec_name]
        elif rec_url in url_lookup:
            matched = url_lookup[rec_url]
        else:
            # Fuzzy name match
            for name, data in name_lookup.items():
                if rec_name in name or name in rec_name:
                    matched = data
                    break
        
        if matched:
            validated.append({
                "name": matched["name"],  # Use exact catalog name
                "url": matched["url"],    # Use exact catalog URL
                "test_type": rec.get("test_type", get_test_type(matched.get("keys", []))),
            })
    
    # Deduplicate by name
    seen = set()
    deduped = []
    for rec in validated:
        if rec["name"] not in seen:
            seen.add(rec["name"])
            deduped.append(rec)
    
    return deduped[:FINAL_TOP_K]


def _build_recommendations_from_results(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Build recommendations directly from retrieval results (fallback)."""
    recs = []
    for r in results[:FINAL_TOP_K]:
        recs.append({
            "name": r["name"],
            "url": r["url"],
            "test_type": get_test_type(r.get("keys", [])),
        })
    return recs
