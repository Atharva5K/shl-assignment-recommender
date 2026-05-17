"""
LLM module: Groq API client for Llama 3.1 8B Instant.

Per Strategy doc Section 3: "The LLM is intentionally used minimally."
Per Strategy doc Section 17: "The LLM never generates recommendations from memory."
"""
import json
import logging
import re
from typing import Dict, Any, Optional

from groq import Groq
import httpx

from app.config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

logger = logging.getLogger(__name__)


class LLMClient:
    """Groq LLM client for grounded generation."""
    
    def __init__(self):
        self.client: Optional[Groq] = None
    
    def initialize(self):
        """Initialize the Groq client."""
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set. LLM calls will fail.")
            return
        # 25s timeout: evaluator allows 30s per call, leave 5s buffer for
        # FAISS search, embedding, and network overhead
        self.client = Groq(
            api_key=GROQ_API_KEY,
            timeout=httpx.Timeout(25.0, connect=10.0),
        )
        logger.info(f"Groq LLM client initialized with model: {GROQ_MODEL}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = GROQ_TEMPERATURE,
        max_tokens: int = GROQ_MAX_TOKENS,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User/task prompt
            system_prompt: System instructions
            temperature: Generation temperature
            max_tokens: Max response tokens
        
        Returns:
            Raw LLM response text
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized. Call initialize() first.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = GROQ_TEMPERATURE,
        max_tokens: int = GROQ_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from the LLM.
        Parses the response and handles common JSON extraction issues.
        """
        raw = self.generate(prompt, system_prompt, temperature, max_tokens)
        return self._parse_json_response(raw)
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Robustly parse JSON from LLM response.
        Handles markdown code blocks, extra text, etc.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object in text
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Failed to parse JSON from LLM response: {text[:200]}")
        # Return safe default
        return {
            "reply": "I apologize, but I encountered an issue processing your request. Could you please rephrase?",
            "recommendations": [],
            "end_of_conversation": False,
        }


# Singleton instance
llm_client = LLMClient()
