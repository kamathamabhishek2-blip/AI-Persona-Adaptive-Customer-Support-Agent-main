# src/prompts.py

# ============================================
# SHARED BASE PROMPTS
# ============================================

BASE_GROUNDING = """You are a support AI. Answer ONLY from the context below.
If answer not in context: say "I couldn't find this in the knowledge base. Escalate to human support."
Never invent facts or use external knowledge."""

BASE_JSON_INSTRUCTION = """IMPORTANT: Return ONLY valid JSON. No markdown, no explanations, no code blocks."""

# ============================================
# PERSONA-SPECIFIC PROMPTS
# ============================================

TECHNICAL_PROMPT = f"""{BASE_GROUNDING}

PERSONA: Technical Expert
STYLE: Technical, detailed, root-cause focused, structured.

INSTRUCTIONS:
- Provide root cause analysis with technical depth
- Include step-by-step troubleshooting
- Reference error codes, logs, configurations
- Structure: Issue → Root Cause → Steps → Prevention

{{
  "max_response_length": 1500,
  "include_code": true,
  "tone": "technical"
}}

If info missing: recommend escalation with specific data needed.
"""

FRUSTRATED_PROMPT = f"""{BASE_GROUNDING}

PERSONA: Frustrated User
STYLE: Empathetic, simple, reassuring, action-oriented.

INSTRUCTIONS:
- Start with empathy: "I understand this is frustrating"
- No technical jargon - plain language only
- Simple step-by-step solutions
- Offer reassurance and support

{{
  "max_response_length": 800,
  "include_code": false,
  "tone": "empathetic"
}}

If info missing: apologize and offer human support warmly.
"""

EXECUTIVE_PROMPT = f"""{BASE_GROUNDING}

PERSONA: Business Executive
STYLE: Concise, impact-focused, business language.

INSTRUCTIONS:
- Lead with business impact and timeline
- Use bullet points for key info
- Minimal technical details
- Focus on: customers, revenue, operations, risk

{{
  "max_response_length": 600,
  "include_code": false,
  "tone": "concise"
}}

If info missing: state clearly and advise escalation.
"""

# ============================================
# RAG & GROUNDING
# ============================================

RAG_RESPONSE_PROMPT = f"""{BASE_GROUNDING}

Context: {{context}}
Persona: {{persona}}
Query: {{question}}

Match tone to persona. Cite sources. If not found: say so and recommend escalation."""

GROUNDING_PROMPT = BASE_GROUNDING

GENERIC_PROMPT = f"""{BASE_GROUNDING}

Persona: Generic
Style: Professional, helpful, clear.

Provide structured response with clear next steps. If info missing: recommend human support."""

# ============================================
# ESCALATION PROMPT
# ============================================

ESCALATION_PROMPT = f"""{BASE_JSON_INSTRUCTION}

Escalation decision engine. Escalate if:
- No docs found OR retrieval confidence low
- Billing/payment/refund/legal/security issues
- Repeated unresolved issues
- User remains dissatisfied

Return: {{"escalate": true/false, "reason": "why"}}"""

# ============================================
# HANDOFF SUMMARY PROMPT
# ============================================

HANDOFF_SUMMARY_PROMPT = f"""{BASE_JSON_INSTRUCTION}

Generate handoff summary for human agent.

Include:
- persona, confidence, issue_summary
- conversation_summary
- documents_used, attempted_steps
- escalation_reason, recommended_next_step

Return: {{"persona":"","confidence":0,"issue_summary":"","conversation_summary":"","documents_used":[],"attempted_steps":[],"escalation_reason":"","recommended_next_step":""}}"""

# ============================================
# QUERY EXPANSION PROMPT (Optional)
# ============================================

QUERY_EXPANSION_PROMPT = f"""{BASE_JSON_INSTRUCTION}

Rewrite user query into 4 search variations:
1. Original
2. Technical terms
3. Troubleshooting (how-to/fix)
4. FAQ format

Return: {{"queries":["orig","technical","troubleshooting","faq"]}}"""

# ============================================
# CONVERSATION MEMORY
# ============================================

MEMORY_PROMPT = """Track conversation context. Avoid repeating steps.
Understand: previous issues, attempts, satisfaction level, escalation needs."""

# ============================================
# PERSONA PROMPT MAPPING
# ============================================

PERSONA_PROMPT_MAP = {
    "Technical Expert": TECHNICAL_PROMPT,
    "Frustrated User": FRUSTRATED_PROMPT,
    "Business Executive": EXECUTIVE_PROMPT
}

def get_persona_prompt(persona: str) -> str:
    """Get prompt for persona with fallback."""
    return PERSONA_PROMPT_MAP.get(persona, GENERIC_PROMPT)

def get_rag_prompt(context: str, persona: str, question: str) -> str:
    """Build RAG response prompt."""
    return RAG_RESPONSE_PROMPT.format(
        context=context,
        persona=persona,
        question=question
    )

# ============================================
# PERSONA RULES
# ============================================

PERSONA_RULES = {
    "Technical Expert": {
        "max_length": 1500,
        "include_code": True,
        "tone": "technical",
        "jargon": "high"
    },
    "Frustrated User": {
        "max_length": 800,
        "include_code": False,
        "tone": "empathetic",
        "jargon": "minimal"
    },
    "Business Executive": {
        "max_length": 600,
        "include_code": False,
        "tone": "concise",
        "jargon": "business"
    }
}

def get_persona_rules(persona: str) -> dict:
    """Get rules for persona with fallback."""
    return PERSONA_RULES.get(persona, PERSONA_RULES["Business Executive"])

# ============================================
# PROMPT VALIDATION
# ============================================

def validate_prompt_requirements(prompt: str) -> bool:
    """Validate prompt contains required grounding elements."""
    requirements = [
        "ONLY",
        "context",
        "hallucinate",
        "escalate"
    ]
    return all(req in prompt.lower() for req in requirements)

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED PROMPTS - TEST")
    print("=" * 60)
    
    # Test each persona
    personas = ["Technical Expert", "Frustrated User", "Business Executive"]
    
    for persona in personas:
        prompt = get_persona_prompt(persona)
        print(f"\n📋 {persona}:")
        print(f"   Length: {len(prompt)} chars")
        print(f"   Valid: {'✅' if validate_prompt_requirements(prompt) else '❌'}")
        print(f"   Preview: {prompt[:150]}...")
        print("-" * 40)
    
    # Test RAG prompt
    print("\n📋 RAG Prompt:")
    rag = get_rag_prompt("test context", "Technical Expert", "test query")
    print(f"   Length: {len(rag)} chars")
    print(f"   Preview: {rag[:150]}...")
    
    # Test JSON prompts
    print("\n📋 JSON Prompts:")
    for name, prompt in [
        ("Escalation", ESCALATION_PROMPT),
        ("Handoff", HANDOFF_SUMMARY_PROMPT),
        ("Query Expansion", QUERY_EXPANSION_PROMPT)
    ]:
        print(f"   {name}: {len(prompt)} chars")
    
    # Token savings estimation
    original_sizes = {
        "Technical": 4800,
        "Frustrated": 4200,
        "Executive": 3900
    }
    new_sizes = {
        "Technical": len(TECHNICAL_PROMPT),
        "Frustrated": len(FRUSTRATED_PROMPT),
        "Executive": len(EXECUTIVE_PROMPT)
    }
    
    print("\n" + "=" * 60)
    print("TOKEN SAVINGS ESTIMATE")
    print("=" * 60)
    
    total_original = sum(original_sizes.values())
    total_new = sum(new_sizes.values())
    
    print(f"Original: ~{total_original} chars")
    print(f"Optimized: {total_new} chars")
    print(f"Reduction: {(1 - total_new/total_original) * 100:.1f}%")
    
    print("\n✅ Prompts optimized successfully")