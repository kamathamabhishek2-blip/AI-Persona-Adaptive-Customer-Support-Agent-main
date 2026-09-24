# app.py

"""
Persona-Adaptive Customer Support Agent - Streamlit UI

A production-grade AI support agent with:
- Persona detection (Technical Expert, Frustrated User, Business Executive)
- RAG retrieval with ChromaDB
- Adaptive response generation with Gemini
- Escalation logic with human handoff
- Analytics dashboard
- Feedback collection
"""

import streamlit as st
import json
import traceback
import socket
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

from src.classifier import classify_persona_with_fallback
from src.generator import generate_response
from src.escalator import EscalationManager
from src.rag_pipeline import build_knowledge_base, get_collection_stats
from src.utils import truncate_text, time_ago, metrics

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="🤖 Persona Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/persona-support-agent",
        "Report a bug": "https://github.com/your-repo/persona-support-agent/issues",
        "About": """
        ## AI Persona-Adaptive Customer Support Agent
        Built for Adsparkx AI Assignment
        
        **Features:**
        - Persona Detection (Technical Expert, Frustrated User, Business Executive)
        - RAG Retrieval with ChromaDB
        - Adaptive Response Generation with Gemini
        - Escalation Logic with Human Handoff
        - Analytics Dashboard
        
        **Tech Stack:**
        - Streamlit
        - Google Gemini 2.5 Flash
        - ChromaDB
        - Sentence Transformers
        """
    }
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def initialize_session_state():
    """Initialize all session state variables."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    if "escalation_manager" not in st.session_state:
        st.session_state.escalation_manager = EscalationManager()
    
    if "knowledge_base_ready" not in st.session_state:
        with st.spinner("Building knowledge base..."):
            result = build_knowledge_base()
            st.session_state.knowledge_base_ready = result["status"] == "success"
            st.session_state.kb_stats = result
    
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    
    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now()


# ============================================
# CACHED FUNCTIONS
# ============================================

@st.cache_data(ttl=60)
def get_cached_collection_stats() -> Dict:
    """Cache collection stats for 60 seconds."""
    return get_collection_stats()


@st.cache_data(ttl=300)
def get_cached_example_queries() -> List[str]:
    """Cache example queries."""
    return [
        "API authentication failure with error logs",
        "I've tried everything and nothing works!",
        "How will this impact business operations?",
        "I need a refund for duplicate payment",
        "How do I reset my password?",
        "What's the SLA for resolving this issue?"
    ]


# ============================================
# SIDEBAR
# ============================================

def render_sidebar():
    """Render the sidebar with all controls and information."""
    with st.sidebar:
        st.title("🤖 Persona Support Agent")
        st.markdown("---")
        
        # Knowledge Base Status
        st.markdown("## 📚 Knowledge Base Status")
        if st.session_state.knowledge_base_ready:
            stats = get_cached_collection_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", stats["total_chunks"])
            with col2:
                st.metric("Status", "✅ Ready")
        else:
            st.error("❌ Knowledge base not ready")
            st.write(st.session_state.kb_stats)
        
        st.markdown("---")
        
        # Analytics Dashboard
        with st.expander("📊 Analytics Dashboard", expanded=False):
            render_analytics_dashboard()
        
        st.markdown("---")
        
        # Supported Personas
        st.markdown("## 🎭 Supported Personas")
        st.markdown("""
        - **Technical Expert** 🛠️
          - Detailed technical responses
          - Root cause analysis
          - Step-by-step troubleshooting
        
        - **Frustrated User** 😤
          - Empathetic responses
          - Simple language
          - Action-oriented steps
        
        - **Business Executive** 📊
          - Concise responses
          - Business impact focused
          - Minimal technical jargon
        """)
        
        st.markdown("---")
        
        # Example Queries
        with st.expander("💡 Example Queries"):
            examples = get_cached_example_queries()
            for query in examples:
                if st.button(query, key=f"example_{query[:20]}"):
                    st.session_state.example_query = query
                    st.rerun()
        
        st.markdown("---")
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.conversation = []
                st.session_state.escalation_manager = EscalationManager()
                st.session_state.query_count = 0
                st.rerun()
        
        with col2:
            if st.button("📤 Export", use_container_width=True):
                export_conversation()
        
        st.markdown("---")
        
        # Access Information
        st.markdown("## 🌐 Access Info")
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            st.code(f"http://localhost:8501", language="text")
            st.code(f"http://{local_ip}:8501", language="text")
            st.info("🔗 For public access: `ngrok http 8501`")
        except Exception:
            pass
        
        st.markdown("---")
        st.caption("Powered by Gemini + ChromaDB + Streamlit")
        st.caption(f"v1.0.0 | {time_ago(st.session_state.start_time.isoformat())}")


def render_analytics_dashboard():
    """Render the analytics dashboard in the sidebar."""
    # Total queries
    total = st.session_state.query_count
    st.metric("Total Queries", total)
    
    # Get metrics from the metrics collector
    metrics_data = metrics.get_metrics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Escalations", metrics_data.get("escalation_counts", 0))
    with col2:
        escalation_rate = metrics_data.get("escalation_rate", 0) * 100
        st.metric("Escalation Rate", f"{escalation_rate:.1f}%")
    
    # Persona distribution
    persona_counts = metrics_data.get("persona_counts", {})
    if persona_counts:
        st.markdown("**Persona Distribution**")
        for persona, count in persona_counts.items():
            pct = (count / max(1, total)) * 100
            st.progress(pct / 100, text=f"{persona}: {pct:.0f}% ({count})")
    
    # Average confidence
    avg_conf = metrics_data.get("avg_confidence", 0)
    st.metric("Avg Confidence", f"{avg_conf:.1%}")


def export_conversation():
    """Export conversation history as JSON."""
    if not st.session_state.conversation:
        st.warning("No conversation to export")
        return
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "total_turns": len(st.session_state.conversation),
        "conversation": st.session_state.conversation
    }
    
    json_str = json.dumps(export_data, indent=2)
    st.download_button(
        label="📥 Download Conversation",
        data=json_str,
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )


# ============================================
# MAIN UI
# ============================================

def render_chat_message(turn: Dict, index: int):
    """Render a single chat message with metadata."""
    is_user = "user" in turn
    
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.write(turn["user"])
        else:
            # Response content
            st.markdown(turn["assistant"])
            
            # Metadata row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption(f"🎭 **Persona:** {turn.get('persona', 'Unknown')}")
            with col2:
                st.caption(f"📊 **Confidence:** {turn.get('confidence', 0):.1%}")
            with col3:
                docs = turn.get("doc_count", 0)
                st.caption(f"📚 **Sources:** {docs}")
            with col4:
                if turn.get("escalated", False):
                    st.caption("⚠️ **Escalated**")
            
            # Sources
            if turn.get("sources"):
                with st.expander("📚 Retrieved Sources", expanded=False):
                    for idx, source in enumerate(turn["sources"], 1):
                        st.markdown(f"**{idx}. {source.get('source', 'Unknown')}**")
                        if source.get('page'):
                            st.caption(f"Page: {source['page']}")
                        st.caption(f"Score: {source.get('score', 0):.2%}")
                        st.text(truncate_text(source.get('text', ''), 200))
                        st.divider()
            
            # Feedback buttons
            feedback_key = f"feedback_{index}"
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"like_{index}"):
                    st.session_state.feedback[feedback_key] = "like"
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎", key=f"dislike_{index}"):
                    st.session_state.feedback[feedback_key] = "dislike"
                    st.warning("We'll improve! Please provide details below.")


def render_handoff_summary(summary: Dict):
    """Render a human handoff summary."""
    with st.expander("📋 Human Handoff Summary", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Detected Persona**")
            st.info(summary.get("persona", "Unknown"))
            
            st.markdown("**Confidence**")
            st.progress(summary.get("confidence", 0), text=f"{summary.get('confidence', 0):.1%}")
            
            st.markdown("**Issue Summary**")
            st.write(summary.get("issue_summary", "No summary"))
        
        with col2:
            st.markdown("**Escalation Reason**")
            st.warning(summary.get("escalation_reason", "No reason"))
            
            st.markdown("**Recommended Next Step**")
            st.info(summary.get("recommended_next_step", "No recommendation"))
        
        st.markdown("**Documents Used**")
        docs = summary.get("documents_used", [])
        st.write(", ".join(docs) if docs else "None")
        
        st.markdown("**Attempted Steps**")
        steps = summary.get("attempted_steps", [])
        for step in steps:
            st.write(f"• {step}")
        
        # Download button
        st.download_button(
            label="📥 Download Handoff Summary",
            data=json.dumps(summary, indent=2),
            file_name=f"handoff_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )


def process_query(query: str) -> Optional[Dict]:
    """
    Process a user query through the full pipeline.
    Returns the result or None on error.
    """
    try:
        # 1. Persona Detection
        persona_data = classify_persona_with_fallback(query)
        
        # 2. Response Generation with RAG
        result = generate_response(query, persona_data)
        
        # 3. Escalation Check
        escalated, reason = st.session_state.escalation_manager.check_escalation(
            query,
            persona_data,
            result["retrieved_chunks"]
        )
        
        # 4. Add to conversation history
        st.session_state.escalation_manager.add_to_history(
            query,
            result["response"],
            result["persona"]
        )
        
        # Record metrics
        metrics.record_query(
            persona=result["persona"],
            confidence=result["confidence"],
            escalated=escalated,
            latency=0  # Would track actual latency
        )
        
        # Increment query count
        st.session_state.query_count += 1
        
        return {
            "query": query,
            "persona_data": persona_data,
            "result": result,
            "escalated": escalated,
            "reason": reason
        }
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        with st.expander("🐞 Debug Details"):
            st.code(traceback.format_exc())
        return None


# ============================================
# MAIN APP
# ============================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🤖 AI Persona-Adaptive Customer Support Agent")
    st.caption("Delivering personalized customer support through AI with real-time persona detection")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("✅ Persona Detection")
    with col2:
        st.success("✅ RAG Retrieval")
    with col3:
        st.success("✅ Response Generation")
    with col4:
        st.success("✅ Escalation Engine")
    
    st.divider()
    
    # Chat container
    chat_container = st.container()
    
    # Display conversation history
    with chat_container:
        for idx, turn in enumerate(st.session_state.conversation):
            render_chat_message(turn, idx)
    
    # Handle example query
    if "example_query" in st.session_state and st.session_state.example_query:
        query = st.session_state.example_query
        st.session_state.example_query = None
        # Process the query (handled below)
    else:
        query = st.chat_input("Describe your issue here...")
    
    # Process query
    if query and query.strip():
        # Add user message
        st.chat_message("user").write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing and generating response..."):
                # Process the query
                result = process_query(query)
                
                if result:
                    # Store in conversation
                    turn = {
                        "user": query,
                        "assistant": result["result"]["response"],
                        "persona": result["result"]["persona"],
                        "confidence": result["result"]["confidence"],
                        "doc_count": len(result["result"]["retrieved_chunks"]),
                        "sources": result["result"]["retrieved_chunks"],
                        "escalated": result["escalated"],
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.conversation.append(turn)
                    
                    # Render the response
                    render_chat_message(turn, len(st.session_state.conversation) - 1)
                    
                    # Show handoff summary if escalated
                    if result["escalated"]:
                        summary = st.session_state.escalation_manager.generate_handoff_summary(
                            query,
                            result["persona_data"],
                            result["result"]["retrieved_chunks"]
                        )
                        render_handoff_summary(summary)
    
    # Footer
    st.divider()
    st.caption("© 2026 Persona Support Agent | Built for Adsparkx AI Assignment")
    st.caption("💡 Try asking about API issues, billing, or general support questions")


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()