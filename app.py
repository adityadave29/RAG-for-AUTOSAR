import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from pathlib import Path
from src.config import DOCS_DIR, GEMINI_API_KEY, GEMINI_MODEL_NAME
from src.retriever import AUTOSARRetriever
from src.generator import GeminiGenerator
from src.vector_store import VectorStoreManager
from src.ingest import run_ingestion

# Page Configuration
st.set_page_config(
    page_title="AUTOSAR RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Blackish-Grey CSS Theme
CUSTOM_CSS = """
<style>
    /* Main Background & Text Colors */
    .stApp {
        background-color: #0d0f12 !important;
        color: #e2e8f0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #14171d !important;
        border-right: 1px solid #222733 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {
        color: #cbd5e1 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }

    /* Cards & Containers */
    .autosar-card {
        background-color: #181b22;
        border: 1px solid #262c38;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    .query-refine-box {
        background-color: #161c28;
        border: 1px solid #2a3854;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.95em;
        color: #93c5fd;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: #151820 !important;
        border: 1px solid #232836 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 14px !important;
    }

    /* Inputs, Buttons, & Controls */
    .stTextInput > div > div > input {
        background-color: #1a1e27 !important;
        color: #f8fafc !important;
        border: 1px solid #2d3546 !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    .stButton > button {
        background-color: #252a36 !important;
        color: #f1f5f9 !important;
        border: 1px solid #363d4e !important;
        border-radius: 8px !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #32394a !important;
        border-color: #4b5568 !important;
        color: #ffffff !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #181c25 !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        border: 1px solid #262c3a !important;
    }
    .streamlit-expanderContent {
        background-color: #12151c !important;
        border: 1px solid #262c3a !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 6px;
        margin-right: 6px;
    }
    .badge-blue { background: #1e293b; color: #60a5fa; border: 1px solid #334155; }
    .badge-green { background: #14281f; color: #4ade80; border: 1px solid #166534; }
    .badge-amber { background: #2a2215; color: #fbbf24; border: 1px solid #78350f; }

    /* Code Block styling */
    pre, code {
        background-color: #12141a !important;
        border: 1px solid #222632 !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### ⚡ **AUTOSAR RAG**")
    st.caption("AI Search & Reasoning over Automotive Specifications")
    st.divider()

    # Knowledge Base Stats
    try:
        vs = VectorStoreManager()
        stats = vs.get_stats()
        st.markdown(f"""
        <div class="autosar-card">
            <span style="font-size: 12px; color: #94a3b8;">KNOWLEDGE BASE STATUS</span>
            <div style="font-size: 24px; font-weight: 700; color: #38bdf8; margin: 4px 0;">{stats['total_chunks']}</div>
            <span style="font-size: 12px; color: #64748b;">Indexed Specification Chunks</span>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load vector store: {e}")

    # Settings
    st.markdown("#### ⚙️ **Engine Settings**")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Reads from .env by default. Enter here to override."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0
    )

    enable_refinement = st.toggle("✨ LLM Query Optimizer", value=True, help="Pass user prompt to LLM first to generate optimized technical AUTOSAR keywords.")
    use_rerank = st.toggle("🎯 FlashRank Re-ranker", value=True, help="Cross-encoder re-ranking for top technical accuracy.")
    top_k = st.slider("Context Chunks (Top-K)", min_value=2, max_value=8, value=4)

    st.divider()

    # Upload & Ingest New Document
    st.markdown("#### 📄 **Add Documents**")
    uploaded_file = st.file_uploader("Upload AUTOSAR PDF / Spec", type=["pdf", "md", "txt"])
    if uploaded_file is not None:
        if st.button("🚀 Ingest Uploaded File", use_container_width=True):
            save_path = DOCS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner(f"Parsing & indexing {uploaded_file.name}..."):
                run_ingestion()
            st.success(f"Indexed {uploaded_file.name} successfully!")
            st.rerun()


# ================= MAIN CHAT UI =================
st.markdown("## 🚗 AUTOSAR Technical Assistant")
st.markdown(
    "<span style='color: #94a3b8;'>Ask questions about AUTOSAR Classic & Adaptive specifications, BSW modules, RTE, MCAL, and state machines.</span>",
    unsafe_allow_html=True
)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # If user message, just show text
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Assistant message
            if "refined_query" in msg and msg["refined_query"]:
                st.markdown(
                    f"""<div class="query-refine-box">
                        <strong>✨ LLM Optimized Search Query:</strong><br/>
                        <code>{msg['refined_query']}</code>
                    </div>""",
                    unsafe_allow_html=True
                )

            st.markdown(msg["content"])

            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Retrieved Specifications & Requirements", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(
                            f"<span class='badge badge-blue'>{s['file']}</span> "
                            f"<span class='badge badge-amber'>Page {s['page']}</span> "
                            f"<span class='badge badge-green'>Score: {s['score']}</span>",
                            unsafe_allow_html=True
                        )
                        st.code(s['snippet'], language="markdown")

# Handle New User Input
if user_prompt := st.chat_input("e.g. How does CAN controller state transition to CAN_CS_STARTED?"):
    # Display user query
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        generator = GeminiGenerator(api_key=api_key_input, model_name=model_name)
        retriever = AUTOSARRetriever()

        # Step 1: Query Refinement via LLM (if enabled)
        if enable_refinement:
            with st.spinner("🔮 LLM optimizing technical query for vector retrieval..."):
                refined_query = generator.refine_query(user_prompt)
        else:
            refined_query = user_prompt

        # Display Refined Query Box
        if enable_refinement and refined_query != user_prompt:
            st.markdown(
                f"""<div class="query-refine-box">
                    <strong>✨ LLM Optimized Search Query:</strong><br/>
                    <code>{refined_query}</code>
                </div>""",
                unsafe_allow_html=True
            )

        # Step 2: Vector Search & Re-ranking with Refined Query
        with st.spinner("🔍 Retrieving and re-ranking relevant AUTOSAR specifications..."):
            chunks = retriever.retrieve(
                query=refined_query,
                top_k_candidates=12,
                top_k_final=top_k,
                use_reranker=use_rerank
            )

        if not chunks:
            response_text = "No relevant AUTOSAR specification chunks found in the knowledge base. Please verify that your documents are ingested."
            st.warning(response_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "refined_query": refined_query if enable_refinement else None
            })
        else:
            sources_data = []
            for c in chunks:
                sources_data.append({
                    "file": c.get("metadata", {}).get("source_file", "Unknown"),
                    "page": c.get("metadata", {}).get("page_number", "N/A"),
                    "score": c.get("rerank_score", c.get("score", "N/A")),
                    "snippet": c["text"]
                })

            context = retriever.format_context(chunks)

            # Step 3: LLM Technical Generation
            with st.spinner("🤖 Generating technical answer with Gemini..."):
                response_text = generator.generate_answer(user_prompt, context)

            st.markdown(response_text)

            # Step 4: Expandable Source Badges
            with st.expander("📚 Retrieved Specifications & Requirements", expanded=False):
                for s in sources_data:
                    st.markdown(
                        f"<span class='badge badge-blue'>{s['file']}</span> "
                        f"<span class='badge badge-amber'>Page {s['page']}</span> "
                        f"<span class='badge badge-green'>Score: {s['score']}</span>",
                        unsafe_allow_html=True
                    )
                    st.code(s['snippet'], language="markdown")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "refined_query": refined_query if enable_refinement else None,
                "sources": sources_data
            })
