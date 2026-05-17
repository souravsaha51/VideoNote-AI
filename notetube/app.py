"""
app.py — VideoNote AI: AI-Powered YouTube Summarization Platform
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.validators import validate_url
from utils.helpers import format_timestamp, build_youtube_url, reading_time_minutes
from modules.rag_pipeline import run_full_analysis, get_llm, LLM_FALLBACK
from modules.chat import answer_question
from modules.summarizer import generate_learning_notes

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VideoNote AI — AI YouTube Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "analysis": None,
        "chat_history": [],
        "current_video_id": None,
        "url_history": [],
        "processing": False,
        "theme": "System",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Custom CSS ────────────────────────────────────────────────────────────────
theme_mode = st.session_state.get("theme", "System")

dark_css = """
    --bg-primary: #0d0d1a;
    --bg-secondary: #12121f;
    --bg-card: rgba(255,255,255,0.04);
    --text-primary: #f0f0ff;
    --text-secondary: #9090b0;
    --border: rgba(124,58,237,0.2);
    --sidebar-bg: linear-gradient(180deg, #0f0f20 0%, #1a0a2e 100%);
    --card-hover: rgba(255,255,255,0.06);
"""

light_css = """
    --bg-primary: #f8fafc;
    --bg-secondary: #ffffff;
    --bg-card: rgba(0,0,0,0.03);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --border: rgba(124,58,237,0.15);
    --sidebar-bg: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
    --card-hover: rgba(0,0,0,0.05);
"""

if theme_mode == "Dark":
    theme_vars = f":root {{ {dark_css} }}"
elif theme_mode == "Light":
    theme_vars = f":root {{ {light_css} }}"
else:
    theme_vars = f"""
    @media (prefers-color-scheme: dark) {{ :root {{ {dark_css} }} }}
    @media (prefers-color-scheme: light) {{ :root {{ {light_css} }} }}
    """

st.markdown(f"<style>{theme_vars}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Common Root Variables */
:root {
    --accent: #7c3aed;
    --accent-light: #a855f7;
    --accent-glow: rgba(124,58,237,0.3);
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --radius: 16px;
}

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-primary); color: var(--text-primary); }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.nt-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s, transform 0.2s;
}
.nt-card:hover { border-color: var(--accent-light); transform: translateY(-2px); }

/* Hero Header */
.nt-hero {
    background: linear-gradient(135deg, #1a0a2e 0%, #0d0d1a 50%, #0a1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.nt-hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}
.nt-hero h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #fff 30%, var(--accent-light));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
}
.nt-hero p { color: var(--text-secondary); font-size: 1.1rem; margin: 0; }

/* Insight chips */
.nt-insight {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(168,85,247,0.08));
    border: 1px solid rgba(168,85,247,0.25);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.93rem;
    line-height: 1.5;
    color: var(--text-primary);
}

/* Timestamp cards */
.nt-timestamp {
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex; align-items: flex-start; gap: 1rem;
}
.nt-ts-badge {
    background: var(--success);
    color: #000; font-weight: 700; font-size: 0.78rem;
    padding: 0.25rem 0.6rem; border-radius: 8px;
    white-space: nowrap; flex-shrink: 0;
}

/* Chat bubbles */
.nt-bubble-user {
    background: linear-gradient(135deg, var(--accent), #6d28d9);
    border-radius: 18px 18px 4px 18px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.93rem;
}
.nt-bubble-ai {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px 18px 18px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.93rem;
}

/* Stat pills */
.nt-stat {
    display: inline-block;
    background: rgba(124,58,237,0.12);
    border: 1px solid var(--border);
    border-radius: 30px;
    padding: 0.35rem 0.9rem;
    font-size: 0.82rem;
    color: var(--accent-light);
    margin-right: 0.5rem;
}

/* Summary box */
.nt-summary {
    background: var(--bg-card);
    border-left: 3px solid var(--accent);
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.4rem;
    line-height: 1.75;
    font-size: 0.97rem;
}

/* Input override */
.stTextInput input, .stSelectbox select {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
    color: white !important; font-weight: 600 !important;
    border: none !important; border-radius: 12px !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px; 
    padding: 0.5rem; 
    gap: 8px;
    border: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important; 
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
    background: transparent !important;
    white-space: nowrap !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
    color: white !important;
}

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; border-radius: 4px; }

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 VideoNote AI")
    st.markdown("---")
    
    api_key = os.getenv("GEMINI_API_KEY", "")

    st.markdown("### ⚙️ Settings")

    st.selectbox(
        "Theme",
        ["System", "Dark", "Light"],
        key="theme"
    )
    
    summary_type = st.selectbox(
        "Summary Type",
        ["short", "detailed"],
        format_func=lambda x: "⚡ Quick Summary" if x == "short" else "📋 Detailed Summary",
    )

    language = st.selectbox(
        "Transcript Language",
        ["en", "hi", "es", "fr", "de", "pt", "ar", "zh", "ja", "ko"],
        format_func=lambda x: {
            "en": "🇬🇧 English", "hi": "🇮🇳 Hindi", "es": "🇪🇸 Spanish",
            "fr": "🇫🇷 French",  "de": "🇩🇪 German", "pt": "🇧🇷 Portuguese",
            "ar": "🇸🇦 Arabic",  "zh": "🇨🇳 Chinese","ja": "🇯🇵 Japanese",
            "ko": "🇰🇷 Korean",
        }.get(x, x),
    )

    st.markdown("---")

    if st.session_state.url_history:
        st.markdown("### 🕐 Recent Videos")
        for entry in reversed(st.session_state.url_history[-5:]):
            st.markdown(
                f'<div class="nt-card" style="padding:0.6rem 0.9rem;font-size:0.8rem;">'
                f'🎬 <code>{entry}</code></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        '<p style="color:#9090b0;font-size:0.78rem;text-align:center;">'
        'Built with Gemini 2.5 Flash Lite + FAISS + LangChain</p>',
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nt-hero">
    <h1>🎬 VideoNote AI</h1>
    <p>Transform any YouTube video into intelligent summaries, insights & interactive Q&A</p>
</div>
""", unsafe_allow_html=True)


# ── URL Input ─────────────────────────────────────────────────────────────────
col_url, col_btn = st.columns([5, 1])
with col_url:
    url_input = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=... or paste any YouTube link",
        label_visibility="collapsed",
    )
with col_btn:
    analyze_clicked = st.button("🚀 Analyze", use_container_width=True)


# ── Analysis Trigger ──────────────────────────────────────────────────────────
if analyze_clicked:
    if not api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar.")
        st.stop()

    is_valid, video_id, err_msg = validate_url(url_input)
    if not is_valid:
        st.error(f"❌ {err_msg}")
        st.stop()

    # Progress bar
    progress_bar = st.progress(0, text="Starting analysis…")
    status_text = st.empty()

    def progress_cb(step, total, msg):
        progress_bar.progress(step / total, text=msg)
        status_text.markdown(f"*{msg}*")

    try:
        with st.spinner(""):
            results = run_full_analysis(
                video_id=video_id,
                language=language,
                api_key=api_key,
                summary_type=summary_type,
                progress_callback=progress_cb,
            )

        progress_bar.progress(1.0, text="✅ Analysis complete!")
        status_text.empty()

        st.session_state.analysis = results
        st.session_state.current_video_id = video_id
        st.session_state.chat_history = []

        if url_input not in st.session_state.url_history:
            st.session_state.url_history.append(video_id)

        st.toast("✅ Video analyzed successfully!", icon="🎬")

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        err = str(e)
        if "transcript" in err.lower() or "disabled" in err.lower():
            st.error(f"📝 {err}")
        elif "api" in err.lower() or "quota" in err.lower() or "429" in err.lower():
            st.error("⚠️ API quota exceeded. Please wait a moment and try again.")
        else:
            st.error(f"❌ Something went wrong: {err}")
        st.stop()


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.analysis:
    data = st.session_state.analysis
    vid = st.session_state.current_video_id

    # Video preview + stats row
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(
            f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
            use_container_width=True,
        )
        st.markdown(
            f'<a href="https://www.youtube.com/watch?v={vid}" target="_blank">'
            f'<button style="width:100%;background:linear-gradient(135deg,#7c3aed,#a855f7);'
            f'color:white;border:none;border-radius:10px;padding:0.5rem;'
            f'font-weight:600;cursor:pointer;margin-top:0.5rem;">▶ Watch on YouTube</button></a>',
            unsafe_allow_html=True,
        )
    with c2:
        words = len(data["transcript"].split())
        read_mins = reading_time_minutes(data["transcript"])
        chunks = len(data.get("segments", []))
        st.markdown(f"""
        <div class="nt-card">
            <div style="margin-bottom:0.8rem;">
                <span class="nt-stat">📝 {words:,} words</span>
                <span class="nt-stat">⏱ ~{read_mins} min read</span>
                <span class="nt-stat">🧩 {chunks} segments</span>
            </div>
            <p style="color:#9090b0;font-size:0.85rem;margin:0;">
                Video ID: <code style="color:#a855f7;">{vid}</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_summary, tab_insights, tab_timestamps, tab_chat, tab_transcript, tab_export = st.tabs([
        "📋 Summary", "💡 Insights", "🕐 Timestamps", "💬 Chat", "📄 Transcript", "📥 Export"
    ])

    # ── Tab: Summary ──────────────────────────────────────────────────────────
    with tab_summary:
        st.markdown("### ⚡ Quick Summary")
        st.markdown(
            f'<div class="nt-summary">{data["summary_short"]}</div>',
            unsafe_allow_html=True,
        )

        if data.get("summary_detailed"):
            st.markdown("### 📋 Detailed Summary")
            st.markdown(data["summary_detailed"])

    # ── Tab: Insights ─────────────────────────────────────────────────────────
    with tab_insights:
        st.markdown("### 💡 Key Insights")
        insights = data.get("insights", [])
        if insights:
            for insight in insights:
                st.markdown(
                    f'<div class="nt-insight">{insight}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No insights extracted. Try re-analyzing with a detailed summary type.")

    # ── Tab: Timestamps ───────────────────────────────────────────────────────
    with tab_timestamps:
        st.markdown("### 🕐 Key Moments")
        timestamps = data.get("timestamps", [])
        segs = data.get("segments", [])
        if segs:
            last_seg = segs[-1]
            last_start = last_seg["start"] if isinstance(last_seg, dict) else getattr(last_seg, "start", 0)
            last_dur = last_seg.get("duration", 0) if isinstance(last_seg, dict) else getattr(last_seg, "duration", 0)
            total_dur = last_start + last_dur
        else:
            total_dur = 0

        pos_map = {"early": 0.15, "middle": 0.5, "late": 0.85}

        if timestamps:
            for ts in timestamps:
                approx_sec = pos_map.get(ts.get("position", "middle"), 0.5) * total_dur
                yt_url = build_youtube_url(vid, approx_sec)
                badge = format_timestamp(approx_sec)
                label = ts.get("label", "Key moment")
                reason = ts.get("reason", "")
                st.markdown(f"""
                <div class="nt-timestamp">
                    <span class="nt-ts-badge">{badge}</span>
                    <div>
                        <strong>{label}</strong><br>
                        <span style="color:#9090b0;font-size:0.85rem;">{reason}</span><br>
                        <a href="{yt_url}" target="_blank"
                           style="color:#a855f7;font-size:0.82rem;">▶ Jump to moment</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No timestamp highlights found for this video.")

    # ── Tab: Chat ─────────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown("### 💬 Chat with this Video")
        st.markdown(
            '<p style="color:#9090b0;font-size:0.88rem;">'
            'Ask anything about the video. Answers are grounded in the transcript.</p>',
            unsafe_allow_html=True,
        )

        # Display chat history
        for msg in st.session_state.chat_history:
            css_class = "nt-bubble-user" if msg["role"] == "user" else "nt-bubble-ai"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(
                f'<div class="{css_class}">{icon} {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        # Input row
        q_col, s_col = st.columns([5, 1])
        with q_col:
            user_q = st.text_input(
                "Ask a question",
                placeholder="e.g. What is the main argument of this video?",
                key="chat_input",
                label_visibility="collapsed",
            )
        with s_col:
            send = st.button("Send", key="send_chat", use_container_width=True)

        if send and user_q.strip():
            with st.spinner("Thinking…"):
                try:
                    answer, _ = answer_question(
                        question=user_q,
                        vectorstore=data["vectorstore"],
                        llm=data["llm"],
                        chat_history=st.session_state.chat_history,
                    )
                    st.session_state.chat_history.append({"role": "user", "content": user_q})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat error: {e}")

        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ── Tab: Transcript ───────────────────────────────────────────────────────
    with tab_transcript:
        st.markdown("### 📄 Full Transcript")
        with st.expander("View transcript", expanded=False):
            st.text_area(
                "Transcript",
                value=data["transcript"],
                height=400,
                label_visibility="collapsed",
            )

    # ── Tab: Export ───────────────────────────────────────────────────────────
    with tab_export:
        st.markdown("### 📥 Export & Download")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.download_button(
                label="📄 Download Transcript",
                data=data["transcript"].encode("utf-8"),
                file_name=f"transcript_{vid}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        summary_md = f"# Video Summary — {vid}\n\n## Quick Summary\n{data['summary_short']}\n"
        if data.get("summary_detailed"):
            summary_md += f"\n{data['summary_detailed']}\n"
        if data.get("insights"):
            summary_md += "\n## Key Insights\n" + "\n".join(f"- {i}" for i in data["insights"])

        with col_b:
            st.download_button(
                label="📋 Download Summary",
                data=summary_md.encode("utf-8"),
                file_name=f"summary_{vid}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_c:
            if st.button("📚 Generate Learning Notes", use_container_width=True):
                with st.spinner("Generating notes…"):
                    try:
                        notes = generate_learning_notes(data["transcript"], data["llm"])
                        st.session_state["notes"] = notes
                    except Exception as e:
                        st.error(f"Notes generation failed: {e}")

        if st.session_state.get("notes"):
            st.markdown("---")
            st.markdown(st.session_state["notes"])
            st.download_button(
                label="📥 Download Notes",
                data=st.session_state["notes"].encode("utf-8"),
                file_name=f"notes_{vid}.md",
                mime="text/markdown",
            )

else:
    # ── Empty State ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">🎬</div>
        <h3 style="color:#9090b0;font-weight:500;">No video analyzed yet</h3>
        <p style="color:#6060a0;">Paste a YouTube URL above and click <strong>Analyze</strong> to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ✨ What VideoNote AI can do")
    f1, f2, f3 = st.columns(3)
    for col, icon, title, desc in [
        (f1, "📋", "AI Summaries", "Short or detailed summaries generated by Gemini 2.5"),
        (f2, "💡", "Key Insights", "Bullet-point takeaways extracted automatically"),
        (f3, "💬", "Chat with Video", "Ask questions — answers grounded in the transcript"),
    ]:
        with col:
            st.markdown(f"""
            <div class="nt-card" style="text-align:center;">
                <div style="font-size:2.2rem;margin-bottom:0.6rem;">{icon}</div>
                <strong>{title}</strong>
                <p style="color:#9090b0;font-size:0.88rem;margin-top:0.4rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
