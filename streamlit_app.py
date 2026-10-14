import os
import base64
import json
from pathlib import Path
from typing import Any

import altair as alt
import requests
import streamlit as st

from app.services.learning.analytics import practice_accuracy_chart, wilson_interval
from app.services.learning.citations import referenced_citations

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT_SECONDS", "300"))

st.set_page_config(page_title="StudySync", page_icon="📘", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
    :root { color-scheme: dark; --ink: #f3f7ff; --muted: #9aacc8; --line: #1b3154; --paper: #030d1d; --surface: #091a32; --blue: #168cff; --violet: #7654ff; --accent: #ff4b4b; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .stApp { background: radial-gradient(ellipse at 86% -10%, #102a53 0, #07162d 33rem, var(--paper) 75rem); }
    [data-testid="stHeader"] { background: rgba(3,13,29,.42); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-bottom: 1px solid rgba(120,170,230,.12); }
    [data-testid="stSidebar"] { background: #06142a; border-right: 1px solid #172a48; }
    [data-testid="stSidebar"] * { color: var(--ink); }
    [data-testid="stSidebar"] hr { border-color: #203451; }
    .block-container { max-width: 1500px; padding: 1rem 2rem 3rem; }
    .hero { padding: 1.35rem 0 1.1rem; margin-bottom: 1rem; }
    .hero h1 { color: var(--ink) !important; font-size: 2rem; line-height: 1.12; margin: .2rem 0 .35rem; }
    .hero p { color: var(--muted); font-size: .95rem; margin: 0; }
    .brand-title { font: 700 1.4rem 'Space Grotesk', sans-serif; letter-spacing: -.04em; color: white; padding: .25rem 0; }
    .brand-title span { color: var(--blue); }
    .eyebrow, .section-kicker { color: #40a9ff; font-weight: 700; font-size: .7rem; text-transform: uppercase; letter-spacing: .12em; }
    .section-kicker { margin-bottom: .35rem; }
    .result { background: #0b1d38; border-left: 3px solid var(--blue); padding: .9rem 1.1rem; border-radius: 10px; color: var(--ink); }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #0c2141, #09172c); border: 1px solid var(--line); padding: .7rem .9rem; border-radius: 12px; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    div.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background: var(--accent); border: 1px solid var(--accent); color: white; border-radius: 10px; font-weight: 700; min-height: 2.8rem; transition: background .15s ease, transform .15s ease; }
    div.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background: #ed3d43; border-color: #ed3d43; transform: translateY(-1px); }
    div.stButton > button { border-radius: 9px; }
    div[data-testid="stFileUploader"] { border: 1px dashed #336195; border-radius: 12px; padding: .35rem; background: rgba(8,24,46,.8); }
    .st-key-study_tutor_composer [data-testid="stChatInput"] { border: 1px solid #29466b; border-radius: 30px; background: #0a1a30; box-shadow: 0 8px 28px rgba(0,0,0,.22); }
    .st-key-study_tutor_composer [data-testid="stChatInput"] textarea { background: transparent; color: var(--ink); }
    .st-key-study_tutor_composer [data-testid="stChatInput"] button { border-radius: 50%; }
    .st-key-study_tutor_composer [data-testid="stChatInput"] button:last-child { background: var(--accent); color: white; }
    .st-key-study_tutor_composer div[data-testid="stFileUploader"] { border: 1px dashed #336195; border-radius: 15px; padding: .35rem; background: rgba(8,24,46,.8); }
    .st-key-study_tutor_composer [data-testid="stPopover"] > button { width: 2.65rem; height: 2.65rem; border-radius: 50%; border: 1px solid #345272; background: #0a1a30; color: var(--ink); font-size: 1.2rem; }
    .st-key-study_tutor_composer [data-testid="stPopover"] > button:hover { border-color: var(--blue); color: #fff; }
    .st-key-study_tutor_composer [data-testid="stTextInput"] input { min-height: 2.8rem; border-radius: 24px; padding: 0 1rem; background: #0a1a30; }
    .st-key-study_tutor_composer [data-testid="stButton"] button { width: 2.65rem; height: 2.65rem; min-height: 2.65rem; padding: 0; border-radius: 50%; font-size: 1.2rem; }
    .auth-layout { min-height: 84vh; align-items: center; }
    .brand-panel { min-height: 570px; padding: 2.3rem 2rem; border: 1px solid #112e58; border-radius: 20px; background: radial-gradient(ellipse at 58% 74%, rgba(0,128,255,.22), transparent 44%), linear-gradient(145deg,#06142c,#030b19); box-shadow: inset 0 0 80px rgba(12,73,153,.15); }
    .brand-panel h1 { font: 700 clamp(2.1rem,3.2vw,3.1rem) 'Space Grotesk',sans-serif; letter-spacing: -.06em; margin: 1.3rem 0 .35rem; color: white; }
    .brand-panel h1 span { color: #218fff; }
    .brand-panel p { color: #bdcbe0; }
    .brand-features { display: grid; gap: .95rem; margin-top: 2rem; color: #e2edff; font-weight: 600; }
    .brand-features span { display: flex; align-items: center; gap: .8rem; }
    .brand-features b { display: grid; place-items: center; width: 2rem; height: 2rem; border: 1px solid #20549a; border-radius: 9px; color: #49adff; background: #0a2143; }
    .auth-mark { color: #39a4ff; font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .dashboard-card { min-height: 142px; padding: 1rem 1.05rem .8rem; border: 1px solid #1a3a66; border-radius: 14px; background: linear-gradient(145deg, #102b51, #0a1931); box-shadow: 0 12px 34px rgba(0,0,0,.2); }
    .dashboard-card small { color: #6dbbff; font-weight: 700; letter-spacing: .07em; }
    .dashboard-card h3 { margin: .45rem 0 .3rem; color: white; font-size: 1.05rem; }
    .dashboard-card p { color: var(--muted); font-size: .82rem; margin: 0; line-height: 1.45; }
    .progress-ring { display: grid; place-items: center; width: 118px; height: 118px; border-radius: 50%; background: conic-gradient(#168cff var(--progress), #183356 0); position: relative; }
    .progress-ring:before { content: ''; position: absolute; inset: 10px; border-radius: 50%; background: #0b1a31; }
    .progress-ring span { position: relative; z-index: 1; font: 700 1.25rem 'Space Grotesk',sans-serif; color: white; }
    [data-testid="stTabs"] { border-bottom: 1px solid #1b3154; }
    [data-testid="stTabs"] button { color: var(--muted); font-weight: 600; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: white; border-bottom-color: var(--blue); }
    [data-testid="stExpander"] { background: rgba(9,26,50,.8); border: 1px solid var(--line); border-radius: 10px; }
    [data-testid="stForm"] { border: 0; padding: 0; }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div { background: var(--surface); color: var(--ink); border-color: var(--line); }
    .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--muted); }
    div[data-testid="stFileUploader"] section, div[data-testid="stFileUploader"] small { color: var(--muted); }
    @media (max-width: 760px) { .block-container { padding: .7rem 1rem 2rem; } .brand-panel { min-height: auto; padding: 1.5rem; } .auth-shell { padding: 1.5rem; margin-top: 1rem; } }
    body:has(.auth-mark) .block-container { padding-top: 4.5rem; padding-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.request(method, f"{API_URL}{path}", headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)


def error_message(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return str(detail) or f"Request failed with status {response.status_code}"


def stream_study_tutor_turn(
    query: str,
    conversation_id: str | None,
    audio: Any | None,
    image: Any | None,
    tutoring_mode: str = "adaptive",
) -> str | None:
    files = {}
    if audio is not None:
        files["audio"] = (
            audio.name,
            audio.getvalue(),
            audio.type or "application/octet-stream",
        )
    if image is not None:
        files["image"] = (
            image.name,
            image.getvalue(),
            image.type or "application/octet-stream",
        )
    data = {}
    if query.strip():
        data["query"] = query.strip()
    if conversation_id:
        data["conversation_id"] = conversation_id
    data["tutoring_mode"] = tutoring_mode

    try:
        response = api_request(
            "POST",
            "/study/tutor/stream",
            data=data,
            files=files or None,
            stream=True,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the study tutor: {exc}")
        return None

    if not response.ok:
        st.error(error_message(response))
        response.close()
        return None

    resolved_conversation_id = conversation_id
    active_event = ""
    answer = ""
    citations = []
    generation_provider = None
    tutoring_mode_used = tutoring_mode
    mastery_band = None
    stream_completed = False
    with st.chat_message("assistant"):
        status = st.empty()
        answer_slot = st.empty()
        status.info("Preparing your course-grounded answer...")
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("event:"):
                    active_event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    payload = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    st.error("The tutor returned an invalid streaming event.")
                    return None

                if active_event == "conversation":
                    resolved_conversation_id = payload.get(
                        "conversation_id",
                        resolved_conversation_id,
                    )
                    if payload.get("transcript"):
                        status.info(f"Transcribed question: {payload['transcript']}")
                    elif payload.get("visual_context"):
                        status.info("Image understood. Retrieving related course material...")
                elif active_event == "context":
                    citations = payload.get("passages", [])
                    status.info("Generating a cited response from your course materials...")
                elif active_event == "token":
                    answer += payload.get("text", "")
                    answer_slot.markdown(answer)
                elif active_event == "replace":
                    answer = payload.get("text", "")
                    answer_slot.markdown(answer)
                elif active_event == "complete":
                    generation_provider = payload.get("generation_provider")
                    tutoring_mode_used = payload.get("tutoring_mode", tutoring_mode_used)
                    mastery_band = payload.get("mastery_band", mastery_band)
                elif active_event == "error":
                    status.empty()
                    st.error(payload.get("detail", "The tutor stream failed."))
                    return resolved_conversation_id
                elif active_event == "done":
                    stream_completed = True
                    generation_provider = payload.get(
                        "generation_provider",
                        generation_provider,
                    )
                    tutoring_mode_used = payload.get(
                        "tutoring_mode",
                        tutoring_mode_used,
                    )
                    mastery_band = payload.get("mastery_band", mastery_band)
        except requests.RequestException as exc:
            status.empty()
            st.error(f"The tutor connection ended before the response completed: {exc}")
            return resolved_conversation_id
        finally:
            response.close()

        status.empty()
        if not stream_completed:
            st.warning("The tutor stream ended before completion. Your question remains in the conversation.")
        elif generation_provider:
            st.caption(f"Response mode: {generation_provider}")
        if stream_completed and mastery_band:
            st.caption(
                f"Tutoring: {tutoring_mode_used.replace('_', ' ')} · "
                f"practice signal: {mastery_band.replace('_', ' ')} "
                "(not a validated mastery measure)"
            )

        for index, citation in referenced_citations(answer, citations):
            location = format_source_location(
                citation["location_type"],
                citation["location_number"],
            )
            with st.expander(f"[{index}] {citation['filename']} · {location}"):
                st.write(citation["content"])
                render_source_open_link(citation, f"stream-citation-{index}")

    return resolved_conversation_id


def format_source_location(location_type: str, location_number: int) -> str:
    if location_type == "timestamp":
        minutes, seconds = divmod(location_number, 60)
        return f"Timestamp {minutes:02d}:{seconds:02d}"
    return f"{location_type.title()} {location_number}"


def render_source_open_link(citation: dict[str, Any], key_prefix: str) -> None:
    source_id = citation.get("source_id")
    if not source_id:
        return

    button_key = (
        f"open-source-{key_prefix}-{source_id}-"
        f"{citation['location_type']}-{citation['location_number']}"
    )
    button_label = (
        "View cited slide"
        if citation.get("location_type") == "slide"
        else "Prepare source link"
    )
    if not st.button(button_label, key=button_key):
        return

    try:
        if citation.get("location_type") == "slide":
            response = api_request(
                "GET",
                f"/study/sources/{source_id}/slides/{citation['location_number']}",
            )
            if not response.ok:
                st.error(error_message(response))
                return
            slide = response.json()
            with st.expander(
                f"{slide['filename']} · Slide {slide['slide_number']} of {slide['slide_count']}",
                expanded=True,
            ):
                if slide["text"].strip():
                    st.text(slide["text"])
                else:
                    st.caption("No selectable text is available on this slide.")
                for image in slide["images"]:
                    st.image(
                        base64.b64decode(image["data"]),
                        use_container_width=True,
                    )
            return

        response = api_request(
            "GET",
            f"/study/sources/{source_id}/open",
            params={
                "location_type": citation["location_type"],
                "location_number": citation["location_number"],
            },
        )
        if not response.ok:
            st.error(error_message(response))
            return
        source_link = response.json()
        if source_link["viewer_supported"]:
            st.link_button("Open at cited location", source_link["url"])
        else:
            st.link_button("Open source file", source_link["url"])
            st.caption(
                "The source file is available, but this browser cannot jump directly "
                "to the cited slide or location."
            )
    except requests.RequestException as exc:
        st.error(f"Could not prepare the source link: {exc}")


def authenticate(endpoint: str, email: str, password: str) -> bool:
    response = requests.post(
        f"{API_URL}/auth/{endpoint}",
        json={"email": email, "password": password},
        timeout=30,
    )
    if response.ok:
        st.session_state.token = response.json()["access_token"]
        return True
    st.error(error_message(response))
    return False


def request_password_reset(email: str) -> bool:
    response = requests.post(
        f"{API_URL}/auth/forgot-password",
        json={"email": email},
        timeout=30,
    )
    if response.status_code == 202:
        st.success("If that email is registered, recovery instructions have been sent.")
        return True
    st.error(error_message(response))
    return False


def reset_password(token: str, password: str) -> bool:
    response = requests.post(
        f"{API_URL}/auth/reset-password",
        json={"token": token, "password": password},
        timeout=30,
    )
    if response.status_code == 204:
        st.success("Password updated. You can now sign in.")
        return True
    st.error(error_message(response))
    return False


_NAVIGATION_TABS = {
    "Roadmap": "Dashboard",
    "Dashboard": "Dashboard",
    "My Library": "My library",
    "AI Tutor": "Dashboard",
    "Quizzes": "Dashboard",
    "Analytics": "Dashboard",
    "RAGAS Evaluation": "RAGAS evaluation",
    "Settings": "System status",
}
_TAB_NAVIGATION = {
    "Dashboard": "Dashboard",
    "My library": "My Library",
    "System status": "Settings",
    "RAGAS evaluation": "RAGAS Evaluation",
}


def _sync_tab_navigation() -> None:
    selected_tab = st.session_state.get("workspace_tabs", "Dashboard")
    st.session_state["workspace_nav"] = _TAB_NAVIGATION.get(selected_tab, "Dashboard")


def _sync_sidebar_navigation() -> None:
    selected_page = st.session_state.get("workspace_nav", "Dashboard")
    st.session_state["workspace_tabs"] = _NAVIGATION_TABS[selected_page]


if "token" not in st.session_state:
    brand_column, auth_column = st.columns([1.05, .95], gap="large")
    with brand_column:
        st.markdown(
            """
            <div class="brand-panel">
              <div class="brand-title">Study<span>Sync</span></div>
              <h1>Study smarter.<br><span>Stay in sync.</span></h1>
              <p>AI-powered learning, grounded in the course materials you trust.</p>
              <div class="brand-features">
                <span><b>▤</b> AI document analysis</span>
                <span><b>✦</b> Source-cited AI tutor</span>
                <span><b>✓</b> Adaptive quiz practice</span>
                <span><b>↗</b> Learning analytics</span>
              </div>
              <div style="margin:2rem auto 0;max-width:330px;text-align:center;font-size:5rem;
                          color:#218fff;text-shadow:0 0 30px rgba(24,135,255,.7)">◉<br>
                <span style="font-size:2.3rem;letter-spacing:.2rem">▰ ▰ ▰</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with auth_column:
        with st.container(border=True):
            st.markdown('<div class="auth-mark">Your learning workspace</div>', unsafe_allow_html=True)
            st.markdown("### Welcome back.")
            st.write(
                "Sign in to open your library, learn with source-cited answers, "
                "and keep track of your progress."
            )
            auth_mode = st.radio(
                "Account action",
                ["Sign in", "Create account", "Recover password"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if auth_mode == "Recover password":
                with st.form("forgot_password"):
                    recovery_email = st.text_input("Account email")
                    recovery_submitted = st.form_submit_button(
                        "Send recovery instructions",
                        type="primary",
                        use_container_width=True,
                    )
                if recovery_submitted and recovery_email:
                    request_password_reset(recovery_email)

                with st.form("reset_password"):
                    reset_token = st.text_input("Recovery token", type="password")
                    new_password = st.text_input("New password", type="password")
                    reset_submitted = st.form_submit_button(
                        "Set new password",
                        use_container_width=True,
                    )
                if reset_submitted:
                    if not reset_token or not new_password:
                        st.warning("Enter the recovery token and a new password.")
                    else:
                        reset_password(reset_token, new_password)
                st.stop()

            with st.form("authentication"):
                email = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input(
                    "Password",
                    type="password",
                    help="Use at least 8 characters with uppercase, lowercase, and a number.",
                )
                submitted = st.form_submit_button(
                    auth_mode,
                    type="primary",
                    use_container_width=True,
                )
            if submitted:
                if not email or not password:
                    st.warning("Enter both email and password.")
                else:
                    endpoint = "login" if auth_mode == "Sign in" else "register"
                    if authenticate(endpoint, email, password):
                        st.rerun()
    st.stop()

with st.sidebar:
    st.markdown('<div class="brand-title">Study<span>Sync</span></div>', unsafe_allow_html=True)
    st.caption("AI-POWERED LEARNING PLATFORM")
    st.divider()
    selected_tab = st.session_state.get("workspace_tabs", "Dashboard")
    selected_page = _TAB_NAVIGATION.get(selected_tab, "Dashboard")
    st.radio(
        "Workspace navigation",
        list(_NAVIGATION_TABS),
        index=list(_NAVIGATION_TABS).index(selected_page),
        key="workspace_nav",
        on_change=_sync_sidebar_navigation,
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

header_brand, header_search, header_activity, header_user = st.columns([1.2, 3.4, .35, .45])
with header_brand:
    st.markdown('<div class="brand-title">Study<span>Sync</span></div>', unsafe_allow_html=True)
with header_search:
    with st.form("course_search", clear_on_submit=False):
        search_column, search_button_column = st.columns([5, .7])
        with search_column:
            global_search_query = st.text_input(
                "Search your course library",
                placeholder="Search your course materials...",
                label_visibility="collapsed",
            )
        with search_button_column:
            search_submitted = st.form_submit_button("⌕", use_container_width=True)
    if search_submitted:
        if not global_search_query.strip():
            st.warning("Enter a topic or phrase to search your course library.")
        else:
            try:
                search_response = api_request(
                    "GET",
                    "/study/search",
                    params={"q": global_search_query.strip(), "limit": 5},
                )
                if search_response.ok:
                    st.session_state["global_search_results"] = search_response.json()["results"]
                else:
                    st.error(error_message(search_response))
            except requests.RequestException as exc:
                st.error(f"Could not search your study library: {exc}")
with header_activity:
    st.markdown("<div style='padding-top:.55rem;color:#9aacc8'>◉</div>", unsafe_allow_html=True)
with header_user:
    st.markdown("<div style='margin-top:.25rem;background:#123b68;border-radius:50%;height:2rem;width:2rem;text-align:center;padding-top:.35rem;color:white;font-weight:700'>S</div>", unsafe_allow_html=True)

if st.session_state.get("global_search_results") is not None:
    with st.expander("Course search results", expanded=True):
        results = st.session_state["global_search_results"]
        if not results:
            st.info("No matching passages found in your uploaded sources.")
        for result in results:
            st.markdown(
                f"**{result['filename']} · "
                f"{format_source_location(result['location_type'], result['location_number'])}**"
            )
            st.write(result["content"])

workspace_tab_labels = [
    "Dashboard",
    "My library",
    "System status",
    "RAGAS evaluation",
]
if st.session_state.get("workspace_tabs") not in workspace_tab_labels:
    st.session_state["workspace_tabs"] = "Dashboard"

study_tab, extract_tab, health_tab, evaluation_tab = st.tabs(
    workspace_tab_labels,
    default=st.session_state["workspace_tabs"],
    key="workspace_tabs",
    on_change=_sync_tab_navigation,
)

with study_tab:
    st.markdown('<div class="hero"><div class="section-kicker">YOUR WORKSPACE</div><h1>Welcome back 👋</h1><p>What would you like to learn today?</p></div>', unsafe_allow_html=True)
    action_columns = st.columns(3)
    action_cards = [
        ("01 · LEARN", "Upload & Learn", "Add course documents and keep every idea linked to its source."),
        ("02 · AI TUTOR", "Ask a question", "Get grounded answers with page, slide, and time citations."),
        ("03 · PRACTICE", "Generate a quiz", "Practise what you are learning and review your progress."),
    ]
    for column, (eyebrow, title, description) in zip(action_columns, action_cards, strict=True):
        with column:
            st.markdown(
                f'<div class="dashboard-card"><small>{eyebrow}</small><h3>{title}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )
    action_buttons = st.columns(3)
    if action_buttons[0].button("Upload Document", key="dashboard_upload", type="primary", use_container_width=True):
        st.session_state["upload_material_open"] = True
        st.rerun()
    if action_buttons[1].button("Ask Question", key="dashboard_tutor", use_container_width=True):
        st.toast("Your course tutor is ready below.")
    if action_buttons[2].button("Generate Quiz", key="dashboard_quiz", use_container_width=True):
        st.toast("Your practice quiz is ready below.")

    dashboard_progress = None
    dashboard_sources = []
    dashboard_courses = []
    try:
        course_response = api_request("GET", "/study/courses")
        progress_response = api_request("GET", "/study/progress")
        sources_response = api_request("GET", "/study/sources", params={"limit": 5})
        if course_response.ok:
            dashboard_courses = course_response.json()
        else:
            st.error(f"Could not load courses: {error_message(course_response)}")
        if progress_response.ok:
            dashboard_progress = progress_response.json()
        else:
            st.error(f"Could not load learning progress: {error_message(progress_response)}")
        if sources_response.ok:
            dashboard_sources = sources_response.json()
        else:
            st.error(f"Could not load your library: {error_message(sources_response)}")
    except requests.RequestException as exc:
        st.error(f"Could not load the live learning dashboard: {exc}")

    if dashboard_courses:
        st.markdown(f"**Active library:** {dashboard_courses[0]['title']} · {dashboard_courses[0]['passage_count']} searchable passages")
    else:
        st.info("Your learning dashboard is ready. Add a PDF or PowerPoint below to start your course library.")

    recent_column, progress_column = st.columns([1.1, 1])
    with recent_column:
        st.markdown("### Recent documents")
        if dashboard_sources:
            for source in dashboard_sources[:4]:
                st.markdown(
                    f"📘 **{source['filename']}**  \n"
                    f"{source['unit_count']} searchable transcript passages"
                    if source.get("source_type") == "video"
                    else f"📘 **{source['filename']}**  \n"
                    f"{source['page_count']} pages/slides · {source['unit_count']} searchable passages"
                )
        else:
            st.caption("Your recently added course materials will appear here.")
    with progress_column:
        st.markdown("### Learning progress")
        if dashboard_progress:
            answered = dashboard_progress["answered_questions"]
            correct = dashboard_progress["correct_answers"]
            if answered:
                accuracy = dashboard_progress["practice_accuracy"]
                interval = wilson_interval(correct, answered)
                st.markdown(
                    f'<div class="progress-ring" style="--progress:{accuracy:.0%}"><span>{accuracy:.0%}</span></div>',
                    unsafe_allow_html=True,
                )
                if interval is not None:
                    st.caption(
                        f"Observed accuracy: {correct}/{answered} correct. "
                        f"With this sample size, the 95% interval is "
                        f"{interval[0]:.0%}–{interval[1]:.0%}."
                    )
            else:
                st.info("No practice answers yet. Accuracy will appear after you answer quiz questions.")
            summary_columns = st.columns(3)
            summary_columns[0].metric("Sources", dashboard_progress["total_sources"])
            summary_columns[1].metric("Practised", dashboard_progress["answered_questions"])
            summary_columns[2].metric("Questions", dashboard_progress["total_questions"])
            st.caption("Accuracy is based on answered quiz questions—not uploaded resources. Mastery is a revision aid, not a validated competency score.")
        if dashboard_progress and dashboard_progress["topics"]:
            accuracy_chart = practice_accuracy_chart(dashboard_progress["topics"])
            if accuracy_chart is not None:
                st.altair_chart(
                    accuracy_chart,
                    use_container_width=True,
                    theme=None,
                    key="dashboard_practice_accuracy_chart",
                )
            else:
                st.caption("Topic accuracy appears after a question in that topic is answered.")
            focus_topics = [
                topic for topic in dashboard_progress["topics"]
                if (
                    topic["recommended_review_in_days"] is not None
                    and topic["recommended_review_in_days"] <= 3
                )
            ]
            if focus_topics:
                review_list = ", ".join(
                    f"{topic['topic']} (in {topic['recommended_review_in_days']} day"
                    f"{'' if topic['recommended_review_in_days'] == 1 else 's'})"
                    for topic in sorted(
                        focus_topics,
                        key=lambda item: item["recommended_review_in_days"] or 0,
                    )[:3]
                )
                st.info(f"Suggested review: {review_list}")
        else:
            st.caption("Answer a practice question to see topic-level progress.")
    upload_study_source = False
    study_file = None
    if st.session_state.get("upload_material_open"):
        st.divider()
        st.markdown('<div class="section-kicker">COURSE LIBRARY</div>', unsafe_allow_html=True)
        st.subheader("Add course material")
        st.caption("Upload a PDF, PowerPoint, or lecture video. Figure text, notes, and timestamped transcripts are indexed with their source locations.")
        with st.form("study_source_upload"):
            study_file = st.file_uploader(
                "Choose course material",
                type=["pdf", "pptx", "mp4", "webm"],
                key="study_source",
            )
            upload_study_source = st.form_submit_button("Add to study library", type="primary")
        if st.button("Close upload panel", key="close_study_upload"):
            st.session_state["upload_material_open"] = False
            st.rerun()
    if upload_study_source:
        if study_file is None:
            st.warning("Choose a PDF, PowerPoint, MP4, or WebM file first.")
        else:
            try:
                with st.spinner("Extracting text and figures, building semantic chunks, and indexing your source..."):
                    response = api_request(
                        "POST",
                        "/study/sources",
                        files={
                            "file": (
                                study_file.name,
                                study_file.getvalue(),
                                study_file.type or "application/octet-stream",
                            )
                        },
                    )
                if response.status_code == 201:
                    source = response.json()
                    st.success(
                        f"Added {source['filename']} ({source['page_count']} source units; "
                        f"{source['unit_count']} searchable passages)."
                    )
                    st.caption(
                        f"Chunking: {source.get('chunking_strategy', 'semantic parent/child')} · "
                        f"{source.get('figure_descriptions') or 0} figure descriptions indexed."
                    )
                    if source.get("vector_indexed") is False:
                        st.warning("Vector indexing is unavailable; BM25 search remains enabled.")
                    if source.get("graph_indexed"):
                        st.caption("Course concepts and source locations were added to the Neo4j knowledge graph.")
                    elif source.get("graph_indexed") is False:
                        st.caption("Neo4j is not configured; the live learning dashboard and study library are unaffected.")
                    if source.get("source_type") == "video":
                        st.caption("Lecture audio was transcribed into timestamped, searchable passages.")
                else:
                    st.error(error_message(response))
            except requests.RequestException as exc:
                st.error(f"Could not reach the study service: {exc}")

    tutor_column, quiz_column = st.columns([1.25, .75])
    with tutor_column:
        st.markdown("#### AI tutor")
        st.caption(
            "Continue a saved conversation, ask by text or video, and attach a figure when "
            "you want the tutor to connect what you see to cited course material."
        )
        try:
            conversation_response = api_request("GET", "/study/conversations")
            if conversation_response.ok:
                study_conversations = conversation_response.json()
            else:
                study_conversations = []
                st.error(f"Could not load study conversations: {error_message(conversation_response)}")
        except requests.RequestException as exc:
            study_conversations = []
            st.error(f"Could not load study conversations: {exc}")

        conversation_titles = {
            item["id"]: item.get("title") or "Study conversation"
            for item in study_conversations
        }
        conversation_options = ["new", *conversation_titles]
        current_selection = st.session_state.get("study_conversation_selection", "new")
        if current_selection not in conversation_options:
            current_selection = "new"
            st.session_state["study_conversation_selection"] = current_selection
        selector_key = "study_conversation_selector"
        if st.session_state.get(selector_key) != current_selection:
            st.session_state[selector_key] = current_selection
        selection_column, new_chat_column = st.columns([1, .3])
        with selection_column:
            selected_conversation = st.selectbox(
                "Conversation",
                options=conversation_options,
                format_func=lambda item: (
                    "Start a new conversation"
                    if item == "new"
                    else conversation_titles[item]
                ),
                key=selector_key,
                label_visibility="collapsed",
            )
        st.session_state["study_conversation_selection"] = selected_conversation
        with new_chat_column:
            if st.button("New chat", key="study_new_conversation", use_container_width=True):
                st.session_state["study_conversation_selection"] = "new"
                st.session_state.pop("study_conversation_id", None)
                st.rerun()

        selected_conversation_id = (
            selected_conversation if selected_conversation != "new" else None
        )
        st.session_state["study_conversation_id"] = selected_conversation_id
        if selected_conversation_id:
            try:
                messages_response = api_request(
                    "GET",
                    f"/study/conversations/{selected_conversation_id}/messages",
                )
                if messages_response.ok:
                    for message in messages_response.json()[-24:]:
                        role = message["role"]
                        if role not in {"user", "assistant"}:
                            continue
                        with st.chat_message(role):
                            st.markdown(message["content"])
                            metadata = message.get("metadata") or {}
                            if metadata.get("transcript"):
                                st.caption("Question transcribed from audio.")
                            if metadata.get("visual_context"):
                                with st.expander("Image understanding used for this turn"):
                                    st.write(metadata["visual_context"])
                            if metadata.get("generation_provider"):
                                st.caption(
                                    f"Response mode: {metadata['generation_provider']}"
                                )
                            for index, citation in referenced_citations(
                                message["content"],
                                metadata.get("citations", []),
                            ):
                                location = format_source_location(
                                    citation["location_type"],
                                    citation["location_number"],
                                )
                                with st.expander(
                                    f"[{index}] {citation['filename']} · {location}"
                                ):
                                    st.write(citation["content"])
                                    render_source_open_link(
                                        citation,
                                        f"conversation-{selected_conversation_id}-{index}",
                                    )
                else:
                    st.error(error_message(messages_response))
            except requests.RequestException as exc:
                st.error(f"Could not load this conversation: {exc}")
        else:
            st.info(
                "Ask a question to start. The tutor will search your uploaded materials "
                "and keep the conversation available for follow-up questions."
            )

        input_version = st.session_state.get("study_tutor_input_version", 0)
        with st.container(key="study_tutor_composer"):
            attach_column, prompt_column, microphone_column, send_column = st.columns(
                [.55, 8, .55, .55],
                vertical_alignment="bottom",
            )
            image_question = None
            with attach_column:
                with st.popover("＋", help="Attach an image or choose a tutoring approach"):
                    image_question = st.file_uploader(
                        "Attach a figure or problem image",
                        max_upload_size=25,
                        type=["jpg", "jpeg", "png", "webp"],
                        key=f"study_tutor_image_{input_version}",
                    )
                    tutoring_style = st.selectbox(
                        "Tutoring approach",
                        options={
                            "Adaptive (recommended)": "adaptive",
                            "Hints first": "hints_first",
                            "Direct explanation": "direct",
                        },
                        format_func=lambda label: label,
                        key=f"study_tutor_style_{input_version}",
                    )
                    selected_tutoring_mode = {
                        "Adaptive (recommended)": "adaptive",
                        "Hints first": "hints_first",
                        "Direct explanation": "direct",
                    }[tutoring_style]

            with microphone_column:
                audio_error_key = f"study_tutor_audio_error_{input_version}"
                with st.popover("🎙", help="Record and transcribe a question"):
                    st.caption(
                        "Allow microphone access for this site, record your question, "
                        "then add the transcript to the composer."
                    )
                    recorded_audio = st.audio_input(
                        "Record a question",
                        sample_rate=16_000,
                        key=f"study_tutor_audio_{input_version}",
                    )
                    transcribe_submitted = st.button(
                        "Transcribe recording",
                        key=f"study_tutor_transcribe_{input_version}",
                        type="primary",
                        disabled=recorded_audio is None,
                        use_container_width=True,
                    )
                    if transcribe_submitted and recorded_audio is not None:
                        try:
                            transcript_response = api_request(
                                "POST",
                                "/study/tutor/transcribe",
                                files={
                                    "audio": (
                                        recorded_audio.name or "study-question.wav",
                                        recorded_audio.getvalue(),
                                        recorded_audio.type or "audio/wav",
                                    )
                                },
                            )
                            if transcript_response.ok:
                                st.session_state[
                                    f"study_tutor_query_{input_version}"
                                ] = transcript_response.json()["transcript"]
                                st.session_state[
                                    f"study_tutor_transcript_ready_{input_version}"
                                ] = True
                                st.session_state.pop(audio_error_key, None)
                            else:
                                st.session_state[audio_error_key] = error_message(
                                    transcript_response
                                )
                        except requests.RequestException as exc:
                            st.session_state[audio_error_key] = (
                                f"Could not transcribe the recording: {exc}"
                            )

            with prompt_column:
                tutor_query = st.text_input(
                    "Your question",
                    max_chars=500,
                    placeholder="Ask a question or follow up on the previous answer...",
                    key=f"study_tutor_query_{input_version}",
                    label_visibility="collapsed",
                )

            with send_column:
                tutor_submitted = st.button(
                    "↑",
                    key=f"study_tutor_send_{input_version}",
                    type="primary",
                    help="Send question",
                    use_container_width=True,
                )

            video_question = st.session_state.get(
                f"study_tutor_video_{input_version}"
            )
            if st.session_state.get(audio_error_key):
                st.error(st.session_state.pop(audio_error_key))
            if st.session_state.get(
                f"study_tutor_transcript_ready_{input_version}"
            ):
                st.caption("Transcript added. Review or edit it before sending.")

            with st.expander("Attach a video (MP4 or WebM)"):
                video_question = st.file_uploader(
                    "Video question",
                    max_upload_size=25,
                    type=["mp4", "webm"],
                    key=f"study_tutor_video_{input_version}",
                )

        if tutor_submitted:
            tutor_query = tutor_query.strip()
            audio_question = video_question
            if not tutor_query and audio_question is None:
                st.warning("Enter a question or attach an MP4/WebM video.")
            else:
                next_conversation_id = stream_study_tutor_turn(
                    tutor_query,
                    selected_conversation_id,
                    audio_question,
                    image_question,
                    selected_tutoring_mode,
                )
                if next_conversation_id:
                    st.session_state["study_conversation_selection"] = next_conversation_id
                    st.session_state["study_conversation_id"] = next_conversation_id
                    st.session_state.pop(
                        f"study_tutor_transcript_ready_{input_version}",
                        None,
                    )
                    st.session_state["study_tutor_input_version"] = input_version + 1
                    st.rerun()

    with quiz_column:
        st.markdown("#### Practice a topic")
        st.caption("Create a source-based passage question and record your answer.")
        with st.form("study_quiz_create"):
            quiz_topic = st.text_input("Topic to practise", max_chars=255)
            quiz_submitted = st.form_submit_button(
                "Generate practice question",
                type="primary",
                use_container_width=True,
            )
        if quiz_submitted:
            if not quiz_topic.strip():
                st.warning("Enter a topic to practise.")
            else:
                try:
                    response = api_request(
                        "POST",
                        "/study/quizzes",
                        json={"topic": quiz_topic.strip()},
                    )
                    if response.status_code == 201:
                        st.session_state["study_quiz_payload"] = response.json()
                    else:
                        st.error(error_message(response))
                except requests.RequestException as exc:
                    st.error(f"Could not reach the study service: {exc}")

        quiz_payload = st.session_state.get("study_quiz_payload")
        if quiz_payload:
            st.write(quiz_payload["prompt"])
            for index, option in enumerate(quiz_payload["options"]):
                with st.expander(
                    f"Passage {chr(65 + index)} · {option['filename']} · "
                    f"{format_source_location(option['location_type'], option['location_number'])}"
                ):
                    st.write(option["content"])
                    render_source_open_link(
                        option,
                        f"quiz-{quiz_payload['id']}-option-{index}",
                    )
            if quiz_payload.get("answered_index") is None:
                selected_index = st.radio(
                    "Choose the best-supported passage",
                    options=range(len(quiz_payload["options"])),
                    format_func=lambda index: f"Passage {chr(65 + index)}",
                    key=f"study_quiz_answer_{quiz_payload['id']}",
                )
                if st.button("Check answer", key=f"study_quiz_submit_{quiz_payload['id']}"):
                    try:
                        response = api_request(
                            "POST",
                            f"/study/quizzes/{quiz_payload['id']}/answer",
                            json={"selected_index": selected_index},
                        )
                        if response.ok:
                            st.session_state["study_quiz_payload"] = response.json()
                            st.rerun()
                        else:
                            st.error(error_message(response))
                    except requests.RequestException as exc:
                        st.error(f"Could not reach the study service: {exc}")
            else:
                if quiz_payload["correct"]:
                    st.success("Correct — this passage best matches the topic.")
                else:
                    st.warning("Not quite. Review the supporting course passage below.")
                explanation = quiz_payload["explanation"]
                st.markdown(
                    f"**{explanation['filename']} · "
                    f"{format_source_location(explanation['location_type'], explanation['location_number'])}**"
                )
                st.write(explanation["content"])

    st.divider()
    st.markdown("#### Learning progress")
    try:
        progress_response = api_request("GET", "/study/progress")
        if progress_response.ok:
            progress = progress_response.json()
            metric_columns = st.columns(4)
            metric_columns[0].metric("Sources", progress["total_sources"])
            metric_columns[1].metric("Questions", progress["total_questions"])
            metric_columns[2].metric("Answered", progress["answered_questions"])
            accuracy_display = (
                f"{progress['practice_accuracy']:.0%}"
                if progress["answered_questions"]
                else "—"
            )
            accuracy_sample = (
                f"{progress['correct_answers']}/{progress['answered_questions']} correct"
                if progress["answered_questions"]
                else "No answers yet"
            )
            metric_columns[3].metric(
                "Observed practice accuracy",
                accuracy_display,
                delta=accuracy_sample,
                delta_color="off",
            )
            st.caption("Mastery estimates use a BKT-inspired update with time-based forgetting; treat them as revision guidance, not validated competency scores.")
            if progress["topics"]:
                accuracy_chart = practice_accuracy_chart(progress["topics"])
                if accuracy_chart is not None:
                    st.altair_chart(
                        accuracy_chart,
                        use_container_width=True,
                        theme=None,
                        key="study_practice_accuracy_chart",
                    )
                else:
                    st.info("Answer practice questions to build a meaningful topic accuracy chart.")
        else:
            st.error(error_message(progress_response))
    except requests.RequestException as exc:
        st.error(f"Could not load learning progress: {exc}")

    st.markdown("#### Search your sources")
    st.caption("Search combines BM25 keyword relevance and Qdrant cosine similarity, with page and slide citations.")
    with st.form("study_source_search"):
        study_query = st.text_input("Search course material", max_chars=500)
        search_submitted = st.form_submit_button("Search")
    if search_submitted:
        if not study_query.strip():
            st.warning("Enter a search phrase.")
        else:
            try:
                response = api_request(
                    "GET",
                    "/study/search",
                    params={"q": study_query.strip(), "limit": 10},
                )
                if response.ok:
                    results = response.json()["results"]
                    if not results:
                        st.info("No matching passages found in your uploaded sources.")
                    for result in results:
                        st.markdown(
                            f"**{result['filename']} — "
                            f"{format_source_location(result['location_type'], result['location_number'])}** "
                            f"(hybrid relevance {result['score']:.2f})"
                        )
                        st.write(result["content"])
                else:
                    st.error(error_message(response))
            except requests.RequestException as exc:
                st.error(f"Could not reach the study service: {exc}")

    st.markdown("#### Your study library")
    try:
        sources_response = api_request("GET", "/study/sources")
        if sources_response.ok:
            sources = sources_response.json()
            if not sources:
                st.info("No study sources yet. Add a course PDF or PowerPoint above.")
            for source in sources:
                source_location = (
                    "video transcript"
                    if source.get("source_type") == "video"
                    else f"{source['page_count']} pages/slides"
                )
                st.write(
                    f"**{source['filename']}** · {source_location} · "
                    f"{source['unit_count']} passages"
                )
        else:
            st.error(error_message(sources_response))
    except requests.RequestException as exc:
        st.error(f"Could not load your study library: {exc}")

with extract_tab:
    st.markdown('<div class="section-kicker">Workflow 01</div>', unsafe_allow_html=True)
    st.subheader("Invoice extraction")
    st.caption("Turn an invoice image into searchable text and structured fields.")
    extraction_mode = st.radio("Processing mode", ["OCR + structured data", "Vision language model"], horizontal=True)
    document = st.file_uploader("Choose a PNG or JPEG", type=["png", "jpg", "jpeg"], key="document")
    if document and st.button("Process document", type="primary"):
        endpoint = "/documents/vlm" if extraction_mode.startswith("Vision") else "/documents/"
        with st.spinner("Reading document..."):
            response = api_request("POST", endpoint, files={"file": (document.name, document.getvalue(), document.type)})
        if response.ok:
            payload = response.json()
            st.success(f"Processed {payload['filename']}")
            st.metric("Processing mode", extraction_mode)
            left, right = st.columns(2)
            with left:
                st.markdown("#### Extracted text")
                st.text_area("Text", payload.get("extracted_text") or "No text returned.", height=280, label_visibility="collapsed")
            with right:
                st.markdown("#### Structured fields")
                invoice = (payload.get("layout_data") or {}).get("invoice_data")
                st.json(invoice or {"status": "No structured invoice fields returned"})
            st.caption(f"Document ID: {payload['id']} | Stored at: {payload['storage_path']}")
        else:
            st.error(error_message(response))

with evaluation_tab:
    st.markdown(
        '<div class="section-kicker">Offline quality evaluation</div>',
        unsafe_allow_html=True,
    )
    st.subheader("RAGAS evaluation")
    report_path = Path(__file__).resolve().parent / "evals" / "ragas-report.json"
    try:
        with report_path.open(encoding="utf-8") as report_file:
            ragas_report = json.load(report_file)
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not load the saved RAGAS report: {exc}")
    else:
        st.warning(
            "Exploratory result only: the cases are machine-generated and not "
            "human-reviewed. The responses labeled full pipeline are not live "
            "runs of the deployed tutor. Do not present these scores as a "
            "reviewed benchmark or compliance proof."
        )
        run_columns = st.columns(4)
        run_columns[0].metric("Evaluation status", ragas_report["evaluation_status"])
        run_columns[1].metric("Candidate cases", ragas_report["case_count"])
        run_columns[2].metric("RAGAS-scored cases", ragas_report["ragas_scored_case_count"])
        run_columns[3].metric("Human reviewed", "Yes" if ragas_report["human_reviewed"] else "No")
        st.caption(
            f"Judge: {ragas_report['judge_model']} · Embeddings: "
            f"{ragas_report['embedding_model']} · Evaluated: "
            f"{ragas_report['evaluated_at_utc']}"
        )

        score_rows = []
        for baseline, results in ragas_report["baselines"].items():
            for metric in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ):
                score = results.get(metric)
                if score is not None:
                    score_rows.append(
                        {
                            "baseline": baseline.replace("_", " ").title(),
                            "metric": metric.replace("_", " ").title(),
                            "score": float(score),
                            "valid_cases": results["metric_valid_case_counts"][metric],
                            "failed_cases": results["metric_failed_case_counts"][metric],
                        }
                    )
        if score_rows:
            score_chart = (
                alt.Chart(alt.Data(values=score_rows))
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X(
                        "score:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%", title="Mean score (0–100%)"),
                    ),
                    y=alt.Y("metric:N", title=None, sort=["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]),
                    yOffset="baseline:N",
                    color=alt.Color("baseline:N", title="Response set"),
                    tooltip=[
                        alt.Tooltip("baseline:N", title="Response set"),
                        alt.Tooltip("metric:N", title="Metric"),
                        alt.Tooltip("score:Q", title="Mean score", format=".1%"),
                        alt.Tooltip("valid_cases:Q", title="Valid cases"),
                        alt.Tooltip("failed_cases:Q", title="Failed cases"),
                    ],
                )
                .properties(height=220)
                .configure_view(stroke=None)
                .configure_axis(labelColor="#d6e5fa", titleColor="#9aacc8", gridColor="#1b3154")
            )
            st.altair_chart(
                score_chart,
                use_container_width=True,
                theme=None,
                key="ragas_baseline_comparison",
            )

        st.markdown("#### Baseline details")
        for baseline, results in ragas_report["baselines"].items():
            with st.expander(baseline.replace("_", " ").title()):
                metrics = st.columns(4)
                for column, metric in zip(
                    metrics,
                    ("faithfulness", "answer_relevancy", "context_precision", "context_recall"),
                    strict=True,
                ):
                    column.metric(
                        metric.replace("_", " ").title(),
                        f"{results[metric]:.1%}",
                        help=(
                            f"{results['metric_valid_case_counts'][metric]} valid and "
                            f"{results['metric_failed_case_counts'][metric]} failed cases"
                        ),
                    )
                st.caption(
                    f"Out-of-scope abstention accuracy: "
                    f"{results['out_of_scope_abstention_accuracy']:.1%} · "
                    f"status: {results['evaluation_status']}"
                )
        st.caption(f"Report file: {report_path.name}")

with health_tab:
    st.markdown('<div class="section-kicker">System observability</div>', unsafe_allow_html=True)
    st.subheader("Runtime status")
    if st.button("Refresh health"):
        for label, endpoint in [("API", "/"), ("Redis", "/health/redis"), ("Full", "/health/full")]:
            response = api_request("GET", endpoint)
            if response.ok:
                st.metric(label, "Healthy")
                st.json(response.json())
            else:
                st.metric(label, f"HTTP {response.status_code}")
