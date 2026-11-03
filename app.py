
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import html
import re

from nlp_model import analyze_complaint
from database import (
    initialize_database,
    save_complaint,
    get_complaint,
    get_all_complaints,
    update_status
)

# ==================================================
# PAGE SETUP
# ==================================================

st.set_page_config(
    page_title="CivicConnect AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_database()

# ==================================================
# DESIGN
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f4f7fb;
    color: #24364b;
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e3eaf2;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 15px !important;
    font-weight: 650 !important;
    color: #344b63 !important;
    padding: 6px 0;
}

h1, h2, h3 {
    color: #193b63 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
}

h1 {
    font-size: 2.5rem !important;
}

h2 {
    font-size: 1.9rem !important;
}

h3 {
    font-size: 1.25rem !important;
}

p, label, li {
    color: #52657b;
    font-size: 15px;
    line-height: 1.7;
}

.material-symbols-rounded {
    font-family: 'Material Symbols Rounded';
    font-weight: normal;
    font-style: normal;
    display: inline-flex;
    vertical-align: middle;
    line-height: 1;
    white-space: nowrap;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}

.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 13px !important;
}

.hero {
    background: linear-gradient(120deg, #e5f3ff 0%, #e8f0ff 55%, #e5f6f1 100%);
    border: 1px solid #d5e5f5;
    border-radius: 20px;
    padding: 40px;
    margin-bottom: 28px;
    box-shadow: 0 5px 18px rgba(35, 70, 110, 0.045);
}

.hero-tag {
    display: inline-block;
    background: #ffffff;
    color: #087f8c;
    padding: 8px 14px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 17px;
}

.hero h1 {
    color: #193b63 !important;
    font-size: 2.9rem !important;
    line-height: 1.2 !important;
    margin: 0 0 12px 0;
}

.hero p {
    color: #526981;
    font-size: 17px;
    max-width: 760px;
    line-height: 1.8;
}

.section-heading {
    font-size: 21px;
    font-weight: 800;
    color: #193b63;
    margin: 27px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 9px;
}

.section-heading .material-symbols-rounded {
    color: #078b91;
    font-size: 25px;
}

.stat-card {
    background: #ffffff;
    border: 1px solid #e0e8f1;
    border-radius: 15px;
    padding: 21px;
    min-height: 137px;
    box-shadow: 0 4px 14px rgba(40, 70, 110, 0.04);
}

.stat-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 15px;
}

.stat-icon {
    width: 43px;
    height: 43px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.stat-icon .material-symbols-rounded {
    font-size: 24px;
}

.stat-label {
    font-size: 14px;
    color: #63768b;
    font-weight: 700;
}

.stat-number {
    font-size: 34px;
    font-weight: 800;
    color: #193b63;
}

.content-card {
    background: #ffffff;
    border: 1px solid #e0e8f1;
    border-radius: 15px;
    padding: 23px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(40, 70, 110, 0.035);
}

.muted {
    color: #718298;
    font-size: 13px;
    line-height: 1.7;
}

.pill {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 750;
    background: #edf1f6;
    color: #435b76;
}

.status-pending {
    background: #fff3d9;
    color: #946000;
}

.status-progress {
    background: #e3efff;
    color: #205eae;
}

.status-resolved {
    background: #e1f5e9;
    color: #08764d;
}

.info-box {
    background: #eaf4fc;
    border: 1px solid #d5e8f8;
    border-radius: 12px;
    padding: 17px 20px;
    color: #315978;
    font-size: 15px;
    line-height: 1.8;
    margin: 15px 0 22px 0;
}

div.stButton > button,
.stFormSubmitButton > button {
    border-radius: 9px;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 0.7rem 1.1rem;
    min-height: 43px;
    border: 1px solid #d5e0eb;
    transition: 0.2s ease;
}

div.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: #087f91;
    color: white;
    border: 1px solid #087f91;
}

div.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    background: #066a79;
    border-color: #066a79;
    color: white;
}

.stTextInput input,
.stTextArea textarea {
    font-size: 15px !important;
    border-radius: 9px !important;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 9px;
    font-size: 15px;
}

.stMetric label {
    font-size: 14px !important;
}

.stMetric [data-testid="stMetricValue"] {
    font-size: 28px !important;
    color: #087f8c;
    font-weight: 800;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

hr {
    border-color: #e3eaf2;
}

footer {
    visibility: hidden;
}

@media (max-width: 768px) {
    .hero {
        padding: 24px;
    }

    .hero h1 {
        font-size: 2.2rem !important;
    }

    h1 {
        font-size: 2rem !important;
    }

    .section-heading {
        font-size: 19px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# ICON AND HELPER FUNCTIONS
# ==================================================

def mi(name, size=22, color="#078b91"):
    return (
        f'<span class="material-symbols-rounded" '
        f'style="font-size:{size}px;color:{color};">'
        f'{name}</span>'
    )


def section_heading(title, icon_name):
    st.markdown(
        f'<div class="section-heading">{mi(icon_name, 25)}'
        f'<span>{html.escape(title)}</span></div>',
        unsafe_allow_html=True
    )


def safe_text(value, default="Not available"):
    if value is None or str(value).strip() == "":
        value = default
    return html.escape(str(value))


def clean_complaint(value):
    if value is None:
        return "No complaint text"

    text = str(value)

    text = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    paragraph = re.search(
        r"<p\b[^>]*>(.*?)</p>",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if paragraph:
        text = paragraph.group(1)

    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]*>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else "No complaint text"


def get_value(record, *keys, default=""):
    if record is None:
        return default

    for key in keys:
        try:
            if key in record.keys():
                value = record[key]
            else:
                continue
        except (AttributeError, TypeError):
            try:
                value = record[key]
            except (KeyError, TypeError, IndexError):
                continue

        if value is not None:
            return value

    return default


def load_reports():
    reports = get_all_complaints()
    return reports if reports is not None else []


def status_class(status):
    status = str(status).lower()

    if status == "resolved":
        return "status-resolved"

    if status == "in progress":
        return "status-progress"

    return "status-pending"


def show_stat_card(label, number, icon_name, color):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-top">
            <div class="stat-label">{html.escape(str(label))}</div>
            <div class="stat-icon" style="background:{color}">
                {mi(icon_name, 23)}
            </div>
        </div>
        <div class="stat-number">{number}</div>
    </div>
    """, unsafe_allow_html=True)


def show_complaint_card(report):
    complaint_id = get_value(
        report, "complaint_id", "id", default="—"
    )

    complaint = clean_complaint(
        get_value(report, "complaint", "text", default="No complaint text")
    )

    category = get_value(report, "category", default="Other")
    department = get_value(report, "department", default="Not assigned")
    priority = get_value(report, "priority", default="Normal")
    status = get_value(report, "status", default="Pending")
    created = get_value(report, "created_at", default="")

    with st.container(border=True):
        left, right = st.columns([3, 1])

        with left:
            st.caption("TRACKING ID")
            st.markdown(
                f"**<span style='color:#087f8c'>"
                f"{safe_text(complaint_id)}</span>**",
                unsafe_allow_html=True
            )

        with right:
            st.markdown(
                f'<span class="pill {status_class(status)}">'
                f'{safe_text(status)}</span>',
                unsafe_allow_html=True
            )

        st.write(complaint)

        st.caption(
            f"Category: {category} | Department: {department} "
            f"| Priority: {priority}"
        )

        st.caption(
            f"Submitted: {created if created else 'Date not available'}"
        )


def show_page_title(title, subtitle):
    st.title(title)

    st.markdown(
        f'<p style="font-size:16px;color:#718298;margin-top:-10px;">'
        f'{safe_text(subtitle)}</p>',
        unsafe_allow_html=True
    )


def save_new_complaint(analysis):
    try:
        return save_complaint(analysis)
    except Exception as error:
        st.error(f"Could not save the complaint: {error}")
        return None


# ==================================================
# SIDEBAR - NATIVE STREAMLIT COMPONENTS
# ==================================================

with st.sidebar:

    if st.session_state.pop("go_to_submit", False):
        st.session_state["main_navigation"] = "Submit a Complaint"

    st.markdown("## CivicConnect AI")
    st.caption("YOUR COMMUNITY. YOUR VOICE.")

    st.divider()

    st.markdown("### MAIN MENU")

    pages = [
        "Overview",
        "AI Grievance Chatbot",
        "Submit a Complaint",
        "Track Complaint",
        "All Reports",
        "Community Analytics",
        "About"
    ]

    page = st.radio(
        "Navigation",
        pages,
        label_visibility="collapsed",
        key="main_navigation"
    )

    st.divider()

    st.markdown("**NEURAL STORM 2026**")
    st.caption("Team: Jinsa")
    st.caption("Problem: HI-02")
    st.caption("Human Impact Track")


# ==================================================
# LOAD REPORTS
# ==================================================

try:
    reports = load_reports()
except Exception as error:
    st.error(f"Unable to load reports: {error}")
    reports = []

total_reports = len(reports)

pending_reports = sum(
    1 for r in reports
    if str(get_value(r, "status", default="Pending")).lower() == "pending"
)

progress_reports = sum(
    1 for r in reports
    if str(get_value(r, "status", default="Pending")).lower() == "in progress"
)

resolved_reports = sum(
    1 for r in reports
    if str(get_value(r, "status", default="Pending")).lower() == "resolved"
)


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">NEURAL STORM 2026 | TEAM JINSA</div>
        <h1>Your Community,<br>Your Voice.</h1>
        <p>
            Have a problem in your neighbourhood?
            Tell us about it. CivicConnect AI helps you report local issues,
            find the right department and follow your complaint.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_heading("Community at a Glance", "monitoring")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_stat_card("Total Reports", total_reports, "description", "#e5f1ff")

    with c2:
        show_stat_card("Waiting for Action", pending_reports, "schedule", "#fff2dc")

    with c3:
        show_stat_card("Being Handled", progress_reports, "engineering", "#e9eaff")

    with c4:
        show_stat_card("Completed", resolved_reports, "task_alt", "#e2f5e9")

    section_heading("How Does It Work?", "account_tree")

    steps = st.columns(5)

    step_data = [
        ("record_voice_over", "STEP 1", "Report a Problem", "Describe the issue."),
        ("psychology", "STEP 2", "AI Analysis", "AI studies your complaint."),
        ("account_balance", "STEP 3", "Find Department", "Get a suggested department."),
        ("confirmation_number", "STEP 4", "Get Your ID", "Receive a tracking number."),
        ("track_changes", "STEP 5", "Track Progress", "Check the latest status.")
    ]

    for col, item in zip(steps, step_data):
        with col:
            st.markdown(f"""
            <div class="content-card" style="height:195px;text-align:center;">
                <div style="margin-top:5px;">{mi(item[0], 34)}</div>
                <div style="font-size:11px;color:#078b91;
                            font-weight:800;margin-top:13px;letter-spacing:1px;">
                    {item[1]}
                </div>
                <div style="font-size:15px;font-weight:800;
                            color:#193b63;margin-top:8px;">
                    {item[2]}
                </div>
                <div class="muted" style="margin-top:7px;">
                    {item[3]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:
        section_heading("Recent Reports", "history")

        if reports:
            for report in reports[:4]:
                show_complaint_card(report)
        else:
            st.markdown("""
            <div class="content-card">
                <h3>Welcome to CivicConnect AI</h3>
                <p>No complaints have been submitted yet.</p>
                <p class="muted">New community reports will appear here.</p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        section_heading("Your Voice Matters", "volunteer_activism")

        st.markdown(f"""
        <div class="content-card">
            <div>{mi("diversity_3", 38)}</div>
            <h3>Let's improve our community.</h3>
            <p>Report problems such as:</p>
            <p>
                {mi("water_drop", 19)} Water shortages<br>
                {mi("add_road", 19)} Damaged roads<br>
                {mi("bolt", 19)} Electricity problems<br>
                {mi("delete", 19)} Waste collection<br>
                {mi("health_and_safety", 19)} Public safety issues
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Report a Problem",
            type="primary",
            use_container_width=True
        ):
            st.session_state["go_to_submit"] = True
            st.rerun()


# ==================================================
# PAGE 2: AI GRIEVANCE CHATBOT
# ==================================================

elif page == "AI Grievance Chatbot":

    show_page_title(
        "AI Grievance Assistant",
        "Describe a problem. The AI will suggest its category and department."
    )

    st.markdown("""
    <div class="info-box">
        <b>How to use:</b> Type your problem in the box below.
        Select Analyse to see what the AI predicts.
        If the result looks correct, submit it to receive a tracking ID.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Describe Your Problem")

    chatbot_text = st.text_area(
        "Complaint description",
        placeholder="Example: There has been no water supply in my street for three days...",
        height=170,
        key="chatbot_complaint"
    )

    if st.button("Analyse My Complaint", type="primary"):

        if len(chatbot_text.strip()) < 8:
            st.warning("Please describe your problem in a little more detail.")
        else:
            try:
                st.session_state["chatbot_analysis"] = analyze_complaint(
                    chatbot_text.strip()
                )

                st.session_state["chatbot_analysis"]["complaint"] = (
                    chatbot_text.strip()
                )

                st.session_state["chatbot_saved"] = False

            except Exception as error:
                st.error(f"Analysis failed: {error}")

    analysis = st.session_state.get("chatbot_analysis")

    if analysis:

        section_heading("What the AI Found", "analytics")

        a1, a2, a3 = st.columns(3)

        with a1:
            st.metric(
                "Language",
                get_value(analysis, "language", default="Unknown")
            )

        with a2:
            st.metric(
                "Problem Type",
                get_value(analysis, "category", default="Other")
            )

        with a3:
            st.metric(
                "Suggested Department",
                get_value(analysis, "department", default="Not assigned")
            )

        st.markdown("""
        <div class="content-card">
            <div class="muted">PRIORITY LEVEL</div>
        """, unsafe_allow_html=True)

        st.subheader(
            get_value(analysis, "priority", default="Normal")
        )

        st.caption("ORIGINAL COMPLAINT")

        st.write(
            clean_complaint(
                get_value(analysis, "complaint", default=chatbot_text)
            )
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if not st.session_state.get("chatbot_saved", False):

            if st.button(
                "Submit Complaint & Get Tracking ID",
                type="primary"
            ):

                new_id = save_new_complaint(analysis)

                if new_id:
                    st.session_state["last_complaint_id"] = new_id
                    st.session_state["chatbot_saved"] = True

                    st.success(
                        f"Your complaint has been submitted! Tracking ID: {new_id}"
                    )

                    st.rerun()

        else:
            st.success(
                f"Your complaint is saved! Tracking ID: "
                f"{st.session_state.get('last_complaint_id', '')}"
            )

    st.caption(
        "Prototype note: AI predictions use a demonstration dataset and "
        "should be checked by a person."
    )


# ==================================================
# PAGE 3: SUBMIT A COMPLAINT
# ==================================================

elif page == "Submit a Complaint":

    show_page_title(
        "Report a Community Problem",
        "Share the issue you are facing. Your report will be saved with a tracking ID."
    )

    st.markdown("""
    <div class="info-box">
        <b>Please be clear:</b> Explain what happened and where it happened.
        This helps the AI suggest the correct problem category.
    </div>
    """, unsafe_allow_html=True)

    with st.form("submit_complaint_form", clear_on_submit=True):

        complaint_text = st.text_area(
            "What is the problem? *",
            placeholder="Example: The road near our school has many potholes...",
            height=170
        )

        location = st.text_input(
            "Where did it happen?",
            placeholder="Example: Main Road, Ward 5"
        )

        submitted = st.form_submit_button(
            "Submit My Complaint",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if len(complaint_text.strip()) < 8:
            st.warning("Please explain your problem before submitting.")

        else:
            try:
                analysis = analyze_complaint(complaint_text.strip())
                analysis["complaint"] = complaint_text.strip()

                new_id = save_new_complaint(analysis)

                if new_id:

                    st.success("Your complaint has been submitted successfully!")

                    st.markdown(f"""
                    <div class="content-card">
                        <div class="muted">YOUR TRACKING ID</div>
                        <h1 style="color:#087f8c !important;">
                            {safe_text(new_id)}
                        </h1>
                        <p>
                            Keep this number safe. You will need it to check your complaint.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    section_heading("AI Analysis Result", "analytics")

                    x1, x2, x3 = st.columns(3)

                    x1.metric(
                        "Language",
                        get_value(analysis, "language", default="Unknown")
                    )

                    x2.metric(
                        "Problem Type",
                        get_value(analysis, "category", default="Other")
                    )

                    x3.metric(
                        "Department",
                        get_value(analysis, "department", default="Not assigned")
                    )

                    if location.strip():
                        st.info(
                            "Your location was entered, but the current database "
                            "does not save location details."
                        )

            except Exception as error:
                st.error(f"Something went wrong: {error}")


# ==================================================
# PAGE 4: TRACK COMPLAINT
# ==================================================

elif page == "Track Complaint":

    show_page_title(
        "Track Your Complaint",
        "Enter your tracking ID to see what is happening with your report."
    )

    st.markdown("""
    <div class="info-box">
        <b>Where can I find my ID?</b> You received a tracking ID
        after submitting your complaint. Enter that ID below.
    </div>
    """, unsafe_allow_html=True)

    tracking_id = st.text_input(
        "Enter your Tracking ID",
        placeholder="Example: CCA-12345678"
    )

    if st.button("Check My Complaint", type="primary"):

        if not tracking_id.strip():
            st.warning("Please enter your tracking ID.")

        else:
            try:
                result = get_complaint(tracking_id.strip())

                if result is not None:
                    try:
                        result = dict(result)
                    except (TypeError, ValueError):
                        pass

                st.session_state["tracked_complaint"] = result

            except Exception as error:
                st.error(f"Unable to track this complaint: {error}")

    result = st.session_state.get("tracked_complaint")

    if result:

        result_id = get_value(
            result, "complaint_id", "id", default=""
        )

        if not result_id:
            st.warning("We could not find that complaint. Please check your ID.")

        else:
            current_status = get_value(result, "status", default="Pending")

            complaint = clean_complaint(
                get_value(result, "complaint", default="No description")
            )

            category = get_value(result, "category", default="Other")
            department = get_value(result, "department", default="Not assigned")
            priority = get_value(result, "priority", default="Normal")
            created = get_value(result, "created_at", default="")

            section_heading("Your Complaint Details", "description")

            st.markdown("""
            <div class="content-card">
            """, unsafe_allow_html=True)

            st.caption("TRACKING ID")
            st.subheader(result_id)

            st.write(complaint)

            st.markdown("---")
            st.write(f"**Problem type:** {category}")
            st.write(f"**Suggested department:** {department}")
            st.write(f"**Priority:** {priority}")
            st.write(f"**Submitted:** {created}")

            st.markdown("</div>", unsafe_allow_html=True)

            section_heading("Complaint Progress", "route")

            stages = ["Pending", "In Progress", "Resolved"]

            stage_index = next(
                (
                    index for index, stage in enumerate(stages)
                    if stage.lower() == str(current_status).lower()
                ),
                0
            )

            progress_cols = st.columns(3)

            for index, (col, stage) in enumerate(zip(progress_cols, stages)):

                with col:

                    if index < stage_index:
                        symbol = "check"
                        color = "#07845e"
                        background = "#def6e9"

                    elif index == stage_index:
                        symbol = "circle"
                        color = "#087fbd"
                        background = "#e0efff"

                    else:
                        symbol = "radio_button_unchecked"
                        color = "#9aabba"
                        background = "#edf1f6"

                    st.markdown(f"""
                    <div style="text-align:center;padding:18px 5px;">
                        <div style="
                            margin:auto;width:52px;height:52px;
                            border-radius:50%;background:{background};
                            display:flex;align-items:center;justify-content:center;">
                            {mi(symbol, 27, color)}
                        </div>
                        <div style="font-size:15px;font-weight:750;
                                    color:#263f5d;margin-top:12px;">
                            {stage}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.progress((stage_index + 1) / 3)

            st.markdown(
                f"**Current status:** "
                f'<span class="pill {status_class(current_status)}">'
                f'{safe_text(current_status)}</span>',
                unsafe_allow_html=True
            )


# ==================================================
# PAGE 5: ALL REPORTS
# ==================================================

elif page == "All Reports":

    show_page_title(
        "All Community Reports",
        "See the problems reported by citizens and update their progress."
    )

    if reports:

        search_text = st.text_input(
            "Search reports",
            placeholder="Search by problem, tracking ID or category..."
        )

        filtered_reports = []

        for report in reports:

            searchable = " ".join([
                str(get_value(report, "complaint_id", "id", default="")),
                clean_complaint(get_value(report, "complaint", default="")),
                str(get_value(report, "category", default="")),
                str(get_value(report, "department", default=""))
            ]).lower()

            if search_text.lower() in searchable:
                filtered_reports.append(report)

        st.markdown(f"### Showing {len(filtered_reports)} report(s)")

        for report in filtered_reports:
            show_complaint_card(report)

        st.markdown("---")

        section_heading("Update a Complaint", "edit_note")

        report_ids = [
            str(get_value(r, "complaint_id", "id", default=""))
            for r in reports
        ]

        report_ids = [r for r in report_ids if r]

        if report_ids:

            selected_id = st.selectbox(
                "Choose a tracking ID",
                report_ids
            )

            new_status = st.selectbox(
                "Choose the new status",
                ["Pending", "In Progress", "Resolved"]
            )

            if st.button("Save Status Update", type="primary"):

                try:
                    update_status(selected_id, new_status)
                    st.success("Complaint status updated successfully!")
                    st.rerun()

                except Exception as error:
                    st.error(f"Could not update status: {error}")

        st.caption(
            "Demo feature: Status updates are made manually in this prototype. "
            "No government department is notified."
        )

    else:

        st.markdown("""
        <div class="content-card">
            <h3>No reports yet</h3>
            <p>When someone submits a complaint, it will appear here.</p>
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# PAGE 6: COMMUNITY ANALYTICS
# ==================================================

elif page == "Community Analytics":

    show_page_title(
        "Community Analytics",
        "See which problems are reported most often in your community."
    )

    if reports:

        data = []

        for report in reports:

            data.append({
                "Problem Type": get_value(report, "category", default="Other"),
                "Status": get_value(report, "status", default="Pending"),
                "Priority": get_value(report, "priority", default="Normal"),
                "Department": get_value(report, "department", default="Not assigned")
            })

        df = pd.DataFrame(data)

        for column in ["Problem Type", "Status", "Priority", "Department"]:

            df[column] = (
                df[column]
                .fillna("Not available")
                .astype(str)
                .str.strip()
                .replace("", "Not available")
            )

        df["Status"] = df["Status"].str.title()

        m1, m2, m3 = st.columns(3)

        m1.metric("Total Reports", total_reports)

        m2.metric(
            "Waiting / In Progress",
            pending_reports + progress_reports
        )

        m3.metric("Completed Reports", resolved_reports)

        def show_pie_chart(counts, title, figsize=(7, 5)):

            counts = counts[counts > 0]

            if counts.empty:
                st.info("No data available for this chart.")
                return

            fig, ax = plt.subplots(figsize=figsize)

            chart_colors = [
                "#159cbd",
                "#17a58d",
                "#8c80dc",
                "#f4b942",
                "#ef8a75",
                "#75a7e8"
            ]

            wedges, texts, autotexts = ax.pie(
                counts.values,
                labels=None,
                autopct=lambda percentage: (
                    f"{percentage:.1f}%" if percentage >= 4 else ""
                ),
                startangle=90,
                counterclock=False,
                colors=chart_colors[:len(counts)]
                if len(counts) <= len(chart_colors) else None,
                pctdistance=0.70,
                wedgeprops={
                    "edgecolor": "white",
                    "linewidth": 2
                },
                textprops={
                    "fontsize": 11,
                    "fontweight": "bold"
                }
            )

            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(10)

            ax.set_title(
                title,
                fontsize=15,
                fontweight="bold",
                color="#193b63",
                pad=18
            )

            legend_labels = [
                f"{label} ({count})"
                for label, count in counts.items()
            ]

            ax.legend(
                wedges,
                legend_labels,
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=9,
                title_fontsize=10,
                frameon=False
            )

            ax.axis("equal")
            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        section_heading("Reports by Problem Type", "category")

        category_counts = df["Problem Type"].value_counts()

        show_pie_chart(
            category_counts,
            "Community Problem Distribution",
            figsize=(9, 5)
        )

        left, right = st.columns(2)

        with left:

            section_heading("Complaint Status", "pending_actions")

            status_counts = df["Status"].value_counts()

            show_pie_chart(
                status_counts,
                "Complaint Status Distribution",
                figsize=(6, 5)
            )

        with right:

            section_heading("Suggested Departments", "account_balance")

            department_counts = df["Department"].value_counts()

            show_pie_chart(
                department_counts,
                "Department Distribution",
                figsize=(6, 5)
            )

        section_heading("Report Information", "table_view")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.markdown("""
        <div class="content-card">
            <h3>Analytics will appear here</h3>
            <p>
                Submit some complaints first. Then you can see charts
                showing the most common problems and their status.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# PAGE 7: ABOUT
# ==================================================

elif page == "About":

    show_page_title(
        "About CivicConnect AI",
        "A simple way for citizens to make their voices heard."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">NEURAL STORM 2026 | HI-02</div>
        <h1>Connecting People<br>With Solutions.</h1>
        <p>
            CivicConnect AI is a citizen grievance chatbot prototype.
            It helps people describe community problems and suggests
            the department that may be responsible.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_heading("What Are We Trying to Do?", "flag")

    st.write(
        "Our goal is to make it easier for people to report everyday "
        "community problems such as water shortages, damaged roads, "
        "electricity issues, waste collection and public safety concerns."
    )

    section_heading("Technology Used", "code")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.markdown(f"""
        <div class="content-card">
            {mi("code", 30)}
            <h3>Python</h3>
            <p>Used to build the main application and process complaints.</p>
        </div>
        """, unsafe_allow_html=True)

    with tech2:

        st.markdown(f"""
        <div class="content-card">
            {mi("psychology", 30)}
            <h3>AI & NLP</h3>
            <p>Used to detect language and predict the problem category.</p>
        </div>
        """, unsafe_allow_html=True)

    with tech3:

        st.markdown(f"""
        <div class="content-card">
            {mi("database", 30)}
            <h3>SQLite</h3>
            <p>Used to save complaints and their tracking status.</p>
        </div>
        """, unsafe_allow_html=True)

    section_heading("Project Information", "info")

    st.markdown("""
    <div class="content-card">
        <p><b>Project:</b> CivicConnect AI</p>
        <p><b>Team:</b> Jinsa</p>
        <p><b>Hackathon:</b> Neural Storm 2026</p>
        <p><b>Problem:</b> HI-02 – Multilingual AI-Powered Citizen Grievance Chatbot</p>
        <p><b>Track:</b> Human Impact</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "CivicConnect AI is a hackathon prototype, not an official government service. "
        "AI predictions are demonstrations and are not connected to real government departments."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center;padding:15px 0;">
    <div style="font-size:21px;font-weight:800;color:#087f91;">
        CivicConnect AI
    </div>
    <div style="font-size:13px;color:#7188a2;margin-top:9px;">
        Your Community. Your Voice. | Neural Storm 2026 | Team Jinsa
    </div>
</div>
""", unsafe_allow_html=True)
