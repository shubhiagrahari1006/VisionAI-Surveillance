"""
VisionAI - Theme Engine (Full Light & Dark Adaptive)
Supports:
- Light Mode: Pure White / Light Slate Sidebar & Canvas, Slate text, Electric Cyan & Mint highlights.
- Dark Mode: Deep Obsidian Sidebar & Canvas, Glowing Cyan accents.
- System Default: Automatically switches based on user OS/browser preferences.
- 100% Seamless synchronization with Streamlit's top-right 3-dots theme toggle.
"""

import streamlit as st


def get_active_theme():
    """Retrieve currently active theme choice."""
    choice = st.session_state.get("app_theme_mode", "Light")
    if "Dark" in choice:
        return "dark"
    elif "System" in choice:
        return "system"
    return "light"


def render_theme_selector():
    """Render a modern segmented theme switcher in the sidebar."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand-header">
            <div class="brand-logo-icon">V</div>
            <div>
                <div class="brand-title">VisionAI Suite</div>
                <div class="brand-status"><span class="status-indicator">●</span> Live Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme_choice = st.sidebar.radio(
        "Theme Preference",
        options=["Light", "Dark", "System"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="app_theme_mode",
    )
    st.sidebar.divider()
    return theme_choice


def apply_custom_theme():
    """Inject dynamic styling ensuring the sidebar & canvas fully adapt to Light and Dark."""
    theme = get_active_theme()

    light_vars = """
        --bg-canvas: #f8fafc;
        --sidebar-bg: #ffffff;
        --sidebar-border: #e2e8f0;
        --sidebar-title: #0f172a;
        --sidebar-text: #1e293b;
        --sidebar-muted: #64748b;
        --sidebar-nav-hover: #f1f5f9;
        --sidebar-nav-active: #e0f2fe;
        --sidebar-nav-active-text: #0284c7;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --card-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
        --text-title: #0f172a;
        --text-body: #334155;
        --text-muted: #64748b;
        --cyan-primary: #00c0fa;
        --cyan-glow: rgba(0, 192, 250, 0.25);
        --mint-primary: #10b981;
        --input-bg: #ffffff;
        --input-border: #cbd5e1;
        --tag-bg: rgba(0, 192, 250, 0.12);
        --tag-text: #0284c7;
        --tag-border: rgba(0, 192, 250, 0.3);
    """

    dark_vars = """
        --bg-canvas: #0b0f19;
        --sidebar-bg: #111524;
        --sidebar-border: #1e263d;
        --sidebar-title: #ffffff;
        --sidebar-text: #f1f5f9;
        --sidebar-muted: #94a3b8;
        --sidebar-nav-hover: #1a2035;
        --sidebar-nav-active: #162a45;
        --sidebar-nav-active-text: #00e5ff;
        --card-bg: #141b2d;
        --card-border: #1f2a44;
        --card-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        --text-title: #ffffff;
        --text-body: #e2e8f0;
        --text-muted: #94a3b8;
        --cyan-primary: #00e5ff;
        --cyan-glow: rgba(0, 229, 255, 0.35);
        --mint-primary: #34d399;
        --input-bg: #141b2d;
        --input-border: #273553;
        --tag-bg: rgba(0, 229, 255, 0.15);
        --tag-text: #00e5ff;
        --tag-border: rgba(0, 229, 255, 0.35);
    """

    if theme == "dark":
        active_root_vars = dark_vars
    elif theme == "light":
        active_root_vars = light_vars
    else:
        active_root_vars = light_vars

    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {{
            {active_root_vars}
        }}

        @media (prefers-color-scheme: dark) {{
            :root:not([data-theme="light"]) {{
                {dark_vars}
            }}
        }}

        /* 100% Synchronized Theme Selectors */
        [data-theme="dark"],
        html[data-theme="dark"],
        body[data-theme="dark"],
        .stApp[data-theme="dark"],
        [data-testid="stAppViewContainer"][data-theme="dark"] {{
            {dark_vars}
        }}

        [data-theme="light"],
        html[data-theme="light"],
        body[data-theme="light"],
        .stApp[data-theme="light"],
        [data-testid="stAppViewContainer"][data-theme="light"] {{
            {light_vars}
        }}

        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        }}

        /* Completely Remove Streamlit Deploy Button */
        [data-testid="stAppDeployButton"],
        .stAppDeployButton,
        .stDeployButton {{
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
        }}

        /* Main app background */
        .stApp {{
            background-color: var(--bg-canvas) !important;
            color: var(--text-body) !important;
            transition: background-color 0.25s ease, color 0.25s ease;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        /* =========================================================
           SIDEBAR STYLING (Fully Light / Dark Adaptive)
           ========================================================= */
        [data-testid="stSidebar"] {{
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--sidebar-border) !important;
            transition: background-color 0.25s ease, border-color 0.25s ease;
        }}

        /* Sidebar Headings and Labels */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p {{
            color: var(--sidebar-text) !important;
        }}

        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] .stCaption p {{
            color: var(--sidebar-muted) !important;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: var(--sidebar-border) !important;
        }}

        /* Sidebar Brand Header */
        .sidebar-brand-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 12px;
            padding: 4px 0;
        }}

        .brand-logo-icon {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(135deg, #00c0fa, #0077ff);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white !important;
            font-size: 16px;
            font-weight: 700;
            box-shadow: 0 0 12px rgba(0, 192, 250, 0.45);
        }}

        .brand-title {{
            font-weight: 700;
            font-size: 14px;
            color: var(--sidebar-title) !important;
            line-height: 1.2;
        }}

        .brand-status {{
            font-size: 11px;
            color: var(--sidebar-muted) !important;
        }}

        .status-indicator {{
            color: var(--mint-primary) !important;
        }}

        /* Sidebar Navigation Links - Prominent, Bold & Highlighted */
        [data-testid="stSidebarNav"] {{
            padding-top: 6px !important;
            padding-bottom: 6px !important;
        }}

        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{
            background: var(--card-bg) !important;
            color: var(--sidebar-title) !important;
            border-radius: 12px !important;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
            font-weight: 700 !important;
            font-size: 16.5px !important;
            padding: 13px 18px !important;
            margin-bottom: 8px !important;
            letter-spacing: -0.3px !important;
            display: flex !important;
            align-items: center !important;
            border: 1px solid var(--sidebar-border) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        }}

        [data-testid="stSidebarNav"] a span,
        [data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span {{
            font-size: 16.5px !important;
            font-weight: 750 !important;
            color: var(--sidebar-title) !important;
        }}

        [data-testid="stSidebarNav"] a:hover,
        [data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {{
            background-color: var(--sidebar-nav-hover) !important;
            border-color: var(--cyan-primary) !important;
            transform: translateX(4px) !important;
            box-shadow: 0 4px 14px var(--cyan-glow) !important;
        }}

        [data-testid="stSidebarNav"] a[aria-current="page"],
        [data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{
            background: linear-gradient(135deg, rgba(0, 192, 250, 0.2) 0%, rgba(0, 119, 255, 0.14) 100%) !important;
            color: var(--cyan-primary) !important;
            font-weight: 800 !important;
            border: 1.5px solid var(--cyan-primary) !important;
            border-left: 5px solid var(--cyan-primary) !important;
            box-shadow: 0 4px 18px var(--cyan-glow) !important;
        }}

        [data-testid="stSidebarNav"] a[aria-current="page"] *,
        [data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] * {{
            color: var(--cyan-primary) !important;
            font-weight: 800 !important;
        }}

        /* Sidebar Inputs & Multiselect */
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] > div {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 10px !important;
            color: var(--sidebar-text) !important;
        }}

        /* Multiselect Selected Tags (Cyan Pill) */
        div[data-baseweb="tag"] {{
            background-color: var(--tag-bg) !important;
            border: 1px solid var(--tag-border) !important;
            color: var(--tag-text) !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
        }}

        div[data-baseweb="tag"] * {{
            color: var(--tag-text) !important;
            fill: var(--tag-text) !important;
        }}

        /* Sliders in sidebar */
        [data-testid="stSidebar"] .stSlider > div {{
            color: var(--sidebar-text) !important;
        }}

        /* Info box in sidebar */
        [data-testid="stSidebar"] [data-testid="stAlert"] {{
            background-color: var(--sidebar-nav-hover) !important;
            border: 1px solid var(--sidebar-border) !important;
            color: var(--sidebar-text) !important;
            border-radius: 10px !important;
        }}

        /* =========================================================
           MAIN CONTAINER & CARDS
           ========================================================= */
        .block-container {{
            max-width: 1560px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }}

        /* Top Header Bar */
        .top-navbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: var(--card-bg);
            border-radius: 16px;
            border: 1px solid var(--card-border);
            box-shadow: var(--card-shadow);
            margin-bottom: 24px;
            transition: background 0.25s ease, border-color 0.25s ease;
        }}

        .top-greeting {{
            font-size: 22px;
            font-weight: 800;
            color: var(--text-title);
            margin: 0;
            letter-spacing: -0.5px;
        }}

        .top-subtitle {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 3px;
        }}

        .badge-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            background: var(--tag-bg);
            color: var(--tag-text);
            border: 1px solid var(--tag-border);
            margin-left: 8px;
        }}

        /* Signature Floating Cards */
        .card-cyan {{
            background: var(--card-bg);
            border-radius: 14px;
            border: 1px solid var(--card-border);
            border-top: 4px solid var(--cyan-primary) !important;
            padding: 18px 20px;
            box-shadow: var(--card-shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.25s ease;
        }}

        .card-mint {{
            background: var(--card-bg);
            border-radius: 14px;
            border: 1px solid var(--card-border);
            border-top: 4px solid var(--mint-primary) !important;
            padding: 18px 20px;
            box-shadow: var(--card-shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.25s ease;
        }}

        .card-blue {{
            background: var(--card-bg);
            border-radius: 14px;
            border: 1px solid var(--card-border);
            border-top: 4px solid #3b82f6 !important;
            padding: 18px 20px;
            box-shadow: var(--card-shadow);
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.25s ease;
        }}

        .card-cyan:hover, .card-mint:hover, .card-blue:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(0, 192, 250, 0.12);
        }}

        /* Featured Summary Card (Solid Cyan Gradient) */
        .card-featured {{
            background: linear-gradient(135deg, #00c0fa 0%, #0077ff 100%) !important;
            border-radius: 14px;
            padding: 20px 22px;
            color: #ffffff !important;
            box-shadow: 0 8px 25px rgba(0, 192, 250, 0.35);
            transition: transform 0.2s ease;
        }}

        .card-featured:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(0, 192, 250, 0.45);
        }}

        .card-featured .metric-label {{
            color: rgba(255, 255, 255, 0.9) !important;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .card-featured .metric-value {{
            color: #ffffff !important;
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -1px;
            margin: 6px 0;
        }}

        .card-featured .metric-note {{
            color: rgba(255, 255, 255, 0.95) !important;
            font-size: 11.5px;
            font-weight: 500;
        }}

        /* Standard Metric Labels */
        .metric-label {{
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-value {{
            font-size: 30px;
            font-weight: 800;
            color: var(--text-title);
            letter-spacing: -1px;
            margin: 6px 0;
        }}

        .metric-note {{
            font-size: 11.5px;
            color: var(--text-muted);
        }}

        .custom-panel {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 22px;
            box-shadow: var(--card-shadow);
            margin-bottom: 20px;
            transition: background 0.25s ease, border-color 0.25s ease;
        }}

        /* =========================================================
           BUTTON STYLING (Flawless Functionality & Contrast)
           ========================================================= */
        .stMain .stButton > button,
        [data-testid="stSidebar"] .stButton > button {{
            border-radius: 10px !important;
            border: none !important;
            background: linear-gradient(135deg, #00c0fa 0%, #0077ff 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 13.5px !important;
            padding: 9px 20px !important;
            box-shadow: 0 4px 14px rgba(0, 192, 250, 0.25) !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
        }}

        .stMain .stButton > button:hover,
        [data-testid="stSidebar"] .stButton > button:hover {{
            background: linear-gradient(135deg, #00a8dc 0%, #0060d4 100%) !important;
            box-shadow: 0 6px 18px rgba(0, 192, 250, 0.4) !important;
            transform: translateY(-1px) !important;
            color: #ffffff !important;
        }}

        .stMain .stButton > button:active,
        [data-testid="stSidebar"] .stButton > button:active {{
            transform: translateY(1px) !important;
        }}

        .stMain .stButton > button:disabled,
        [data-testid="stSidebar"] .stButton > button:disabled {{
            opacity: 0.6 !important;
            cursor: not-allowed !important;
            transform: none !important;
        }}

        /* Download Buttons */
        div[data-testid="stDownloadButton"] button {{
            border-radius: 10px !important;
            border: none !important;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 13.5px !important;
            padding: 9px 20px !important;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }}

        div[data-testid="stDownloadButton"] button:hover {{
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            box-shadow: 0 6px 18px rgba(16, 185, 129, 0.4) !important;
            color: #ffffff !important;
            transform: translateY(-1px) !important;
        }}

        /* Main Canvas Select & Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            border-radius: 10px !important;
            color: var(--text-title) !important;
        }}

        /* Dataframe */
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--card-border) !important;
            border-radius: 12px !important;
            background-color: var(--card-bg) !important;
            overflow: hidden !important;
        }}

        /* Section Headings */
        .section-header {{
            font-size: 17px;
            font-weight: 800;
            color: var(--text-title);
            margin-top: 14px;
            margin-bottom: 3px;
            letter-spacing: -0.3px;
        }}

        .section-desc {{
            font-size: 12.5px;
            color: var(--text-muted);
            margin-bottom: 16px;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 500;
            padding: 24px 0 10px 0;
            border-top: 1px solid var(--card-border);
            margin-top: 36px;
        }}
        </style>
    """

    st.markdown(css, unsafe_allow_html=True)
