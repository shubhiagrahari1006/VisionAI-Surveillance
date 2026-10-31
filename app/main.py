"""
VisionAI - Intelligent Computer Vision & Traffic Analytics Platform
Unified Entrypoint with Theme Engine
"""

from pathlib import Path
import sys

# -------------------------------------------------------------
# Windows Asyncio ProactorEventLoop WinError 10054 Fix
# Silences ConnectionResetError when browser refreshes or closes tab
# -------------------------------------------------------------
if sys.platform == "win32":
    try:
        import asyncio.proactor_events

        _orig_call_connection_lost = (
            asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost
        )

        def _silent_call_connection_lost(self, exc):
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                # Socket was abruptly closed by remote browser (refresh/tab close); safe to ignore
                try:
                    if self._sock is not None:
                        self._sock.close()
                        self._sock = None
                except Exception:
                    pass
                self._called_connection_lost = True

        asyncio.proactor_events._ProactorBasePipeTransport._call_connection_lost = (
            _silent_call_connection_lost
        )
    except Exception:
        pass

# Ensure current directory is in path for imports
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from dashboard import render_dashboard_page
from detection_view import render_detection_page
from ui_theme import apply_custom_theme, render_theme_selector

# Page configuration
st.set_page_config(
    page_title="VisionAI | Surveillance & Analytics",
    page_icon=":material/visibility:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar: Theme Selector (Light, Dark, System)
render_theme_selector()

# Inject Dynamic Styling
apply_custom_theme()

# Core navigation pages
pages = [
    st.Page(
        render_detection_page,
        title="Detection Hub",
        icon=":material/radar:",
        default=True,
    ),
    st.Page(
        render_dashboard_page,
        title="Analytics Dashboard",
        icon=":material/insights:",
    ),
]

navigation = st.navigation(pages)
navigation.run()
