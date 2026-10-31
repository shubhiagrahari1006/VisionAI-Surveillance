"""
VisionAI - Detection & Traffic Analytics Dashboard
Image 3 Inspired UI: Electric Cyan & Mint highlights, floating cards, multi-theme support.
Telemetry features:
- Inbound vs Outbound directional analysis.
- Speed monitoring & Overspeeding violations breakdown.
- Temporal activity & classification distributions.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from reporting import generate_pdf_report
from ui_theme import apply_custom_theme, get_active_theme

PROJECT_DIR = Path(__file__).resolve().parent.parent
HISTORY_FILE = PROJECT_DIR / "data" / "detection_history.csv"


@st.cache_data(ttl=3)
def load_history(file_path, modified_time):
    """Load and preprocess detection history records from CSV."""
    del modified_time

    if not Path(file_path).exists():
        return pd.DataFrame()

    try:
        data = pd.read_csv(file_path, on_bad_lines="skip")
        if data.empty:
            return data

        for column in [
            "timestamp",
            "source",
            "object_class",
            "confidence",
            "tracking_id",
            "event",
            "speed_kmh",
            "direction",
        ]:
            if column not in data.columns:
                data[column] = None

        data["object_class"] = data["object_class"].fillna("unknown").astype(str)
        data["source"] = data["source"].fillna("unknown").astype(str)
        data["event"] = data["event"].fillna("detected").astype(str)
        data["direction"] = data["direction"].fillna("").astype(str)
        data["confidence"] = pd.to_numeric(data["confidence"], errors="coerce")
        data["speed_kmh"] = pd.to_numeric(data["speed_kmh"], errors="coerce")
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        return data
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def style_chart(fig, height=330):
    """Apply dynamic theme-aware styling to Plotly figures."""
    theme = get_active_theme()

    if theme == "dark":
        font_color = "#f1f5f9"
        grid_color = "#1f2a44"
        title_color = "#ffffff"
    else:
        font_color = "#334155"
        grid_color = "#e8ecf4"
        title_color = "#0f172a"

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Plus Jakarta Sans, -apple-system, sans-serif",
            color=font_color,
            size=11,
        ),
        margin=dict(l=15, r=15, t=42, b=15),
        title_font=dict(size=14, color=title_color, family="Plus Jakarta Sans"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=font_color),
        ),
        xaxis=dict(
            gridcolor=grid_color,
            linecolor=grid_color,
            zerolinecolor=grid_color,
        ),
        yaxis=dict(
            gridcolor=grid_color,
            linecolor=grid_color,
            zerolinecolor=grid_color,
        ),
    )
    return fig


def render_dashboard_page():
    """Render the full Analytics Dashboard page with Image 3 styling."""
    apply_custom_theme()

    modified_time = HISTORY_FILE.stat().st_mtime if HISTORY_FILE.exists() else 0
    df = load_history(str(HISTORY_FILE), modified_time)

    # --------------------------------------------------
    # TOP NAVBAR HEADER (Image 3 Aesthetic)
    # --------------------------------------------------
    st.markdown(
        """
        <div class="top-navbar">
            <div>
                <h2 class="top-greeting">VisionAI Analytics Dashboard</h2>
                <div class="top-subtitle">Real-time object telemetry, traffic volume, speed & directional flow</div>
            </div>
            <div>
                <span class="badge-pill">● Live Telemetry</span>
                <span class="badge-pill">YOLO11n</span>
                <span class="badge-pill">ByteTrack ITS</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # SIDEBAR CONTROLS & FILTERS
    # --------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-top: 6px; margin-bottom: 8px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00c0fa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                </svg>
                <span style="font-weight: 700; font-size: 14.5px; color: var(--sidebar-title);">Dashboard Controls</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Refresh Telemetry Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px; margin-top: 14px; margin-bottom: 8px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00c0fa" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
                </svg>
                <span style="font-weight: 700; font-size: 14.5px; color: var(--sidebar-title);">Data Filters</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if df.empty:
            st.info("No detection history found yet.")
            filtered = df.copy()
        else:
            sources = sorted(df["source"].dropna().unique().tolist())
            classes = sorted(df["object_class"].dropna().unique().tolist())

            selected_sources = st.multiselect(
                "Input Source",
                sources,
                default=sources,
            )
            selected_classes = st.multiselect(
                "Object Classes",
                classes,
                default=classes,
            )

            filtered = df[
                df["source"].isin(selected_sources)
                & df["object_class"].isin(selected_classes)
            ].copy()

            st.caption(f"Showing {len(filtered):,} of {len(df):,} records")

        st.divider()

    # --------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------
    if df.empty:
        st.markdown(
            """
            <div class="custom-panel" style="text-align: center; padding: 40px 20px;">
                <div style="margin-bottom: 14px;">
                    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#00c0fa" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                </div>
                <h3 style="margin-bottom: 6px;">No Telemetry Records Yet</h3>
                <p style="color: var(--text-muted); font-size: 13.5px; max-width: 520px; margin: 0 auto 20px auto;">
                    Run detection or video tracking from the <b>Detection Hub</b> to generate live telemetry. 
                    Your real-time stats and distributions will populate here automatically.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if filtered.empty:
        st.warning("No records match the active filter criteria. Please adjust your filters.")
        return

    # --------------------------------------------------
    # SIGNATURE IMAGE 3 METRIC CARDS
    # --------------------------------------------------
    total_records = len(filtered)
    unique_classes = filtered["object_class"].nunique()

    # Direction counts
    inbound_count = int(filtered["event"].str.contains("inbound", case=False, na=False).sum())
    outbound_count = int(filtered["event"].str.contains("outbound", case=False, na=False).sum())
    violations_count = int(filtered["event"].str.contains("overspeeding", case=False, na=False).sum())
    avg_conf = filtered["confidence"].mean()
    max_speed = filtered["speed_kmh"].dropna().max() if not filtered["speed_kmh"].dropna().empty else 0.0

    m_col1, m_col2, m_col3, m_col4 = st.columns(4, gap="medium")

    with m_col1:
        st.markdown(
            f"""
            <div class="card-featured">
                <div class="metric-label">Total Detections</div>
                <div class="metric-value">{total_records:,}</div>
                <div class="metric-note">↑ Logged telemetry events</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col2:
        st.markdown(
            f"""
            <div class="card-cyan">
                <div class="metric-label">Inbound (Entry)</div>
                <div class="metric-value">{inbound_count:,}</div>
                <div class="metric-note">↓ Campus entrance crossings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col3:
        st.markdown(
            f"""
            <div class="card-mint">
                <div class="metric-label">Outbound (Exit)</div>
                <div class="metric-value">{outbound_count:,}</div>
                <div class="metric-note">↑ Campus exit crossings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m_col4:
        st.markdown(
            f"""
            <div class="card-blue">
                <div class="metric-label">Speed Violations</div>
                <div class="metric-value" style="color: {'#ef4444' if violations_count > 0 else 'inherit'};">{violations_count:,}</div>
                <div class="metric-note">Peak speed: {max_speed:.1f} km/h</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------
    # DISTRIBUTION CHARTS
    # --------------------------------------------------
    st.markdown('<div class="section-header">Detection & Flow Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Class frequency distribution and source allocation</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1], gap="large")

    class_counts = (
        filtered["object_class"]
        .value_counts()
        .rename_axis("Object Class")
        .reset_index(name="Records")
    )

    with left:
        fig_class = px.bar(
            class_counts.head(10).sort_values("Records"),
            x="Records",
            y="Object Class",
            orientation="h",
            text="Records",
            color="Records",
            color_continuous_scale=["#00c0fa", "#0077ff"],
            title="Top Detected Object Classes",
        )
        fig_class.update_traces(textposition="outside", cliponaxis=False)
        fig_class.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Detections",
            yaxis_title="",
        )
        st.plotly_chart(style_chart(fig_class, 360), use_container_width=True)

    with right:
        source_counts = (
            filtered["source"]
            .value_counts()
            .rename_axis("Input Source")
            .reset_index(name="Records")
        )
        fig_source = px.pie(
            source_counts,
            names="Input Source",
            values="Records",
            hole=0.68,
            title="Records by Input Source",
            color_discrete_sequence=[
                "#00c0fa",
                "#10b981",
                "#3b82f6",
                "#8b5cf6",
                "#f59e0b",
            ],
        )
        fig_source.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate="%{label}<br>Records: %{value}<extra></extra>",
        )
        fig_source.add_annotation(
            text=f"<b>{total_records:,}</b><br><span style='font-size:11px'>Total</span>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=19),
        )
        st.plotly_chart(style_chart(fig_source, 360), use_container_width=True)

    # --------------------------------------------------
    # DIRECTIONAL FLOW & SPEED ANALYTICS
    # --------------------------------------------------
    st.markdown('<div class="section-header">Directional Flow & Speed Telemetry</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Analysis of Inbound vs Outbound traffic and velocity distribution</div>', unsafe_allow_html=True)

    dir_col, speed_col = st.columns(2, gap="large")

    with dir_col:
        dir_data = pd.DataFrame([
            {"Direction": "Inbound (Entry)", "Count": inbound_count},
            {"Direction": "Outbound (Exit)", "Count": outbound_count},
        ])
        fig_dir = px.bar(
            dir_data,
            x="Direction",
            y="Count",
            color="Direction",
            title="Campus Traffic Direction Split",
            color_discrete_map={
                "Inbound (Entry)": "#10b981",
                "Outbound (Exit)": "#00c0fa",
            },
        )
        fig_dir.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig_dir, 290), use_container_width=True)

    with speed_col:
        speed_records = filtered["speed_kmh"].dropna()
        speed_records = speed_records[speed_records > 0]

        if not speed_records.empty:
            fig_speed = px.histogram(
                speed_records,
                x=speed_records,
                nbins=12,
                title="Vehicle Speed Distribution (km/h)",
                color_discrete_sequence=["#00c0fa"],
            )
            fig_speed.update_layout(
                xaxis_title="Speed (km/h)",
                yaxis_title="Observed Frames",
            )
            st.plotly_chart(style_chart(fig_speed, 290), use_container_width=True)
        else:
            st.info("No numerical speed logs available in the current filter.")

    # --------------------------------------------------
    # TIME TREND (Image 3 Style Area Chart)
    # --------------------------------------------------
    st.markdown('<div class="section-header">Activity Trends Over Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Temporal volume of detection events across timestamps</div>', unsafe_allow_html=True)

    timed = filtered.dropna(subset=["timestamp"]).copy()

    if not timed.empty:
        timed["Time"] = timed["timestamp"].dt.floor("h")
        trend = (
            timed.groupby("Time")
            .size()
            .reset_index(name="Records")
            .sort_values("Time")
        )

        fig_trend = px.area(
            trend,
            x="Time",
            y="Records",
            title="Hourly Detection Flow",
            markers=True,
        )
        fig_trend.update_traces(
            line=dict(color="#00c0fa", width=3),
            fillcolor="rgba(0, 192, 250, 0.16)",
            marker=dict(size=6, color="#0077ff"),
        )
        fig_trend.update_layout(
            xaxis_title="Time of Day",
            yaxis_title="Detections Logged",
        )
        st.plotly_chart(style_chart(fig_trend, 290), use_container_width=True)
    else:
        st.info("Timestamp information not available in current records.")

    # --------------------------------------------------
    # DETECTION HISTORY TABLE
    # --------------------------------------------------
    st.markdown('<div class="section-header">Detection History Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Filter, search, and export detection telemetry</div>', unsafe_allow_html=True)

    search_col, class_col, limit_col = st.columns([2, 1, 1])

    with search_col:
        search_text = st.text_input(
            "Search Records",
            placeholder="Search class, source, event or direction...",
            label_visibility="collapsed",
        )

    with class_col:
        sort_order = st.selectbox(
            "Sort Order",
            ["Newest first", "Highest confidence", "Highest speed"],
            label_visibility="collapsed",
        )

    with limit_col:
        display_limit = st.selectbox(
            "Rows Limit",
            [10, 25, 50, 100],
            index=1,
            label_visibility="collapsed",
        )

    table_data = filtered.copy()

    if search_text.strip():
        search_mask = (
            table_data["object_class"].str.contains(search_text, case=False, na=False)
            | table_data["source"].str.contains(search_text, case=False, na=False)
            | table_data["event"].str.contains(search_text, case=False, na=False)
            | table_data["direction"].str.contains(search_text, case=False, na=False)
        )
        table_data = table_data[search_mask]

    if sort_order == "Newest first":
        table_data = table_data.sort_values(
            "timestamp", ascending=False, na_position="last"
        )
    elif sort_order == "Highest confidence":
        table_data = table_data.sort_values(
            "confidence", ascending=False, na_position="last"
        )
    else:
        table_data = table_data.sort_values(
            "speed_kmh", ascending=False, na_position="last"
        )

    display_data = table_data.head(display_limit).copy()
    display_data["confidence"] = display_data["confidence"].map(
        lambda val: f"{val:.1%}" if pd.notna(val) else "N/A"
    )
    display_data["speed_kmh"] = display_data["speed_kmh"].map(
        lambda val: f"{val:.1f} km/h" if pd.notna(val) and val > 0 else "-"
    )
    display_data["timestamp"] = display_data["timestamp"].map(
        lambda val: val.strftime("%d %b %Y, %H:%M:%S") if pd.notna(val) else "N/A"
    )
    display_data = display_data[
        [
            "timestamp",
            "source",
            "object_class",
            "confidence",
            "speed_kmh",
            "direction",
            "tracking_id",
            "event",
        ]
    ]
    display_data.columns = [
        "Timestamp",
        "Source",
        "Object Class",
        "Confidence",
        "Speed",
        "Direction",
        "Tracking ID",
        "Event",
    ]

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
        height=360,
    )

    st.write("")
    st.markdown('<div class="section-header">Export Telemetry & Compliance Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Generate official executive PDF audit summaries and download structured telemetry data</div>', unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns([1, 1], gap="medium")

    with exp_col1:
        if st.button("Generate Official PDF Audit Report", use_container_width=True, key="btn_gen_pdf"):
            with st.spinner("Rendering multi-page vector PDF report..."):
                pdf_bytes = generate_pdf_report(table_data, speed_limit=30.0)
                st.session_state["cached_pdf"] = pdf_bytes
                st.toast("Official PDF Report compiled successfully!")

        if "cached_pdf" in st.session_state:
            st.download_button(
                label="Download Official Audit Report (PDF)",
                data=st.session_state["cached_pdf"],
                file_name="VisionAI_Traffic_Audit_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="btn_download_pdf",
            )

    with exp_col2:
        csv_data = table_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Telemetry (CSV)",
            data=csv_data,
            file_name="visionai_telemetry.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_download_csv",
        )

    st.markdown(
        """
        <div class="footer">
            VisionAI · Real-time Intelligent Computer Vision & Surveillance
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="VisionAI | Detection Analytics",
        page_icon=":material/insights:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_dashboard_page()
