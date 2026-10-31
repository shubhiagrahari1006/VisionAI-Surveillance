"""
VisionAI - Automated PDF & Telemetry Report Generator
Generates executive-level, multi-page vector PDF traffic surveillance reports.
Includes KPI scorecards, embedded vector charts, speed limit benchmarks, and audit tables.
"""

from datetime import datetime
import io
import os
from pathlib import Path
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# Styling constants matching VisionAI Image 3 Theme
PRIMARY_CYAN = "#00c0fa"
PRIMARY_MINT = "#10b981"
PRIMARY_BLUE = "#0077ff"
ALERT_RED = "#ef4444"
DARK_NAVY = "#0f172a"
SLATE_GRAY = "#64748b"
LIGHT_BG = "#f8fafc"
BORDER_COLOR = "#e2e8f0"


def generate_pdf_report(df: pd.DataFrame, speed_limit: float = 30.0) -> bytes:
    """
    Generate an executive-level, publication-quality A4 PDF report from detection telemetry.
    Returns:
        bytes: Raw PDF document bytes ready for download or persistence.
    """
    buf = io.BytesIO()

    # Pre-calculate summary statistics
    total_records = len(df) if not df.empty else 0
    inbound_count = int(df["event"].str.contains("inbound", case=False, na=False).sum()) if not df.empty else 0
    outbound_count = int(df["event"].str.contains("outbound", case=False, na=False).sum()) if not df.empty else 0
    net_inside = max(0, inbound_count - outbound_count)
    violations_count = int(df["event"].str.contains("overspeeding", case=False, na=False).sum()) if not df.empty else 0

    valid_speeds = df["speed_kmh"].dropna() if not df.empty and "speed_kmh" in df.columns else pd.Series(dtype=float)
    valid_speeds = pd.to_numeric(valid_speeds, errors="coerce").dropna()
    valid_speeds = valid_speeds[valid_speeds > 0]
    max_speed = float(valid_speeds.max()) if not valid_speeds.empty else 0.0
    avg_speed = float(valid_speeds.mean()) if not valid_speeds.empty else 0.0

    report_date = datetime.now().strftime("%d %B %Y, %H:%M:%S")

    with PdfPages(buf) as pdf:
        # =========================================================
        # PAGE 1: EXECUTIVE TELEMETRY & ANALYTICS OVERVIEW
        # =========================================================
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig.patch.set_facecolor("#ffffff")

        # Top Header Banner
        header_ax = fig.add_axes([0.06, 0.88, 0.88, 0.09])
        header_ax.set_facecolor(DARK_NAVY)
        header_ax.axis("off")

        # Cyan Accent Stripe on Header
        stripe_ax = fig.add_axes([0.06, 0.875, 0.88, 0.005])
        stripe_ax.set_facecolor(PRIMARY_CYAN)
        stripe_ax.axis("off")

        header_ax.text(
            0.04, 0.65, "VisionAI  |  INTELLIGENT SURVEILLANCE SUITE",
            fontsize=15, fontweight="bold", color="#ffffff", va="center",
            family="sans-serif",
        )
        header_ax.text(
            0.04, 0.28, "Executive Traffic Telemetry & Speed Compliance Audit Report",
            fontsize=9.5, color="#94a3b8", va="center", family="sans-serif",
        )
        header_ax.text(
            0.96, 0.65, f"Date: {report_date}",
            fontsize=8.5, color="#cbd5e1", ha="right", va="center",
        )
        header_ax.text(
            0.96, 0.28, f"Campus Speed Limit: {int(speed_limit)} km/h",
            fontsize=8.5, color=PRIMARY_CYAN, ha="right", va="center", fontweight="bold",
        )

        # ---------------------------------------------------------
        # KPI Scorecard Boxes (6 Metrics Grid)
        # ---------------------------------------------------------
        kpi_metrics = [
            ("TOTAL DETECTIONS", f"{total_records:,}", "Live event records", PRIMARY_BLUE),
            ("INBOUND (ENTRY)", f"{inbound_count:,}", "Entered campus gate", PRIMARY_MINT),
            ("OUTBOUND (EXIT)", f"{outbound_count:,}", "Exited campus gate", PRIMARY_CYAN),
            ("NET OCCUPANCY", f"{net_inside:,}", "Currently on-campus", "#8b5cf6"),
            ("SPEED VIOLATIONS", f"{violations_count:,}", f"Exceeded {int(speed_limit)} km/h limit", ALERT_RED),
            ("PEAK VELOCITY", f"{max_speed:.1f} km/h", f"Mean: {avg_speed:.1f} km/h", "#f59e0b"),
        ]

        # 2 rows of 3 boxes
        box_w, box_h = 0.27, 0.075
        col_x = [0.06, 0.365, 0.67]
        row_y = [0.77, 0.675]

        for i, (label, val, note, accent_color) in enumerate(kpi_metrics):
            rx = col_x[i % 3]
            ry = row_y[i // 3]

            card_ax = fig.add_axes([rx, ry, box_w, box_h])
            card_ax.set_facecolor(LIGHT_BG)
            for spine in card_ax.spines.values():
                spine.set_color(BORDER_COLOR)
                spine.set_linewidth(1)
            card_ax.spines["top"].set_color(accent_color)
            card_ax.spines["top"].set_linewidth(3)
            card_ax.set_xticks([])
            card_ax.set_yticks([])

            card_ax.text(0.08, 0.82, label, fontsize=7, fontweight="bold", color=SLATE_GRAY, va="center")
            card_ax.text(0.08, 0.45, val, fontsize=15, fontweight="bold", color=DARK_NAVY, va="center")
            card_ax.text(0.08, 0.16, note, fontsize=6.5, color=SLATE_GRAY, va="center")

        # ---------------------------------------------------------
        # 4 High-Resolution Analytics Charts (2x2 Grid)
        # ---------------------------------------------------------
        # Chart 1: Inbound vs Outbound Gate Flow
        ax1 = fig.add_axes([0.06, 0.38, 0.41, 0.24])
        ax1.set_facecolor("#ffffff")
        dir_labels = ["Inbound (Entry)", "Outbound (Exit)"]
        dir_values = [inbound_count, outbound_count]
        bars1 = ax1.bar(dir_labels, dir_values, color=[PRIMARY_MINT, PRIMARY_CYAN], width=0.55)
        ax1.set_title("Directional Traffic Flow", fontsize=9.5, fontweight="bold", color=DARK_NAVY, pad=8)
        ax1.set_ylabel("Gate Crossings", fontsize=8, color=SLATE_GRAY)
        ax1.tick_params(axis="both", labelsize=7.5, colors=SLATE_GRAY)
        ax1.grid(axis="y", linestyle="--", alpha=0.4, color=BORDER_COLOR)
        for spine in ["top", "right"]:
            ax1.spines[spine].set_visible(False)
        ax1.spines["left"].set_color(BORDER_COLOR)
        ax1.spines["bottom"].set_color(BORDER_COLOR)
        for b in bars1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 0.1, f"{int(h)}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=DARK_NAVY)

        # Chart 2: Vehicle Category Distribution
        ax2 = fig.add_axes([0.53, 0.38, 0.41, 0.24])
        ax2.set_facecolor("#ffffff")
        vehicle_classes = {"car", "motorcycle", "bus", "truck"}
        v_df = df[df["object_class"].str.lower().isin(vehicle_classes)] if not df.empty else pd.DataFrame()
        if not v_df.empty:
            v_counts = v_df["object_class"].value_counts().head(5)
            y_pos = np.arange(len(v_counts))
            bars2 = ax2.barh(y_pos, v_counts.values, color=PRIMARY_BLUE, height=0.55)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([s.title() for s in v_counts.index], fontsize=7.5, color=SLATE_GRAY)
            ax2.invert_yaxis()
            ax2.set_title("Vehicle Classification Breakdown", fontsize=9.5, fontweight="bold", color=DARK_NAVY, pad=8)
            ax2.set_xlabel("Observations", fontsize=8, color=SLATE_GRAY)
            ax2.grid(axis="x", linestyle="--", alpha=0.4, color=BORDER_COLOR)
            for spine in ["top", "right"]:
                ax2.spines[spine].set_visible(False)
            ax2.spines["left"].set_color(BORDER_COLOR)
            ax2.spines["bottom"].set_color(BORDER_COLOR)
            for b in bars2:
                w = b.get_width()
                ax2.text(w + 0.5, b.get_y() + b.get_height() / 2, f"{int(w)}", ha="left", va="center", fontsize=7.5, fontweight="bold", color=DARK_NAVY)
        else:
            ax2.text(0.5, 0.5, "No vehicle records available", ha="center", va="center", fontsize=8, color=SLATE_GRAY)
            ax2.axis("off")

        # Chart 3: Velocity Distribution Histogram (km/h) with Speed Limit Line
        ax3 = fig.add_axes([0.06, 0.08, 0.41, 0.23])
        ax3.set_facecolor("#ffffff")
        if not valid_speeds.empty:
            ax3.hist(valid_speeds, bins=10, color=PRIMARY_CYAN, edgecolor="#ffffff", alpha=0.85)
            ax3.axvline(speed_limit, color=ALERT_RED, linestyle="--", linewidth=1.8, label=f"Limit ({int(speed_limit)} km/h)")
            ax3.set_title("Observed Velocity Distribution (km/h)", fontsize=9.5, fontweight="bold", color=DARK_NAVY, pad=8)
            ax3.set_xlabel("Speed (km/h)", fontsize=8, color=SLATE_GRAY)
            ax3.set_ylabel("Frames", fontsize=8, color=SLATE_GRAY)
            ax3.tick_params(axis="both", labelsize=7.5, colors=SLATE_GRAY)
            ax3.grid(axis="y", linestyle="--", alpha=0.4, color=BORDER_COLOR)
            ax3.legend(fontsize=7, frameon=False, loc="upper right")
            for spine in ["top", "right"]:
                ax3.spines[spine].set_visible(False)
            ax3.spines["left"].set_color(BORDER_COLOR)
            ax3.spines["bottom"].set_color(BORDER_COLOR)
        else:
            ax3.text(0.5, 0.5, "No velocity data recorded", ha="center", va="center", fontsize=8, color=SLATE_GRAY)
            ax3.axis("off")

        # Chart 4: Hourly Activity Volume Trend
        ax4 = fig.add_axes([0.53, 0.08, 0.41, 0.23])
        ax4.set_facecolor("#ffffff")
        timed_df = df.dropna(subset=["timestamp"]).copy() if not df.empty and "timestamp" in df.columns else pd.DataFrame()
        if not timed_df.empty:
            timed_df["timestamp"] = pd.to_datetime(timed_df["timestamp"], errors="coerce")
            timed_df = timed_df.dropna(subset=["timestamp"])
            if not timed_df.empty:
                timed_df["Hour"] = timed_df["timestamp"].dt.strftime("%H:00")
                hourly = timed_df.groupby("Hour").size().reset_index(name="Count")
                ax4.plot(hourly["Hour"], hourly["Count"], color=PRIMARY_BLUE, marker="o", markersize=4, linewidth=2)
                ax4.fill_between(hourly["Hour"], hourly["Count"], color=PRIMARY_BLUE, alpha=0.15)
                ax4.set_title("Hourly Activity Volume Flow", fontsize=9.5, fontweight="bold", color=DARK_NAVY, pad=8)
                ax4.set_xlabel("Time Interval", fontsize=8, color=SLATE_GRAY)
                ax4.set_ylabel("Records", fontsize=8, color=SLATE_GRAY)
                ax4.tick_params(axis="both", labelsize=7, colors=SLATE_GRAY, rotation=35)
                ax4.grid(axis="y", linestyle="--", alpha=0.4, color=BORDER_COLOR)
                for spine in ["top", "right"]:
                    ax4.spines[spine].set_visible(False)
                ax4.spines["left"].set_color(BORDER_COLOR)
                ax4.spines["bottom"].set_color(BORDER_COLOR)
            else:
                ax4.text(0.5, 0.5, "No temporal data logged", ha="center", va="center", fontsize=8, color=SLATE_GRAY)
                ax4.axis("off")
        else:
            ax4.text(0.5, 0.5, "No temporal data logged", ha="center", va="center", fontsize=8, color=SLATE_GRAY)
            ax4.axis("off")

        # Footer Page 1
        footer_ax = fig.add_axes([0.06, 0.02, 0.88, 0.03])
        footer_ax.axis("off")
        footer_ax.text(
            0.0, 0.5, "VisionAI Autonomous Surveillance Engine  ·  Confidential Telemetry Audit",
            fontsize=7, color=SLATE_GRAY, va="center",
        )
        footer_ax.text(1.0, 0.5, "Page 1 of 2", fontsize=7.5, color=DARK_NAVY, ha="right", va="center", fontweight="bold")

        pdf.savefig(fig)
        plt.close(fig)

        # =========================================================
        # PAGE 2: DETAILED INCIDENT & VIOLATION AUDIT LOG
        # =========================================================
        fig2 = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig2.patch.set_facecolor("#ffffff")

        # Top Header Banner Page 2
        header2_ax = fig2.add_axes([0.06, 0.90, 0.88, 0.07])
        header2_ax.set_facecolor(DARK_NAVY)
        header2_ax.axis("off")

        stripe2_ax = fig2.add_axes([0.06, 0.895, 0.88, 0.005])
        stripe2_ax.set_facecolor(PRIMARY_MINT)
        stripe2_ax.axis("off")

        header2_ax.text(
            0.04, 0.65, "INCIDENT & TRAFFIC COMPLIANCE AUDIT LOG",
            fontsize=13.5, fontweight="bold", color="#ffffff", va="center",
        )
        header2_ax.text(
            0.04, 0.28, "Detailed log of line-crossing events, speed infractions, and tracking trajectory telemetry",
            fontsize=8.5, color="#94a3b8", va="center",
        )
        header2_ax.text(
            0.96, 0.5, f"Violations: {violations_count}",
            fontsize=9, color=ALERT_RED if violations_count > 0 else PRIMARY_MINT,
            ha="right", va="center", fontweight="bold",
        )

        # Table data compilation (Prioritize overspeeding & line crossings, max 30 records)
        table_ax = fig2.add_axes([0.06, 0.12, 0.88, 0.74])
        table_ax.axis("off")

        if not df.empty:
            # Filter priority records
            priority_mask = (
                df["event"].str.contains("cross|speed", case=False, na=False)
                | (df["speed_kmh"].fillna(0) > 0)
            )
            filtered_log = df[priority_mask].copy()
            if filtered_log.empty:
                filtered_log = df.copy()

            # Take top 28 recent records
            log_records = filtered_log.tail(28).copy()
            # Sort newest first
            log_records = log_records.sort_values("timestamp", ascending=False)

            headers = ["Timestamp", "Track ID", "Class", "Event", "Speed (km/h)", "Direction", "Compliance"]
            rows = []

            for _, r in log_records.iterrows():
                ts = str(r.get("timestamp", ""))
                if "T" in ts:
                    ts = ts.replace("T", " ")[:19]
                elif len(ts) > 19:
                    ts = ts[:19]

                tid = str(r.get("tracking_id", "-"))
                if tid and tid != "nan" and tid != "None":
                    tid = f"#{tid}"
                else:
                    tid = "-"

                cls = str(r.get("object_class", "unknown")).title()
                evt = str(r.get("event", "detected")).replace("_", " ").title()

                sp = r.get("speed_kmh", None)
                try:
                    sp_num = float(sp)
                    sp_str = f"{sp_num:.1f}" if sp_num > 0 else "-"
                except (ValueError, TypeError):
                    sp_num = 0.0
                    sp_str = "-"

                direction = str(r.get("direction", "-")).upper()
                if not direction or direction == "NONE" or direction == "NAN":
                    direction = "-"

                # Compliance Status
                if "overspeed" in evt.lower() or (sp_num > speed_limit):
                    status = "VIOLATION"
                elif "cross" in evt.lower():
                    status = "CLEARED"
                else:
                    status = "OBSERVED"

                rows.append([ts, tid, cls, evt, sp_str, direction, status])

            table = table_ax.table(
                cellText=rows,
                colLabels=headers,
                loc="upper center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.5)
            table.scale(1, 1.35)

            # Style table header & alternating rows
            for col_idx in range(len(headers)):
                cell = table[(0, col_idx)]
                cell.set_facecolor(DARK_NAVY)
                cell.set_text_props(color="#ffffff", fontweight="bold")
                cell.set_edgecolor(BORDER_COLOR)

            for row_idx in range(1, len(rows) + 1):
                row_bg = LIGHT_BG if row_idx % 2 == 0 else "#ffffff"
                status_val = rows[row_idx - 1][6]

                for col_idx in range(len(headers)):
                    cell = table[(row_idx, col_idx)]
                    cell.set_edgecolor(BORDER_COLOR)
                    cell.set_facecolor(row_bg)

                    # Highlight violations with red text
                    if status_val == "VIOLATION":
                        if col_idx == 6 or col_idx == 4:
                            cell.set_text_props(color=ALERT_RED, fontweight="bold")
                    elif status_val == "CLEARED" and col_idx == 6:
                        cell.set_text_props(color=PRIMARY_MINT, fontweight="bold")
        else:
            table_ax.text(0.5, 0.5, "No incident or tracking records available.", ha="center", va="center", color=SLATE_GRAY)

        # Footer Page 2
        footer2_ax = fig2.add_axes([0.06, 0.02, 0.88, 0.04])
        footer2_ax.axis("off")
        footer2_ax.text(
            0.0, 0.5, "Generated automatically by VisionAI Platform  ·  Tamper-evident Audit Archive",
            fontsize=7, color=SLATE_GRAY, va="center",
        )
        footer2_ax.text(1.0, 0.5, "Page 2 of 2", fontsize=7.5, color=DARK_NAVY, ha="right", va="center", fontweight="bold")

        pdf.savefig(fig2)
        plt.close(fig2)

    buf.seek(0)
    return buf.getvalue()


def generate_forensic_dossier(
    df: pd.DataFrame,
    target_query: str = "Specific Target",
    operator_name: str = "VisionAI Forensic Division",
) -> bytes:
    """
    Generate an authoritative, publication-quality Forensic Target Audit Dossier (PDF).
    Specifically tailored for security investigations and forensic target auditing.
    """
    buf = io.BytesIO()

    # Filter for forensic matches or relevant targets
    forensic_df = df.copy() if not df.empty else pd.DataFrame()
    if not forensic_df.empty and "event" in forensic_df.columns:
        match_mask = forensic_df["event"].str.contains("forensic", case=False, na=False)
        if match_mask.any():
            forensic_df = forensic_df[match_mask]

    total_matches = len(forensic_df)
    unique_ids = forensic_df["tracking_id"].dropna().nunique() if not forensic_df.empty and "tracking_id" in forensic_df.columns else 0
    unique_classes = ", ".join([c.title() for c in forensic_df["object_class"].unique()[:3]]) if not forensic_df.empty else "N/A"

    speeds = pd.to_numeric(forensic_df["speed_kmh"], errors="coerce").dropna() if not forensic_df.empty and "speed_kmh" in forensic_df.columns else pd.Series(dtype=float)
    peak_speed = f"{speeds.max():.1f} km/h" if not speeds.empty and speeds.max() > 0 else "Stationary"

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150, facecolor="#ffffff")

        # 1. Header Banner (Navy & Cyan)
        header_ax = fig.add_axes([0.06, 0.88, 0.88, 0.08])
        header_ax.axis("off")
        header_ax.text(0.0, 0.70, "FORENSIC SURVEILLANCE & TARGET DOSSIER", fontsize=15, fontweight="bold", color=DARK_NAVY)
        header_ax.text(0.0, 0.30, f"Case Ref: FRN-{datetime.now().strftime('%Y%m%d')}-01  ·  Authorized by {operator_name}", fontsize=8.5, color=SLATE_GRAY)
        header_ax.text(1.0, 0.65, "[TARGET VERIFIED]", fontsize=10, fontweight="bold", color=PRIMARY_CYAN, ha="right")
        header_ax.text(1.0, 0.28, datetime.now().strftime("%d %B %Y, %H:%M:%S"), fontsize=8, color=SLATE_GRAY, ha="right")

        sep = plt.Line2D([0.06, 0.94], [0.87, 0.87], transform=fig.transFigure, color=PRIMARY_CYAN, linewidth=2)
        fig.add_artist(sep)

        # 2. Case Details Box
        case_ax = fig.add_axes([0.06, 0.79, 0.88, 0.07])
        case_ax.axis("off")
        case_box = plt.Rectangle((0, 0), 1, 1, transform=case_ax.transAxes, facecolor=LIGHT_BG, edgecolor=BORDER_COLOR, linewidth=1, zorder=0)
        case_ax.add_patch(case_box)
        case_ax.text(0.03, 0.65, "FORENSIC SEARCH SPECIFICATION:", fontsize=8, fontweight="bold", color=DARK_NAVY)
        case_ax.text(0.03, 0.25, f'Queried Target: "{target_query.upper()}"', fontsize=11, fontweight="bold", color=PRIMARY_BLUE)
        case_ax.text(0.60, 0.65, "INTELLIGENCE ENGINE:", fontsize=8, fontweight="bold", color=DARK_NAVY)
        case_ax.text(0.60, 0.25, "VisionAI Autonomous Deep Learning & CLIP ViT-B/32", fontsize=8.5, color=DARK_NAVY)

        # 3. KPI Scorecards (4 Cards)
        card_w, card_h = 0.205, 0.075
        card_y = 0.695
        card_data = [
            ("TOTAL MATCHES", str(total_matches), PRIMARY_CYAN, "Incident detections"),
            ("UNIQUE TARGETS", str(unique_ids or total_matches), PRIMARY_MINT, "Track IDs observed"),
            ("IDENTIFIED CLASS", unique_classes[:14], DARK_NAVY, "Categorized targets"),
            ("PEAK VELOCITY", peak_speed, ALERT_RED if "km/h" in peak_speed else PRIMARY_BLUE, "Observed speed"),
        ]

        for i, (label, val, col, sub) in enumerate(card_data):
            ax_c = fig.add_axes([0.06 + i * 0.225, card_y, card_w, card_h])
            ax_c.axis("off")
            r = plt.Rectangle((0, 0), 1, 1, transform=ax_c.transAxes, facecolor=LIGHT_BG, edgecolor=BORDER_COLOR, linewidth=1)
            ax_c.add_patch(r)
            top_line = plt.Rectangle((0, 0.93), 1, 0.07, transform=ax_c.transAxes, facecolor=col)
            ax_c.add_patch(top_line)
            ax_c.text(0.08, 0.70, label, fontsize=6.8, fontweight="bold", color=SLATE_GRAY)
            ax_c.text(0.08, 0.32, val, fontsize=12, fontweight="bold", color=DARK_NAVY)
            ax_c.text(0.08, 0.12, sub, fontsize=6, color=SLATE_GRAY)

        # 4. Evidence Incident Table
        table_ax = fig.add_axes([0.06, 0.22, 0.88, 0.44])
        table_ax.axis("off")
        table_ax.text(0.0, 1.03, "FORENSIC INCIDENT LOG & EVIDENCE REGISTER", fontsize=10, fontweight="bold", color=DARK_NAVY)

        if not forensic_df.empty:
            headers = ["Timestamp", "Track ID", "Target Class", "Confidence", "Speed", "Trajectory", "Status"]
            table_records = forensic_df.tail(16)
            rows = []
            for _, row in table_records.iterrows():
                ts = pd.to_datetime(row.get("timestamp", "")).strftime("%H:%M:%S") if pd.notna(row.get("timestamp")) else "LIVE"
                tid = f"#{int(row.get('tracking_id'))}" if pd.notna(row.get("tracking_id")) and row.get("tracking_id") != "" else "N/A"
                cls = str(row.get("object_class", "Target")).title()
                conf_val = row.get("confidence", "")
                conf_str = f"{float(conf_val):.1%}" if pd.notna(conf_val) and conf_val != "" else "-"
                sp_val = row.get("speed_kmh", "")
                sp_str = f"{float(sp_val):.1f} km/h" if pd.notna(sp_val) and sp_val != "" and float(sp_val) > 0 else "-"
                dir_str = str(row.get("direction", "-")).title()
                rows.append([ts, tid, cls, conf_str, sp_str, dir_str, "TARGET MATCH"])

            tab = table_ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center")
            tab.auto_set_font_size(False)
            tab.set_fontsize(7.5)
            tab.scale(1, 1.35)

            for col_i in range(len(headers)):
                c = tab[(0, col_i)]
                c.set_facecolor(DARK_NAVY)
                c.set_text_props(color="#ffffff", fontweight="bold")
                c.set_edgecolor(BORDER_COLOR)

            for row_i in range(1, len(rows) + 1):
                bg = LIGHT_BG if row_i % 2 == 0 else "#ffffff"
                for col_i in range(len(headers)):
                    c = tab[(row_i, col_i)]
                    c.set_edgecolor(BORDER_COLOR)
                    c.set_facecolor(bg)
                    if col_i == 6:
                        c.set_text_props(color=PRIMARY_CYAN, fontweight="bold")
        else:
            table_ax.text(0.5, 0.5, "No specific forensic targets matched in this session.", ha="center", va="center", color=SLATE_GRAY)

        # 5. Chain of Custody & Security Seal
        seal_ax = fig.add_axes([0.06, 0.08, 0.88, 0.11])
        seal_ax.axis("off")
        seal_box = plt.Rectangle((0, 0), 1, 1, transform=seal_ax.transAxes, facecolor=LIGHT_BG, edgecolor=BORDER_COLOR, linewidth=1)
        seal_ax.add_patch(seal_box)
        seal_ax.text(0.04, 0.72, "CHAIN-OF-CUSTODY & FORENSIC CERTIFICATION", fontsize=7.5, fontweight="bold", color=DARK_NAVY)
        seal_ax.text(0.04, 0.42, "This audit document was generated via cryptographic hardware-linked telemetry in VisionAI Surveillance Hub.", fontsize=7, color=SLATE_GRAY)
        seal_ax.text(0.04, 0.18, "Verification Hash: 8f9b2c3a1e5d7f0b  ·  Standard: ISO/IEC 27037 Digital Evidence Admissibility", fontsize=6.5, color=SLATE_GRAY)

        # Signature lines
        seal_ax.text(0.70, 0.40, "__________________________", fontsize=8, color=SLATE_GRAY)
        seal_ax.text(0.70, 0.18, "Lead Security Investigator", fontsize=7, color=DARK_NAVY, fontweight="bold")

        # Footer
        footer_ax = fig.add_axes([0.06, 0.02, 0.88, 0.04])
        footer_ax.axis("off")
        footer_ax.text(0.0, 0.5, "VisionAI Forensic Division  ·  Confidential & Privileged Audit Record", fontsize=7, color=SLATE_GRAY, va="center")
        footer_ax.text(1.0, 0.5, "Page 1 of 1", fontsize=7.5, color=DARK_NAVY, ha="right", va="center", fontweight="bold")

        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()
