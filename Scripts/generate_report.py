"""
Member 4 - Automation: build a one-page HTML report after each pipeline run.

Reads Data/clean/hotel_bookings_scored.csv and reports/model_metrics.json,
writes reports/report.html (downloadable from GitHub Actions).

Run:  python Scripts/generate_report.py
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ("Data" if (ROOT / "Data").exists() else "data")
SCORED = DATA / "clean" / "hotel_bookings_scored.csv"
METRICS = ROOT / "reports" / "model_metrics.json"
REPORT = ROOT / "reports" / "report.html"

RISK_ORDER = ["Low", "Medium", "High"]


def main():
    df = pd.read_csv(SCORED)
    metrics = json.loads(METRICS.read_text())
    df["outcome"] = df["is_canceled"].map({0: "Kept", 1: "Cancelled"})

    # Headline numbers
    kept = df[df["is_canceled"] == 0]
    high = df[df["risk_level"] == "High"]
    kpis = {
        "Bookings": f"{len(df):,}",
        "Cancellation rate": f"{df['is_canceled'].mean():.1%}",
        "Average ADR (kept bookings)": f"{kept['adr'].mean():.0f}",
        "High-risk bookings": f"{len(high):,} ({len(high) / len(df):.1%})",
        "Actual cancellation rate in High risk group": f"{high['is_canceled'].mean():.1%}" if len(high) else "-",
    }

    # Charts
    risk = (df.groupby("risk_level", observed=True)
              .agg(bookings=("is_canceled", "size"), actual_cancel_rate=("is_canceled", "mean"))
              .reindex(RISK_ORDER).reset_index())
    fig_risk = px.bar(risk, x="risk_level", y="bookings", text="bookings",
                      hover_data={"actual_cancel_rate": ":.1%"},
                      title="Bookings per predicted risk level")
    fig_hist = px.histogram(df, x="cancel_probability", color="outcome", nbins=40, barmode="overlay",
                            title="Predicted cancellation probability vs actual outcome")
    seg = (df.groupby("market_segment")["is_canceled"].mean().sort_values().reset_index())
    fig_seg = px.bar(seg, x="is_canceled", y="market_segment", orientation="h",
                     labels={"is_canceled": "Cancellation rate"}, title="Cancellation rate by market segment")
    imp = pd.Series(metrics["feature_importance"]).sort_values().reset_index()
    imp.columns = ["feature", "importance"]
    fig_imp = px.bar(imp, x="importance", y="feature", orientation="h",
                     title=f"What drives the prediction ({metrics['best_model']})")

    # Model comparison table
    comp = pd.DataFrame(metrics["comparison"]).T
    comp.index.name = "Model"

    kpi_html = "".join(f"<div class='kpi'><span>{k}</span><b>{v}</b></div>" for k, v in kpis.items())
    charts = [fig_risk, fig_hist, fig_imp, fig_seg]
    charts_html = "".join(f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                          for i, f in enumerate(charts))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hotel Booking Report</title>
<style>
 body {{font-family: Arial, sans-serif; max-width: 1100px; margin: auto; padding: 20px; color: #222}}
 .kpis {{display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0}}
 .kpi {{border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; min-width: 180px}}
 .kpi span {{display: block; color: #666; font-size: 13px}} .kpi b {{font-size: 22px}}
 table {{border-collapse: collapse}} td, th {{border: 1px solid #ddd; padding: 6px 12px; text-align: right}}
</style></head><body>
<h1>Hotel Booking Report</h1>
<p>Generated {datetime.now():%Y-%m-%d %H:%M} by the automated pipeline.</p>
<div class="kpis">{kpi_html}</div>
<h2>Model comparison (test set, {metrics['test_rows']:,} bookings)</h2>
<p>Selected model: <b>{metrics['best_model']}</b> (highest ROC AUC).</p>
{comp.to_html()}
{charts_html}
</body></html>"""

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(html, encoding="utf-8")
    print(f"Saved report -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
