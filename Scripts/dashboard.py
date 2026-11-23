
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output


# ============================================================
# 1. PROJECT PATHS AND DATA
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "indicators"

FILES = {
    "segment_cancel": "cancellation_by_segment.csv",
    "lead_cancel": "cancellation_by_lead_time.csv",
    "deposit_cancel": "cancellation_by_deposit.csv",
    "customer_cancel": "cancellation_by_customer.csv",
    "hotel": "booking_by_hotel.csv",
    "segment_booking": "booking_by_segment.csv",
    "monthly": "booking_by_month.csv",
}

def load_indicator(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Please check that all seven CSV files are in Data/indicators/."
        )

    return pd.read_csv(path)


segment_cancel = load_indicator(FILES["segment_cancel"])
lead_cancel = load_indicator(FILES["lead_cancel"])
deposit_cancel = load_indicator(FILES["deposit_cancel"])
customer_cancel = load_indicator(FILES["customer_cancel"])

hotel_data = load_indicator(FILES["hotel"])
segment_booking = load_indicator(FILES["segment_booking"])
monthly_data = load_indicator(FILES["monthly"])


# Check required columns before starting the dashboard.
REQUIRED_COLUMNS = {
    "segment_cancel": ["market_segment", "booking_count", "cancellation_rate"],
    "lead_cancel": ["lead_time_category", "booking_count", "cancellation_rate"],
    "deposit_cancel": ["deposit_type", "booking_count", "cancellation_rate"],
    "customer_cancel": ["customer_type", "booking_count", "cancellation_rate"],
    "hotel": [
        "hotel", "booking_count", "average_adr",
        "average_nights", "average_booking_value"
    ],
    "segment_booking": [
        "market_segment", "booking_count", "average_adr",
        "average_nights", "average_booking_value"
    ],
    "monthly": [
        "arrival_date_year", "arrival_date_month", "booking_count",
        "average_adr", "average_nights", "average_booking_value"
    ],
}

DATASETS = {
    "segment_cancel": segment_cancel,
    "lead_cancel": lead_cancel,
    "deposit_cancel": deposit_cancel,
    "customer_cancel": customer_cancel,
    "hotel": hotel_data,
    "segment_booking": segment_booking,
    "monthly": monthly_data,
}

for dataset_name, required in REQUIRED_COLUMNS.items():
    missing = set(required) - set(DATASETS[dataset_name].columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {sorted(missing)}"
        )


# ============================================================
# 2. PREPARE MONTHLY DATA
# ============================================================

MONTH_ORDER = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

monthly_data["arrival_date_year"] = pd.to_numeric(
    monthly_data["arrival_date_year"], errors="coerce"
)

monthly_data["month_number"] = (
    pd.Categorical(
        monthly_data["arrival_date_month"],
        categories=MONTH_ORDER,
        ordered=True
    ).codes + 1
)

monthly_data = monthly_data.sort_values(
    ["arrival_date_year", "month_number"]
).copy()

monthly_data["period"] = pd.to_datetime(
    monthly_data["arrival_date_year"].astype(int).astype(str)
    + "-"
    + monthly_data["month_number"].astype(str).str.zfill(2)
    + "-01"
)

YEARS = sorted(
    monthly_data["arrival_date_year"].dropna().astype(int).unique()
)


# ============================================================
# 3. DASHBOARD DESIGN
# ============================================================

NAVY = "#101827"
CARD = "#1B2738"
CARD_LIGHT = "#223249"
BLUE = "#4F8CFF"
TEAL = "#21C7A8"
PURPLE = "#A78BFA"
ORANGE = "#F5A65B"
WHITE = "#F5F7FB"
MUTED = "#9BAAC0"
BORDER = "#2B3B52"

FONT = "Inter, Segoe UI, Arial, sans-serif"

app = Dash(__name__)
app.title = "Hotel Booking Analytics Dashboard"

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                background: #101827;
                font-family: Inter, Segoe UI, Arial, sans-serif;
            }
            .Select-control, .Select-menu-outer {
                background-color: #223249 !important;
                color: #F5F7FB !important;
            }
            .Select-value-label, .Select-placeholder {
                color: #F5F7FB !important;
            }
            .Select-input input { color: #F5F7FB !important; }
            .VirtualizedSelectOption {
                background-color: #223249 !important;
                color: #F5F7FB !important;
            }
            .VirtualizedSelectFocusedOption {
                background-color: #344761 !important;
            }
            .dash-dropdown { color: #F5F7FB; }
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-thumb {
                background: #344761;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


# ============================================================
# 4. REUSABLE DESIGN COMPONENTS
# ============================================================

def card_style():
    return {
        "backgroundColor": CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "14px",
        "padding": "20px",
        "boxShadow": "0 5px 18px rgba(0,0,0,0.15)",
    }


def section_heading(number, title, subtitle):
    return html.Div(
        [
            html.Div(
                number,
                style={
                    "backgroundColor": BLUE,
                    "color": "white",
                    "padding": "9px 13px",
                    "borderRadius": "9px",
                    "fontWeight": "bold",
                    "fontSize": "15px",
                },
            ),
            html.Div(
                [
                    html.H2(
                        title,
                        style={
                            "color": WHITE,
                            "fontSize": "23px",
                            "margin": "0 0 5px 0",
                        },
                    ),
                    html.P(
                        subtitle,
                        style={
                            "color": MUTED,
                            "fontSize": "13px",
                            "margin": 0,
                        },
                    ),
                ],
                style={"marginLeft": "13px"},
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "margin": "32px 0 18px 0",
        },
    )


def dropdown(label, component_id, options, value, width="220px"):
    return html.Div(
        [
            html.Label(
                label,
                style={
                    "display": "block",
                    "color": MUTED,
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "marginBottom": "8px",
                },
            ),
            dcc.Dropdown(
                id=component_id,
                options=options,
                value=value,
                clearable=False,
                style={
                    "width": width,
                    "color": "#172033",
                    "fontSize": "13px",
                },
            ),
        ]
    )


def graph_card(title, subtitle, graph_id, height=350):
    return html.Div(
        [
            html.H3(
                title,
                style={
                    "color": WHITE,
                    "fontSize": "16px",
                    "margin": "0 0 6px 0",
                },
            ),
            html.P(
                subtitle,
                style={
                    "color": MUTED,
                    "fontSize": "12px",
                    "margin": "0 0 12px 0",
                },
            ),
            dcc.Graph(
                id=graph_id,
                config={"displayModeBar": False, "responsive": True},
                style={"height": f"{height}px"},
            ),
        ],
        style=card_style(),
    )


def insight_card(title, message, accent=TEAL):
    return html.Div(
        [
            html.Div(
                title.upper(),
                style={
                    "fontSize": "10px",
                    "fontWeight": "bold",
                    "letterSpacing": "1.2px",
                    "color": accent,
                    "marginBottom": "9px",
                },
            ),
            html.P(
                message,
                style={
                    "color": WHITE,
                    "fontSize": "13px",
                    "lineHeight": "1.7",
                    "margin": 0,
                },
            ),
        ],
        style={
            **card_style(),
            "borderLeft": f"3px solid {accent}",
            "flex": "1",
            "minWidth": "220px",
        },
    )


def kpi_card(label, value, description, accent):
    return html.Div(
        [
            html.P(
                label,
                style={
                    "color": MUTED,
                    "fontSize": "12px",
                    "margin": "0 0 10px 0",
                },
            ),
            html.H2(
                value,
                style={
                    "color": WHITE,
                    "fontSize": "27px",
                    "margin": "0 0 8px 0",
                    "fontWeight": "700",
                },
            ),
            html.P(
                description,
                style={
                    "color": accent,
                    "fontSize": "11px",
                    "margin": 0,
                },
            ),
        ],
        style={
            **card_style(),
            "flex": "1",
            "minWidth": "180px",
        },
    )


# ============================================================
# 5. CHART FORMATTING
# ============================================================

def style_figure(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"family": FONT, "color": MUTED, "size": 11},
        margin={"l": 45, "r": 20, "t": 20, "b": 55},
        height=height,
        hoverlabel={
            "bgcolor": CARD_LIGHT,
            "font": {"color": WHITE, "family": FONT},
        },
        legend={
            "orientation": "h",
            "y": 1.12,
            "x": 0,
            "font": {"size": 10},
        },
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=BORDER,
    )

    fig.update_yaxes(
        gridcolor=BORDER,
        zeroline=False,
        linecolor=BORDER,
    )

    return fig


# ============================================================
# 6. KPI CALCULATIONS FROM EXISTING INDICATORS
# ============================================================

total_bookings = int(hotel_data["booking_count"].sum())

weighted_adr = (
    (hotel_data["average_adr"] * hotel_data["booking_count"]).sum()
    / total_bookings
)

weighted_estimated_value = (
    (
        hotel_data["average_booking_value"]
        * hotel_data["booking_count"]
    ).sum()
    / total_bookings
)

total_cancellation_bookings = int(
    segment_cancel["booking_count"].sum()
)

overall_cancellation_rate = (
    (
        segment_cancel["cancellation_rate"]
        * segment_cancel["booking_count"]
    ).sum()
    / total_cancellation_bookings
)


# ============================================================
# 7. MAIN DASHBOARD LAYOUT
# ============================================================

app.layout = html.Div(
    [
        # HEADER
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "HOTEL",
                                    style={"color": TEAL},
                                ),
                                html.Span(
                                    " INSIGHTS",
                                    style={"color": WHITE},
                                ),
                            ],
                            style={
                                "fontSize": "12px",
                                "fontWeight": "bold",
                                "letterSpacing": "2px",
                            },
                        ),
                        html.H1(
                            "Hotel Booking Analytics",
                            style={
                                "color": WHITE,
                                "fontSize": "30px",
                                "margin": "12px 0 7px 0",
                            },
                        ),
                        html.P(
                            "Understand cancellations, booking demand "
                            "and estimated booking value.",
                            style={
                                "color": MUTED,
                                "fontSize": "13px",
                                "margin": 0,
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Span(
                            "● DATA LOADED",
                            style={
                                "color": TEAL,
                                "fontSize": "11px",
                                "fontWeight": "bold",
                                "border": f"1px solid {BORDER}",
                                "padding": "9px 13px",
                                "borderRadius": "20px",
                            },
                        ),
                        html.P(
                            "Source: 7 pre-calculated indicator CSV files",
                            style={
                                "color": MUTED,
                                "fontSize": "10px",
                                "marginTop": "10px",
                                "textAlign": "right",
                            },
                        ),
                    ],
                    style={"textAlign": "right"},
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "15px",
                "padding": "28px 30px",
                "backgroundColor": CARD,
                "borderBottom": f"1px solid {BORDER}",
            },
        ),

        html.Div(
            [
                # OVERVIEW KPIs
                html.Div(
                    [
                        kpi_card(
                            "TOTAL BOOKINGS",
                            f"{total_bookings:,}",
                            "Across both hotel types",
                            BLUE,
                        ),
                        kpi_card(
                            "OVERALL CANCELLATION RATE",
                            f"{overall_cancellation_rate:.1f}%",
                            "Weighted from segment indicators",
                            ORANGE,
                        ),
                        kpi_card(
                            "WEIGHTED AVERAGE ADR",
                            f"{weighted_adr:.2f}",
                            "Average daily rate",
                            TEAL,
                        ),
                        kpi_card(
                            "ESTIMATED BOOKING VALUE",
                            f"{weighted_estimated_value:.2f}",
                            "Average estimated value per booking",
                            PURPLE,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "15px",
                        "flexWrap": "wrap",
                        "marginBottom": "25px",
                    },
                ),

                # =================================================
                # Q1: CANCELLATION ANALYSIS
                # =================================================

                section_heading(
                    "Q1",
                    "Cancellation Analysis",
                    "Explore the factors associated with hotel booking cancellations.",
                ),

                html.Div(
                    [
                        dropdown(
                            "Display metric",
                            "q1-metric",
                            [
                                {
                                    "label": "Cancellation rate (%)",
                                    "value": "cancellation_rate",
                                },
                                {
                                    "label": "Booking count",
                                    "value": "booking_count",
                                },
                            ],
                            "cancellation_rate",
                        ),
                    ],
                    style={"marginBottom": "16px"},
                ),

                html.Div(
                    [
                        html.Div(
                            graph_card(
                                "Cancellation by Market Segment",
                                "Compare cancellation behaviour across booking channels.",
                                "segment-cancel-chart",
                            ),
                            style={"flex": "1", "minWidth": "320px"},
                        ),
                        html.Div(
                            graph_card(
                                "Cancellation by Lead Time",
                                "Understand how advance booking time relates to cancellations.",
                                "lead-cancel-chart",
                            ),
                            style={"flex": "1", "minWidth": "320px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "flexWrap": "wrap",
                        "marginBottom": "16px",
                    },
                ),

                html.Div(
                    [
                        html.Div(
                            graph_card(
                                "Cancellation by Deposit Type",
                                "Compare cancellation rates across deposit policies.",
                                "deposit-cancel-chart",
                            ),
                            style={"flex": "1", "minWidth": "320px"},
                        ),
                        html.Div(
                            graph_card(
                                "Cancellation by Customer Type",
                                "Identify differences between customer categories.",
                                "customer-cancel-chart",
                            ),
                            style={"flex": "1", "minWidth": "320px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "flexWrap": "wrap",
                        "marginBottom": "16px",
                    },
                ),

                html.Div(
                    [
                        insight_card(
                            "Lead time insight",
                            "The existing indicators show higher cancellation "
                            "rates for bookings made further in advance. "
                            "This suggests that long-lead bookings may need "
                            "closer monitoring.",
                            ORANGE,
                        ),
                        insight_card(
                            "Deposit insight",
                            "Non-refundable bookings show a very high "
                            "cancellation rate in the available indicators. "
                            "However, this category has a much smaller "
                            "booking count, so interpret the rate carefully.",
                            PURPLE,
                        ),
                        insight_card(
                            "Segment insight",
                            "Online TA has a higher cancellation rate than "
                            "Direct bookings. Reviewing booking policies "
                            "and customer communication may help reduce risk.",
                            BLUE,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "15px",
                        "flexWrap": "wrap",
                        "marginBottom": "38px",
                    },
                ),

                # =================================================
                # Q2: BOOKING DEMAND, ADR AND ESTIMATED VALUE
                # =================================================

                section_heading(
                    "Q2",
                    "Booking Demand, ADR & Estimated Booking Value",
                    "Compare hotel performance, market segments and monthly trends.",
                ),

                html.Div(
                    [
                        dropdown(
                            "Comparison metric",
                            "q2-metric",
                            [
                                {
                                    "label": "Booking demand",
                                    "value": "booking_count",
                                },
                                {
                                    "label": "Average ADR",
                                    "value": "average_adr",
                                },
                                {
                                    "label": "Estimated booking value",
                                    "value": "average_booking_value",
                                },
                                {
                                    "label": "Average nights",
                                    "value": "average_nights",
                                },
                            ],
                            "booking_count",
                        ),
                        dropdown(
                            "Monthly trend year",
                            "q2-year",
                            [
                                {"label": "All years", "value": "all"}
                            ] + [
                                {"label": str(year), "value": int(year)}
                                for year in YEARS
                            ],
                            "all",
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "20px",
                        "flexWrap": "wrap",
                        "marginBottom": "16px",
                    },
                ),

                html.Div(
                    [
                        html.Div(
                            graph_card(
                                "Hotel Performance Comparison",
                                "Compare City Hotel and Resort Hotel using the selected metric.",
                                "hotel-chart",
                            ),
                            style={"flex": "1", "minWidth": "320px"},
                        ),
                        html.Div(
                            graph_card(
                                "Market Segment Performance",
                                "Compare booking channels using the selected metric.",
                                "market-chart",
                            ),
                            style={"flex": "1", "minWidth": "320px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "flexWrap": "wrap",
                        "marginBottom": "16px",
                    },
                ),

                html.Div(
                    graph_card(
                        "Monthly Booking and Value Trends",
                        "Follow booking demand, ADR and estimated booking value over time.",
                        "monthly-chart",
                        height=410,
                    ),
                    style={"marginBottom": "16px"},
                ),

                html.Div(
                    [
                        insight_card(
                            "Hotel comparison",
                            "Resort Hotel has a higher average estimated "
                            "booking value, while City Hotel has a higher "
                            "average ADR. Longer average stays at Resort "
                            "Hotel contribute to its higher estimated value.",
                            TEAL,
                        ),
                        insight_card(
                            "Booking demand",
                            "Online TA is the largest market segment by "
                            "booking count in the available indicators. "
                            "This highlights the importance of online "
                            "distribution channels.",
                            BLUE,
                        ),
                        insight_card(
                            "Seasonality",
                            "Monthly comparisons help identify periods of "
                            "stronger demand and changes in average daily "
                            "rates. Use these patterns to support seasonal "
                            "pricing and capacity planning.",
                            PURPLE,
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "15px",
                        "flexWrap": "wrap",
                        "marginBottom": "30px",
                    },
                ),

                # FOOTER
                html.Div(
                    [
                        html.P(
                            "HOTEL BOOKING ANALYTICS | Q1 & Q2",
                            style={
                                "color": MUTED,
                                "fontSize": "10px",
                                "letterSpacing": "1.5px",
                                "margin": "0 0 8px 0",
                            },
                        ),
                        html.P(
                            "Booking value is an estimate calculated in "
                            "the source indicators, not actual revenue. "
                            "All displayed indicators are pre-calculated "
                            "CSV outputs.",
                            style={
                                "color": MUTED,
                                "fontSize": "11px",
                                "lineHeight": "1.6",
                                "margin": 0,
                            },
                        ),
                    ],
                    style={
                        "borderTop": f"1px solid {BORDER}",
                        "padding": "20px 0",
                    },
                ),
            ],
            style={
                "maxWidth": "1500px",
                "margin": "0 auto",
                "padding": "25px 30px",
            },
        ),
    ],
    style={
        "backgroundColor": NAVY,
        "minHeight": "100vh",
        "color": WHITE,
        "fontFamily": FONT,
    },
)


# ============================================================
# 8. Q1: CANCELLATION CHART CALLBACKS
# ============================================================

def cancellation_figure(df, category_column, metric, color):
    if metric == "cancellation_rate":
        values = df["cancellation_rate"]
        y_title = "Cancellation rate (%)"
        labels = [f"{v:.1f}%" for v in values]
        hover_template = (
            "%{x}<br>Cancellation rate: %{y:.2f}%"
            "<br>Booking count: %{customdata:,.0f}<extra></extra>"
        )
    else:
        values = df["booking_count"]
        y_title = "Booking count"
        labels = [f"{v:,.0f}" for v in values]
        hover_template = (
            "%{x}<br>Booking count: %{y:,.0f}"
            "<br>Cancellation rate: %{customdata:.2f}%"
            "<extra></extra>"
        )

    if metric == "cancellation_rate":
        customdata = df["booking_count"]
    else:
        customdata = df["cancellation_rate"]

    fig = go.Figure(
        go.Bar(
            x=df[category_column],
            y=values,
            marker={
                "color": color,
                "line": {"width": 0},
            },
            text=labels,
            textposition="outside",
            customdata=customdata,
            hovertemplate=hover_template,
        )
    )

    fig.update_layout(
        yaxis_title=y_title,
        xaxis_title="",
        showlegend=False,
    )

    return style_figure(fig)


@app.callback(
    Output("segment-cancel-chart", "figure"),
    Output("lead-cancel-chart", "figure"),
    Output("deposit-cancel-chart", "figure"),
    Output("customer-cancel-chart", "figure"),
    Input("q1-metric", "value"),
)
def update_cancellation_charts(metric):
    return (
        cancellation_figure(
            segment_cancel, "market_segment", metric, BLUE
        ),
        cancellation_figure(
            lead_cancel, "lead_time_category", metric, TEAL
        ),
        cancellation_figure(
            deposit_cancel, "deposit_type", metric, PURPLE
        ),
        cancellation_figure(
            customer_cancel, "customer_type", metric, ORANGE
        ),
    )


# ============================================================
# 9. Q2: HOTEL AND MARKET SEGMENT COMPARISONS
# ============================================================

METRIC_LABELS = {
    "booking_count": "Booking count",
    "average_adr": "Average ADR",
    "average_booking_value": "Average estimated booking value",
    "average_nights": "Average nights",
}

METRIC_FORMATS = {
    "booking_count": ",.0f",
    "average_adr": ".2f",
    "average_booking_value": ".2f",
    "average_nights": ".2f",
}


def comparison_figure(df, category_column, metric, color):
    fig = px.bar(
        df,
        x=category_column,
        y=metric,
        text=metric,
        color_discrete_sequence=[color],
    )

    fig.update_traces(
        texttemplate="%{y:" + METRIC_FORMATS[metric] + "}",
        textposition="outside",
        hovertemplate=(
            "%{x}<br>"
            + METRIC_LABELS[metric]
            + ": %{y:"
            + METRIC_FORMATS[metric]
            + "}<extra></extra>"
        ),
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title=METRIC_LABELS[metric],
        showlegend=False,
    )

    return style_figure(fig)


@app.callback(
    Output("hotel-chart", "figure"),
    Output("market-chart", "figure"),
    Input("q2-metric", "value"),
)
def update_comparison_charts(metric):
    return (
        comparison_figure(
            hotel_data, "hotel", metric, TEAL
        ),
        comparison_figure(
            segment_booking, "market_segment", metric, BLUE
        ),
    )


# ============================================================
# 10. Q2: MONTHLY TRENDS
# ============================================================

@app.callback(
    Output("monthly-chart", "figure"),
    Input("q2-year", "value"),
)
def update_monthly_chart(selected_year):
    if selected_year == "all":
        df = monthly_data.copy()
    else:
        df = monthly_data[
            monthly_data["arrival_date_year"] == int(selected_year)
        ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df["period"],
            y=df["booking_count"],
            name="Booking demand",
            marker_color=BLUE,
            opacity=0.8,
            yaxis="y",
            hovertemplate=(
                "Month: %{x|%b %Y}"
                "<br>Bookings: %{y:,.0f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["average_adr"],
            name="Average ADR",
            mode="lines+markers",
            line={"color": TEAL, "width": 3},
            marker={"size": 7},
            yaxis="y2",
            hovertemplate=(
                "Month: %{x|%b %Y}"
                "<br>Average ADR: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["period"],
            y=df["average_booking_value"],
            name="Estimated booking value",
            mode="lines+markers",
            line={"color": PURPLE, "width": 3, "dash": "dot"},
            marker={"size": 7},
            yaxis="y2",
            hovertemplate=(
                "Month: %{x|%b %Y}"
                "<br>Estimated booking value: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font={"family": FONT, "color": MUTED, "size": 11},
        margin={"l": 55, "r": 65, "t": 35, "b": 55},
        height=410,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.15,
            "x": 0,
        },
        xaxis={
            "title": "Arrival month",
            "gridcolor": BORDER,
        },
        yaxis={
            "title": "Booking count",
            "gridcolor": BORDER,
            "rangemode": "tozero",
        },
        yaxis2={
            "title": "Average value",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "rangemode": "tozero",
        },
    )

    return fig


# ============================================================
# 11. START THE DASHBOARD
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("HOTEL BOOKING ANALYTICS DASHBOARD")
    print("=" * 55)
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Indicator folder: {DATA_DIR}")
    print("All seven indicator files loaded successfully.")
    print("Open your browser at: http://127.0.0.1:8050/")
    print("=" * 55)

    app.run(debug=True, port=8050)
