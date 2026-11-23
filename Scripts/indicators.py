import pandas as pd

df = pd.read_csv("data/clean/hotel_bookings_clean.csv")

## Calculate the overall cancellation rate
cancellation_rate = df["is_canceled"].mean() * 100

print("Overall cancellation rate:", round(cancellation_rate, 2), "%")


## Cancellation rate by market segment
segment_analysis = df.groupby("market_segment")["is_canceled"].agg(
    booking_count="count",
    cancellation_rate="mean"
)

segment_analysis["cancellation_rate"] *= 100

print("\nCancellation rate by market segment:")
print(segment_analysis.round(2))


# Create lead-time categories
bins = [-1, 7, 30, 90, 180, 365, float("inf")]
labels = ["0-7 days", "8-30 days", "31-90 days",
          "91-180 days", "181-365 days", "Over 365 days"]

df["lead_time_category"] = pd.cut(
    df["lead_time"],
    bins=bins,
    labels=labels
)

# Calculate cancellation rate by lead-time category
lead_time_analysis = df.groupby(
    "lead_time_category", observed=True
)["is_canceled"].agg(
    booking_count="count",
    cancellation_rate="mean"
)

lead_time_analysis["cancellation_rate"] *= 100

print("\nCancellation rate by lead time:")
print(lead_time_analysis.round(2))

