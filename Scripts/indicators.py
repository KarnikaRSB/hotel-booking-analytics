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
