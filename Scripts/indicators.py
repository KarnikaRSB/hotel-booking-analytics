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

## Calculate cancellation rate by lead-time category
lead_time_analysis = df.groupby(
    "lead_time_category", observed=True
)["is_canceled"].agg(
    booking_count="count",
    cancellation_rate="mean"
)

lead_time_analysis["cancellation_rate"] *= 100

print("\nCancellation rate by lead time:")
print(lead_time_analysis.round(2))


# Cancellation rate by deposit type
deposit_analysis = df.groupby("deposit_type")["is_canceled"].agg(
    booking_count="count",
    cancellation_rate="mean"
)

deposit_analysis["cancellation_rate"] *= 100

print("\nCancellation rate by deposit type:")
print(deposit_analysis.round(2))


## Cancellation rate by customer type
customer_analysis = df.groupby("customer_type")["is_canceled"].agg(
    booking_count="count",
    cancellation_rate="mean"
)

customer_analysis["cancellation_rate"] *= 100

print("\nCancellation rate by customer type:")
print(customer_analysis.round(2))


# Question 2: Booking demand, ADR and estimated booking value

# Calculate total nights booked
df["total_nights"] = (
    df["stays_in_weekend_nights"] +
    df["stays_in_week_nights"]
)

# Calculate estimated booking value
df["estimated_booking_value"] = (
    df["adr"] * df["total_nights"]
)

# Display a sample
print("\nSample of Q2 indicators:")
print(df[["hotel", "adr", "total_nights",
          "estimated_booking_value"]].head())


# Question 2: Analysis by hotel type

hotel_analysis = df.groupby("hotel").agg(
    booking_count=("hotel", "size"),
    average_adr=("adr", "mean"),
    average_nights=("total_nights", "mean"),
    average_booking_value=("estimated_booking_value", "mean")
)

print("\nBooking analysis by hotel type:")
print(hotel_analysis.round(2).to_string(index=True))


# Question 2: Booking analysis by market segment

segment_booking_analysis = df.groupby("market_segment").agg(
    booking_count=("hotel", "size"),
    average_adr=("adr", "mean"),
    average_nights=("total_nights", "mean"),
    average_booking_value=("estimated_booking_value", "mean")
)

print("\nBooking analysis by market segment:")
print(segment_booking_analysis.round(2).T.to_string())


# Question 2: Booking analysis by year and month

monthly_analysis = df.groupby(
    ["arrival_date_year", "arrival_date_month"]
).agg(
    booking_count=("hotel", "size"),
    average_adr=("adr", "mean"),
    average_nights=("total_nights", "mean"),
    average_booking_value=("estimated_booking_value", "mean")
)

print("\nBooking analysis by year and month:")
print(monthly_analysis.round(2).to_string())


# Arrange months in chronological order
month_order = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

monthly_analysis = monthly_analysis.reset_index()

monthly_analysis["arrival_date_month"] = pd.Categorical(
    monthly_analysis["arrival_date_month"],
    categories=month_order,
    ordered=True
)

monthly_analysis = monthly_analysis.sort_values(
    ["arrival_date_year", "arrival_date_month"]
)

print("\nMonthly analysis in chronological order:")
print(monthly_analysis.round(2).to_string(index=False))


# Save calculated indicators for the dashboard

from pathlib import Path

output_folder = Path("Data/indicators")
output_folder.mkdir(parents=True, exist_ok=True)

segment_analysis.to_csv(output_folder / "cancellation_by_segment.csv")
lead_time_analysis.to_csv(output_folder / "cancellation_by_lead_time.csv")
deposit_analysis.to_csv(output_folder / "cancellation_by_deposit.csv")
customer_analysis.to_csv(output_folder / "cancellation_by_customer.csv")

hotel_analysis.to_csv(output_folder / "booking_by_hotel.csv")
segment_booking_analysis.to_csv(output_folder / "booking_by_segment.csv")
monthly_analysis.to_csv(output_folder / "booking_by_month.csv", index=False)

print("\nAll indicator files saved successfully!")