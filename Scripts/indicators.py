import pandas as pd

df = pd.read_csv("data/clean/hotel_bookings_clean.csv")

## Calculate the overall cancellation rate
cancellation_rate = df["is_canceled"].mean() * 100

print("Overall cancellation rate:", round(cancellation_rate, 2), "%")

