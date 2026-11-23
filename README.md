# Hotel-booking-analytics
Hotel booking analytics: data examination and cleaning, business indicators, interactive Dash dashboard, GitHub Actions automation and batch data processing.

## Kaggle data set link:
    https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

## Questions to solve:

**Q1: Which booking characteristics are associated with higher cancellation rates, and how do these patterns differ across reservation segments?**

Indicators: Cancellation rate by market segment, lead-time category, deposit type and customer type; booking counts per segment; and relative cancellation risk compared with the overall cancellation rate.

**Q2: How do booking demand, average daily rates and estimated booking value vary across time, hotel types and market segments?**

Indicators: Booking volume, average daily rate (ADR), average length of stay, estimated booking value (ADR × total nights), estimated non-cancelled booking value and total booked nights, compared by month, hotel type and market segment.

## Data examination and cleaning

**Script:** `Scripts/data_exam.py`
**Output:** `data/clean/hotel_bookings_clean.csv`

The original dataset has [119,390] bookings and [32] columns. After cleaning, [86,520] bookings remain.

### What we did and why

**1. Filled in missing values**

Some columns have empty cells. We handled each column separately instead of deleting every row with a gap, because that would have removed most of the data.

- `children` ([4] empty): set to 0, since an empty value means no children were declared.
- `country` ([488] empty): set to "Unknown", so we keep the booking without guessing a country.
- `agent` and `company` (many empty): set to 0, meaning "no agent" or "no company". The `company` column is almost entirely empty, so dropping those rows was not an option.

**2. Merged labels that mean the same thing**

- In `meal`, "Undefined" and "SC" both mean "no meal package", so we merged them into "SC".
- "Undefined" in `market_segment` and `distribution_channel` was renamed "Other".
- Some countries appeared under two codes (for example CN and CHN). We kept one code, so each country is counted as a single group.
- We removed extra spaces in text values, so "Resort Hotel" and "Resort Hotel " are not treated as different hotels.

**3. Removed records that make no sense**

- Bookings with no guests at all (0 adults, 0 children, 0 babies).
- Stays of 0 nights, because they generate no revenue or occupancy.
- Bookings with a negative price (`adr`).
- Extreme prices, such as an `adr` of 5400. We removed the top 0.1% of prices, because a few extreme values would distort the average price and revenue.

**4. Removed duplicates**

We removed [32,870]  duplicate rows. Keeping them would count the same booking twice and bias the cancellation rate.

**5. Fixed the date format**

`reservation_status_date` was stored as text. We converted it to a real date so it can be sorted and filtered by time.

### Important: data leakage

The columns `reservation_status` and `reservation_status_date` reveal whether a booking was cancelled. We keep them in the cleaned file, but we will **not** use them in the machine learning model, otherwise the model would simply read the answer.

### How to run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data_exam.py
```
