# Hotel Booking Analytics

End-to-end analytics project on hotel booking data: data examination and cleaning, business indicators, an interactive Dash dashboard, automated batch processing with GitHub Actions, and a machine learning model that predicts booking cancellations.

**Team:** [add names of group members]
**Course:** Data Science Toolbox, MSc Data & Business Analytics, Rennes School of Business

## Dataset

Kaggle: Hotel Booking Demand
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

The raw file contains 119,390 bookings and 32 columns for a city hotel and a resort hotel (arrival dates, lead time, length of stay, price per night, market segment, deposit type, cancellation status and more).

## Business questions

**Q1: Which booking characteristics are associated with higher cancellation rates, and how do these patterns differ across reservation segments?**

Indicators: overall cancellation rate; cancellation rate and booking count by market segment, lead-time category, deposit type and customer type.

**Q2: How do booking demand, average daily rates and estimated booking value vary across time, hotel types and market segments?**

Indicators: booking volume, average daily rate (ADR), average length of stay and estimated booking value (ADR × total nights), compared by hotel type, market segment and month (year and month).

## Project structure

```
hotel-booking-analytics/
├── .github/workflows/            # GitHub Actions pipeline
├── Data/
│   ├── raw/                      # original Kaggle file (hotel_bookings.csv)
│   ├── clean/                    # hotel_bookings_clean.csv and hotel_bookings_scored.csv
│   ├── indicators/               # 7 indicator tables read by the dashboard
│   └── batches/
│       ├── incoming/             # new batches waiting to be processed
│       └── processed/            # batches already added to the dataset
├── Scripts/
│   ├── data_exam.py              # examine and clean the raw data
│   ├── indicators.py             # calculate the business indicators
│   ├── dashboard.py              # interactive Dash dashboard
│   ├── simulate_batch.py         # temporary batch generator (until the course BatchSimulator is available)
│   ├── process_batch.py          # clean new batches and append them to the dataset
│   ├── ml_model.py               # compare models, score bookings with a cancellation probability
│   └── generate_report.py        # build the HTML report
├── Local/
│   └── run_all.py                # run the pipeline locally with one command
├── models/                       # cancel_model.joblib (saved model)
├── reports/                      # model_metrics.json and report.html
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3 and Git.

```bash
git clone https://github.com/KarnikaRSB/hotel-booking-analytics.git
cd hotel-booking-analytics
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Make sure the Kaggle file is at `Data/raw/hotel_bookings.csv`.

## How to run

Run every script **from the repository root**, so relative paths resolve correctly.

| Step | Command | Result |
|---|---|---|
| 1. Examine and clean the raw data | `python Scripts/data_exam.py` | `Data/clean/hotel_bookings_clean.csv` |
| 2. Calculate indicators | `python Scripts/indicators.py` | 7 tables in `Data/indicators/` |
| 3. Start the dashboard | `python Scripts/dashboard.py` | dashboard at http://127.0.0.1:8050/ |
| 4. Create a test batch (temporary) | `python Scripts/simulate_batch.py 500` | a CSV in `Data/batches/incoming/` |
| 5. Process new batches | `python Scripts/process_batch.py` | clean dataset updated, batch moved to `processed/` |
| 6. Train the model and score bookings | `python Scripts/ml_model.py` | `models/cancel_model.joblib`, `Data/clean/hotel_bookings_scored.csv`, `reports/model_metrics.json` |
| 7. Build the report | `python Scripts/generate_report.py` | `reports/report.html` |

To run the whole pipeline in order with one command:

```bash
python Local/run_all.py
```

`run_all.py` runs the scripts one after the other, stops at the first error, and finishes by starting the dashboard, which keeps running until you stop it with Ctrl+C. To start only the dashboard, use step 3.

**Note:** `data_exam.py` rebuilds the clean dataset from the raw file, so running it discards bookings that earlier batches added. Use it for the initial setup, and use steps 5 to 7 for new data.

## 1. Data examination and cleaning

**Script:** `Scripts/data_exam.py`
**Output:** `Data/clean/hotel_bookings_clean.csv`

The original dataset has 119,390 bookings and 32 columns. After cleaning, 86,520 bookings remain, so the cleaning steps below removed 32,870 rows in total.

The script first examines the data (shape, first and last rows, column types, minimum and maximum values, missing values, duplicates, values of the categorical columns, possible misspellings and suspicious records), then cleans it with the `clean()` function. The same `clean()` function is reused for every new batch.

### What we did and why

**1. Filled in missing values**

Some columns have empty cells. We handled each column separately instead of deleting every row with a gap, because that would have removed most of the data.

- `children` (4 empty): set to 0, since an empty value means no children were declared.
- `country` (488 empty): set to "Unknown", so we keep the booking without guessing a country.
- `agent` and `company` (many empty): set to 0, meaning "no agent" or "no company". The `company` column is almost entirely empty, so dropping those rows was not an option.

**2. Merged labels that mean the same thing**

- In `meal`, "Undefined" and "SC" both mean "no meal package", so we merged them into "SC".
- "Undefined" in `market_segment` and `distribution_channel` was renamed "Other".
- Some countries appeared under two codes (for example CN and CHN). We kept one code, so each country is counted as a single group.
- We removed extra spaces in column names and text values, so "Resort Hotel" and "Resort Hotel " are not treated as different hotels.

**3. Removed records that make no sense**

- Bookings with no guests at all (0 adults, 0 children, 0 babies).
- Stays of 0 nights, because they generate no revenue or occupancy.
- Bookings with a negative price (`adr`).
- Extreme prices, such as an `adr` of 5400. We removed the top 0.1% of prices, because a few extreme values would distort the average price and revenue.

**4. Removed duplicates**

We removed exact duplicate rows (the script prints how many the raw file contains). Keeping them would count the same booking twice and bias the cancellation rate.

**5. Fixed the date format**

`reservation_status_date` was stored as text. We converted it to a real date so it can be sorted and filtered by time.

### Important: data leakage

The columns `reservation_status` and `reservation_status_date` reveal whether a booking was cancelled. We keep them in the cleaned file, but we do **not** use them in the machine learning model, otherwise the model would simply read the answer.

## 2. Indicators

**Script:** `Scripts/indicators.py`
**Output:** seven indicator tables in `Data/indicators/`

| File | Content | Question |
|---|---|---|
| `cancellation_by_segment.csv` | booking count and cancellation rate (%) per market segment | Q1 |
| `cancellation_by_lead_time.csv` | the same, per lead-time category | Q1 |
| `cancellation_by_deposit.csv` | the same, per deposit type | Q1 |
| `cancellation_by_customer.csv` | the same, per customer type | Q1 |
| `booking_by_hotel.csv` | booking count, average ADR, average nights and average booking value per hotel | Q2 |
| `booking_by_segment.csv` | the same, per market segment | Q2 |
| `booking_by_month.csv` | the same, per year and month in chronological order | Q2 |

Definitions:

- **Cancellation rate** = share of bookings with `is_canceled = 1`, shown in percent.
- **Lead time** = days between booking and arrival, grouped into six categories: 0-7, 8-30, 31-90, 91-180, 181-365 and over 365 days.
- **Total nights** = weekend nights + week nights.
- **Estimated booking value** = ADR × total nights. It is an estimate, not accounting revenue.

## 3. Dashboard

**Script:** `Scripts/dashboard.py`

```bash
python Scripts/dashboard.py
```

Open http://127.0.0.1:8050/ in your browser (the script also tries to open it automatically). Stop it with Ctrl+C in the terminal.

The dashboard reads the seven indicator tables and is organised as a story:

- **Overview KPIs:** total bookings, overall cancellation rate, weighted average ADR and average estimated booking value.
- **Q1: Cancellation analysis.** Charts of cancellation rate by market segment, lead time, deposit type and customer type. A dropdown switches between cancellation rate and booking count. Insight cards summarise the main findings.
- **Q2: Booking demand, ADR and estimated booking value.** Hotel and market-segment comparisons with a metric dropdown (booking demand, average ADR, estimated booking value, average nights), plus a monthly trend chart with a year filter.

At start-up the dashboard checks that all seven files and their required columns exist, and stops with a clear message if something is missing.

## 4. Batch processing

New bookings arrive as CSV files in `Data/batches/incoming/`. `Scripts/process_batch.py` handles every file it finds:

1. cleans it with the same `clean()` function as the original data,
2. appends it to `Data/clean/hotel_bookings_clean.csv` and removes duplicates,
3. moves it to `Data/batches/processed/` with a timestamp, so it is never processed twice.

If there is no new batch, the script says so and changes nothing.

Until the course BatchSimulator is available, `Scripts/simulate_batch.py` creates a temporary batch by sampling rows from the raw dataset and slightly changing lead time and ADR (default 500 rows, for example `python Scripts/simulate_batch.py 1000`). It will be replaced by the BatchSimulator.

## 5. Machine learning: cancellation prediction

**Script:** `Scripts/ml_model.py`

The model predicts whether a booking will be cancelled and adds two indicators to the data: `cancel_probability` (0 to 1) and `risk_level` (Low up to 0.3, Medium from 0.3 to 0.6, High above 0.6).

- **Target:** `is_canceled`
- **Models compared:** Logistic Regression (simple baseline) and Gradient Boosting. The one with the higher ROC AUC on the test set is kept.
- **Validation:** 80/20 train/test split, stratified so both sets keep the same cancellation rate.
- **Metrics:** ROC AUC, accuracy, precision, recall and F1, saved in `reports/model_metrics.json`.
- **Features (known at booking time):** hotel, meal, market segment, distribution channel, deposit type, customer type, reserved room type, arrival month, lead time, total nights, total guests, ADR, repeated guest, previous cancellations, previous non-cancelled bookings, days in waiting list, parking spaces and special requests.
- **Excluded to avoid data leakage:** `reservation_status` and `reservation_status_date` (they reveal the outcome), `assigned_room_type` (decided at check-in), `booking_changes` (can change until arrival) and `country`.
- **Honest scores:** each booking in `hotel_bookings_scored.csv` is scored by a model that did not see it during training (5-fold cross-validation). The final model, trained on all bookings, is saved to `models/cancel_model.joblib` for scoring future batches.
- **Interpretation:** permutation importance shows which inputs drive the prediction (top 10 in the metrics file and the report).

**Results:** the selected model, its ROC AUC, accuracy, precision, recall and F1, and the comparison with the other model are saved in `reports/model_metrics.json` after every run and shown in `reports/report.html`.

The ML output is shown in `reports/report.html`: headline numbers (bookings, cancellation rate, average ADR, high-risk bookings), the model comparison table, bookings per predicted risk level, predicted probability versus actual outcome, the main drivers of the prediction and the cancellation rate by market segment.

Limitations: cancellation behaviour changes over time and the data covers only two hotels, which is why the pipeline retrains the model on every run.

## 6. Automation with GitHub Actions

**Workflow:** `.github/workflows/` (data pipeline)

| Trigger | Why |
|---|---|
| Push to `main` that changes `Data/batches/incoming/**`, `Scripts/**` or `requirements.txt` | A new batch or a code change refreshes the results. |
| Every Monday at 06:00 UTC | Simulates the weekly arrival of new bookings. |
| Manual "Run workflow" button | Allows a rerun at any time from the Actions tab. |

Steps of each run:

1. Check out the latest `main` and install Python and the packages from `requirements.txt`.
2. On scheduled runs only, create a temporary batch with `simulate_batch.py`.
3. Process new batches (`process_batch.py`).
4. Retrain the model and score all bookings (`ml_model.py`).
5. Build the report (`generate_report.py`) and upload it as a downloadable artifact.
6. Commit the updated data, model and report back to the repository.

Design choices:

- **Concurrency group:** two pipeline runs never overlap, so they cannot overwrite each other's results.
- **`[skip ci]` in the automated commit message:** the bot's own commit does not trigger a new run.
- **Report as an artifact:** the report can be downloaded from the run page.
- **The dashboard is not a workflow step.** It is a server that runs until stopped, so the workflow only keeps its input data up to date.

## Teamwork with Git and GitHub

- `main` always contains working code.
- Each task is developed on its own branch (for example `feature/data-exam`) and merged through a Pull Request.
- Commit messages describe what changed and why.

## Known limitations

- The indicator tables used by the dashboard are created by `Scripts/indicators.py`. The automated pipeline updates the clean dataset, the model, the scores and the report, so after new batches run `python Scripts/indicators.py` to refresh the dashboard indicators.
- The dashboard shows the cancellation and booking indicators. The machine learning results are presented in `reports/report.html`.

- ADR is a price per night, so estimated booking value is an approximation, and the dataset does not state a currency.
- Nights booked is only a proxy for demand. The dataset has no room inventory, so a true occupancy rate cannot be calculated.
- The dataset covers only two hotels and a limited period.

## Tech stack

Python, pandas, scikit-learn, Plotly and Dash, Git and GitHub, GitHub Actions, VS Code.
