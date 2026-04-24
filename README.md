# Predicting Stock Breakout Success
### ECON 3916 — Final Project | Northeastern University | Spring 2026

Can pre-breakout price and volume characteristics predict whether a stock's breakout will succeed or fail at the moment the signal fires?

---

## Project Overview

This project builds a binary classifier that predicts whether a stock breakout will succeed (positive 5-day return) or fail at the moment the breakout signal fires. A breakout is defined as a daily close above the highest closing price of the prior 20 trading days, with a 5-day cooldown filter to prevent consecutive signals from the same trend move being counted multiple times.

The project has two components:

**Descriptive analysis** — measures what historically happens after breakouts across 5 tickers (AAPL, MSFT, NVDA, XLF, XLV), producing an Excel workbook with event tables, summary statistics, and annotated charts.

**ML prediction** — trains a binary classifier on 10 tickers to predict at the moment of signal whether the breakout will succeed, using only information available at that moment (no look-ahead bias).

**Key finding:** Both models scored ROC-AUC approximately 0.50, statistically indistinguishable from random guessing. This null result is consistent with the semi-strong form of the Efficient Market Hypothesis — publicly observable price and volume patterns in liquid large-cap stocks are already priced in at the moment they become detectable.

---

## Repository Structure

```
├── final_project.py               # Complete ML pipeline (all code, no markdown)
├── app.py                         # Streamlit interactive app
├── model.pkl                      # Saved logistic regression model
├── scaler.pkl                     # Saved StandardScaler
├── requirements.txt               # Python dependencies
├── project_report.docx            # Final report (Word)
├── ai_appendix.docx               # AI Methodology Appendix (Word)
└── README.md                      # This file
```

---

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python final_project.py
```

The script will automatically download 5 years of daily price data from Yahoo Finance and run end-to-end. No API key or account required. Runtime is approximately 3-5 minutes on a standard laptop CPU.

### 4. Run the Streamlit app (optional)

```bash
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Data

No data file is included. Data is downloaded programmatically at runtime from Yahoo Finance via `yfinance`.

**Tickers:** AAPL, MSFT, NVDA, XLF, XLV, AMZN, GOOGL, META, JPM, UNH

**Period:** 5 years of daily OHLCV data (~1,256 trading days per ticker)

**Raw rows:** 6,280 (before event filtering)

**Final N:** 683 breakout events (after 20-day lookback, 5-day cooldown, and .dropna())

**Breakout definition:** Daily close above the highest close of the prior 20 trading days, with a 5-day cooldown filter suppressing consecutive signals from the same trend move.

**Target variable:** success = 1 if ret_5d > 0, else success = 0

**Access date:** April 2026

---

## Features

All features are computed at the moment of the breakout signal (t = 0) using only prior information. No future price data is used anywhere in the pipeline.

| Feature | Description | Why it matters |
|---|---|---|
| `breakout_strength` | (Close minus 20d high) / 20d high | How convincingly price cleared resistance |
| `volume_ratio` | Today's volume / 20-day avg volume | Whether buying conviction supported the move |
| `ret_prior_5d` | Return over 5 days leading into breakout | Momentum context entering the signal |
| `ret_prior_20d` | Return over 20 days leading into breakout | Broader trend context |
| `volatility_20d` | Rolling 20-day std of daily returns | Signal noise level |
| `atr_ratio` | Average True Range / Close | Normalized intraday noise |
| `ticker_*` | One-hot encoded ticker dummies (drop_first=True) | Structural differences across instruments |

---

## Methodology

**Train/test split:** Temporal (not random) — 70% earlier events for training, 30% more recent events for testing. Random splitting on financial time series creates look-ahead bias and is methodologically incorrect.

**Standardization:** StandardScaler fit on training data only, then applied to test data via .transform().

**Models:**
- Model 1: LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42) — baseline; chosen for interpretability and low variance on small N
- Model 2: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42) — captures non-linear feature interactions

**Cross-validation:** 5-fold CV on training set, scoring metric: F1

**Class imbalance handling:** class_weight='balanced' on both models. The initial run without this produced CV F1 = 0.00 for logistic regression (model collapsed to majority class prediction).

---

## Results

| Model | CV F1 (mean) | CV F1 (std) | Test Accuracy | Test Precision | Test Recall | Test F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.5226 | 0.1117 | 0.5024 | 0.5700 | 0.4914 | 0.5278 | 0.503 |
| Random Forest | 0.6513 | 0.0290 | 0.5512 | 0.5811 | 0.7414 | 0.6515 | 0.492 |

**Interpretation:** Despite moderate F1 scores achieved through class-weighted prediction, ROC-AUC approximately 0.50 for both models indicates that the predicted probability rankings carry no real discriminative signal. The F1 improvement reflects the models predicting both classes more evenly — not genuine predictive ability over unseen data.

---

## Key Design Decisions

**Temporal split over random split** — financial time series must be split by date. Random splitting allows future market conditions to leak into training, inflating all test metrics. This was corrected after the initial suggestion of train_test_split(random_state=42) was identified as introducing look-ahead bias.

**Target relaxation** — the original target required (ret_5d > 0) AND (failed_5d == 0), which was too strict and created severe class imbalance. Relaxing to ret_5d > 0 alone improved the success rate from heavily skewed to approximately 55/45, enabling meaningful model training.

**Cooldown filter** — a 5-day cooldown is applied after each breakout signal to avoid counting the same upward trend move multiple times. Without it, a 10-day NVDA rally would produce 10 breakout events that are structurally one observation.

**Outliers retained** — extreme values in volume_ratio (large volume surges) are not removed. They represent genuine market events and StandardScaler normalizes scale without discarding observations.

**Null result framing** — ROC-AUC approximately 0.50 is interpreted as a genuine finding consistent with the semi-strong Efficient Market Hypothesis, not a model failure requiring further tuning.

---

## Limitations

- Small N (~683 events) — wide confidence intervals on all metrics
- 5-year window covers limited market regimes
- Features restricted to price and volume — no fundamental, macro, or sentiment data
- ROC-AUC approximately 0.50 indicates no genuine discriminative ability in the current feature set

---

## Potential Extensions

- Add market-wide features: VIX level, SPY return on breakout day
- Add sector momentum: sector ETF return in the prior week
- Add earnings proximity: days until next earnings announcement
- Expand ticker universe to increase event count
- Test across different market regime windows (2020 crash, 2022 bear, 2023-24 bull) separately

---

## Requirements

```
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
streamlit>=1.31.0
openpyxl>=3.1.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Streamlit App

The app loads the saved model.pkl and scaler.pkl and allows a user to input six breakout characteristics via sliders. It returns a predicted probability of success and a plain-English signal quality label (STRONG / MODERATE / WEAK).

**Important:** The feature input must match the scaler exactly — 6 real features + 9 ticker dummies = 15 total inputs. If you retrain the model with a different ticker set, regenerate model.pkl and scaler.pkl and update the dummy count in app.py accordingly.

**Live deployment:** [Add your Streamlit Cloud URL here after deployment]

---

## Author

Frank Zhang
Northeastern University — ECON 3916
Spring 2026
