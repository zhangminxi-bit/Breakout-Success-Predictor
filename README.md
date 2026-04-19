# Predicting Stock Breakout Success
### ECON 3916 — Final Project

Can pre-breakout price and volume characteristics predict whether a stock's breakout will succeed or fail at the moment the signal fires?

---

## Project Overview

This project builds a binary classifier that predicts whether a stock breakout will succeed (positive 5-day return with no reversal) or fail (price closes back below the breakout level within 5 days). The model uses only information available at the moment the signal fires — no future price data is used.

**Tickers:** AAPL, MSFT, NVDA, XLF, XLV  
**Data:** 5 years of daily OHLCV data from Yahoo Finance  
**Models:** Logistic Regression (baseline) + Random Forest  
**Task:** Binary classification

---

## Repository Structure

```
├── breakout_final_project.ipynb   # Main notebook — all analysis
├── app.py                         # Streamlit interactive app
├── model.pkl                      # Saved logistic regression model
├── scaler.pkl                     # Saved StandardScaler
├── requirements.txt               # Python dependencies
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

### 3. Run the notebook

Open `breakout_final_project.ipynb` in Jupyter or Google Colab and run all cells top to bottom. The notebook will:

- Download 5 years of daily price data from Yahoo Finance automatically via `yfinance` — no API key required
- Detect breakout events using a 20-day rolling high with a 5-day cooldown filter
- Engineer 6 features from the price and volume history
- Train and evaluate both models
- Save `model.pkl` and `scaler.pkl`

> **Note:** Data is downloaded live from Yahoo Finance each time the notebook runs. Results may differ slightly from the original submission if Yahoo Finance updates historical data.

### 4. Run the Streamlit app (optional)

```bash
streamlit run app.py
```

---

## Data

No data file is included in this repository. Data is downloaded programmatically from Yahoo Finance using the `yfinance` library (free, no account required).

```python
import yfinance as yf
raw = yf.download(['AAPL', 'MSFT', 'NVDA', 'XLF', 'XLV'],
                  period='5y', interval='1d', auto_adjust=True)
```

**Dataset construction:**
- Raw source: Yahoo Finance daily OHLCV (https://finance.yahoo.com)
- Breakout definition: daily close above the highest close of the prior 20 trading days
- Cooldown filter: 5 days between consecutive signals to avoid counting the same trend move multiple times
- Final N: ~125–250 breakout events across all five tickers
- Target variable: `success = 1` if ret_5d > 0 AND price never closed below the breakout level within 5 days, else `success = 0`

---

## Features

All features are computed at the moment of the breakout signal using only prior information — no look-ahead bias.

| Feature | Description |
|---|---|
| `breakout_strength` | How far above the 20-day high did price close (%) |
| `volume_ratio` | Today's volume divided by 20-day average volume |
| `ret_prior_5d` | Return over the 5 days leading into the breakout |
| `ret_prior_20d` | Return over the 20 days leading into the breakout |
| `volatility_20d` | Rolling 20-day standard deviation of daily returns |
| `atr_ratio` | Average True Range divided by Close — normalized intraday noise |
| `ticker_*` | One-hot encoded ticker dummies (drop_first=True) |

---

## Results

| Model | CV F1 (mean ± std) | Test Accuracy | Test Precision | Test Recall | Test F1 |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |

> Fill in the table with your actual results after running the notebook.

---

## Key Design Decisions

**Temporal train/test split** — data is split by date (70% earlier events for training, 30% more recent events for testing), not randomly. Random splitting on financial time series creates look-ahead bias by allowing future market conditions to leak into training.

**Outliers retained** — extreme values in `volume_ratio` (large volume surges) are kept because they represent genuine market events that are likely the most informative observations for predicting breakout success.

**Prediction, not causation** — this model predicts which breakouts are likely to succeed based on historical patterns. It does not claim that high volume *causes* success. The same pattern may not persist under different market regimes.

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
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Author

Frank  
Northeastern University — ECON 3916  
Spring 2026
