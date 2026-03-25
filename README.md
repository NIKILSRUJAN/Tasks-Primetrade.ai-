# Tasks-Primetrade.ai- 📊

A comprehensive data analysis project featuring deep dives into cryptocurrency market sentiment and trading performance analytics.

---

## 📋 Project Overview

This repository contains two independent but complementary analysis tasks:

- **TASK-1**: Fear & Greed Index Analysis - Temporal sentiment analysis of Bitcoin market psychology
- **TASK-2**: Historical Trader Performance - Behavioral segmentation and profitability analysis

Both projects provide actionable insights through statistical analysis, clustering, and advanced visualizations.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pandas, numpy, matplotlib, scikit-learn

### Installation

```bash
# Clone the repository
git clone https://github.com/NIKILSRUJAN/Tasks-Primetrade.ai-.git
cd Tasks-Primetrade.ai-

# Install dependencies (optional - scripts will prompt if needed)
pip install pandas numpy matplotlib scikit-learn
```

### Running the Analysis

```bash
# TASK-1: Fear & Greed Index Analysis
cd TASK-1
python analysis.py

# TASK-2: Historical Trader Performance
cd ../TASK-2
python analysis.py
```

---

## 📊 TASK-1: Fear & Greed Index Analysis

### Overview
Comprehensive analysis of the Bitcoin Fear & Greed Index from 2018-2025, examining market sentiment patterns, transition probabilities, and temporal trends.

### Objective
- Analyze historical Fear/Greed sentiment distribution
- Identify sentiment persistence and transition patterns
- Compute momentum indicators and rolling averages
- Segment data by classification, temporal features, and extremes

### Input Data
- **File**: `TASK-1/fear_greed_index.csv`
- **Records**: 2,644 days of data
- **Date Range**: 2018-02-01 to 2025-05-02
- **Columns**: timestamp, value, classification, date

### Key Metrics

| Metric | Value |
|--------|-------|
| **Fear Days** | 1,289 (48.8%) |
| **Greed Days** | 959 (36.3%) |
| **Neutral Days** | 396 (15.0%) |
| **Fear Persistence** | 93.3% (stays in Fear) |
| **Greed Persistence** | 91.3% (stays in Greed) |
| **Longest Fear Streak** | 151 consecutive days |
| **Longest Greed Streak** | 97 consecutive days |

### Output Files

```
TASK-1/outputs/
├── cleaned_fear_greed_index.csv          # Processed dataset with features
├── task1_fig1_overview.png               # Distribution & classification charts
├── task1_fig2_trends.png                 # Time series & momentum analysis
└── task1_fig3_yearly.png                 # Yearly sentiment breakdown
```

### Visualizations

1. **Figure 1 - Overview & Distribution**
   - Classification frequency bar chart
   - F&G value distribution by class histogram
   - Sentiment transition matrix heatmap
   - Monthly average heatmap by year

2. **Figure 2 - Trends & Momentum**
   - Full time series with 7-day and 30-day moving averages
   - Yearly average bar chart
   - Consecutive day streak density distribution

3. **Figure 3 - Yearly Sentiment Breakdown**
   - Stacked bar chart: yearly sentiment composition
   - Box plot: F&G value distribution by year

### Key Insights

✓ **High Sentiment Autocorrelation** - Market sentiment exhibits strong momentum (93% persistence), making it useful for predictive models

✓ **Fear Dominance** - Nearly half of all trading days are classified as "Fear," suggesting bearish market conditions are more frequent

✓ **Extreme Events** - 508 days of Extreme Fear (<25) and 279 days of Extreme Greed (>75), indicating rare but impactful market movements

✓ **Cyclical Patterns** - Clear multi-year sentiment cycles aligned with Bitcoin market cycles

---

## 💰 TASK-2: Historical Trader Performance

### Overview
Analysis of 211,224 trades from 32 unique trading accounts, examining profitability, behavioral patterns, and trader archetypes using unsupervised clustering.

### Objective
- Quantify account-level profitability and trading activity
- Identify trader behavioral patterns and performance drivers
- Segment traders into archetypal groups using K-Means clustering
- Analyze directional bias, position sizing, and temporal patterns

### Input Data
- **File**: `TASK-2/historical_data.csv`
- **Records**: 211,224 trades
- **Date Range**: 2023-05-01 to 2025-05-01
- **Columns**: Account, Timestamp, Side, Coin, Size USD, Closed PnL, Fee, Direction, etc.

### Key Metrics

| Metric | Value |
|--------|-------|
| **Gross PnL** | $10,296,958.94 |
| **Net PnL (After Fees)** | $10,051,101.22 |
| **Total Fees** | $245,857.72 |
| **Profit Factor** | 4.49x |
| **Profitable Accounts** | 29 of 32 (90.6%) |
| **Long Avg PnL** | $102.81 |
| **Short Avg PnL** | $96.41 |
| **HYPE Trades** | 68,005 (32.2% of total) |

### Trading Archetypes

Four distinct trader profiles identified through K-Means clustering:

1. **Whale Traders** 🐋
   - Highest total PnL
   - Large position sizes
   - Lower trade frequency
   - Strategic, longer-term positions

2. **HFT Bots** ⚡
   - Highest trade frequency
   - Smaller average position sizes
   - Consistent daily activity
   - Algorithmic execution style

3. **Precision Players** 🎯
   - Highest win rate
   - Medium position sizes
   - Selective trade approach
   - Technical/analytical focus

4. **Casual Traders** 👥
   - Diverse trading patterns
   - Variable position sizes
   - Lower activity levels
   - Retail/hobby traders

### Output Files

```
TASK-2/outputs/
├── task2_fig1_eda.png                    # EDA overview charts
├── task2_fig2_pnl.png                    # PnL deep dive analysis
├── task2_fig3_behavior.png               # Trader behavior patterns
└── task2_fig4_accounts.png               # Archetype segmentation
```

### Visualizations

1. **Figure 1 - EDA Overview**
   - Trade count by direction
   - Top 10 coins by volume
   - Position size distribution pie chart
   - Hourly trading pattern
   - Day-of-week analysis
   - Closed PnL distribution histogram

2. **Figure 2 - PnL Analysis**
   - Daily aggregate PnL with 30-day MA
   - Top & bottom coins by total PnL
   - Win rate by position size
   - Account performance scatter

3. **Figure 3 - Trader Behavior**
   - Long vs Short average PnL comparison
   - Total fees paid by account
   - Gross vs Net PnL breakdown

4. **Figure 4 - Trader Archetypes**
   - Average total PnL by archetype
   - Trades/Day vs Average Position Size scatter plot

### Key Insights

✓ **Exceptional Profitability** - 4.49x profit factor indicates highly skilled or selective trading (industry benchmark is ~2.0x)

✓ **Fee Efficiency** - Fees represent only 2.3% of gross + fees combined, demonstrating good trade management

✓ **Directional Bias** - Slight long bias ($102.81 vs $96.41 avg), but both directions profitable

✓ **Diversified Success** - 90.6% of accounts are profitable, suggesting systematic edge rather than luck

✓ **HYPE Dominance** - Single coin represents 32.2% of all trades, indicating concentrated speculative activity

---

## 📁 Project Structure

```
Tasks-Primetrade.ai-/
│
├── README.md                              # This file
│
├── TASK-1/
│   ├── analysis.py                        # Main analysis script
│   ├── fear_greed_index.csv               # Source dataset
│   └── outputs/
│       ├── cleaned_fear_greed_index.csv   # Processed data
│       ├── task1_fig1_overview.png
│       ├── task1_fig2_trends.png
│       └── task1_fig3_yearly.png
│
├── TASK-2/
│   ├── analysis.py                        # Main analysis script
│   ├── historical_data.csv                # Source dataset
│   └── outputs/
│       ├── task2_fig1_eda.png
│       ├── task2_fig2_pnl.png
│       ├── task2_fig3_behavior.png
│       └── task2_fig4_accounts.png
│
├── TASK-1_Insights.pdf                    # Detailed insights report
└── TASK-2_Insights.pdf                    # Detailed insights report
```

---

## 🛠️ Technical Stack

| Component | Tool/Library |
|-----------|-------------|
| Data Processing | pandas, numpy |
| Analysis | scikit-learn (clustering), scipy (statistics) |
| Visualization | matplotlib |
| Dataset Format | CSV |
| Language | Python 3.12 |

---

## 📈 Analysis Methodology

### TASK-1 Approach
1. **Data Cleaning** - Handle missing values, validate timestamps
2. **Feature Engineering** - Temporal features (year, month, quarter, hour, day_of_week)
3. **Rolling Statistics** - Compute 7-day and 30-day moving averages
4. **Transition Analysis** - Build sentiment transition matrix
5. **Streak Analysis** - Compute consecutive day streaks
6. **Visualization** - Multi-faceted charting and reporting

### TASK-2 Approach
1. **Data Cleaning** - Normalize timestamps, validate trade data
2. **Account Aggregation** - Compute PnL, win rate, trading frequency per account
3. **Unsupervised Learning** - K-Means clustering with 4 clusters
4. **Feature Scaling** - StandardScaler for clustering features
5. **Archetype Labeling** - Dynamic labeling based on cluster characteristics
6. **Behavior Analysis** - Directional bias, position sizing, temporal patterns

---

## 🔍 How to Use the Output Data

### TASK-1 Cleaned Dataset
The cleaned CSV includes:
- Temporal features for trend analysis
- Rolling averages for momentum assessment
- Streak lengths for persistence analysis
- Can be imported into ML models for prediction tasks

### TASK-2 Account Segmentation
Use the archetype classifications for:
- Portfolio management (allocate based on trader type)
- Risk assessment (profile-based sizing)
- Signal generation (combine account types)
- Performance benchmarking

---

## 📋 Requirements

```
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.5.0
scipy>=1.10.0
```

Install via:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Submit bug reports
- Propose feature enhancements
- Optimize analysis workflows
- Add additional visualizations or metrics

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👤 Author

**NIKILSRUJAN**

For questions or feedback, please open an issue on GitHub.

---

## 📚 References & Resources

- [Bitcoin Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
- [K-Means Clustering Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Visualization Guide](https://matplotlib.org/)

---

**Last Updated**: March 25, 2026  
**Status**: ✅ Active & Maintained