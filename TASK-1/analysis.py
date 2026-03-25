# 1: SETUP

import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg') # Keeps plots from rendering inline. Comment out if you want to see them in Colab!
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Create outputs directory
OUTPUT_DIR = Path('./outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Directory ready: {OUTPUT_DIR}")

# Global Plotting Configuration
FIG_BG = '#0D1117'
AX_BG = '#161B22'
TEXT = '#E6EDF3'
GRID = '#21262D'

COLORS = {
    'Extreme Fear': '#8B0000', 
    'Fear': '#E84040',
    'Neutral': '#7F8C8D',
    'Greed': '#27AE60', 
    'Extreme Greed': '#145A32'
}
SENT_COLORS = {'Fear': '#E84040', 'Neutral': '#7F8C8D', 'Greed': '#27AE60'}

plt.rcParams.update({
    'figure.facecolor': FIG_BG, 
    'axes.facecolor': AX_BG,
    'axes.edgecolor': GRID, 
    'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 
    'ytick.color': TEXT,
    'text.color': TEXT, 
    'grid.color': GRID, 
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
})

# 2: DATA LOADING & PREPROCESSING

print("=" * 60)
print("TASK 1 — FEAR/GREED INDEX ANALYSIS")
print("=" * 60)

file_path = 'fear_greed_index.csv'

# Ensure file exists before proceeding
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Please upload '{file_path}' to the Colab environment.")

df = pd.read_csv(file_path)

print(f"\n[1] Raw Dataset Overview")
print(f"    Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Columns: {list(df.columns)}\n")

print("─" * 40)
print("[2] DATA CLEANING & FEATURE ENGINEERING")

# Handle Missing values & Duplicates
print(f"    Missing values :\n{df.isnull().sum().to_string()}")
print(f"    Duplicate rows : {df.duplicated().sum()}")

# Time formatting
df['date'] = pd.to_datetime(df['date'])
df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

# Temporal features
df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df['month_name'] = df['date'].dt.strftime('%b')
df['quarter'] = df['date'].dt.to_period('Q').astype(str)

# Consolidated Sentiment Mapping
df['sentiment'] = df['classification'].map({
    'Extreme Fear': 'Fear',
    'Fear':         'Fear',
    'Neutral':      'Neutral',
    'Greed':        'Greed',
    'Extreme Greed':'Greed',
})

print(f"\n    Date range     : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"    Total days     : {len(df)}")
print(f"    Years covered  : {sorted(df['year'].unique())}")

# 3: DESCRIPTIVE STATISTICS & MOMENTUM

print("─" * 40)
print("[3] DESCRIPTIVE STATISTICS")

print("\n    F&G Value Stats:")
print(df['value'].describe().round(2).to_string())

print("\n    Consolidated Sentiment Distribution:")
sent_dist = df['sentiment'].value_counts()
for s, cnt in sent_dist.items():
    pct = cnt / len(df) * 100
    print(f"      {s:<12} {cnt:>5} days  ({pct:.1f}%)")

print("\n    Avg F&G Value per Classification:")
print(df.groupby('classification')['value'].agg(['mean','median','std']).round(2).to_string())

print("\n─" * 40)
print("[4] SENTIMENT → PERFORMANCE MAPPING")

# Rolling Averages & Momentum
df = df.sort_values('date').reset_index(drop=True)
df['rolling_7d']  = df['value'].rolling(7,  min_periods=3).mean()
df['rolling_30d'] = df['value'].rolling(30, min_periods=7).mean()
df['momentum'] = df['value'] - df['rolling_7d']

# Streaks
df['sentiment_change'] = (df['sentiment'] != df['sentiment'].shift(1)).cumsum()
df['streak'] = df.groupby('sentiment_change').cumcount() + 1
streak_stats = df.groupby('sentiment')['streak'].max()

# Transition Matrix
df['next_sentiment'] = df['sentiment'].shift(-1)
transition = pd.crosstab(df['sentiment'], df['next_sentiment'], normalize='index').round(3)

print("\n    Longest Streak by Sentiment:")
print(streak_stats.to_string())
print("\n    Sentiment Transition Matrix (Row → Next Day):")
print(transition.to_string())

# Save Cleaned Data to Outputs Folder
clean_path = OUTPUT_DIR / 'cleaned_fear_greed_index.csv'
df.to_csv(clean_path, index=False)
print(f"\n    ✓ Cleaned dataset saved to {clean_path}")

# CELL 4: FIGURE 1 - OVERVIEW & DISTRIBUTION

print("\n[5] Generating Figures...")

fig = plt.figure(figsize=(20, 14), facecolor=FIG_BG)
gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
               left=0.07, right=0.97, top=0.90, bottom=0.08)

cats = ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed']
col_list = [COLORS.get(c, '#FFFFFF') for c in cats]

# Subplot 1: Classification Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
counts = df['classification'].value_counts().reindex(cats)
bars = ax1.bar(cats, counts.values, color=col_list, edgecolor='#30363D', linewidth=0.8, zorder=3)
for bar, val in zip(bars, counts.values):
    pct = val / len(df) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8.5, color=TEXT, fontweight='bold')
ax1.set_title('Days per Classification', fontsize=11, color=TEXT, fontweight='bold')
ax1.set_ylabel('Number of Days', fontsize=10)
ax1.set_xticklabels(cats, rotation=15, ha='right', fontsize=9)
ax1.grid(axis='y', zorder=0); ax1.set_facecolor(AX_BG)

# Subplot 2: Value Distribution Histogram
ax2 = fig.add_subplot(gs[0, 1])
for cls in cats:
    sub = df[df['classification'] == cls]['value']
    ax2.hist(sub, bins=20, alpha=0.7, color=COLORS[cls], label=cls, zorder=3)
ax2.set_title('F&G Value Distribution by Class', fontsize=11, color=TEXT, fontweight='bold')
ax2.set_xlabel('Fear/Greed Index Value (0–100)', fontsize=9)
ax2.set_ylabel('Frequency', fontsize=9)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(axis='y', zorder=0); ax2.set_facecolor(AX_BG)

# Subplot 3: Transition Heatmap
ax3 = fig.add_subplot(gs[0, 2])
sents = ['Fear','Neutral','Greed']
trans_mat = transition.reindex(index=sents, columns=sents).fillna(0).values
im = ax3.imshow(trans_mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax3.set_xticks(range(3)); ax3.set_xticklabels(sents, fontsize=10)
ax3.set_yticks(range(3)); ax3.set_yticklabels(sents, fontsize=10)
ax3.set_xlabel('Next Day Sentiment', fontsize=9)
ax3.set_ylabel('Today Sentiment', fontsize=9)
for i in range(3):
    for j in range(3):
        ax3.text(j, i, f'{trans_mat[i,j]:.2f}', ha='center', va='center',
                 fontsize=12, color='black', fontweight='bold')
plt.colorbar(im, ax=ax3, label='Transition Probability')
ax3.set_title('Sentiment Transition Matrix', fontsize=11, color=TEXT, fontweight='bold')

# Subplot 4: Monthly Average Heatmap
ax4 = fig.add_subplot(gs[1, :])
monthly_avg = df.groupby(['year','month'])['value'].mean().unstack(fill_value=np.nan)
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
im2 = ax4.imshow(monthly_avg.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax4.set_xticks(range(12)); ax4.set_xticklabels(month_names, fontsize=10)
ax4.set_yticks(range(len(monthly_avg.index))); ax4.set_yticklabels(monthly_avg.index, fontsize=10)

for i, yr in enumerate(monthly_avg.index):
    for j, m in enumerate(range(1, 13)):
        val = monthly_avg.loc[yr, m] if m in monthly_avg.columns else np.nan
        if not np.isnan(val):
            ax4.text(j, i, f'{val:.0f}', ha='center', va='center',
                     fontsize=8.5, color='black', fontweight='bold')
plt.colorbar(im2, ax=ax4, label='Avg F&G Value', orientation='vertical', fraction=0.015)
ax4.set_title('Monthly Average Fear/Greed Index by Year', fontsize=11, color=TEXT, fontweight='bold')

fig.suptitle('TASK 1 — FEAR/GREED INDEX: OVERVIEW & DISTRIBUTION', fontsize=14, color=TEXT, fontweight='bold', y=0.97)

# Save output
fig1_path = OUTPUT_DIR / 'task1_fig1_overview.png'
plt.savefig(fig1_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print(f"    ✓ {fig1_path.name} saved successfully.")

# 5: FIGURE 2 - TRENDS & MOMENTUM
fig2 = plt.figure(figsize=(20, 12), facecolor=FIG_BG)
gs2  = GridSpec(2, 2, figure=fig2, hspace=0.40, wspace=0.32,
                left=0.07, right=0.97, top=0.90, bottom=0.08)

# Full time series with sentiment shading
ax = fig2.add_subplot(gs2[0, :])
ax.plot(df['date'], df['value'], color='#A0A8B0', linewidth=0.6, alpha=0.5, label='Daily Value')
ax.plot(df['date'], df['rolling_30d'], color='#FFD700', linewidth=1.8, label='30-Day MA')
ax.plot(df['date'], df['rolling_7d'],  color='#4FC3F7', linewidth=1.2, alpha=0.8, label='7-Day MA')

# Shade zones
ax.axhspan(0,  25, alpha=0.07, color='#8B0000')
ax.axhspan(25, 45, alpha=0.07, color='#E84040')
ax.axhspan(45, 55, alpha=0.07, color='#7F8C8D')
ax.axhspan(55, 75, alpha=0.07, color='#27AE60')
ax.axhspan(75, 100,alpha=0.07, color='#145A32')

for y, lbl, col in [(12,'Extreme Fear','#8B0000'), (35,'Fear','#E84040'),
                    (50,'Neutral','#7F8C8D'), (65,'Greed','#27AE60'), (87,'Extreme Greed','#145A32')]:
    ax.text(df['date'].iloc[5], y, lbl, fontsize=8, color=col, alpha=0.8, va='center')

ax.set_ylabel('Fear/Greed Value (0–100)', fontsize=10)
ax.set_title('Bitcoin Fear/Greed Index — Full History with Rolling Averages', fontsize=12, color=TEXT, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(0, 100)
ax.grid(axis='y', zorder=0); ax.set_facecolor(AX_BG)

# Yearly avg F&G value (bar)
ax2b = fig2.add_subplot(gs2[1, 0])
yr_avg = df.groupby('year')['value'].mean()
yr_colors = ['#E84040' if v < 45 else '#27AE60' if v > 55 else '#7F8C8D' for v in yr_avg.values]
bars = ax2b.bar(yr_avg.index.astype(str), yr_avg.values, color=yr_colors, edgecolor='#30363D', zorder=3)
ax2b.axhline(50, color='#FFD700', linestyle='--', linewidth=1.2, alpha=0.8, label='Neutral (50)')

for bar, val in zip(bars, yr_avg.values):
    ax2b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=TEXT, fontweight='bold')
ax2b.set_title('Average F&G Index by Year', fontsize=11, color=TEXT, fontweight='bold')
ax2b.set_ylabel('Avg F&G Value', fontsize=9)
ax2b.set_ylim(0, 100)
ax2b.legend(fontsize=9); ax2b.grid(axis='y', zorder=0); ax2b.set_facecolor(AX_BG)

# Streak Distribution
ax2c = fig2.add_subplot(gs2[1, 1])
for s, c in SENT_COLORS.items():
    streak_data = df[df['sentiment'] == s]['streak']
    ax2c.hist(streak_data, bins=range(1, 40), alpha=0.6, color=c, label=s, density=True, zorder=3)
ax2c.set_title('Consecutive Day Streak Density', fontsize=11, color=TEXT, fontweight='bold')
ax2c.set_xlabel('Streak Length (days)', fontsize=9)
ax2c.set_ylabel('Density', fontsize=9)
ax2c.legend(fontsize=9); ax2c.grid(axis='y', zorder=0); ax2c.set_facecolor(AX_BG)

fig2.suptitle('TASK 1 — TREND ANALYSIS & MOMENTUM', fontsize=14, color=TEXT, fontweight='bold', y=0.97)

fig2_path = OUTPUT_DIR / 'task1_fig2_trends.png'
plt.savefig(fig2_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print(f"    ✓ {fig2_path.name} saved successfully.")

# 6: FIGURE 3 - YEARLY BREAKDOWN

fig3 = plt.figure(figsize=(20, 10), facecolor=FIG_BG)
gs3  = GridSpec(1, 2, figure=fig3, hspace=0.3, wspace=0.32,
                left=0.07, right=0.97, top=0.88, bottom=0.10)

# Yearly Sentiment Breakdown (%)
yearly = df.groupby(['year','sentiment']).size().unstack(fill_value=0)
yearly_pct = yearly.div(yearly.sum(axis=1), axis=0).round(3) * 100

ax3a = fig3.add_subplot(gs3[0, 0])
yearly_pct_plot = yearly_pct[['Fear','Neutral','Greed']] if 'Fear' in yearly_pct.columns else yearly_pct
bottom = np.zeros(len(yearly_pct_plot))

for s, c in [('Fear','#E84040'),('Neutral','#7F8C8D'),('Greed','#27AE60')]:
    if s in yearly_pct_plot.columns:
        vals = yearly_pct_plot[s].values
        bars = ax3a.bar(yearly_pct_plot.index.astype(str), vals, bottom=bottom, color=c,
                        label=s, edgecolor='#30363D', linewidth=0.5, zorder=3)
        for bar, val, bot in zip(bars, vals, bottom):
            if val > 5:
                ax3a.text(bar.get_x() + bar.get_width()/2, bot + val/2,
                          f'{val:.0f}%', ha='center', va='center', fontsize=8.5,
                          color='white', fontweight='bold')
        bottom += vals

ax3a.set_title('Yearly Sentiment Composition (%)', fontsize=12, color=TEXT, fontweight='bold')
ax3a.set_ylabel('Percentage of Days', fontsize=10)
ax3a.set_xlabel('Year', fontsize=9)
ax3a.legend(fontsize=9, loc='upper right')
ax3a.set_ylim(0, 105); ax3a.grid(axis='y', zorder=0); ax3a.set_facecolor(AX_BG)

# Boxplot of Values by Year
ax3b = fig3.add_subplot(gs3[0, 1])
years = sorted(df['year'].unique())
data_by_year = [df[df['year'] == y]['value'].values for y in years]

bp = ax3b.boxplot(data_by_year, patch_artist=True, notch=False,
                  medianprops=dict(color='#FFD700', linewidth=2),
                  whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
                  flierprops=dict(markerfacecolor='#E84040', markersize=3, alpha=0.5))

colors_box = ['#E84040' if yr_avg.get(y,50) < 45 else '#27AE60' if yr_avg.get(y,50) > 55 else '#7F8C8D' for y in years]
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c); patch.set_alpha(0.7)

ax3b.set_xticklabels([str(y) for y in years], fontsize=9)
ax3b.axhline(50, color='#FFD700', linestyle='--', alpha=0.6, linewidth=1.2)
ax3b.set_title('F&G Value Distribution by Year (Box Plot)', fontsize=12, color=TEXT, fontweight='bold')
ax3b.set_ylabel('Fear/Greed Value', fontsize=10)
ax3b.set_xlabel('Year', fontsize=9)
ax3b.grid(axis='y', zorder=0); ax3b.set_facecolor(AX_BG)

fig3.suptitle('TASK 1 — YEARLY SENTIMENT BREAKDOWN', fontsize=14, color=TEXT, fontweight='bold', y=0.97)

fig3_path = OUTPUT_DIR / 'task1_fig3_yearly.png'
plt.savefig(fig3_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close()
print(f"    ✓ {fig3_path.name} saved successfully.")

# 7: KEY INSIGHTS & SUMMARY

print("\n" + "=" * 60)
print("[6] KEY INSIGHTS — FEAR/GREED INDEX")
print("=" * 60)

fear_days   = sent_dist.get('Fear', 0)
greed_days  = sent_dist.get('Greed', 0)
neutral_days= sent_dist.get('Neutral', 0)
total_days  = len(df)

# Guard against missing data in transition matrix
fear_trans = transition.loc['Fear','Fear'] * 100 if 'Fear' in transition.index else 0
greed_trans = transition.loc['Greed','Greed'] * 100 if 'Greed' in transition.index else 0

print(f"""
  1. TIME IN MARKET (FEAR VS GREED)
     Fear   : {fear_days} days ({fear_days/total_days*100:.1f}%)
     Greed  : {greed_days} days ({greed_days/total_days*100:.1f}%)
     Neutral: {neutral_days} days ({neutral_days/total_days*100:.1f}%)

  2. SENTIMENT AUTOCORRELATION (STICKINESS)
     Once in Fear, market stays in Fear {fear_trans:.0f}% of the time the next day.
     Once in Greed, market stays in Greed {greed_trans:.0f}% of the time the next day.
     (Sentiment regimes have high momentum, highly useful for predictive models).

  3. LONGEST HISTORICAL STREAKS
     Fear   : {streak_stats.get('Fear',0)} consecutive days
     Greed  : {streak_stats.get('Greed',0)} consecutive days
     Neutral: {streak_stats.get('Neutral',0)} consecutive days

  4. RECENT MARKET CONDITIONS
     Aligning with Bitcoin's cycles, recent years generally show heavier skews 
     toward Greed following bearish conditions.

  5. EXTREME VALUES FREQUENCY
     Extreme Fear (<25) : {len(df[df['value']<25])} days
     Extreme Greed (>75): {len(df[df['value']>75])} days
""")

print("Task 1 completed successfully. All assets have been exported to the './outputs' directory.\n")