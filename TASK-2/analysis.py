import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg') # Comment out to display plots inline in Colab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Configuration and output directory setup
OUTPUT_DIR = Path('./outputs')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_BG = '#0D1117'
AX_BG = '#161B22'
TEXT = '#E6EDF3'
GRID = '#21262D'
ACCENT = ['#4FC3F7', '#FFB74D', '#80CBC4', '#F48FB1', '#CE93D8', '#E84040', '#27AE60']

plt.rcParams.update({
    'figure.facecolor': FIG_BG, 'axes.facecolor': AX_BG,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT,
    'text.color': TEXT, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
})

print("Loading dataset...")
df = pd.read_csv('historical_data.csv')
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

print("Cleaning and engineering features...")

df['datetime'] = pd.to_datetime(df['Timestamp IST'], format='%d-%m-%Y %H:%M', dayfirst=True)
df['date'] = df['datetime'].dt.normalize()
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.day_name()
df['week'] = df['datetime'].dt.to_period('W').astype(str)
df['month'] = df['datetime'].dt.to_period('M').astype(str)

# Trading outcome flags
df['is_win'] = df['Closed PnL'] > 0
df['is_loss'] = df['Closed PnL'] < 0
df['is_closed_trade'] = df['Closed PnL'] != 0
df['is_long'] = df['Side'] == 'BUY'
df['net_pnl'] = df['Closed PnL'] - df['Fee']

# Position sizing (Leverage proxy)
df['size_bucket'] = pd.cut(
    df['Size USD'],
    bins=[0, 500, 2000, 10000, 50000, np.inf],
    labels=['Micro (<$500)', 'Small ($500-2K)', 'Medium ($2K-10K)', 'Large ($10K-50K)', 'Whale (>$50K)']
)

closed = df[df['is_closed_trade']]

print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Total trades: {len(df):,}")

print("Processing aggregations and clustering...")

# Account-level performance metrics
acct_pnl = df.groupby('Account').agg(
    total_pnl=('Closed PnL', 'sum'),
    net_pnl=('net_pnl', 'sum'),
    total_trades=('Closed PnL', 'count'),
    closed_trades=('is_closed_trade', 'sum'),
    wins=('is_win', 'sum'),
    total_fees=('Fee', 'sum'),
    avg_size=('Size USD', 'mean'),
    active_days=('date', 'nunique'),
    total_volume=('Size USD', 'sum'),
).reset_index()

acct_pnl['win_rate'] = acct_pnl['wins'] / (acct_pnl['closed_trades'] + 1e-9)
acct_pnl['trades_per_day'] = acct_pnl['total_trades'] / acct_pnl['active_days']
acct_pnl['short_addr'] = acct_pnl['Account'].str[:8] + '...'

# K-Means Clustering for Trader Archetypes
feats = ['total_pnl', 'win_rate', 'trades_per_day', 'avg_size', 'active_days']
Xc = StandardScaler().fit_transform(acct_pnl[feats].fillna(0))
acct_pnl['cluster'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(Xc)

cluster_map = acct_pnl.groupby('cluster')[['total_pnl', 'win_rate', 'trades_per_day']].mean()

# Dynamically assign labels based on cluster centroids
labels = {
    cluster_map['total_pnl'].idxmax(): 'Whale Traders',
    cluster_map['trades_per_day'].idxmax(): 'HFT Bots',
    cluster_map['win_rate'].nlargest(2).index[-1]: 'Precision Players',
}
for c in range(4):
    if c not in labels:
        labels[c] = 'Casual Traders'
acct_pnl['archetype'] = acct_pnl['cluster'].map(labels)

# Daily PnL Tracking
daily_pnl = df.groupby('date').agg(
    total_pnl=('Closed PnL', 'sum'),
    rolling_pnl_30d=('Closed PnL', lambda x: x.sum()) 
).reset_index()
daily_pnl['rolling_pnl_30d'] = daily_pnl['total_pnl'].rolling(30, min_periods=7).mean()

# Coin-level aggregations
coin_pnl = closed.groupby('Coin').agg(
    total_pnl=('Closed PnL', 'sum')
).sort_values('total_pnl', ascending=False)

print("Aggregations complete.")

print("Generating EDA and PnL visual reports...")

# --- FIGURE 1: EDA Overview ---
fig1 = plt.figure(figsize=(20, 14), facecolor=FIG_BG)
gs1 = GridSpec(2, 3, figure=fig1, hspace=0.42, wspace=0.35, left=0.07, right=0.97, top=0.90, bottom=0.08)

# 1A: Direction Count
ax1 = fig1.add_subplot(gs1[0, 0])
dirs = df['Direction'].value_counts().head(8)
ax1.barh(dirs.index, dirs.values, color=ACCENT[:len(dirs)], edgecolor='#30363D')
ax1.set_title('Trade Count by Direction', color=TEXT, fontweight='bold')
ax1.set_facecolor(AX_BG)

# 1B: Top Coins
ax2 = fig1.add_subplot(gs1[0, 1])
top10 = df['Coin'].value_counts().head(10)
ax2.barh(top10.index[::-1], top10.values[::-1], color=ACCENT[0], edgecolor='#30363D')
ax2.set_title('Top 10 Coins by Volume', color=TEXT, fontweight='bold')
ax2.set_facecolor(AX_BG)

# 1C: Size Distribution
ax3 = fig1.add_subplot(gs1[0, 2])
size_counts = df.groupby('size_bucket', observed=True).size()
ax3.pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%', textprops={'color': TEXT})
ax3.set_title('Position Size Distribution', color=TEXT, fontweight='bold')
ax3.set_facecolor(AX_BG)

# 1D: Hourly Pattern
ax4 = fig1.add_subplot(gs1[1, 0])
hourly_data = df.groupby('hour').size()
ax4.bar(hourly_data.index, hourly_data.values, color=ACCENT[0], edgecolor='#30363D')
ax4.set_title('Trades by Hour (IST)', color=TEXT, fontweight='bold')
ax4.set_facecolor(AX_BG)

# 1E: Day of Week
ax5 = fig1.add_subplot(gs1[1, 1])
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_data = df.groupby('day_of_week').size().reindex(dow_order)
ax5.bar(range(7), dow_data.values, color=[ACCENT[1] if d in ['Saturday', 'Sunday'] else ACCENT[0] for d in dow_order])
ax5.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax5.set_title('Trades by Day of Week', color=TEXT, fontweight='bold')
ax5.set_facecolor(AX_BG)

# 1F: PnL Distribution
ax6 = fig1.add_subplot(gs1[1, 2])
pnl_clip = df['Closed PnL'].clip(-5000, 20000)
ax6.hist(pnl_clip[pnl_clip != 0], bins=80, color=ACCENT[2], edgecolor='#30363D')
ax6.axvline(0, color='#FFD700', linestyle='--')
ax6.set_title('Closed PnL Distribution', color=TEXT, fontweight='bold')
ax6.set_facecolor(AX_BG)

fig1.suptitle('TASK 2 — EDA OVERVIEW', fontsize=14, color=TEXT, fontweight='bold', y=0.97)
fig1.savefig(OUTPUT_DIR / 'task2_fig1_eda.png', dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close(fig1)

# --- FIGURE 2: PnL Deep Dive ---
fig2 = plt.figure(figsize=(20, 12), facecolor=FIG_BG)
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.35, left=0.07, right=0.97, top=0.90, bottom=0.08)

# 2A: Daily PnL
ax7 = fig2.add_subplot(gs2[0, :])
ax7.bar(daily_pnl['date'], daily_pnl['total_pnl'], color=np.where(daily_pnl['total_pnl'] >= 0, '#27AE60', '#E84040'))
ax7.plot(daily_pnl['date'], daily_pnl['rolling_pnl_30d'], color='#FFD700', label='30-Day MA')
ax7.set_title('Aggregate Daily PnL', color=TEXT, fontweight='bold')
ax7.set_facecolor(AX_BG)

# 2B: Coin PnL
ax8 = fig2.add_subplot(gs2[1, 0])
combined = pd.concat([coin_pnl.head(10), coin_pnl.tail(5)])
ax8.barh(combined.index[::-1], combined['total_pnl'].values[::-1], color=['#27AE60' if v > 0 else '#E84040' for v in combined['total_pnl']][::-1])
ax8.set_title('Top & Bottom Coins by Total PnL', color=TEXT, fontweight='bold')
ax8.set_facecolor(AX_BG)

# 2C: Win Rate by Size
ax9 = fig2.add_subplot(gs2[1, 1])
wr_by_size = df.groupby('size_bucket', observed=True)['is_win'].mean() * 100
ax9.bar(range(len(wr_by_size)), wr_by_size.values, color=ACCENT[0])
ax9.set_xticks(range(len(wr_by_size)))
ax9.set_xticklabels(wr_by_size.index, rotation=20, ha='right')
ax9.set_title('Win Rate by Position Size', color=TEXT, fontweight='bold')
ax9.set_facecolor(AX_BG)

fig2.suptitle('TASK 2 — PNL ANALYSIS', fontsize=14, color=TEXT, fontweight='bold', y=0.97)
fig2.savefig(OUTPUT_DIR / 'task2_fig2_pnl.png', dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close(fig2)

print(f"Figures 1 and 2 saved to {OUTPUT_DIR}")

print("Generating Behavior and Segmentation visual reports...")

# --- FIGURE 3: Behavior Analysis ---
fig3 = plt.figure(figsize=(20, 12), facecolor=FIG_BG)
gs3 = GridSpec(2, 3, figure=fig3, hspace=0.42, wspace=0.38, left=0.07, right=0.97, top=0.90, bottom=0.08)

# 3A & 3B: Long vs Short
ax10 = fig3.add_subplot(gs3[0, 0])
ls_data = closed.groupby('is_long')['Closed PnL'].mean()
ax10.bar(['Short', 'Long'], ls_data.values, color=['#E84040', '#27AE60'])
ax10.set_title('Avg Closed PnL: Long vs Short', color=TEXT, fontweight='bold')
ax10.set_facecolor(AX_BG)

# 3D: Total Fees
ax11 = fig3.add_subplot(gs3[1, 0])
fee_data = acct_pnl.sort_values('total_fees', ascending=False)
ax11.barh(range(len(fee_data)), fee_data['total_fees'], color=ACCENT[3])
ax11.set_yticks(range(len(fee_data)))
ax11.set_yticklabels(fee_data['short_addr'], fontsize=7)
ax11.set_title('Total Fees Paid', color=TEXT, fontweight='bold')
ax11.set_facecolor(AX_BG)

# 3E: Gross vs Net PnL
ax12 = fig3.add_subplot(gs3[1, 1])
sorted_acct = acct_pnl.sort_values('total_pnl', ascending=False)
ax12.bar(range(len(sorted_acct)), sorted_acct['total_pnl'], color='#27AE60', label='Gross')
ax12.bar(range(len(sorted_acct)), sorted_acct['net_pnl'], color='#4FC3F7', label='Net')
ax12.set_title('Gross vs Net PnL per Account', color=TEXT, fontweight='bold')
ax12.set_facecolor(AX_BG)

fig3.suptitle('TASK 2 — TRADER BEHAVIOR', fontsize=14, color=TEXT, fontweight='bold', y=0.97)
fig3.savefig(OUTPUT_DIR / 'task2_fig3_behavior.png', dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close(fig3)

# --- FIGURE 4: Account Segmentation ---
fig4 = plt.figure(figsize=(20, 6), facecolor=FIG_BG)
gs4 = GridSpec(1, 2, figure=fig4, hspace=0.3, wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.12)
arch_colors = {'Whale Traders': '#4FC3F7', 'HFT Bots': '#FFB74D', 'Precision Players': '#80CBC4', 'Casual Traders': '#F48FB1'}

# 4A: Archetype PnL
ax13 = fig4.add_subplot(gs4[0, 0])
arch_pnl = acct_pnl.groupby('archetype')['total_pnl'].mean().sort_values(ascending=False)
ax13.bar(arch_pnl.index, arch_pnl.values, color=[arch_colors.get(a, '#888') for a in arch_pnl.index])
ax13.set_title('Avg Total PnL by Archetype', color=TEXT, fontweight='bold')
ax13.set_facecolor(AX_BG)

# 4B: Trades vs Size Scatter
ax14 = fig4.add_subplot(gs4[0, 1])
for arch, grp in acct_pnl.groupby('archetype'):
    ax14.scatter(grp['trades_per_day'], grp['avg_size'], label=arch, color=arch_colors.get(arch, '#888'), s=100)
ax14.set_title('Trades/Day vs Avg Position Size', color=TEXT, fontweight='bold')
ax14.legend(loc='upper right')
ax14.set_facecolor(AX_BG)

fig4.suptitle('TASK 2 — TRADER ARCHETYPES', fontsize=14, color=TEXT, fontweight='bold', y=0.97)
fig4.savefig(OUTPUT_DIR / 'task2_fig4_accounts.png', dpi=150, bbox_inches='tight', facecolor=FIG_BG)
plt.close(fig4)

print(f"Figures 3 and 4 saved to {OUTPUT_DIR}")

print("=" * 60)
print("KEY INSIGHTS — HISTORICAL TRADER DATA")
print("=" * 60)

gross_wins = closed[closed['is_win']]['Closed PnL'].sum()
gross_losses = abs(closed[closed['is_loss']]['Closed PnL'].sum())
profit_factor = gross_wins / (gross_losses + 1e-9)

print(f"""
1. MARKET PROFITABILITY
   Gross PnL    : ${df['Closed PnL'].sum():,.2f}
   Total Fees   : ${df['Fee'].sum():,.2f}
   Net PnL      : ${df['net_pnl'].sum():,.2f}
   Profit Factor: {profit_factor:.2f}

2. DIRECTIONAL BIAS
   Long avg PnL : ${closed[closed['is_long']]['Closed PnL'].mean():,.2f}
   Short avg PnL: ${closed[~closed['is_long']]['Closed PnL'].mean():,.2f}

3. COIN DOMINANCE
   HYPE Trades  : {df[df['Coin']=='HYPE'].shape[0]:,} ({df[df['Coin']=='HYPE'].shape[0]/len(df)*100:.1f}%)

4. ACCOUNT DEMOGRAPHICS
   {(acct_pnl['total_pnl'] > 0).sum()} of {len(acct_pnl)} accounts are net profitable.
   Fees represent {df['Fee'].sum() / (df['Fee'].sum() + df['Closed PnL'].sum()) * 100:.1f}% of gross+fees combined.
""")

print("Task 2 complete. All assets have been exported to the './outputs' directory.")

