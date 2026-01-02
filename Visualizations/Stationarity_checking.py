from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

plt.rcParams['font.family'] = 'Arial'
# Set plot style
sns.set(style="whitegrid")

# Use your existing DataFrame
volume = df['Volume']

# --- Preprocessing ---

# 1. Log transformation (handle negatives)
min_val = volume.min()
shift = abs(min_val) + 1e-6 if min_val <= 0 else 0
volume_log = np.log(volume + shift)

# 2. Winsorizing (2% tails)
volume_winsorized = pd.Series(winsorize(volume, limits=[0.05, 0.05]), index=volume.index)

# 3. Z-score scaling
scaler = StandardScaler()
volume_zscore = pd.Series(scaler.fit_transform(volume.values.reshape(-1, 1)).flatten(), index=volume.index)

# --- ADF Stationarity Test ---
def adf_test_text(series):
    result = adfuller(series.dropna(), autolag='AIC')
    pval = result[1]
    return f"ADF p = {pval:.3f}\n{' Stationary' if pval < 0.05 else '❌ Non-Stationary'}"

# --- Plot Setup ---
fig, axes = plt.subplots(2, 2, figsize=(22, 16), sharex=True)

fontsize_title = 30
fontsize_label = 25
fontsize_text = 25
fontsize_ticks = 25

# --- Plot Helper ---
def plot_series(ax, series, title, ylabel, color):
    sns.lineplot(x=series.index, y=series.values, ax=ax, color=color)
    ax.set_title(title, fontsize=fontsize_title)
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.set_xlabel("Index", fontsize=fontsize_label)
    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    ax.text(0.98, 0.97, adf_test_text(series),
            transform=ax.transAxes, fontsize=fontsize_text,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

# --- Plot Each Variant ---
plot_series(axes[0, 0], volume, "Original Volume", "Volume", "#1f77b4")
plot_series(axes[0, 1], volume_log, "Log-Transformed Volume", "Log(Volume)", "#ff7f0e")
plot_series(axes[1, 0], volume_winsorized, "Winsorized Volume (5%)", "Winsorized", "#2ca02c")
plot_series(axes[1, 1], volume_zscore, "Z-score Scaled Volume", "Z-score", "#d62728")

plt.tight_layout()
plt.show()
