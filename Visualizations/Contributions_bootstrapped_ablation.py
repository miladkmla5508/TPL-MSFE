import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# ---------------------------
# Style settings
# ---------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 23,
    'axes.titlesize': 23,
    'axes.labelsize': 23,
    'xtick.labelsize': 23,
    'ytick.labelsize': 23,
})

# ---------------------------
# Data
# ---------------------------
sortino_ratio = {
    "Without Fusing": [-1.7825, -2.91, -2.59, -1.77, -2.07, -2.8, -2.72, -1.49, -1.47, -1.31,
                       -2.45, -2.06, -1.69, -2.36, -2.3, -1.68, -2.62, -2.74, -2.4, -2.12,
                       -1.74, -1.44, -2.89, -1.18, -1.6, -2.23, -1.1, -2.77, -1.7, -2.59],
    "Fusing just TPL": [25.38, 6.36, 8.64, 30.09, 19.13, 25.4, 18.45, 113.44, 38.66, 89.53,
                   12.71, 32.93, 46.08, 22.03, 34.45, 31.59, 19.35, 29.16, 25.75, 41.55,
                   37.05, 69.32, 20.52, 22.39, 63.39, 22.69, 59.39, 26.96, 37.97, 33.24],
    "Fusing TPL+MSFE": [28.63, 30.81, 27.22, 51.6, 51.48, 43.05, 28.1, 244.06, 64.81, 130.86,
                        45.59, 66.5, 48.62, 34.41, 55.79, 54.21, 31.5, 61.02, 32.73, 61.08,
                        40.13, 98.61, 42.8, 32.66, 99.68, 48.42, 107.91, 44.55, 40.18, 33.21]
}

calmar_ratio = {
    "Without Fusing our Perspectives": [-0.5741, -0.79, -0.71, -0.59, -0.66, -0.77, -0.7, -0.56, -0.63, -0.51,
                       -0.73, -0.63, -0.68, -0.67, -0.64, -0.64, -0.68, -0.77, -0.69, -0.69,
                       -0.56, -0.54, -0.7, -0.47, -0.63, -0.6, -0.47, -0.78, -0.49, -0.64],
    "Fusing just TPL": [31.14, 4.07, 5.53, 18.65, 9.78, 20.63, 18.39, 75.65, 25.26, 128.97,
                   4.82, 21.67, 31.43, 26.9, 45.96, 28.57, 18.95, 24.03, 17.95, 55.7,
                   50.03, 83.63, 15.53, 14.43, 79.69, 26.07, 70.81, 21.16, 32, 38.12],
    "Fusing TPL+MSFE": [34.36, 22.01, 19.67, 46.66, 63.08, 40.41, 26.42, 244.07, 54.44, 107.48,
                        58.87, 68.53, 46.88, 42.89, 68.21, 48.38, 27.79, 98.87, 19.38, 34.3,
                        56.17, 92.99, 53.19, 37.76, 94.49, 61.55, 147.31, 64.32, 49.53, 35.77]
}

average_return = {
    "Without Fusing our Perspectives": [-0.3684, -0.5819, -0.4493, -0.4713, -0.3714, -0.5507, -0.5384, -0.4613, -0.543, -0.3471,
                       -0.5592, -0.5904, -0.8508, -0.4727, -0.495, -0.4491, -0.4794, -0.5673, -0.4612, -0.4833,
                       -0.3801, -0.3905, -0.5354, -0.2394, -0.5133, -0.4035, -0.3024, -0.6011, -0.3685, -0.4952],
    "Fusing just TPL": [3.4531, 1.1297, 1.4381, 5.0298, 2.5301, 3.733, 2.432, 23.2047, 10.0118, 15.4753,
                   2.2556, 5.3782, 9.0237, 2.7745, 4.6862, 5.6448, 2.8357, 3.6826, 3.5652, 6.166,
                   4.3461, 11.4545, 2.7511, 3.962, 12.0586, 3.0124, 10.7926, 3.6595, 5.3088, 3.7992],
    "Fusing TPL+MSFE": [3.7991, 3.9287, 3.091, 7.709, 4.9665, 4.7351, 3.4725, 37.7701, 15.3368, 22.7651,
                        5.3394, 9.2802, 9.6414, 3.4582, 6.9731, 8.5292, 3.9764, 5.5976, 4.3542, 7.9858,
                        4.4854, 14.8411, 4.0542, 5.1604, 18.5706, 5.2128, 16.9178, 5.2981, 5.5997, 3.8948]
}

maximum_drawdown = {
    "Without Fusing our Perspectives": [-0.6765, -0.7659, -0.6636, -0.8373, -0.5927, -0.7365, -0.7935, -0.8629, -0.8936, -0.7266,
                       -0.7882, -0.967, -0.8843, -0.7401, -0.7961, -0.7309, -0.7373, -0.7609, -0.6935, -0.7347,
                       -0.72, -0.7586, -0.7915, -0.5533, -0.8522, -0.7116, -0.6721, -0.7936, -0.7913, -0.8104],
    "Fusing just TPL": [-0.1102, -0.2727, -0.2563, -0.2684, -0.2577, -0.1798, -0.1312, -0.3071, -0.3973, -0.1194,
                   -0.4643, -0.247, -0.2865, -0.1024, -0.1019, -0.1969, -0.1486, -0.1517, -0.1977, -0.1101,
                   -0.0869, -0.1367, -0.1758, -0.2724, -0.1511, -0.1148, -0.1534, -0.1722, -0.1651, -0.0993],
    "Fusing TPL+MSFE": [-0.1102, -0.178, -0.1562, -0.165, -0.0786, -0.1178, -0.1312, -0.1548, -0.2817, -0.2121,
                        -0.0905, -0.1351, -0.2054, -0.0802, -0.102, -0.1761, -0.1427, -0.0564, -0.224, -0.2325,
                        -0.0797, -0.1596, -0.0759, -0.1363, -0.1967, -0.0845, -0.1149, -0.0824, -0.1129, -0.1085]
}

# ---------------------------
# Helper functions
# ---------------------------
def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    bootstrap_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return bootstrap_means, lower, upper

def create_summary_stats(data_dict, metric_name):
    results = []
    for method, values in data_dict.items():
        values = np.array(values)
        bootstrap_means, ci_lower, ci_upper = bootstrap_ci(values)
        results.append({
            'Method': method,
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std': np.std(values, ddof=1),
            'CI_Lower (95%)': ci_lower,
            'CI_Upper (95%)': ci_upper,
            'Min': np.min(values),
            'Max': np.max(values)
        })
    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"{metric_name} - Summary Statistics")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    return df

def plot_bootstrap_distributions(data_dict, metric_name, n_bootstrap=10000):
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    for idx, (method, values) in enumerate(data_dict.items()):
        bootstrap_means, ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap)
        ax = axes[idx]
        ax.hist(bootstrap_means, bins=70, alpha=0.7, color=colors[idx], edgecolor='black')
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='black', linestyle='-', linewidth=4, label=f'Mean: {mean_val:.2f}')
        ax.axvline(ci_lower, color='black', linestyle='--', linewidth=4, label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
        ax.axvline(ci_upper, color='black', linestyle='--', linewidth=4)
        ax.set_title(f'{method}', fontsize=30, fontweight='bold')
        ax.set_xlabel('Bootstrap Mean')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.7)
    plt.suptitle(f'{metric_name}', fontsize=40, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ---------------------------
# Main analysis
# ---------------------------
print("\n" + "="*80)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("="*80)

metrics = {
    'Sortino ratio': sortino_ratio,
    'Calmar ratio': calmar_ratio,
    'Average rate of return': average_return,
    'Maximum drawdown': maximum_drawdown
}

for metric_name, data_dict in metrics.items():
    create_summary_stats(data_dict, metric_name)
    plot_bootstrap_distributions(data_dict, metric_name)

print("\nAnalysis complete!")
