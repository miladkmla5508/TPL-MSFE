import matplotlib.ticker as ticker

# Set a professional style
sns.set(style='whitegrid')

# Define T_C and B_r
T_C = [0.050, 0.045, 0.040, 0.035, 0.030, 0.025, 0.020, 0.015, 0.010, 0.005]
B_r = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0045, 0.0050]

# Define the AAR matrix (10x10)
AAR_values = [
    [-0.4483, -0.1899, -0.1496, -0.1588, -0.2379, -0.0663, -0.1532, -0.0548, -0.02693, 0.1144],
    [-0.0065, -0.2336, -0.4232, -0.3082, -0.2689, -0.2467, -0.2637, -0.1891, -0.0067, -0.0592],
    [0.01566, 3.27727, 0.01426, 2.1981, 0.19, 0.4591, 0.1497, 0.1693, 0.007, -0.8426],
    [-0.1818, 0.42, 0.4131, 0.1977, 0.56, 0.6505, 0.4664, 0.6194, 0.3331, -0.1444],
    [-0.03295, 0.1781, 0.1778, 0.3726, 0.8575, 0.1793, 0.6193, 5.1175, 0.2924, -0.08494],
    [-0.0287, 0.1764, 0.4726, 0.4058, 0.4111, 0.6861, 1.4003, 0.8846, 0.7359, 2.7657],
    [-0.0041, -0.0033, 0.2552, 0.1762, 0.4855, 0.9201, 1.8583, 1.2589, 4.0907, 1.7614],
    [0.04944, 0.1306, 0.1139, 0.2749, 0.4112, 0.5438, 1.1592, 1.8239, 2.3781, 1.2323],
    [-0.0267, 0.04539, 0.5774, 0.84811, 1.7529, 2.546, 3.8134, 5.4097, 7.718, 9.7306],
    [-0.1098, -0.0545, 0.04304, 0.264, 0.2518, 0.5189, 0.6791, 5.169707, 7.1039, 8.7758]
]

# Define MDD values
MDD_values = [
    [-0.7251, -0.6973, -0.5739, -0.5028, -0.5521, -0.5701, -0.538,  -0.541,  -0.4682, -0.4562],
    [-0.6338, -0.667,  -0.6759, -0.7046, -0.64,   -0.6096, -0.569,  -0.4919, -0.4936, -0.5044],
    [-0.5875, -0.4585, -0.5,    -0.6671, -0.3467, -0.4376, -0.5282, -0.45,   -0.4637, -0.6237],
    [-0.6333, -0.2855, -0.385,  -0.4638, -0.4487, -0.3941, -0.4318, -0.4278, -0.4355, -0.5064],
    [-0.6014, -0.6596, -0.3378, -0.3732, -0.4065, -0.5606, -0.48,   -0.1175, -0.3727, -0.6297],
    [-0.5092, -0.3163, -0.5193, -0.3046, -0.3426, -0.5194, -0.3683, -0.4142, -0.4774, -0.1928],
    [-0.5369, -0.5111, -0.3127, -0.5942, -0.4313, -0.3198, -0.2783, -0.3142, -0.1827, -0.2058],
    [-0.514,  -0.431,  -0.5105, -0.4611, -0.4174, -0.2993, -0.2523, -0.3948, -0.2649, -0.2408],
    [-0.5292, -0.6363, -0.2774, -0.3001, -0.2053, -0.2053, -0.1758, -0.078,  -0.0676, -0.078],
    [-0.6376, -0.6966, -0.476,  -0.3458, -0.3688, -0.4043, -0.3417, -0.0912, -0.0687, -0.0676]
]

# Create meshgrid
T_C_grid, B_r_grid = np.meshgrid(T_C, B_r)
AAR_array = np.array(AAR_values)
MDD_array = np.array(MDD_values)

# Create figure with subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# Coordinates to mark
x_mark = 0.015
y_mark = 0.0025

# **FIXED: Plot AAR (left subplot) - removed 'shading' parameter**
ax = axes[0]
levels = np.linspace(np.min(AAR_array), np.max(AAR_array), 100)
contour_filled = ax.contourf(T_C_grid, B_r_grid, AAR_array, levels=levels, cmap='Greens', alpha=0.9)
contour_lines = ax.contour(T_C_grid, B_r_grid, AAR_array, levels=4, colors='black', linewidths=0.9)
ax.clabel(contour_lines, inline=True, fontsize=18, fmt="%.2f")
cbar = fig.colorbar(contour_filled, ax=ax, pad=0.02, format="%.2f")
cbar.ax.tick_params(labelsize=18)
ax.set_xlabel('Total transaction costs', fontsize=20, labelpad=10)
ax.set_ylabel('Balance rate', fontsize=20, labelpad=10)
ax.set_title('Average Annual Return (AAR)', fontsize=22, pad=15)
ax.grid(True, linestyle='--', linewidth=0.9, alpha=0.9)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax.tick_params(axis='both', labelsize=18)

# Mark point on AAR plot
ax.plot(x_mark, y_mark, '*', markersize=0)
ax.annotate('selected point',
            xy=(x_mark, y_mark),
            xytext=(x_mark - 0.009, y_mark - 0.0009),
            fontsize=20,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))

# **FIXED: Plot MDD (right subplot) - removed 'shading' parameter**
ax = axes[1]
levels = np.linspace(np.min(MDD_array), np.max(MDD_array), 100)
contour_filled = ax.contourf(T_C_grid, B_r_grid, MDD_array, levels=levels, cmap='Reds_r', alpha=0.9)
contour_lines = ax.contour(T_C_grid, B_r_grid, MDD_array, levels=5, colors='black', linewidths=0.9, linestyles='solid')
ax.clabel(contour_lines, inline=True, fontsize=18, fmt="%.2f")
cbar = fig.colorbar(contour_filled, ax=ax, pad=0.02, format="%.2f")
cbar.ax.tick_params(labelsize=18)
ax.set_xlabel('Total transaction costs', fontsize=20, labelpad=10)
ax.set_ylabel('Balance rate', fontsize=20, labelpad=10)
ax.set_title('Maximum Drawdown (MDD)', fontsize=22, pad=15)
ax.grid(True, linestyle='--', linewidth=0.9, alpha=0.9)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax.tick_params(axis='both', labelsize=18)

# Mark point on MDD plot
ax.plot(x_mark, y_mark, '*', markersize=0)
ax.annotate('selected point',
            xy=(x_mark, y_mark),
            xytext=(x_mark - 0.009, y_mark - 0.0009),
            fontsize=20,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))

# Adjust layout
plt.tight_layout()
plt.show()
