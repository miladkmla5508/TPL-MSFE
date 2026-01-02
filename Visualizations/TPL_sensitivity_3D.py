from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.gridspec as gridspec

# Apply seaborn style and context BEFORE plotting
sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.3)

# --- Data for Sells Plot ---
T_C = np.array([0.050, 0.045, 0.040, 0.035, 0.030, 0.025, 0.020, 0.015, 0.010, 0.005])
B_r = np.array([0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005])

# Sells data
PS_data = np.array([
    [0.671, 0.771, 0.1365, 0.1345, 0.2681, 0.0576, 0.2316, 0.1596, 0.3231, 0.1477],
    [4.5327, 4.093, 2.7896, 2.2226, 1.9246, 2.1141, 1.467, 1.1911, 0.6784, 0.6106],
    [7.0743, 6.3932, 5.1379, 4.9591, 3.7813, 3.4372, 2.6837, 2.0835, 1.0837, 1.5292],
    [8.167, 9.5361, 7.8358, 6.888, 6.6638, 5.7277, 4.019, 3.1763, 1.882, 2.2933],
    [9.9067, 9.909, 9.8568, 8.5368, 8.7896, 7.3803, 6.4455, 7.6774, 3.5332, 3.1877],
    [10.742, 10.75, 10.6815, 10.2216, 9.4064, 8.5551, 8.3979, 5.8891, 5.2682, 12.2975],
    [10.6826, 10.8468, 10.8556, 9.9336, 10.2743, 9.7868, 9.3924, 7.8513, 9.4049, 12.5266],
    [11.1435, 11.0437, 10.5983, 10.5819, 10.527, 10.5962, 10.6366, 9.6298, 10.8771, 12.0427],
    [10.7672, 10.6362, 11.0343, 10.9557, 10.8464, 11.07, 10.9985, 10.963, 11.1788, 13.1861],
    [10.8351, 10.9583, 10.9836, 11.0738, 11.062, 11.0534, 11.2686, 11.749, 11.7951, 13.6387]
])

LS_data = np.array([
    [-4.1121, -2.3117, -1.3581, -0.9748, -1.7445, -0.7730, -1.0644, -0.4887, -0.4134, -0.1036],
    [-6.1669, -6.0279, -6.1837, -4.3701, -4.0576, -4.1254, -3.3563, -2.4221, -1.8873, -0.8276],
    [-8.3861, -6.9895, -6.2857, -5.6092, -4.6448, -3.3975, -3.5170, -2.4171, -1.5665, -2.3475],
    [-10.088, -9.2096, -8.1498, -7.2970, -5.8004, -4.8133, -3.5567, -2.5661, -1.5591, -2.9100],
    [-9.9618, -9.9928, -9.8151, -7.8053, -7.1237, -7.7955, -5.4319, -1.6990, -3.4431, -4.4095],
    [-11.047, -10.4597, -9.4278, -9.0105, -8.7349, -6.9638, -5.8261, -4.7534, -3.8520, -5.5175],
    [-11.0246, -11.1696, -10.2333, -9.4676, -8.9034, -7.1322, -5.7213, -5.4150, -3.2510, -6.8936],
    [-10.9604, -10.5706, -10.5288, -9.8021, -9.2412, -8.7526, -7.5171, -5.5919, -5.5775, -7.8393],
    [-11.2722, -10.7136, -9.0277, -8.4545, -6.8024, -5.6622, -4.4989, -3.1719, -1.8971, -1.3323],
    [-11.6724, -11.4808, -11.0103, -10.0538, -10.1338, -9.0646, -8.8554, -3.3516, -2.0814, -1.7133]
])

# --- Data for Buys Plot ---
PB_data = np.array([
    [9.2677, 10.1269, 10.9159, 10.7416, 10.2469, 11.3455, 10.6014, 11.0653, 11.078, 11.9011],
    [7.0564, 6.3076, 6.7884, 7.9827, 8.4836, 8.1694, 8.8961, 9.2497, 10.2526, 10.357],
    [4.4323, 5.3893, 5.8978, 6.5089, 7.9806, 8.8867, 8.9817, 9.6171, 10.4347, 8.6907],
    [2.5569, 3.1485, 4.6922, 4.7663, 5.8371, 6.6019, 8.2984, 9.4975, 10.3484, 8.0334],
    [2.0561, 2.1081, 2.104, 3.5146, 4.313, 4.0685, 5.6595, 7.6246, 8.1826, 7.083],
    [0.5887, 1.2808, 1.9456, 1.8792, 2.7398, 3.92, 5.2841, 6.5751, 7.3064, 2.9412],
    [0.7656, 0.8164, 1.2872, 1.8141, 2.0313, 3.1484, 5.1028, 5.5494, 6.5226, 1.787],
    [0.5202, 0.7833, 1.2447, 1.4892, 1.7696, 1.9358, 2.9309, 4.8102, 3.9179, 1.8007],
    [0.7638, 0.9152, 1.6766, 2.28, 3.6293, 3.8861, 5.0404, 5.7039, 6.3869, 5.0374],
    [0.4638, 0.5658, 0.7329, 1.0102, 1.0568, 1.4307, 1.5877, 4.7148, 5.2998, 4.3119]
])

LB_data = np.array([
    [-9.7047, -9.7365, -10.5111, -10.7995, -10.4198, -10.7891, -10.619, -10.8283, -10.9244, -10.8679],
    [-5.1352, -5.9189, -6.9245, -8.1563, -8.2271, -7.8872, -8.891, -9.2331, -9.6414, -10.266],
    [-2.7512, -3.0526, -4.4044, -4.415, -5.6554, -5.9791, -6.9187, -7.8812, -9.1104, -9.4107],
    [-1.7781, -0.6996, -1.6979, -2.8536, -3.287, -3.7183, -5.7772, -6.4892, -8.3487, -8.2743],
    [-1.0881, -0.5588, -0.7403, -1.7896, -1.3482, -2.2114, -3.0249, -0.7022, -6.1826, -6.2544],
    [-0.1753, -0.0919, -0.1777, -0.4143, -0.698, -1.6367, -1.5015, -2.9936, -4.5659, -0.1611],
    [-0.1998, -0.1489, -0.0696, -0.8455, -0.3307, -0.9622, -1.1304, -2.0473, -1.0208, -0.0861],
    [-0.0558, -0.0889, -0.3042, -0.2628, -0.3408, -0.4403, -0.3668, -1.2835, -0.4168, -0.1405],
    [-0.1971, -0.2304, -0.1922, -0.18, -0.2856, -0.1422, -0.2511, -0.2203, -0.24, -0.0167],
    [-0.1183, -0.1134, -0.0933, -0.0858, -0.1038, -0.2014, -0.1338, -0.1058, -0.1144, -0.0142]
])

# Create meshgrid for plotting
T_C_mesh, B_r_mesh = np.meshgrid(T_C, B_r)

# Create zero_plane BEFORE using it
zero_plane = np.zeros_like(T_C_mesh)

# Selected coordinate
selected_tc = 0.015
selected_br = 0.0025

# Find the index of these values
i_tc = np.where(T_C == selected_tc)[0][0]
i_br = np.where(B_r == selected_br)[0][0]

# Get the corresponding Z values
selected_ps = PS_data[i_br, i_tc]
selected_ls = LS_data[i_br, i_tc]
selected_pb = PB_data[i_br, i_tc]
selected_lb = LB_data[i_br, i_tc]

# Create figure with GridSpec for better spacing control
fig = plt.figure(figsize=(15, 10), dpi=100)

# Use GridSpec with minimal horizontal spacing
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.2)

# --- First subplot: Buys ---
ax1 = fig.add_subplot(gs[0], projection='3d')

# Plot PB surface
ax1.plot_surface(T_C_mesh, B_r_mesh, PB_data,
                 cmap='Blues_r',
                 alpha=0.6,
                 linewidth=0.2,
                 edgecolors='darkblue',
                 antialiased=True)

# Plot LB surface
ax1.plot_surface(T_C_mesh, B_r_mesh, LB_data,
                 cmap='Reds',
                 alpha=0.6,
                 linewidth=0.2,
                 edgecolors='darkred',
                 antialiased=True)

# Add a reference gray plane at z = 0
ax1.plot_surface(T_C_mesh, B_r_mesh, zero_plane,
                 color='gray', alpha=0.1)

# Labels and title
ax1.set_xlabel('Total transaction costs', fontsize=16, labelpad=10)
ax1.set_ylabel('Balance rate', fontsize=16, labelpad=10)
ax1.set_zlabel('Profit / Loss', fontsize=16, labelpad=5)
ax1.set_title('Profits and Losses in Buys', fontsize=20, fontweight='bold')

ax1.view_init(elev=20, azim=150)

# Increase font size of axis numbers
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.tick_params(axis='z', labelsize=14)

# Plot stars for selected points
ax1.scatter(selected_tc, selected_br, selected_pb,
            color='navy', marker='*', s=150)
ax1.scatter(selected_tc, selected_br, selected_lb,
            color='darkred', marker='*', s=150)

# Annotate the points
ax1.text(selected_tc, selected_br, selected_pb + 0.7,
         f'{selected_pb:.2f}', color='navy', fontsize=20)
ax1.text(selected_tc, selected_br, selected_lb - 0.7,
         f'{selected_lb:.2f}', color='darkred', fontsize=20)

# --- Second subplot: Sells ---
ax2 = fig.add_subplot(gs[1], projection='3d')

# Plot PS surface (Profits in Sells)
ax2.plot_surface(T_C_mesh, B_r_mesh, PS_data,
                 cmap='Blues_r',
                 alpha=0.5,
                 linewidth=0.2,
                 edgecolors='darkblue',
                 antialiased=True)

# Plot LS surface (Losses in Sells)
ax2.plot_surface(T_C_mesh, B_r_mesh, LS_data,
                 cmap='Reds',
                 alpha=0.5,
                 linewidth=0.2,
                 edgecolors='darkred',
                 antialiased=True)

# Add a reference gray plane at z = 0
ax2.plot_surface(T_C_mesh, B_r_mesh, zero_plane,
                 color='gray', alpha=0.1)

# Labels and title
ax2.set_xlabel('Total transaction costs', fontsize=16, labelpad=10)
ax2.set_ylabel('Balance rate', fontsize=16, labelpad=10)
ax2.set_zlabel('Profit / Loss', fontsize=16, labelpad=5)
ax2.set_title('Profits and Losses in Sells', fontsize=20, fontweight='bold')

ax2.view_init(elev=20, azim=145)

# Increase font size of axis numbers
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.tick_params(axis='z', labelsize=14)

# Plot stars for selected points
ax2.scatter(selected_tc, selected_br, selected_ps,
            color='navy', marker='*', s=150)
ax2.scatter(selected_tc, selected_br, selected_ls,
            color='darkred', marker='*', s=150)

# Annotate the points
ax2.text(selected_tc, selected_br, selected_ps + 0.7,
         f'{selected_ps:.2f}', color='navy', fontsize=20)
ax2.text(selected_tc, selected_br, selected_ls - 0.7,
         f'{selected_ls:.2f}', color='darkred', fontsize=20)

# Create a single legend for both plots
legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label='Profits'),
    Patch(facecolor='red', alpha=0.7, label='Losses'),
    Line2D([0], [0], marker='*', color='black', label='Selected Point\n(TC=0.015, Br=0.0025)',
           markerfacecolor='black', markersize=15, linestyle='None')
]

# Add legend to the figure (not to individual axes)
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.2),
           ncol=3, fontsize=20, frameon=True, fancybox=True, shadow=True)

plt.show()
