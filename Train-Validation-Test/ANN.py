import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             classification_report, roc_auc_score)
from itertools import product

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=Warning, module='scipy')
warnings.filterwarnings('ignore', category=Warning, module='ReliefF')
warnings.filterwarnings('ignore', category=Warning, module='mlxtend')
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# START TIMER
# ============================================================================
start = time.time()

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
initial_window = 1260  # Initial training window size (5 years of trading days)
test_window = 252      # Test window size (1 year of trading days)
num_windows = 7        # Number of sliding windows
num_features = 50      # Number of features to select
topk = 200             # Initial feature selection threshold
riskfree = 0.02        # Risk-free rate for performance metrics

# Walk-Forward Cross-Validation Parameters
cv_window = 252        # Size of each CV fold (1 year)

# MLP Hyperparameter Grid for Tuning
param_grid = {
    'hidden_layer_sizes': [(16, 8), (32, 16), (32, 16, 8), (64, 32, 16)],  # Network architectures
    'activation': ['relu', 'tanh', 'logistic'],  # Activation functions
    'solver': ['adam', 'sgd', 'lbfgs'],  # Optimization algorithms
    'alpha': [ 0.001, 0.01, 0.1],  # L2 regularization strength
    'batch_size': [32, 64, 128],  # Mini-batch sizes
    'learning_rate': ['constant', 'adaptive', 'invscaling'],  # Learning rate schedules
    'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rates
    'max_iter': [200, 400, 600],  # Maximum iterations
    'early_stopping': [True],  # Early stopping
    #'validation_fraction': [0.1, 0.15, 0.2],  # Validation set size
    #'n_iter_no_change': [10, 20, 50],  # Early stopping patience
    #'tol': [1e-4, 1e-3, 1e-2],  # Tolerance for optimization
    #'momentum': [0.8, 0.9, 0.95],  # Momentum for SGD
    #'nesterovs_momentum': [True, False],  # Nesterov momentum
    #'beta_1': [0.9, 0.95, 0.99],  # Adam beta1
    #'beta_2': [0.999, 0.9999],  # Adam beta2
    #'epsilon': [1e-8, 1e-7],  # Numerical stability
    #'power_t': [0.5, 0.6],  # Power for inverse scaling
    'random_state': [42],  # Random seed
    'shuffle': [False],  # Shuffle training data
    'warm_start': [False],  # Warm start
    'verbose': [False]  # Verbosity
}

# Trading parameters
cb = 0.0075    # Buy commission
cs = 0.0075    # Sell commission

# Initialize scaler - MLP requires careful scaling for optimal performance
scaler = StandardScaler()

# ============================================================================
# EXTRACT FEATURE NAMES
# ============================================================================
list_features = list(df.loc[:, 'SMA_6':'UI_20'].columns)
print(f"Total features available: {len(list_features)}")

# ============================================================================
# INITIALIZE STORAGE VARIABLES
# ============================================================================
yptt = []              # Store predictions for each window
modelval_MLP = []       # Store trading system values over time
y_test_window_MLP = []  # Store all test predictions concatenated
training_loss_history = []       # Store training loss curves across windows
best_params_history = []  # Store best parameters from CV for each window
cv_results_history = []  # Store CV results for each window
cv_folds_history = []    # Track number of CV folds used per window

# ============================================================================
# SLIDING WINDOW TRAINING AND PREDICTION
# ============================================================================
print("\n" + "="*80)
print("STARTING SLIDING WINDOW ANALYSIS WITH MULTI-LAYER PERCEPTRON AND DYNAMIC CROSS-VALIDATION")
print("="*80 + "\n")

for window_idx in range(num_windows):
    print(f"\n{'='*60}")
    print(f"PROCESSING WINDOW {window_idx + 1}/{num_windows}")
    print(f"{'='*60}\n")
    
    # Calculate window boundaries
    offset = test_window * window_idx
    train_end = initial_window + offset
    test_start = train_end
    test_end = test_start + test_window
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    X_train_raw = df.drop('LABEL', axis=1)[:train_end]
    X_test_raw = df.drop('LABEL', axis=1)[test_start:test_end]
    y_train = df['LABEL'][:train_end]
    y_test = df['LABEL'][test_start:test_end]
    
    # Scale data - CRITICAL for MLP performance
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
    
    # ========================================================================
    # FEATURE SELECTION - STEP 1: UNIVARIATE METHODS
    # ========================================================================
    print("\n--- Feature Selection Step 1: Univariate Methods ---")
    
    x_main = X_train.copy()
    x_main_raw = X_train_raw.values
    
    # ANOVA F-test
    select_k_best_anova = SelectKBest(f_classif, k=topk)
    select_k_best_anova.fit(x_main, y_train)
    selected_indices_anova = select_k_best_anova.get_support(indices=True)
    selected_features_anova = [list_features[i] for i in selected_indices_anova if i < len(list_features)]
    print(f"ANOVA selected {len(selected_features_anova)} features")
    
    # Mutual Information
    select_k_best_mi = SelectKBest(mutual_info_classif, k=topk)
    select_k_best_mi.fit(x_main, y_train)
    selected_indices_mi = select_k_best_mi.get_support(indices=True)
    selected_features_mi = [list_features[i] for i in selected_indices_mi if i < len(list_features)]
    print(f"Mutual Info selected {len(selected_features_mi)} features")
    
    # Chi-Square test
    if np.any(x_main_raw < 0):
        minmax_scaler = MinMaxScaler()
        x_main_chi2 = minmax_scaler.fit_transform(x_main_raw)
    else:
        x_main_chi2 = x_main_raw
    
    select_k_best_chi2 = SelectKBest(chi2, k=topk)
    select_k_best_chi2.fit(x_main_chi2, y_train)
    selected_indices_chi2 = select_k_best_chi2.get_support(indices=True)
    selected_features_chi2 = [list_features[i] for i in selected_indices_chi2 if i < len(list_features)]
    print(f"Chi-Square selected {len(selected_features_chi2)} features")
    
    # ========================================================================
    # FEATURE SELECTION - STEP 2: INTERSECTION
    # ========================================================================
    print("\n--- Feature Selection Step 2: Finding Common Features ---")
    
    common = list(set(set(selected_features_anova).intersection(selected_features_mi)).intersection(selected_features_chi2))
    print(f"Common features found: {len(common)}")
    
    if len(common) < num_features:
        print(f"WARNING: Only {len(common)} common features found, needed {num_features}")
        common_union = list(set(set(selected_features_anova).union(selected_features_mi)).union(selected_features_chi2))
        print(f"Using union instead: {len(common_union)} features")
        common = common_union[:num_features]
    
    feat_idx = sorted([list_features.index(f) for f in common])
    X_train_common = X_train[:, feat_idx]
    X_test_common = X_test[:, feat_idx]
    
    # ========================================================================
    # FEATURE SELECTION - STEP 3: RELIEFF
    # ========================================================================
    print("\n--- Feature Selection Step 3: ReliefF ---")
    
    try:
        from sklearn_relief import ReliefF
        
        fs = ReliefF(n_features=num_features)
        fs.fit(X_train_common, y_train.values)
        
        if hasattr(fs, 'feature_importances_'):
            feature_scores = fs.feature_importances_
            top_indices = np.argsort(feature_scores)[-num_features:][::-1]
            selected_feature_names_relief = [common[i] for i in top_indices]
        else:
            X_train_df = pd.DataFrame(X_train_common, columns=common).reset_index(drop=True)
            X_train_transformed = fs.transform(X_train_common)
            X_transformed_df = pd.DataFrame(X_train_transformed).reset_index(drop=True)
            
            selected_feature_names_relief = []
            for col_idx in range(X_transformed_df.shape[1]):
                transformed_col = X_transformed_df.iloc[:, col_idx].values
                for original_col in X_train_df.columns:
                    original_values = X_train_df[original_col].values
                    if np.allclose(transformed_col, original_values, rtol=1e-5, atol=1e-8):
                        selected_feature_names_relief.append(original_col)
                        break
        
        print(f"ReliefF selected {len(selected_feature_names_relief)} features")
        
    except Exception as e:
        print(f"ReliefF error: {str(e)}, using top features from common")
        selected_feature_names_relief = common[:num_features]
    
    # ========================================================================
    # FEATURE SELECTION - STEP 4: SFFS
    # ========================================================================
    print("\n--- Feature Selection Step 4: SFFS ---")
    
    try:
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS
        
        X_sffs = df[selected_feature_names_relief][:train_end].values
        y_sffs = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Use default MLP for SFFS
        sfs = SFS(
            MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=500, random_state=42, verbose=False),
            k_features=(1, 20),
            forward=True,
            floating=True,
            verbose=0,
            scoring='accuracy',
            cv=None,
            n_jobs=-1
        )
        
        sfs.fit(X_sffs, y_sffs)
        final_features = [selected_feature_names_relief[i] for i in sfs.k_feature_idx_]
        print(f"SFFS selected {len(final_features)} features")
        
    except ImportError:
        print("mlxtend not installed, using top 20 from ReliefF")
        final_features = selected_feature_names_relief[:20]
    
    # ========================================================================
    # CREATE FINAL DATASET WITH SELECTED FEATURES
    # ========================================================================
    df_novel = pd.concat([df[final_features], df['LABEL']], axis=1)
    
    X_train_final = df_novel.drop('LABEL', axis=1)[:train_end]
    X_test_final = df_novel.drop('LABEL', axis=1)[test_start:test_end]
    y_train_final = df_novel['LABEL'][:train_end]
    y_test_final = df_novel['LABEL'][test_start:test_end]
    
    # ========================================================================
    # DYNAMIC CROSS-VALIDATION SETUP
    # ========================================================================
    print("\n" + "="*60)
    print(f"DYNAMIC CROSS-VALIDATION - WINDOW {window_idx + 1}")
    print("="*60 + "\n")
    
    # Calculate maximum number of CV folds based on available training data
    max_possible_folds = len(X_train_final) // cv_window
    cv_folds = max_possible_folds
    
    if cv_folds < 2:
        print(f"WARNING: Not enough data for CV. Available: {len(X_train_final)} days, needed: {cv_window*2} days")
        cv_folds = 2  # Minimum 2 folds for validation
    
    print(f"Using {cv_folds} CV folds for this window")
    print(f"Training data size: {len(X_train_final)} days ({len(X_train_final)/252:.1f} years)")
    print(f"CV fold size: {cv_window} days (1 year each)")
    
    # Store CV folds info
    cv_folds_history.append(cv_folds)
    
    # ========================================================================
    # NEURAL NETWORK PARAMETER OPTIMIZATION WITH DYNAMIC CV
    # ========================================================================
    print("\n" + "="*60)
    print(f"NEURAL NETWORK PARAMETER OPTIMIZATION - WINDOW {window_idx + 1}")
    print("="*60 + "\n")
    
    # Generate all parameter combinations
    param_combinations = []
    keys = list(param_grid.keys())
    for values in product(*[param_grid[k] for k in keys]):
        param_dict = dict(zip(keys, values))
        
        # Filter invalid combinations
        # Early stopping requires validation_fraction
        if param_dict['early_stopping'] and param_dict['validation_fraction'] <= 0:
            continue
        
        # Some solvers don't support certain parameters
        if param_dict['solver'] == 'lbfgs':
            # lbfgs doesn't use batch_size, momentum, or nesterovs_momentum
            param_dict['batch_size'] = 'auto'
        
        param_combinations.append(param_dict)
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    # Storage for CV results
    cv_results = []
    
    # Test each parameter combination
    for param_idx, params in enumerate(param_combinations):
        fold_accuracies = []
        
        # Perform walk-forward CV
        for fold in range(1, cv_folds):
            train_cv_end = cv_window * fold
            val_cv_start = train_cv_end
            val_cv_end = val_cv_start + cv_window
            
            if val_cv_end > len(X_train_final):
                continue
                
            # Extract CV data
            X_cv_train = X_train_final.iloc[:train_cv_end]
            y_cv_train = y_train_final.iloc[:train_cv_end]
            X_cv_val = X_train_final.iloc[val_cv_start:val_cv_end]
            y_cv_val = y_train_final.iloc[val_cv_start:val_cv_end]
            
            # Scale CV data
            cv_scaler = StandardScaler()
            X_cv_train_scaled = cv_scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = cv_scaler.transform(X_cv_val)
            
            try:
                # Create MLP with current parameters
                # Reduce max_iter for faster CV
                cv_params = params.copy()
                cv_params['max_iter'] = min(200, cv_params['max_iter'])  # Reduce for CV
                cv_params['verbose'] = False
                
                model = MLPClassifier(**cv_params)
                model.fit(X_cv_train_scaled, y_cv_train)
                y_cv_pred = model.predict(X_cv_val_scaled)
                fold_acc = accuracy_score(y_cv_val, y_cv_pred)
                fold_accuracies.append(fold_acc)
                
            except Exception as e:
                print(f"  Warning: MLP failed for params {params} on fold {fold}: {str(e)}")
                fold_accuracies.append(0.0)
        
        # Only consider combinations that were tested on all folds
        if len(fold_accuracies) > 0:
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            
            # Store results
            cv_results.append({
                'params': params,
                'fold_accuracies': fold_accuracies,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'num_folds': len(fold_accuracies)
            })
        
        # Progress update
        if (param_idx + 1) % 10 == 0:
            print(f"  Tested {param_idx + 1}/{len(param_combinations)} combinations...")
    
    # Find best parameters
    if cv_results:
        best_result = max(cv_results, key=lambda x: x['mean_accuracy'])
        best_params = best_result['params']
        best_mean_acc = best_result['mean_accuracy']
        best_fold_accs = best_result['fold_accuracies']
        
        print(f"\n{'='*60}")
        print(f"BEST NEURAL NETWORK PARAMETERS FOUND:")
        print(f"{'='*60}")
        for key, value in best_params.items():
            if key not in ['verbose']:  # Skip verbose in display
                print(f"  {key}: {value}")
        print(f"\nCross-Validation Accuracies by Fold:")
        for fold_idx, acc in enumerate(best_fold_accs):
            print(f"  Fold {fold_idx + 1}: {acc:.4f}")
        print(f"Mean CV Accuracy: {best_mean_acc:.4f} (+/- {best_result['std_accuracy']:.4f})")
        print(f"Number of CV folds used: {best_result['num_folds']}")
        print(f"{'='*60}\n")
    else:
        print("ERROR: No valid CV results. Using safe default parameters.")
        best_params = {
            'hidden_layer_sizes': (32, 16, 8),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 64,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'random_state': 42,
            'shuffle': False,
            'warm_start': False,
            'verbose': False
        }
        best_mean_acc = 0.0
        best_fold_accs = []
    
    # Store CV results for this window
    best_params_history.append(best_params)
    cv_results_history.append(cv_results)
    
    # ========================================================================
    # VISUALIZE CV RESULTS
    # ========================================================================
    if best_fold_accs:
        fig, ax = plt.subplots(figsize=(7, 5))
        fold_indices = np.arange(1, len(best_fold_accs) + 1)
        
        ax.scatter(fold_indices, best_fold_accs, s=50, c='green', edgecolors='black', zorder=3)
        ax.plot(fold_indices, best_fold_accs, 'o-', linewidth=1.5, markersize=0, color='gray', alpha=0.5)
        ax.axhline(y=best_mean_acc, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, 
                   label=f'Mean Accuracy: {best_mean_acc:.4f}')
        
        ax.set_xlabel('CV Fold', fontsize=15)
        ax.set_ylabel('Accuracy', fontsize=15)
        ax.set_title(f'Window {window_idx + 1}: MLP CV Results ({len(best_fold_accs)} folds)', fontsize=18)
        ax.set_xticks(fold_indices)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # TRAIN FINAL NEURAL NETWORK WITH OPTIMIZED PARAMETERS
    # ========================================================================
    print("\n--- Training Final Neural Network with Optimized Parameters ---")
    
    # Scale the full training data
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    X_test_final_scaled = final_scaler.transform(X_test_final)
    
    # Train MLP with best parameters
    try:
        # Restore full max_iter for final training
        train_params = best_params.copy()
        if 'max_iter' in train_params and train_params['max_iter'] < 500:
            train_params['max_iter'] = 1000  # Use more iterations for final training
        
        model_mlp = MLPClassifier(**train_params)
        model_mlp.fit(X_train_final_scaled, y_train_final)
        
        # Display network architecture
        total_params = 0
        for i, coef in enumerate(model_mlp.coefs_):
            total_params += coef.size + model_mlp.intercepts_[i].size
        
        print(f"Neural Network trained with architecture: {best_params['hidden_layer_sizes']}")
        print(f"Activation function: {best_params['activation']}")
        print(f"Optimizer: {best_params['solver']}")
        print(f"Learning rate: {best_params['learning_rate']} ({best_params['learning_rate_init']})")
        print(f"Regularization (alpha): {best_params['alpha']}")
        print(f"Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"Warning: MLP failed to train with best params: {str(e)}")
        print("Using fallback parameters...")
        
        # Fallback to simple MLP
        fallback_params = {
            'hidden_layer_sizes': (32, 16, 8),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 64,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'early_stopping': True,
            'validation_fraction': 0.15,
            'random_state': 42,
            'shuffle': False,
            'warm_start': False,
            'verbose': False
        }
        
        model_mlp = MLPClassifier(**fallback_params)
        model_mlp.fit(X_train_final_scaled, y_train_final)
    
    # Make predictions
    y_pred_test = model_mlp.predict(X_test_final_scaled)
    y_pred_train = model_mlp.predict(X_train_final_scaled)
    y_pred_prob_test = model_mlp.predict_proba(X_test_final_scaled)[:, 1]
    
    # Evaluate model
    test_accuracy = accuracy_score(y_test_final, y_pred_test)
    train_accuracy = accuracy_score(y_train_final, y_pred_train)
    
    try:
        auc_roc = roc_auc_score(y_test_final, y_pred_prob_test)
        print(f"AUC-ROC Score: {auc_roc:.4f}")
    except:
        auc_roc = None
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_final, y_pred_test)}")
    
    # Store training loss curve
    if hasattr(model_mlp, 'loss_curve_') and model_mlp.loss_curve_ is not None:
        training_loss_history.append({
            'window': window_idx,
            'loss_curve': model_mlp.loss_curve_,
            'final_loss': model_mlp.loss_curve_[-1],
            'iterations': len(model_mlp.loss_curve_)
        })
        print(f"Final training loss: {model_mlp.loss_curve_[-1]:.4f}")
        print(f"Training iterations: {len(model_mlp.loss_curve_)}")
    
    # Store predictions
    yptt.append(y_pred_test)
    
    # Display layer information
    print(f"\nNeural Network Layer Information:")
    for i, (coef, intercept) in enumerate(zip(model_mlp.coefs_, model_mlp.intercepts_)):
        layer_type = "Input → Hidden" if i == 0 else f"Hidden {i} → Hidden {i+1}" if i < len(model_mlp.coefs_)-1 else "Hidden → Output"
        print(f"  Layer {i+1} ({layer_type}): {coef.shape[0]}×{coef.shape[1]} weights, {intercept.shape[0]} biases")

# ============================================================================
# CONSOLIDATE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("CONSOLIDATING PREDICTIONS")
print("="*80 + "\n")

for i in range(num_windows):
    y_test_window_MLP.extend(yptt[i])

print(f"Total predictions: {len(y_test_window_MLP)}")

# ============================================================================
# CV FOLDS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION FOLDS ANALYSIS")
print("="*80 + "\n")

if cv_folds_history:
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = np.arange(1, num_windows + 1)
    
    ax.bar(windows, cv_folds_history, color='dodgerblue', edgecolor='black', alpha=0.7)
    ax.plot(windows, cv_folds_history, 'o-', color='blue', linewidth=2, markersize=8)
    
    ax.set_xlabel('Window', fontsize=12)
    ax.set_ylabel('Number of CV Folds', fontsize=12)
    ax.set_title('Number of Cross-Validation Folds per Window (Neural Network)', fontsize=14)
    ax.set_xticks(windows)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(cv_folds_history):
        ax.text(i + 1, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCV Folds Summary for Neural Network:")
    for window_idx, num_folds in enumerate(cv_folds_history):
        training_years = (initial_window + test_window * window_idx) / 252
        print(f"Window {window_idx + 1}: {training_years:.1f} years training data → {num_folds} CV folds")

# ============================================================================
# OVERALL MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("OVERALL MODEL PERFORMANCE WITH PARAMETER OPTIMIZATION AND DYNAMIC CV")
print("="*80 + "\n")

y_true = df['LABEL'][initial_window:]
conf_matrix = confusion_matrix(y_true, y_test_window_MLP)
acc = accuracy_score(y_true, y_test_window_MLP)
precision = precision_score(y_true, y_test_window_MLP)
recall = recall_score(y_true, y_test_window_MLP)
f1 = f1_score(y_true, y_test_window_MLP)
mcc = matthews_corrcoef(y_true, y_test_window_MLP)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"\n{classification_report(y_true, y_test_window_MLP, digits=4)}")

# ============================================================================
# VISUALIZATION: CONFUSION MATRIX
# ============================================================================
group_names = ['True Sell', 'False Buy', 'False Sell', 'True Buy']
group_counts = [f"{value:0.0f}" for value in conf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=labels, linewidths=1, fmt='', 
            cmap='coolwarm', annot_kws={"size": 14})
ax.set_xlabel('Predicted Labels', fontsize=15)
ax.set_ylabel('Actual Labels', fontsize=15)
ax.set_title('Neural Network - Confusion Matrix (With Dynamic CV)', fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# NEURAL NETWORK PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV
# ============================================================================
print("\n" + "="*80)
print("NEURAL NETWORK PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV")
print("="*80 + "\n")

print("Best Parameters Selected for Each Window:")
print("-" * 60)

hidden_layers_stats = {}
activation_stats = {}
solver_stats = {}
alpha_stats = {}
batch_size_stats = {}
learning_rate_stats = {}
learning_rate_init_stats = {}
max_iter_stats = {}

for window_idx, params in enumerate(best_params_history):
    print(f"\nWindow {window_idx + 1}:")
    print(f"  hidden_layer_sizes: {params.get('hidden_layer_sizes', 'N/A')}")
    print(f"  activation: {params.get('activation', 'N/A')}")
    print(f"  solver: {params.get('solver', 'N/A')}")
    print(f"  alpha: {params.get('alpha', 'N/A')}")
    print(f"  batch_size: {params.get('batch_size', 'N/A')}")
    print(f"  learning_rate: {params.get('learning_rate', 'N/A')}")
    print(f"  learning_rate_init: {params.get('learning_rate_init', 'N/A')}")
    print(f"  max_iter: {params.get('max_iter', 'N/A')}")
    print(f"  CV folds used: {cv_folds_history[window_idx]}")
    
    # Collect statistics
    hidden_layers = str(params.get('hidden_layer_sizes', 'N/A'))
    activation = params.get('activation', 'N/A')
    solver = params.get('solver', 'N/A')
    alpha = params.get('alpha', 'N/A')
    batch_size = params.get('batch_size', 'N/A')
    learning_rate = params.get('learning_rate', 'N/A')
    learning_rate_init = params.get('learning_rate_init', 'N/A')
    max_iter = params.get('max_iter', 'N/A')
    
    hidden_layers_stats[hidden_layers] = hidden_layers_stats.get(hidden_layers, 0) + 1
    activation_stats[activation] = activation_stats.get(activation, 0) + 1
    solver_stats[solver] = solver_stats.get(solver, 0) + 1
    alpha_stats[alpha] = alpha_stats.get(alpha, 0) + 1
    batch_size_stats[batch_size] = batch_size_stats.get(batch_size, 0) + 1
    learning_rate_stats[learning_rate] = learning_rate_stats.get(learning_rate, 0) + 1
    learning_rate_init_stats[learning_rate_init] = learning_rate_init_stats.get(learning_rate_init, 0) + 1
    max_iter_stats[max_iter] = max_iter_stats.get(max_iter, 0) + 1

# Print statistics
print("\n" + "="*60)
print("NEURAL NETWORK PARAMETER SELECTION STATISTICS WITH DYNAMIC CV:")
print("="*60)

print("\nHidden Layer Architectures:")
for layers, count in sorted(hidden_layers_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {layers}: {count} windows ({percentage:.1f}%)")

print("\nActivation Functions:")
for activation, count in sorted(activation_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {activation}: {count} windows ({percentage:.1f}%)")

print("\nOptimization Algorithms:")
for solver, count in sorted(solver_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {solver}: {count} windows ({percentage:.1f}%)")

print("\nRegularization Strength (Alpha):")
for alpha, count in sorted(alpha_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {alpha}: {count} windows ({percentage:.1f}%)")

print("\nBatch Sizes:")
for batch_size, count in sorted(batch_size_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {batch_size}: {count} windows ({percentage:.1f}%)")

print("\nLearning Rate Schedules:")
for lr_schedule, count in sorted(learning_rate_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {lr_schedule}: {count} windows ({percentage:.1f}%)")

print("\nInitial Learning Rates:")
for lr_init, count in sorted(learning_rate_init_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {lr_init}: {count} windows ({percentage:.1f}%)")

print("\nMaximum Iterations:")
for iterations, count in sorted(max_iter_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {iterations}: {count} windows ({percentage:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Hidden layers selection
layers, layer_counts = zip(*sorted(hidden_layers_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 0].bar(range(len(layers)), layer_counts, color='dodgerblue')
axes[0, 0].set_xlabel('Hidden Layers', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].set_title('Network Architecture Selection', fontsize=12)
axes[0, 0].set_xticks(range(len(layers)))
axes[0, 0].set_xticklabels(layers, rotation=45, fontsize=8)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Activation functions selection
activations, activation_counts = zip(*sorted(activation_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 1].bar(activations, activation_counts, color='steelblue')
axes[0, 1].set_xlabel('Activation Function', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].set_title('Activation Function Selection', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Solvers selection
solvers, solver_counts = zip(*sorted(solver_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 2].bar(solvers, solver_counts, color='lightblue')
axes[0, 2].set_xlabel('Optimizer', fontsize=10)
axes[0, 2].set_ylabel('Frequency', fontsize=10)
axes[0, 2].set_title('Optimizer Selection', fontsize=12)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Alpha values selection
alphas, alpha_counts = zip(*sorted(alpha_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 3].bar([str(a) for a in alphas], alpha_counts, color='skyblue')
axes[0, 3].set_xlabel('Regularization (Alpha)', fontsize=10)
axes[0, 3].set_ylabel('Frequency', fontsize=10)
axes[0, 3].set_title('Regularization Selection', fontsize=12)
axes[0, 3].grid(True, alpha=0.3, axis='y')

# Batch sizes selection
batch_sizes, batch_counts = zip(*sorted(batch_size_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 0].bar([str(b) for b in batch_sizes], batch_counts, color='deepskyblue')
axes[1, 0].set_xlabel('Batch Size', fontsize=10)
axes[1, 0].set_ylabel('Frequency', fontsize=10)
axes[1, 0].set_title('Batch Size Selection', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Learning rate schedules selection
lr_schedules, lr_counts = zip(*sorted(learning_rate_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 1].bar(lr_schedules, lr_counts, color='lightskyblue')
axes[1, 1].set_xlabel('Learning Rate Schedule', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].set_title('Learning Rate Schedule Selection', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Initial learning rates selection
lr_inits, lr_init_counts = zip(*sorted(learning_rate_init_stats.items(), key=lambda x: x[1], reverse=True))
axes[1, 2].bar([str(lr) for lr in lr_inits], lr_init_counts, color='aliceblue')
axes[1, 2].set_xlabel('Initial Learning Rate', fontsize=10)
axes[1, 2].set_ylabel('Frequency', fontsize=10)
axes[1, 2].set_title('Initial Learning Rate Selection', fontsize=12)
axes[1, 2].grid(True, alpha=0.3, axis='y')

# CV Folds Evolution
windows = np.arange(1, num_windows + 1)
axes[1, 3].plot(windows, cv_folds_history, 'o-', linewidth=2, markersize=8, color='darkblue')
axes[1, 3].set_xlabel('Window', fontsize=12)
axes[1, 3].set_ylabel('Number of CV Folds', fontsize=12)
axes[1, 3].set_title('CV Folds Evolution (Neural Network)', fontsize=14)
axes[1, 3].set_xticks(windows)
axes[1, 3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# TRAINING LOSS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("NEURAL NETWORK TRAINING LOSS ANALYSIS")
print("="*80 + "\n")

if training_loss_history:
    # Plot training loss curves
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, loss_data in enumerate(training_loss_history[:min(8, len(training_loss_history))]):
        row = idx // 4
        col = idx % 4
        window_idx = loss_data['window']
        loss_curve = loss_data['loss_curve']
        
        axes[row, col].plot(loss_curve, color='blue', linewidth=1.5)
        axes[row, col].set_xlabel('Iteration', fontsize=9)
        axes[row, col].set_ylabel('Loss', fontsize=9)
        axes[row, col].set_title(f'Window {window_idx + 1} Loss Curve', fontsize=10)
        axes[row, col].grid(True, alpha=0.3)
        
        # Mark final loss
        final_loss = loss_curve[-1]
        axes[row, col].scatter(len(loss_curve)-1, final_loss, color='red', s=30, zorder=5)
    
    # Hide empty subplots
    for idx in range(len(training_loss_history), 8):
        row = idx // 4
        col = idx % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display training statistics
    final_losses = [loss_data['final_loss'] for loss_data in training_loss_history]
    iterations = [loss_data['iterations'] for loss_data in training_loss_history]
    
    print(f"Training Statistics:")
    print(f"  Average final loss: {np.mean(final_losses):.4f}")
    print(f"  Average iterations: {np.mean(iterations):.0f}")
    print(f"  Best final loss: {np.min(final_losses):.4f} (Window {np.argmin(final_losses)+1})")
    print(f"  Worst final loss: {np.max(final_losses):.4f} (Window {np.argmax(final_losses)+1})")


# ============================================================================
# TRADING SIMULATION
# ============================================================================
print("\n" + "="*80)
print("TRADING SIMULATION WITH OPTIMIZED NEURAL NETWORK AND DYNAMIC CV")
print("="*80 + "\n")

initial_capital = 1000
sm1 = initial_capital
f = 0
ns = 0
nn = test_window
n = initial_window

open_prices = df['Open'].values
close_prices = df['Close'].values

modelval_MLP = []

for j in range(num_windows):
    for i in range(len(yptt[j])):
        price_idx = j * nn + i + n
        open_price = open_prices[price_idx]
        close_price = close_prices[price_idx]
        signal = yptt[j][i]
        
        if signal == 1 and f == 0:
            ns = (sm1 / (1 + cb)) / open_price
            sm1 = ns * close_price
            modelval_MLP.append(sm1)
            f = 1
        elif signal == 1 and f == 1:
            sm1 = ns * close_price
            modelval_MLP.append(sm1)
        elif signal == 0 and f == 0:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = sm1 * (1 - price_change_pct)
            modelval_MLP.append(sm1)
        elif signal == 0 and f == 1:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = ns * open_price * (1 - cs) * (1 - price_change_pct)
            modelval_MLP.append(sm1)
            f = 0
            ns = 0

total_return = ((modelval_MLP[-1] - initial_capital) / initial_capital) * 100

print(f"{'='*60}")
print(f"TRADING RESULTS WITH OPTIMIZED NEURAL NETWORK AND DYNAMIC CV")
print(f"{'='*60}")
print(f"Initial Investment:     ${initial_capital:.2f}")
print(f"Final Investment Value: ${modelval_MLP[-1]:.2f}")
print(f"Total Return:           {total_return:.2f}%")
print(f"Absolute Profit/Loss:   ${modelval_MLP[-1] - initial_capital:.2f}")
print(f"{'='*60}\n")

plt.figure(figsize=(12, 6))
plt.plot(modelval_MLP, label='Investment Value', linewidth=2, color='blue')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Investment')
plt.xlabel('Trading Day')
plt.ylabel('Investment Value ($)')
plt.title('Trading System Performance with Optimized Neural Network (Dynamic CV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE METRICS WITH DYNAMIC CV")
print("="*80 + "\n")

def max_drawdown(series, window=252):
    roll_max = series.rolling(window, min_periods=1).max()
    daily_drawdown = series / roll_max - 1
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()
    return max_daily_drawdown

df_modelval_MLP = pd.DataFrame(modelval_MLP)
num_years = num_windows
aar = (((modelval_MLP[-1] / modelval_MLP[0]) ** (1 / num_years)) - 1) * 100
print(f"Average Annual Return (AAR): {aar:.2f}%")

mdd_series = max_drawdown(df_modelval_MLP)
mdd_value = mdd_series.min()
if isinstance(mdd_value, pd.Series):
    mdd_value = mdd_value.iloc[0]
print(f"Maximum Drawdown (MDD): {mdd_value:.4f}")

modelval_MLP_list = modelval_MLP.copy()
modelval_MLP_list.insert(0, initial_capital)

port_value = pd.DataFrame(modelval_MLP_list, columns=['Value'])
ret_data = port_value.pct_change()

annal_return_MLP = []
for year_idx in range(num_years):
    start_idx = year_idx * test_window
    end_idx = start_idx + test_window
    start_val = port_value.iloc[start_idx, 0]
    end_val = port_value.iloc[end_idx, 0]
    annual_ret = (end_val - start_val) / start_val
    annal_return_MLP.append(annual_ret)

annal_return_MLP_df = pd.DataFrame(annal_return_MLP, columns=['Return'])

dffff = pd.DataFrame(np.array(ret_data), columns=['Returns'])
dffff['downside_returns'] = 0.0
dffff_clean = dffff.dropna()
dffff_clean.loc[dffff_clean['Returns'] < 0, 'downside_returns'] = abs(dffff_clean['Returns'])

negative_returns = dffff_clean[dffff_clean['Returns'] < 0]['Returns']
std_neg = negative_returns.std() if len(negative_returns) > 0 else 0

annual_downside_std = dffff_clean['downside_returns'].std() * np.sqrt(252)
print(f"Annualized Downside Std: {annual_downside_std:.4f}")

if std_neg > 0:
    sortino = ((((modelval_MLP_list[-1] / modelval_MLP_list[0]) ** (1 / num_years)) - 1) - riskfree) / (std_neg * np.sqrt(252))
    print(f"Sortino Ratio: {sortino:.4f}")

annual_mean_return = (((modelval_MLP_list[-1] / modelval_MLP_list[0]) ** (1 / num_years)) - 1)
annual_return_std = annal_return_MLP_df['Return'].std()

sharp_ratio = ((annual_mean_return - riskfree) / annual_return_std) if annual_return_std > 0 else 0
calmar_ratio = ((annual_mean_return - riskfree) / (-mdd_value)) if mdd_value != 0 else 0
sortino_ratio = ((annual_mean_return - riskfree) / annual_downside_std) if annual_downside_std > 0 else 0

print(f"\nSharpe Ratio: {sharp_ratio:.4f}")
print(f"Calmar Ratio: {calmar_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")

# ============================================================================
# EXECUTION TIME
# ============================================================================
end = time.time()
execution_time = (end - start) / 60
print("\n" + "="*80)
print(f"Total Execution Time: {execution_time:.2f} minutes")
print("="*80)
