import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, 
                             classification_report, roc_auc_score)
from itertools import product
from operator import itemgetter

# ============================================================================
# SUPPRESS WARNINGS
# ============================================================================
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

# Hyperparameter Grid for Support Vector Machine
param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0, 100.0],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel types
    'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],  # Kernel coefficient
    'degree': [2, 3, 4],  # For polynomial kernel
    'coef0': [0.0, 0.1, 0.5, 1.0],  # For poly and sigmoid kernels
    #'shrinking': [True, False],
    'probability': [True],
    #'cache_size': [200],
    #'class_weight': [None, 'balanced'],
    'max_iter': [10000],  # Increased for convergence
    'random_state': [0],
    #'tol': [1e-3, 1e-4],
    'verbose': [False]
}

# Trading parameters
cb = 0.0075    # Buy commission
cs = 0.0075    # Sell commission

# Initialize scaler
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
modelval_SVM = []      # Store trading system values over time
y_test_window_SVM = [] # Store all test predictions concatenated
feature_importance_history = []  # Store feature importance across windows
support_vectors_history = []     # Store number of support vectors per window
best_params_history = []  # Store best parameters from CV for each window
cv_results_history = []  # Store CV results for each window
convergence_history = []  # Track convergence issues
cv_folds_history = []    # Track number of CV folds used per window

# ============================================================================
# SLIDING WINDOW TRAINING AND PREDICTION
# ============================================================================
print("\n" + "="*80)
print("STARTING SLIDING WINDOW ANALYSIS WITH SVM AND DYNAMIC CROSS-VALIDATION")
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
    
    # Scale data
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
        
        # Use default SVM for SFFS
        sfs = SFS(
            SVC(C=1.0, kernel='rbf', probability=True, random_state=0),
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
    # SVM PARAMETER OPTIMIZATION
    # ========================================================================
    print("\n" + "="*60)
    print(f"SVM PARAMETER OPTIMIZATION - WINDOW {window_idx + 1}")
    print("="*60 + "\n")
    
    # Generate valid parameter combinations
    param_combinations = []
    keys = list(param_grid.keys())
    for values in product(*[param_grid[k] for k in keys]):
        param_dict = dict(zip(keys, values))
        
        # Filter invalid parameter combinations
        # degree parameter is only relevant for polynomial kernel
        if param_dict['kernel'] != 'poly' and param_dict['degree'] != 2:
            continue
        # coef0 is mainly for poly and sigmoid kernels
        if param_dict['kernel'] not in ['poly', 'sigmoid'] and param_dict['coef0'] != 0.0:
            continue
        
        param_combinations.append(param_dict)
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    # Storage for CV results
    cv_results = []
    
    # Test each parameter combination
    for param_idx, params in enumerate(param_combinations):
        fold_accuracies = []
        fold_converged = []
        
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
                model = SVC(**params)
                model.fit(X_cv_train_scaled, y_cv_train)
                y_cv_pred = model.predict(X_cv_val_scaled)
                fold_acc = accuracy_score(y_cv_val, y_cv_pred)
                fold_accuracies.append(fold_acc)
                fold_converged.append(True)
                
            except Exception as e:
                print(f"  Warning: SVM failed for params {params} on fold {fold}: {str(e)}")
                fold_accuracies.append(0.0)
                fold_converged.append(False)
        
        # Only consider combinations that were tested on all folds
        if len(fold_accuracies) > 0:
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            convergence_rate = sum(fold_converged) / len(fold_converged)
            
            # Store results
            cv_results.append({
                'params': params,
                'fold_accuracies': fold_accuracies,
                'fold_converged': fold_converged,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'convergence_rate': convergence_rate,
                'num_folds': len(fold_accuracies)
            })
        
        # Progress update
        if (param_idx + 1) % 10 == 0:
            print(f"  Tested {param_idx + 1}/{len(param_combinations)} combinations...")
    
    # Find best parameters - prioritize convergence
    if cv_results:
        # First filter for models that converged in all folds
        fully_converged = [r for r in cv_results if all(r['fold_converged'])]
        
        if fully_converged:
            best_result = max(fully_converged, key=lambda x: x['mean_accuracy'])
            print("Selected from fully converged models")
        else:
            best_result = max(cv_results, key=lambda x: x['mean_accuracy'])
            print(f"Warning: No model converged in all folds. Best model converged in {best_result['convergence_rate']*100:.1f}% of folds")
        
        best_params = best_result['params']
        best_mean_acc = best_result['mean_accuracy']
        best_fold_accs = best_result['fold_accuracies']
        
        print(f"\n{'='*60}")
        print(f"BEST SVM PARAMETERS FOUND:")
        print(f"{'='*60}")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nMean CV Accuracy: {best_mean_acc:.4f} (+/- {best_result['std_accuracy']:.4f})")
        print(f"Convergence Rate: {best_result['convergence_rate']*100:.1f}%")
        print(f"Number of CV folds used: {best_result['num_folds']}")
        print(f"{'='*60}\n")
    else:
        print("ERROR: No valid CV results. Using safe default parameters.")
        best_params = {
            'C': 1.0,
            'kernel': 'linear',
            'gamma': 'scale',
            'probability': True,
            'shrinking': True,
            'max_iter': 10000,
            'random_state': 0,
            'tol': 1e-3,
            'cache_size': 200,
            'verbose': False
        }
        best_mean_acc = 0.0
        best_fold_accs = []
    
    # Store convergence info
    convergence_history.append({
        'window': window_idx,
        'best_params': best_params,
        'convergence_rate': best_result['convergence_rate'] if cv_results else 0.0,
        'mean_accuracy': best_mean_acc,
        'num_folds': best_result['num_folds'] if cv_results else 0
    })
    
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
        ax.set_title(f'Window {window_idx + 1}: SVM CV Results ({len(best_fold_accs)} folds)', fontsize=18)
        ax.set_xticks(fold_indices)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()        
    # ========================================================================
    # TRAIN FINAL SVM MODEL WITH OPTIMIZED PARAMETERS
    # ========================================================================
    print("\n--- Training Final SVM Model with Optimized Parameters ---")
    
    # Scale the full training data
    final_scaler = StandardScaler()
    X_train_final_scaled = final_scaler.fit_transform(X_train_final)
    X_test_final_scaled = final_scaler.transform(X_test_final)
    
    # Train SVM with best parameters
    try:
        model_svm = SVC(**best_params)
        model_svm.fit(X_train_final_scaled, y_train_final)
        converged_successfully = True
        print("SVM model trained successfully")
        
    except Exception as e:
        print(f"Warning: SVM failed to converge with best params: {str(e)}")
        print("Using fallback parameters...")
        
        # Fallback to simple linear SVM
        fallback_params = {
            'C': 1.0,
            'kernel': 'linear',
            'gamma': 'scale',
            'probability': True,
            'shrinking': True,
            'max_iter': 10000,
            'random_state': 0,
            'tol': 1e-3,
            'cache_size': 200,
            'verbose': False
        }
        
        model_svm = SVC(**fallback_params)
        model_svm.fit(X_train_final_scaled, y_train_final)
        converged_successfully = False
    
    # Make predictions
    y_pred_test = model_svm.predict(X_test_final_scaled)
    y_pred_train = model_svm.predict(X_train_final_scaled)
    y_pred_prob_test = model_svm.predict_proba(X_test_final_scaled)[:, 1]
    
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
    
    # Store feature importance (coefficients for linear kernel)
    if best_params.get('kernel') == 'linear' and hasattr(model_svm, 'coef_') and len(model_svm.coef_) > 0:
        feature_importance = dict(zip(final_features, model_svm.coef_[0]))
        feature_importance_history.append({
            'features': final_features,
            'coefficients': model_svm.coef_[0],
            'best_params': best_params,
            'window_idx': window_idx,
            'converged': converged_successfully
        })
    else:
        # Store variance as proxy for non-linear kernels
        feature_variance = {feature: np.std(X_train_final_scaled[:, idx]) 
                           for idx, feature in enumerate(final_features)}
        feature_importance_history.append({
            'features': final_features,
            'variances': feature_variance,
            'best_params': best_params,
            'window_idx': window_idx,
            'converged': converged_successfully
        })
    
    # Store support vector information
    n_support_vectors = np.sum(model_svm.n_support_) if hasattr(model_svm, 'n_support_') else 0
    support_vectors_history.append({
        'window': window_idx,
        'n_support_vectors': n_support_vectors,
        'n_support_per_class': model_svm.n_support_ if hasattr(model_svm, 'n_support_') else None,
        'best_params': best_params
    })
    
    print(f"Number of support vectors: {n_support_vectors}")
    
    # Store predictions
    yptt.append(y_pred_test)
    
    # Display top 10 features
    print("\nTop 10 features:")
    if best_params.get('kernel') == 'linear' and hasattr(model_svm, 'coef_') and len(model_svm.coef_) > 0:
        print("(Based on SVM coefficients for linear kernel):")
        feature_importance = list(zip(final_features, abs(model_svm.coef_[0])))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for feature, importance in feature_importance[:10]:
            print(f"  {feature}: {importance:.4f}")
    else:
        print(f"(Based on variance for {best_params.get('kernel')} kernel):")
        feature_variance = {feature: np.std(X_train_final_scaled[:, idx]) 
                           for idx, feature in enumerate(final_features)}
        sorted_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)
        for feature, variance in sorted_features[:10]:
            print(f"  {feature}: {variance:.4f}")

# ============================================================================
# CONSOLIDATE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("CONSOLIDATING PREDICTIONS")
print("="*80 + "\n")

for i in range(num_windows):
    y_test_window_SVM.extend(yptt[i])

print(f"Total predictions: {len(y_test_window_SVM)}")

# ============================================================================
# CV FOLDS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION FOLDS ANALYSIS")
print("="*80 + "\n")

if cv_folds_history:
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = np.arange(1, num_windows + 1)
    
    ax.bar(windows, cv_folds_history, color='darkviolet', edgecolor='black', alpha=0.7)
    ax.plot(windows, cv_folds_history, 'o-', color='purple', linewidth=2, markersize=8)
    
    ax.set_xlabel('Window', fontsize=12)
    ax.set_ylabel('Number of CV Folds', fontsize=12)
    ax.set_title('Number of Cross-Validation Folds per Window (SVM)', fontsize=14)
    ax.set_xticks(windows)
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(cv_folds_history):
        ax.text(i + 1, v + 0.1, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\nCV Folds Summary for SVM:")
    for window_idx, num_folds in enumerate(cv_folds_history):
        training_years = (initial_window + test_window * window_idx) / 252
        print(f"Window {window_idx + 1}: {training_years:.1f} years training data → {num_folds} CV folds")

# ============================================================================
# CONVERGENCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SVM CONVERGENCE ANALYSIS REPORT WITH DYNAMIC CV")
print("="*80 + "\n")

if convergence_history:
    conv_rates = [ch['convergence_rate'] for ch in convergence_history]
    mean_conv_rate = np.mean(conv_rates)
    successful_windows = sum(1 for rate in conv_rates if rate == 1.0)
    
    print(f"Average Convergence Rate: {mean_conv_rate:.2%}")
    print(f"Windows with 100% convergence: {successful_windows}/{num_windows}")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = np.arange(1, num_windows + 1)
    conv_rates = [ch['convergence_rate'] for ch in convergence_history]
    mean_accs = [ch['mean_accuracy'] for ch in convergence_history]
    
    ax.bar(windows - 0.2, conv_rates, width=0.4, label='Convergence Rate', color='mediumorchid', alpha=0.7)
    ax.bar(windows + 0.2, mean_accs, width=0.4, label='Mean Accuracy', color='darkviolet', alpha=0.7)
    
    ax.set_xlabel('Window', fontsize=12)
    ax.set_ylabel('Rate / Accuracy', fontsize=12)
    ax.set_title('SVM Convergence Rate vs Mean Accuracy (Dynamic CV)', fontsize=14)
    ax.set_xticks(windows)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# OVERALL MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*80)
print("OVERALL MODEL PERFORMANCE WITH PARAMETER OPTIMIZATION")
print("="*80 + "\n")

y_true = df['LABEL'][initial_window:]
conf_matrix = confusion_matrix(y_true, y_test_window_SVM)
acc = accuracy_score(y_true, y_test_window_SVM)
precision = precision_score(y_true, y_test_window_SVM)
recall = recall_score(y_true, y_test_window_SVM)
f1 = f1_score(y_true, y_test_window_SVM)
mcc = matthews_corrcoef(y_true, y_test_window_SVM)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"\n{classification_report(y_true, y_test_window_SVM, digits=4)}")

# ============================================================================
# VISUALIZATION: CONFUSION MATRIX
# ============================================================================
group_names = ['True Sell', 'False Buy', 'False Sell', 'True Buy']
group_counts = [f"{value:0.0f}" for value in conf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2, 2)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=labels, linewidths=1, fmt='', 
            cmap='Purples', annot_kws={"size": 14})
ax.set_xlabel('Predicted Labels', fontsize=15)
ax.set_ylabel('Actual Labels', fontsize=15)
ax.set_title('SVM - Confusion Matrix (With Dynamic CV)', fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# SVM PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV
# ============================================================================
print("\n" + "="*80)
print("SVM PARAMETER OPTIMIZATION SUMMARY WITH DYNAMIC CV")
print("="*80 + "\n")

print("Best Parameters Selected for Each Window:")
print("-" * 60)

kernel_stats = {}
c_stats = {}
gamma_stats = {}
shrinking_stats = {}
class_weight_stats = {}

for window_idx, params in enumerate(best_params_history):
    print(f"\nWindow {window_idx + 1}:")
    print(f"  kernel: {params.get('kernel', 'N/A')}")
    print(f"  C: {params.get('C', 'N/A')}")
    print(f"  gamma: {params.get('gamma', 'N/A')}")
    print(f"  shrinking: {params.get('shrinking', 'N/A')}")
    print(f"  class_weight: {params.get('class_weight', 'N/A')}")
    print(f"  CV folds used: {cv_folds_history[window_idx]}")
    
    # Collect statistics
    kernel = params.get('kernel', 'N/A')
    c_val = params.get('C', 'N/A')
    gamma = params.get('gamma', 'N/A')
    shrinking = params.get('shrinking', 'N/A')
    class_weight = params.get('class_weight', 'N/A')
    
    kernel_stats[kernel] = kernel_stats.get(kernel, 0) + 1
    c_stats[c_val] = c_stats.get(c_val, 0) + 1
    gamma_stats[gamma] = gamma_stats.get(gamma, 0) + 1
    shrinking_stats[shrinking] = shrinking_stats.get(shrinking, 0) + 1
    class_weight_stats[class_weight] = class_weight_stats.get(class_weight, 0) + 1

# Print statistics
print("\n" + "="*60)
print("SVM PARAMETER SELECTION STATISTICS:")
print("="*60)

print("\nKernel Type Selection:")
for kernel, count in sorted(kernel_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  {kernel}: {count} windows ({percentage:.1f}%)")

print("\nRegularization Strength (C) Selection:")
for c_val, count in sorted(c_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  C={c_val}: {count} windows ({percentage:.1f}%)")

print("\nGamma Parameter Selection:")
for gamma, count in sorted(gamma_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    print(f"  gamma={gamma}: {count} windows ({percentage:.1f}%)")

print("\nShrinking Heuristic Selection:")
for shrinking, count in sorted(shrinking_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    shrinking_str = 'Enabled' if shrinking else 'Disabled'
    print(f"  {shrinking_str}: {count} windows ({percentage:.1f}%)")

print("\nClass Weight Selection:")
for weight, count in sorted(class_weight_stats.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / num_windows) * 100
    weight_str = 'balanced' if weight == 'balanced' else 'None'
    print(f"  {weight_str}: {count} windows ({percentage:.1f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Kernel selection
kernels, kernel_counts = zip(*sorted(kernel_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 0].bar(kernels, kernel_counts, color='mediumpurple')
axes[0, 0].set_xlabel('Kernel Type', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('SVM Kernel Selection', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# C values
c_vals, c_counts = zip(*sorted(c_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 1].bar([str(c) for c in c_vals], c_counts, color='orchid')
axes[0, 1].set_xlabel('C Value', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Regularization Strength (C)', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Gamma values
gammas, gamma_counts = zip(*sorted(gamma_stats.items(), key=lambda x: x[1], reverse=True))
axes[0, 2].bar([str(g) for g in gammas], gamma_counts, color='violet')
axes[0, 2].set_xlabel('Gamma Value', fontsize=12)
axes[0, 2].set_ylabel('Frequency', fontsize=12)
axes[0, 2].set_title('Kernel Coefficient (Gamma)', fontsize=14)
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Shrinking selection
shrinking_vals, shrinking_counts = zip(*sorted(shrinking_stats.items(), key=lambda x: x[1], reverse=True))
shrinking_labels = ['Enabled' if s else 'Disabled' for s in shrinking_vals]
axes[1, 0].bar(shrinking_labels, shrinking_counts, color='plum')
axes[1, 0].set_xlabel('Shrinking Heuristic', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Shrinking Heuristic Selection', fontsize=14)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Class weight selection
weights, weight_counts = zip(*sorted(class_weight_stats.items(), key=lambda x: x[1], reverse=True))
weight_labels = ['balanced' if w == 'balanced' else 'None' for w in weights]
axes[1, 1].bar(weight_labels, weight_counts, color='lavender')
axes[1, 1].set_xlabel('Class Weight', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Class Weight Selection', fontsize=14)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# CV Folds Evolution
windows = np.arange(1, num_windows + 1)
axes[1, 2].plot(windows, cv_folds_history, 'o-', linewidth=2, markersize=8, color='darkviolet')
axes[1, 2].set_xlabel('Window', fontsize=12)
axes[1, 2].set_ylabel('Number of CV Folds', fontsize=12)
axes[1, 2].set_title('CV Folds Evolution (SVM)', fontsize=14)
axes[1, 2].set_xticks(windows)
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS WITH OPTIMIZATION")
print("="*80 + "\n")

if feature_importance_history:
    # Check if we have linear kernel models
    linear_windows = [fh for fh in feature_importance_history if fh['best_params'].get('kernel') == 'linear']
    
    if linear_windows:
        print("Linear Kernel Windows Detected - Using Coefficients:")
        aggregated_importance = {}
        feature_occurrence = {}
        
        for window_data in linear_windows:
            features = window_data['features']
            coefficients = window_data['coefficients']
            
            for feature, coef in zip(features, coefficients):
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = []
                    feature_occurrence[feature] = 0
                aggregated_importance[feature].append(abs(coef))
                feature_occurrence[feature] += 1
        
        avg_importance = {feature: np.mean(importances) for feature, importances in aggregated_importance.items()}
        top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        print("Top 15 Features with Highest Average Coefficient Magnitude:")
        for feature, importance in top_features:
            occurrence_pct = (feature_occurrence[feature] / len(linear_windows)) * 100
            print(f"  {feature}: {importance:.4f} (in {occurrence_pct:.1f}% of linear windows)")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        features, importances = zip(*top_features)
        y_pos = np.arange(len(features))
        
        ax1.barh(y_pos, importances, color='darkviolet')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features)
        ax1.invert_yaxis()
        ax1.set_xlabel('Average Coefficient Magnitude', fontsize=12)
        ax1.set_title('Top 15 Feature Importance (Linear Kernel)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        occurrence_counts = [feature_occurrence[f] for f in features]
        ax2.barh(y_pos, occurrence_counts, color='mediumpurple')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features)
        ax2.invert_yaxis()
        ax2.set_xlabel('Number of Windows Selected', fontsize=12)
        ax2.set_title('Feature Selection Frequency', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No linear kernel windows found. Using variance as proxy for importance.")
        
        # Use variance for all windows
        aggregated_variance = {}
        feature_occurrence = {}
        
        for window_data in feature_importance_history:
            features = window_data['features']
            variances = window_data.get('variances', {})
            
            if not variances:
                # Calculate variance if not already stored
                continue
            
            for feature, variance in variances.items():
                if feature not in aggregated_variance:
                    aggregated_variance[feature] = []
                    feature_occurrence[feature] = 0
                aggregated_variance[feature].append(variance)
                feature_occurrence[feature] += 1
        
        if aggregated_variance:
            avg_variance = {feature: np.mean(variances) for feature, variances in aggregated_variance.items()}
            top_features = sorted(avg_variance.items(), key=lambda x: x[1], reverse=True)[:15]
            
            print("Top 15 Features with Highest Average Variance:")
            for feature, variance in top_features:
                occurrence_pct = (feature_occurrence[feature] / len(feature_importance_history)) * 100
                print(f"  {feature}: {variance:.4f} (in {occurrence_pct:.1f}% of windows)")
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            features, variances = zip(*top_features)
            y_pos = np.arange(len(features))
            
            ax1.barh(y_pos, variances, color='darkviolet')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features)
            ax1.invert_yaxis()
            ax1.set_xlabel('Average Variance', fontsize=12)
            ax1.set_title('Top 15 Feature Variance', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            occurrence_counts = [feature_occurrence[f] for f in features]
            ax2.barh(y_pos, occurrence_counts, color='mediumpurple')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(features)
            ax2.invert_yaxis()
            ax2.set_xlabel('Number of Windows Selected', fontsize=12)
            ax2.set_title('Feature Selection Frequency', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

# ============================================================================
# SUPPORT VECTORS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SUPPORT VECTORS ANALYSIS WITH DYNAMIC CV")
print("="*80 + "\n")

if support_vectors_history:
    n_support_list = [sv['n_support_vectors'] for sv in support_vectors_history]
    kernels_used = [sv['best_params'].get('kernel', 'unknown') for sv in support_vectors_history]
    
    print(f"Support vectors per window: {n_support_list}")
    print(f"Average support vectors per window: {np.mean(n_support_list):.1f}")
    print(f"Min support vectors: {np.min(n_support_list)}")
    print(f"Max support vectors: {np.max(n_support_list)}")
    print(f"Percentage of training samples as support vectors: {np.mean(n_support_list) / initial_window * 100:.1f}%")
    
    # Analyze by kernel type
    kernel_support_stats = {}
    for sv, kernel in zip(support_vectors_history, kernels_used):
        if kernel not in kernel_support_stats:
            kernel_support_stats[kernel] = []
        kernel_support_stats[kernel].append(sv['n_support_vectors'])
    
    print("\nSupport Vectors by Kernel Type:")
    for kernel, support_counts in kernel_support_stats.items():
        print(f"  {kernel}: {np.mean(support_counts):.1f} average support vectors")
    
    # Plot support vectors trend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Trend plot
    windows = np.arange(1, num_windows + 1)
    ax1.plot(windows, n_support_list, marker='o', linewidth=2, color='darkviolet')
    ax1.axhline(y=np.mean(n_support_list), color='r', linestyle='--', alpha=0.7, 
                label=f'Average: {np.mean(n_support_list):.1f}')
    ax1.set_xlabel('Window Number', fontsize=12)
    ax1.set_ylabel('Number of Support Vectors', fontsize=12)
    ax1.set_title('Support Vectors Trend Across Windows', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(windows)
    ax1.legend()
    
    # Percentage plot
    support_percentage = [n / initial_window * 100 for n in n_support_list]
    ax2.bar(windows, support_percentage, color='mediumorchid')
    ax2.axhline(y=np.mean(support_percentage), color='r', linestyle='--', alpha=0.7, 
                label=f'Average: {np.mean(support_percentage):.1f}%')
    ax2.set_xlabel('Window Number', fontsize=12)
    ax2.set_ylabel('Percentage of Training Samples', fontsize=12)
    ax2.set_title('Support Vectors as % of Training Data', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(windows)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# TRADING SIMULATION
# ============================================================================
print("\n" + "="*80)
print("TRADING SIMULATION WITH OPTIMIZED SVM")
print("="*80 + "\n")

initial_capital = 1000
sm1 = initial_capital
f = 0
ns = 0
nn = test_window
n = initial_window

open_prices = df['Open'].values
close_prices = df['Close'].values

modelval_SVM = []

for j in range(num_windows):
    for i in range(len(yptt[j])):
        price_idx = j * nn + i + n
        open_price = open_prices[price_idx]
        close_price = close_prices[price_idx]
        signal = yptt[j][i]
        
        if signal == 1 and f == 0:
            ns = (sm1 / (1 + cb)) / open_price
            sm1 = ns * close_price
            modelval_SVM.append(sm1)
            f = 1
        elif signal == 1 and f == 1:
            sm1 = ns * close_price
            modelval_SVM.append(sm1)
        elif signal == 0 and f == 0:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = sm1 * (1 - price_change_pct)
            modelval_SVM.append(sm1)
        elif signal == 0 and f == 1:
            price_change_pct = (close_price - open_price) / open_price
            sm1 = ns * open_price * (1 - cs) * (1 - price_change_pct)
            modelval_SVM.append(sm1)
            f = 0
            ns = 0

total_return = ((modelval_SVM[-1] - initial_capital) / initial_capital) * 100

print(f"{'='*60}")
print(f"TRADING RESULTS WITH OPTIMIZED SVM AND DYNAMIC CV")
print(f"{'='*60}")
print(f"Initial Investment:     ${initial_capital:.2f}")
print(f"Final Investment Value: ${modelval_SVM[-1]:.2f}")
print(f"Total Return:           {total_return:.2f}%")
print(f"Absolute Profit/Loss:   ${modelval_SVM[-1] - initial_capital:.2f}")
print(f"{'='*60}\n")

plt.figure(figsize=(12, 6))
plt.plot(modelval_SVM, label='Investment Value', linewidth=2, color='darkviolet')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Investment')
plt.xlabel('Trading Day')
plt.ylabel('Investment Value ($)')
plt.title('Trading System Performance with Optimized SVM (Dynamic CV)')
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

df_modelval_SVM = pd.DataFrame(modelval_SVM)
num_years = num_windows
aar = (((modelval_SVM[-1] / modelval_SVM[0]) ** (1 / num_years)) - 1) * 100
print(f"Average Annual Return (AAR): {aar:.2f}%")

mdd_series = max_drawdown(df_modelval_SVM)
mdd_value = mdd_series.min()
if isinstance(mdd_value, pd.Series):
    mdd_value = mdd_value.iloc[0]
print(f"Maximum Drawdown (MDD): {mdd_value:.4f}")

modelval_SVM_list = modelval_SVM.copy()
modelval_SVM_list.insert(0, initial_capital)

port_value = pd.DataFrame(modelval_SVM_list, columns=['Value'])
ret_data = port_value.pct_change()

annal_return_SVM = []
for year_idx in range(num_years):
    start_idx = year_idx * test_window
    end_idx = start_idx + test_window
    start_val = port_value.iloc[start_idx, 0]
    end_val = port_value.iloc[end_idx, 0]
    annual_ret = (end_val - start_val) / start_val
    annal_return_SVM.append(annual_ret)

annal_return_SVM_df = pd.DataFrame(annal_return_SVM, columns=['Return'])

dffff = pd.DataFrame(np.array(ret_data), columns=['Returns'])
dffff['downside_returns'] = 0.0
dffff_clean = dffff.dropna()
dffff_clean.loc[dffff_clean['Returns'] < 0, 'downside_returns'] = abs(dffff_clean['Returns'])

negative_returns = dffff_clean[dffff_clean['Returns'] < 0]['Returns']
std_neg = negative_returns.std() if len(negative_returns) > 0 else 0

annual_downside_std = dffff_clean['downside_returns'].std() * np.sqrt(252)
print(f"Annualized Downside Std: {annual_downside_std:.4f}")

if std_neg > 0:
    sortino = ((((modelval_SVM_list[-1] / modelval_SVM_list[0]) ** (1 / num_years)) - 1) - riskfree) / (std_neg * np.sqrt(252))
    print(f"Sortino Ratio: {sortino:.4f}")

annual_mean_return = (((modelval_SVM_list[-1] / modelval_SVM_list[0]) ** (1 / num_years)) - 1)
annual_return_std = annal_return_SVM_df['Return'].std()

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
