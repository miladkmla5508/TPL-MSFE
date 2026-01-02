import time
import numpy as np
import pandas as pd

# ============================================================================
# START TIMER
# ============================================================================
start = time.time()

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
balance_rate = 0.0025         # 0.5% balance rate (opportunity cost threshold)
total_transaction = 1.5         # 2% total transaction cost (buy + sell)
window_size = 6              # Maximum lookahead window size (days)

print("="*80)
print("LABEL GENERATION CONFIGURATION")
print("="*80)
print(f"Balance Rate: {balance_rate*100:.2f}%")
print(f"Total Transaction Cost: {total_transaction:.2f}%")
print(f"Lookahead Window Size: {window_size} days")
print("="*80 + "\n")

# ============================================================================
# LABEL GENERATION - EFFICIENT VECTORIZED APPROACH
# ============================================================================
print("Generating trading labels...")

# Initialize label array (work with numpy array, then assign to df once)
labels = np.full(len(df), np.nan)
labels[0] = 0

# Extract close prices as numpy array for faster computation
close_prices = df['Close'].values
cret1_1 = df['CRet1_1'].values

# Pre-compute all future returns for lookahead windows
n = len(df)
future_returns = np.zeros((n, window_size + 1))

print(f"Pre-computing future returns for {n} data points...")
for days_ahead in range(2, window_size + 1):
    # Calculate percentage return: ((future_price / current_price) - 1) * 100
    future_returns[:-days_ahead, days_ahead] = (
        (close_prices[days_ahead:] / close_prices[:-days_ahead]) - 1
    ) * 100

print("Pre-computation complete.\n")

# ============================================================================
# LABEL ASSIGNMENT LOGIC
# ============================================================================

def should_buy_from_sell(i, cret_val, future_rets):
    """
    Determine if we should buy when currently in sell position (LABEL=0)
    
    Logic:
    1. If immediate return < balance_rate: stay in sell (0)
    2. If immediate return >= balance_rate + total_transaction: buy (1)
    3. Otherwise: look ahead up to window_size days to find profitable buy opportunity
    
    Parameters:
    -----------
    i : int
        Current index in the dataset
    cret_val : float
        Current period return (CRet1_1)
    future_rets : ndarray
        Pre-computed future returns array
        
    Returns:
    --------
    int : 0 (sell/stay out) or 1 (buy)
    """
    # Quick decision based on immediate return
    if cret_val < balance_rate:
        return 0
    if cret_val >= (total_transaction + balance_rate):
        return 1
    
    # Look ahead for profitable opportunity
    for days in range(2, window_size + 1):
        expected_return = future_rets[i, days]
        threshold_return = days * balance_rate
        
        # Check if return meets threshold
        if expected_return >= threshold_return:
            # Check if it covers transaction cost
            if expected_return >= (threshold_return + total_transaction):
                return 1
        else:
            # If this period doesn't meet threshold, no future period will
            return 0
    
    return 0


def should_sell_from_buy(i, cret_val, future_rets):
    """
    Determine if we should sell when currently in buy position (LABEL=1)
    
    Logic:
    1. If immediate return < balance_rate - total_transaction: sell (0)
    2. If immediate return >= balance_rate: stay in buy (1)
    3. Otherwise: look ahead up to window_size days to decide
    
    Parameters:
    -----------
    i : int
        Current index in the dataset
    cret_val : float
        Current period return (CRet1_1)
    future_rets : ndarray
        Pre-computed future returns array
        
    Returns:
    --------
    int : 0 (sell) or 1 (stay in buy)
    """
    # Quick decision based on immediate return
    if cret_val < (balance_rate - total_transaction):
        return 0
    if cret_val >= balance_rate:
        return 1
    
    # Look ahead to see if we should exit position
    for days in range(2, window_size + 1):
        expected_return = future_rets[i, days]
        threshold_return = days * balance_rate
        
        # Check if return falls below threshold
        if expected_return < threshold_return:
            # Check if it's below threshold minus transaction cost
            if expected_return < (threshold_return - total_transaction):
                return 0
        else:
            # If this period meets threshold, stay in position
            return 1
    
    return 1


# ============================================================================
# MAIN LABELING LOOP
# ============================================================================
print("Processing labels...")

for i in range(1, n - window_size):
    previous_label = labels[i - 1]
    current_cret = cret1_1[i]
    
    if previous_label == 0:
        # Currently in sell position, check if we should buy
        labels[i] = should_buy_from_sell(i, current_cret, future_returns)
    else:
        # Currently in buy position, check if we should sell
        labels[i] = should_sell_from_buy(i, current_cret, future_returns)
    
    # Progress indicator
    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{n - window_size} rows...")

print(f"  Processed {n - window_size}/{n - window_size} rows... Done!\n")

# Assign labels to dataframe in single operation
df['LABEL'] = labels

# ============================================================================
# OUTPUT STATISTICS
# ============================================================================
buy_count = df['LABEL'].sum()
total_labeled = (~df['LABEL'].isna()).sum()
sell_count = total_labeled - buy_count

print("="*80)
print("LABEL GENERATION COMPLETE")
print("="*80)
print(f"Buy Signals (LABEL=1):  {int(buy_count):>8} ({buy_count/total_labeled*100:>6.2f}%)")
print(f"Sell Signals (LABEL=0): {int(sell_count):>8} ({sell_count/total_labeled*100:>6.2f}%)")
print(f"Total Labeled:          {int(total_labeled):>8}")
print(f"Unlabeled (NaN):        {int(df['LABEL'].isna().sum()):>8}")
print("="*80 + "\n")

# ============================================================================
# DATA CLEANUP
# ============================================================================
print("Cleaning data...")

# Store original length
original_length = len(df)

# Remove rows with NaN labels
df = df.dropna().copy()  # Use copy() to avoid fragmentation warnings

# Drop the intermediate calculation column
if 'CRet1_1' in df.columns:
    df = df.drop(columns=['CRet1_1'])

# Convert LABEL to categorical codes (ensures 0 and 1 values)
df['LABEL'] = df['LABEL'].astype(int)

print(f"Removed {original_length - len(df)} rows with NaN values")
print(f"Final dataset size: {len(df)} rows\n")

# ============================================================================
# LABEL DISTRIBUTION ANALYSIS
# ============================================================================
print("="*80)
print("FINAL LABEL DISTRIBUTION")
print("="*80)
label_counts = df['LABEL'].value_counts().sort_index()
for label, count in label_counts.items():
    label_name = "Sell" if label == 0 else "Buy"
    print(f"LABEL={label} ({label_name}): {count:>8} ({count/len(df)*100:>6.2f}%)")
print("="*80 + "\n")

# ============================================================================
# EXECUTION TIME
# ============================================================================
end = time.time()
execution_time = (end - start) / 60

print("="*80)
print("PERFORMANCE SUMMARY")
print("="*80)
print(f"Total Execution Time: {execution_time:.4f} minutes")
print(f"                      ({(end - start):.2f} seconds)")
print(f"Processing Speed:     {len(df)/execution_time:.0f} rows/minute")
print("="*80 + "\n")
