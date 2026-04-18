import pandas as pd
import numpy as np

def preprocess_data(df, is_train=True):
    """
    Preprocesses the blog post dataset.
    :param df: pandas DataFrame containing the data
    :param is_train: boolean, True for training data, False for prediction data
    :return: cleaned and preprocessed DataFrame
    """
    print(f"--- Starting Preprocessing (is_train={is_train}) ---")
    
    # ---------------------------------------------------------
    # 1. Handle Missing or Null Values
    # ---------------------------------------------------------
    # Content is the most likely to be null. We fill it with an empty string.
    if 'content' in df.columns:
        missing_content = df['content'].isnull().sum()
        if missing_content > 0:
            print(f"Filling {missing_content} missing 'content' values with empty string.")
            df['content'] = df['content'].fillna('')
            
    # Drop rows where critical IDs or time are missing
    critical_cols = ['uid', 'mid', 'time']
    initial_len = len(df)
    df = df.dropna(subset=critical_cols)
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing critical IDs/Time.")

    # ---------------------------------------------------------
    # 2. Parse and Convert Time Fields
    # ---------------------------------------------------------
    print("Converting 'time' field to datetime objects...")
    # The format is "Year-Month-Day Hour-Minute-Second"
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Drop rows where datetime parsing failed
    df = df.dropna(subset=['time'])
    
    # Sort data chronologically (Important for time-series/historical feature engineering later)
    df = df.sort_values(by=['time']).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Clip and Transform Skewed Target Variables (Train Only)
    # ---------------------------------------------------------
    if is_train:
        targets = ['forward_count', 'comment_count', 'like_count']
        
        for col in targets:
            # Ensure target variables are integers and fill any weird NaNs with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
            # --- CLIPPING ---
            # Viral posts act as massive outliers. We clip them to the 99.9th percentile.
            # This prevents a single post with 100,000 likes from dominating the model's loss.
            upper_bound = df[col].quantile(0.999)
            clipped_col_name = f'{col}_clipped'
            df[clipped_col_name] = df[col].clip(upper=upper_bound)
            print(f"Clipped '{col}' at 99.9th percentile: {upper_bound:.1f}")
            
            # --- LOG1P TRANSFORMATION ---
            # np.log1p calculates log(1 + x). It's perfect because our data contains 0s.
            log_col_name = f'{col}_log'
            df[log_col_name] = np.log1p(df[clipped_col_name])
            
        print("Target variables clipped and log1p transformed.")
        
    print(f"--- Preprocessing Complete. Final shape: {df.shape} ---\n")
    return df

# ==========================================
# How to use the function:
# ==========================================

# Assuming df_train is the dataframe you loaded in Step 1
# df_train = pd.read_csv('weibo_train_data.txt', sep='\t', header=None, names=['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content'])

# Apply preprocessing to training data
# df_train_clean = preprocess_data(df_train, is_train=True)

# Later, when you load the prediction/test data (which doesn't have targets):
# df_test = pd.read_csv('weibo_predict_data.txt', sep='\t', header=None, names=['uid', 'mid', 'time', 'content'])
# df_test_clean = preprocess_data(df_test, is_train=False)

# Let's peek at the new target columns
# display(df_train_clean[['forward_count', 'forward_count_clipped', 'forward_count_log']].head())