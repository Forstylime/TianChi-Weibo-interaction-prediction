import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def create_features(df_train, df_test):
    """
    Combines train and test sets to generate features, preventing data leakage,
    and then splits them back.
    """
    print("--- Starting Feature Engineering ---")
    
    # Add a flag to split them later
    df_train['is_test'] = 0
    df_test['is_test'] = 1
    
    # Combine datasets for consistent feature engineering
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Fill missing content
    df['content'] = df['content'].fillna('')
    
    # Ensure chronological order (CRITICAL for historical features)
    df = df.sort_values(by=['time']).reset_index(drop=True)
    
    # =========================================================
    # 1. Time Features
    # =========================================================
    print("Extracting Time Features...")
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Time of day bins (0-3, 4-8, 9-14, 15-19, 20-21, 22-23)
    df['time_segment'] = pd.cut(df['hour'], bins=[-1, 3, 8, 14, 19, 21, 23], labels=[0, 1, 2, 3, 4, 5]).astype(int)

    # =========================================================
    # 2. Content Features
    # =========================================================
    print("Extracting Content Features...")
    # Post length
    df['content_len'] = df['content'].apply(len)
    
    # Number of URLs (usually contains 'http' or 't.cn' for Weibo)
    df['url_count'] = df['content'].apply(lambda x: len(re.findall(r'(http://|https://|t\.cn)', x)))
    
    # Number of Hashtags (usually enclosed in #...#)
    df['hashtag_count'] = df['content'].apply(lambda x: len(re.findall(r'#.*?#', x)))
    
    # Mentions (usually starts with @)
    df['mention_count'] = df['content'].apply(lambda x: len(re.findall(r'@', x)))

    # =========================================================
    # 3. NLP Features (TF-IDF + PCA/SVD)
    # =========================================================
    # Note: Running full TF-IDF on millions of rows will crash your RAM. 
    # We restrict it to max_features=50 or use TruncatedSVD to compress it.
    print("Extracting NLP Features (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=20, analyzer='char_wb', ngram_range=(1,2)) 
    tfidf_matrix = tfidf.fit_transform(df['content'])
    
    # Add top 5 PCA/SVD components of text to dataframe
    svd = TruncatedSVD(n_components=5, random_state=42)
    text_svd = svd.fit_transform(tfidf_matrix)
    for i in range(5):
        df[f'text_svd_{i}'] = text_svd[:, i]

    # =========================================================
    # 4. User Interaction History (The Secret Sauce)
    # =========================================================
    print("Extracting User Historical Features (Safely)...")
    
    # Sort by UID and Time so we can calculate rolling history safely
    df = df.sort_values(by=['uid', 'time']).reset_index(drop=True)
    
    targets = ['forward_count', 'comment_count', 'like_count']

    # 计算距离上次发博的时间间隔（秒）
    df['time_diff_from_last'] = df.groupby('uid')['time'].diff().dt.total_seconds()
    df['time_diff_from_last'] = df['time_diff_from_last'].fillna(-1)  # 第一次发博填-1

    # 是否包含抽奖/红包词汇
    df['has_lucky_draw'] = df['content'].apply(lambda x: 1 if re.search(r'抽奖|红包|转发抽|送出', str(x)) else 0)

    # 是否包含书名号/方括号（通常代表文章标题或重磅新闻）
    df['has_title_bracket'] = df['content'].apply(lambda x: 1 if re.search(r'【.*?】|《.*?》', str(x)) else 0)

    # 是否是互动问句
    df['has_question_mark'] = df['content'].apply(lambda x: 1 if '？' in str(x) or '?' in str(x) else 0)

    # 4c. Total posts by user (calculated once)
    df['user_post_count'] = df.groupby('uid').cumcount()

    for col in targets:
        # 4a. Cumulative Average (Expanding Mean) strictly BEFORE the current post
        df[f'user_past_avg_{col}'] = df.groupby('uid')[col].transform(lambda x: x.shift(1).expanding().mean())
        
        # 4b. 90th Percentile strictly BEFORE the current post
        df[f'user_past_q90_{col}'] = df.groupby('uid')[col].transform(lambda x: x.shift(1).expanding().quantile(0.9))
        
        # 4d. Recent 5 Average strictly BEFORE the current post
        df[f'user_recent_5_avg_{col}'] = df.groupby('uid')[col].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )

        # 4e. Historical Median strictly BEFORE the current post
        df[f'user_past_median_{col}'] = df.groupby('uid')[col].transform(
            lambda x: x.shift(1).expanding().median()
        )

        # Fill NaNs (for the user's very first post) with 0
        hist_cols = [f'user_past_avg_{col}', f'user_past_q90_{col}', 
                     f'user_recent_5_avg_{col}', f'user_past_median_{col}']
        df[hist_cols] = df[hist_cols].fillna(0)

    # Note on TEST SET targets: 
    # Because df_test targets are empty/NaN, historical calculations on the full df will be NaN for test posts.
    # We map the FINAL train profiles to the test set.
    print("Mapping Train profiles to Test users...")
    
    # Calculate profiles from Train only
    train_df = df[df['is_test'] == 0]
    user_stats = train_df.groupby('uid').agg({
        'forward_count': ['mean', ('q90', lambda x: x.quantile(0.9)), 'median'],
        'comment_count': ['mean', ('q90', lambda x: x.quantile(0.9)), 'median'],
        'like_count': ['mean', ('q90', lambda x: x.quantile(0.9)), 'median']
    })
    
    # Flatten columns: e.g., ('forward_count', 'mean') -> 'train_mean_forward_count'
    user_stats.columns = [f'train_{stat}_{c}' for c, stat in user_stats.columns]
    df = df.merge(user_stats, on='uid', how='left')
    
    for col in targets:
        # Fill missing users (Cold Start) with global train median
        for stat in ['mean', 'q90', 'median']:
            feat_name = f'train_{stat}_{col}'
            global_median = user_stats[feat_name].median()
            df[feat_name] = df[feat_name].fillna(global_median)
        
        # Override test set historical features with their final train profiles
        test_mask = (df['is_test'] == 1)
        df.loc[test_mask, f'user_past_avg_{col}'] = df.loc[test_mask, f'train_mean_{col}']
        df.loc[test_mask, f'user_past_q90_{col}'] = df.loc[test_mask, f'train_q90_{col}']
        df.loc[test_mask, f'user_past_median_{col}'] = df.loc[test_mask, f'train_median_{col}']
        df.loc[test_mask, f'user_recent_5_avg_{col}'] = df.loc[test_mask, f'train_mean_{col}']

    # Drop the temporary training profile columns
    drop_cols = [f'train_{stat}_{col}' for col in targets for stat in ['mean', 'q90', 'median']]
    df.drop(columns=drop_cols, inplace=True)

    # =========================================================
    # 5. Split Back into Train and Test
    # =========================================================
    print("Splitting back into Train and Test sets...")
    
    # Sort back by time just to be clean
    df = df.sort_values(by='time').reset_index(drop=True)
    
    df_train_engineered = df[df['is_test'] == 0].drop(columns=['is_test']).copy()
    df_test_engineered  = df[df['is_test'] == 1].drop(columns=['is_test']).copy()
    
    print(f"Engineered Train shape: {df_train_engineered.shape}")
    print(f"Engineered Test shape: {df_test_engineered.shape}")
    print(f"Engineered Columns: {df_train_engineered.columns.tolist()}")
    print("--- Feature Engineering Complete ---")
    
    return df_train_engineered, df_test_engineered

# ==========================================
# How to use:
# ==========================================
# df_train_clean and df_test_clean are the outputs from Step 2.
# df_train_final, df_test_final = create_features(df_train_clean, df_test_clean)