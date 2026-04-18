import pandas as pd
import numpy as np
import os
import pickle
from data_process import preprocess_data
from feature_eng import create_features
from model_optmized import run_pipeline

# Paths
TRAIN_DATA_PATH = 'dataset/weibo_train_data.txt'
PREDICT_DATA_PATH = 'dataset/weibo_predict_data.txt'
TRAIN_PKL_PATH = 'dataset/weibo_train_features.pkl'
PREDICT_PKL_PATH = 'dataset/weibo_predict_features.pkl'
SUBMISSION_PATH = 'submission_optimized.txt'

def main():
    # 1. Load or Generate Features
    if os.path.exists(TRAIN_PKL_PATH) and os.path.exists(PREDICT_PKL_PATH):
        print("Loading pre-engineered features from pickle files...")
        with open(TRAIN_PKL_PATH, 'rb') as f:
            df_train_final = pickle.load(f)
        with open(PREDICT_PKL_PATH, 'rb') as f:
            df_test_final = pickle.load(f)
    else:
        print("Generating features from scratch...")
        # Load raw data
        print("Loading raw data...")
        df_train = pd.read_csv(TRAIN_DATA_PATH, sep='\t', header=None, 
                               names=['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content'])
        df_test = pd.read_csv(PREDICT_DATA_PATH, sep='\t', header=None, 
                              names=['uid', 'mid', 'time', 'content'])
        
        # Preprocess
        df_train_clean = preprocess_data(df_train, is_train=True)
        df_test_clean = preprocess_data(df_test, is_train=False)
        
        # Feature Engineering
        df_train_final, df_test_final = create_features(df_train_clean, df_test_clean)
        
        # Save for next time
        print("Saving features to pickle files...")
        with open(TRAIN_PKL_PATH, 'wb') as f:
            pickle.dump(df_train_final, f)
        with open(PREDICT_PKL_PATH, 'wb') as f:
            pickle.dump(df_test_final, f)

    print(f"Final Train Shape: {df_train_final.shape}")
    print(f"Final Test Shape: {df_test_final.shape}")

    # 2. Run Training Pipeline
    print("Starting Optimized Model Pipeline...")
    final_preds = run_pipeline(df_train_final, df_test_final)

    # 3. Save Submission
    print(f"Generating submission in correct format...")
    # Format: uid\tmid\tforward_count,comment_count,like_count
    final_preds['counts'] = final_preds['forward_count'].astype(str) + ',' + \
                            final_preds['comment_count'].astype(str) + ',' + \
                            final_preds['like_count'].astype(str)
    
    submission = final_preds[['uid', 'mid', 'counts']]
    
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, sep='\t', index=False, header=False)
    print("Done!")

if __name__ == "__main__":
    main()
