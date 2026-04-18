import pandas as pd
import numpy as np
import os
import pickle
import warnings
from data_process import preprocess_data
from feature_eng import create_features
from modle import run_pipeline

warnings.filterwarnings('ignore')

# 配置路径
TRAIN_DATA_PATH = 'dataset/weibo_train_data.txt'
PREDICT_DATA_PATH = 'dataset/weibo_predict_data.txt'
TRAIN_PKL_PATH = 'dataset/weibo_train_features.pkl'
PREDICT_PKL_PATH = 'dataset/weibo_predict_features.pkl'
SUBMISSION_PATH = 'submission.txt'

def main():
    # 1. 数据加载与特征工程
    if os.path.exists(TRAIN_PKL_PATH) and os.path.exists(PREDICT_PKL_PATH):
        print("--- 加载已缓存的特征文件 ---")
        with open(TRAIN_PKL_PATH, 'rb') as f:
            df_train_final = pickle.load(f)
        with open(PREDICT_PKL_PATH, 'rb') as f:
            df_test_final = pickle.load(f)
    else:
        print("--- 从原始数据生成特征 ---")
        # 读取数据
        print("加载原始数据...")
        df_train = pd.read_csv(TRAIN_DATA_PATH, sep='\t', header=None, 
                               names=['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content'])
        df_test = pd.read_csv(PREDICT_DATA_PATH, sep='\t', header=None, 
                              names=['uid', 'mid', 'time', 'content'])
        
        # 预处理
        df_train_clean = preprocess_data(df_train, is_train=True)
        df_test_clean = preprocess_data(df_test, is_train=False)
        
        # 特征工程
        df_train_final, df_test_final = create_features(df_train_clean, df_test_clean)
        
        # 缓存特征
        print("保存特征到 pickle 文件...")
        with open(TRAIN_PKL_PATH, 'wb') as f:
            pickle.dump(df_train_final, f)
        with open(PREDICT_PKL_PATH, 'wb') as f:
            pickle.dump(df_test_final, f)

    print(f"训练集维度: {df_train_final.shape}")
    print(f"预测集维度: {df_test_final.shape}")

    # 2. 运行模型训练与融合流程
    print("\n--- 开始优化模型训练流程 ---")
    final_preds = run_pipeline(df_train_final, df_test_final)

    # 3. 生成提交文件
    print(f"\n--- 生成提交文件: {SUBMISSION_PATH} ---")
    # 格式: uid\tmid\tforward_count,comment_count,like_count
    final_preds['counts'] = final_preds['forward_count'].astype(str) + ',' + \
                            final_preds['comment_count'].astype(str) + ',' + \
                            final_preds['like_count'].astype(str)
    
    submission = final_preds[['uid', 'mid', 'counts']]
    submission.to_csv(SUBMISSION_PATH, sep='\t', index=False, header=False)
    
    print("训练及预测完成！")

if __name__ == "__main__":
    main()
