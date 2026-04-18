import pandas as pd
import numpy as np

def generate_submission(test_predictions, filename='weibo_submission.txt'):
    """
    将预测结果格式化并保存为比赛要求的 .txt 文件格式
    test_predictions: 包含 uid, mid, forward_count, comment_count, like_count 的 DataFrame
    """
    print(f"--- 正在生成提交文件: {filename} ---")
    
    # 1. 确保不会修改原始的预测数据
    sub_df = test_predictions.copy()
    
    # 2. 确保预测值是正数，并且严格转换为整数 (去除任何可能的小数点)
    # 对于极少数模型预测出负数的情况，使用 clip(lower=0) 修正
    targets = ['forward_count', 'comment_count', 'like_count']
    for col in targets:
        sub_df[col] = sub_df[col].clip(lower=0)
        sub_df[col] = np.round(sub_df[col]).astype(int)
    
    # 3. 拼接字符串格式：将 转,评,赞 用逗号拼接成一列
    # 格式示例: "1,0,3"
    sub_df['predict_str'] = sub_df['forward_count'].astype(str) + ',' + \
                            sub_df['comment_count'].astype(str) + ',' + \
                            sub_df['like_count'].astype(str)
                            
    # 4. 只保留需要的列：uid, mid, 拼接好的预测字符串
    final_sub = sub_df[['uid', 'mid', 'predict_str']]
    
    # 5. 保存为 txt 文件
    # sep='\t' 表示使用 Tab 分隔 uid 和 mid 等
    # header=False 表示不输出列名 (uid, mid等表头不需要)
    # index=False 表示不输出行号
    final_sub.to_csv(filename, sep='\t', header=False, index=False)
    
    print("✅ 提交文件生成完毕！")
    print("文件头部预览：")
    
    # 打印前 5 行检查格式
    with open(filename, 'r') as f:
        for _ in range(5):
            print(repr(f.readline())) # 使用 repr 可以看到隐藏的 \t 和 \n

# ==========================================
# 如何调用：
# ==========================================
# 假设 final_test_predictions 是你在 Step 4 得到的最终测试集预测 DataFrame
# generate_submission(final_test_predictions, 'my_final_submission.txt')