# Weibo Interaction Prediction: Model & Feature Summary

This document summarizes the current technical implementation for predicting Weibo interactions (forward, comment, and like counts).

## 1. Feature Engineering Pipeline

The feature engineering process is designed to capture temporal patterns, content quality, and user behavioral consistency.

### A. Temporal Features
- **Time Decomposition**: Extraction of `hour`, `day_of_week`, `month`, and `week_of_year`.
- **User Habits**: `is_weekend` flag and `time_segment` (binned hours) to capture peak activity periods.

### B. Content & Metadata Features
- **Quantity Metrics**: `content_len`, `url_count`, `hashtag_count`, and `mention_count`.
- **Engagement Triggers**: 
  - `has_lucky_draw`: Detection of keywords like "抽奖" or "红包".
  - `has_title_bracket`: Detection of "【】" or "《》" indicating formal titles or news.
  - `has_question_mark`: Detection of interactive queries.

### C. NLP Features (Text Representation)
- **Dimensionality Reduction**: A combination of `TfidfVectorizer` (char-level n-grams) and `TruncatedSVD` (PCA-like compression) to extract the top 5 latent semantic components from the post content.

### E. User Historical Features (The "Secret Sauce")
To prevent data leakage, historical statistics are calculated strictly on past data:
- **Interaction History**: `user_past_avg`, `user_past_q90`, and `user_past_median` for each interaction type.
- **Recency**: `user_recent_5_avg` (rolling window) and `time_diff_from_last` (time since previous post).
- **Cold Start Strategy**: New users or users with no history are assigned global median statistics from the training set.

---

## 2. Modeling Strategy

### A. Preprocessing
- **Outlier Handling**: Target variables are clipped at the **99.9th percentile** to prevent viral posts from skewing the loss function.
- **Target Transformation**: `log1p` transformation is applied to target counts to handle the heavy-tailed distribution.

### B. Architecture: Model Ensemble
The system uses a weighted ensemble of two gradient boosting frameworks:
1. **LightGBM (60%)**: Optimized for speed and large datasets, using `MAE` as the objective function.
2. **XGBoost (40%)**: Uses `reg:absoluteerror` to provide robust predictions and capture different variance patterns.

### C. Validation & Evaluation
- **Time-Based Split**: Data is split chronologically (Training: Feb–June; Validation: July) to simulate the real-world prediction task.
- **Scoring Function**: Evaluation is performed using the official competition formula.

---

## 3. Current Pipeline Status
- [x] **Data Loading**: Pandas-based ingestion with `.pkl` caching for features.
- [x] **Feature Engineering**: Modular pipeline in `feature_eng.py`.
- [x] **Model Training**: Multi-target regression with LGBM/XGB fusion.
- [x] **Evaluation**: Local validation score calculation integrated.
- [x] **Submission**: Automated formatting of `.txt` results for competition upload.

---
## 4. Score Record

| Model         | Score    | 绝对提升  |对应结果文件|
|---------------|----------|----------|--------------|
| Baseline      | 27.5265% | -        | -            |
| Version 1.0   | 30.9287% | 3.4022%  | submission/submission_1.0.txt |
| Version 1.1   | 30.6816% | 3.1551%  | submission/submission_1.1.txt |