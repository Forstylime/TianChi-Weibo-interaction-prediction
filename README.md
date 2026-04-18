# TianChi-Weibo-interaction-prediction

### 1. Competition Background and Task

* **Background**: For original blog posts, interactions such as reposts, comments, and likes are important indicators of post popularity and user interest.

* **Task**: Build a blog post interaction model to predict the total number of reposts, comments, and likes for sampled users' original blog posts one week after publication.

### 2. Data Description

* **Training Data**: Original blog post data covering the period from February 1, 2015 to July 31, 2015 (fields include user ID, post ID, posting time, number of reposts, comments, likes one week later, and post content).

* **Prediction Data**: Original blog post data covering the period from August 1, 2015 to August 31, 2015 (fields include user ID, post ID, posting time, and post content). * **Data Description:** User IDs and post IDs have been sampled and encrypted. Time format is "Year-Month-Day Hour-Minute-Second".

### 3. Submission Requirements

* **Format:** Must be a text file (.txt).

* **Content:** uid, mid, forward_count, comment_count, like_count.

* **Format Details:**

* Use tabs to separate fields.

* Use commas to separate forward_count, comment_count, and like_count fields.

* Prediction results must be **integers**; floating-point numbers are not accepted.