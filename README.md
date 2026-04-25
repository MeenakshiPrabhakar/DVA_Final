# Group Project

This repository contains our Airbnb vs. hotel analysis workflow.

Core scripts:
- `score_reviews.py`: builds keyword-based frequency features from review text.
- `predict_rating.py`: trains an XGBoost regressor and writes scored outputs to `model_output/`.
- `evaluate_predict_rating.py`: runs cross-validation and baseline comparisons.
- `airbnb_bertopic_topics.py`: fits BERTopic and exports topic summaries.
- `compare_feature_scores.py`: compares Airbnb and hotel feature performance.

Algorithm pipeline:
1. Collect review data (Airbnb for SF, SD, LA, and NY, plus hotel data).
2. Run BERTopic (unsupervised) on reviews to discover major themes.
3. Select the top 5 features/themes from that analysis.
4. Compute feature scores for each listing/review.
5. Train an XGBoost model using those feature scores to predict standardized ratings.
6. Use model outputs to build the ZIP-level visualization maps.

Visualization pipeline lives in `Visualization/`.
