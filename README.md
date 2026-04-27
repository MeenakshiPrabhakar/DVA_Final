##Team 27 — Comparative Visualization for Airbnbs and Hotels
Polly Baugh, Anwitha Kollipara, Niti Mirkhelkar, Vibha Narasayya,
Meenakshi Prabhakar, Skye Solomon

Repository: https://github.com/MeenakshiPrabhakar/DVA_Final

##DESCRIPTION
This package compares Airbnb listings and hotels across four U.S. metros 
San Francisco, San Diego, Los Angeles, and New York City by mining what
guests actually write in their reviews and presenting the result as an
interactive ZIP-code choropleth. Existing platforms only support comparison
within a single platform; our system provides a unified, location-aware
view of accommodation value across both Airbnb and hotels, helping a user
see at a glance which property type tends to outperform expectations in
each neighborhood.

The analysis pipeline runs in three stages. First, a SentenceTransformer
(all-MiniLM-L6-v2) embeds a 100,000-review sample and BERTopic clusters
those embeddings into emergent themes; the discovered topics are
cross-validated against five theoretically motivated dimensions:
location, transport, space, amenities, and host. These topics produce the final
feature set. Second, every review is scored on those five dimensions
using a negation-aware signed frequency method that goes beyond binary
sentiment classification. Keyword counts are normalized by review length
and signed by surrounding negation context, yielding freq_location,
freq_transport, freq_space, freq_amenities, and freq_host scores. Third,
an XGBoost regressor is trained on those features (80/20 train-test split,
5-fold cross-validation) to predict a standardized rating on a 0–100 scale,
and the residual between predicted and observed rating is interpreted as a
quality signal: properties that overdeliver relative to what their reviews
suggest, and those that underdeliver.

The visualization layer aggregates listing-level predictions to the
ZIP-code level for each city and renders a GeoJSON choropleth. ZIPs are
colored pink where Airbnbs score higher than hotels on the model's
multidimensional rating, blue where hotels score higher, and gray where
data is missing for one or both property types. A city dropdown, legend,
and hover tooltip support interactive exploration. The final maps reveal
clear geographic clusters. For example, NYC is more Airbnb-stronger than
expected outside hotel-dense districts, while LA shows highly
neighborhood-dependent patterns, letting travelers reason about
accommodation value beyond a simple star rating.


##INSTALLATION
First clone the repo: 
git clone https://github.com/MeenakshiPrabhakar/DVA_Final.git
cd DVA_Final

Prerequisites:
    - Python 3.10 or higher
Required libraries include:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - bertopic


    Airbnb listings, reviews, and neighborhood GeoJSONs (Inside Airbnb):
        https://insideairbnb.com/get-the-data/
        Pull the SF, SD, LA, and NYC archives and place under data/airbnb/.

    TripAdvisor hotel reviews (CMU CS, ~878K reviews, 4,333 hotels):
        https://www.cs.cmu.edu/~jiweil/html/hotel-review.html
        Place the offering and review JSONs under data/hotels/.

The preprocessing step left-joins reviews to listings/offerings for each
source, concatenates across the four cities, and merges Airbnb and hotel
records into a single CSV consumed by the scoring scripts.


##EXECUTION
Run the scripts in pipeline order from the repository root with the
virtual environment activated:

Step 1 — Discover review themes with BERTopic:

    python airbnb_bertopic_topics.py

    Embeds a 100,000-review sample with all-MiniLM-L6-v2, clusters with
    BERTopic, and writes topic summaries. The discovered topics are
    mapped onto the five semantic dimensions (location, transport,
    space, amenities, host) used downstream.

Step 2 — Score every review on the five dimensions:

    python score_reviews.py

    Computes negation-aware signed frequency scores per review
    (freq_location, freq_transport, freq_space, freq_amenities,
    freq_host) and writes the scored tables to disk.

Step 3 — Train the XGBoost regressor and generate predictions:

    python predict_rating.py

    Trains an XGBoost rating regressor on the five feature scores plus
    property type (Airbnb vs. hotel), uses an 80/20 train-test split,
    predicts a normalized 0–100 score per listing, computes the
    residual against the observed rating, and writes results to
    `model_output/`.

Step 4 — Evaluate the model:

    python evaluate_predict_rating.py

    Runs 5-fold cross-validation and reports MAE, RMSE, R², exact
    rounded accuracy, accuracy within 0.5 stars, and accuracy within
    1 star, alongside global-mean and type-mean baselines. Expected
    XGBoost numbers from our run: MAE 0.243, RMSE 0.475, R² 0.399.

Step 5 — Compare Airbnb vs. hotel feature performance:

    python compare_feature_scores.py

    Produces side-by-side feature-score reports between the two
    populations to verify the model is not systematically biased
    toward one property type.

Step 6 — Generate the ZIP-level maps:
 
    cp model_output/predictions_all.csv Visualization/
    cd Visualization
    ./run_maps.sh
 
    This runs `generate_city_comparison_maps.py`, which reads
    `predictions_all.csv` plus `data/us_zip_codes_geo.json` and the
    neighborhood GeoJSONs in `airbnb_geojson/`, aggregates Airbnb
    scores from neighborhood to ZIP via polygon overlap, and writes
    one HTML map per city plus a combined map under
    `Visualization/maps/`.
 
Step 7 — View the result:
 
    Open Visualization/maps/all_cities_postal_comparison_map.html in a
    browser.






##MORE INFO: 
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
