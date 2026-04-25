"""
BERTopic pipeline for Airbnb reviews.

This script:
1. Loads a CSV file with Airbnb reviews
2. Cleans and prepares the review text
3. Randomly samples up to 100,000 reviews
4. Creates sentence embeddings with SentenceTransformer
5. Fits a BERTopic model
6. Saves topic assignments and topic summaries

Edit the configuration section below before running.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# =========================
# Configuration
# =========================
INPUT_CSV = "Hotel_Data/hotel_merged_redone.csv"
TEXT_COLUMN = "text"
OUTPUT_DIR = "bertopic_output_hotel"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 10
SAMPLE_SIZE = 100000
RANDOM_STATE = 42

# Rule-based candidate labels.
# You can extend these vocabularies based on your domain.
CANDIDATE_LABEL_TERMS = {
    "location": {
        "nyc", "manhattan", "brooklyn", "harlem", "williamsburg", "city",
        "neighborhood", "area", "downtown", "midtown", "district", "spot",
        "heart", "central", "bridge", "street", "access", "walking distance"
    },

    "transport": {
        "subway", "station", "stations", "line", "lines", "metro", "walk",
        "close", "near", "commute", "transit", "train", "public transport"
    },

    "host": {
        "host", "hosts", "responsive", "communicative", "attentive",
        "helpful", "friendly", "kind", "welcome"
    },

    "cleanliness": {
        "clean", "dirty", "smell", "stain", "dust", "mess", "hygiene",
        "tidy", "filthy"
    },

    "comfort": {
        "bed", "beds", "mattress", "comfortable", "comfy", "sleep", "slept",
        "pillows", "soft", "firm", "cozy", "cosy",
        "spacious", "quiet", "peaceful", "relaxing", "relax",
        "air conditioning", "heating", "warm", "cool"
    },

    "service": {
        "service", "staff", "desk", "lobby", "checkin", "checkout",
        "support", "management"
    },

    "value": {
        "price", "money", "value", "worth", "budget", "expensive", "cheap",
        "cost", "deal", "worth the price"
    },

    "return_intent": {
        "again", "back", "definitely", "would", "will", "coming", "return"
    },

    "airport": {
        "airport", "jfk", "layover", "flight", "uber", "overnight"
    },

    "space": {
        "studio", "apartment", "room", "small", "basement", "modern", "den"
    },

    "bathroom": {
        "bathroom", "restroom", "toilet", "shower", "bathtub", "tub",
        "sink", "water pressure", "hot water", "cold water",
        "leak", "leaky", "drain", "clogged", "mold", "mildew",
        "towels", "toiletries"
    },

    "safety": {
        "safe", "unsafe", "security", "secure", "lock", "locks",
        "door", "key", "keypad", "sketchy", "crime", "alarm", "camera"
    },

    "noise": {
        "noise", "noisy", "loud", "quiet", "silence",
        "traffic", "street noise", "neighbors", "party",
        "thin walls", "soundproof", "construction"
    },

    "wifi": {
        "wifi", "wi-fi", "internet", "connection", "signal",
        "fast", "slow", "speed", "unstable", "reliable",
        "working remotely", "zoom", "streaming"
    },

    "kitchen": {
        "kitchen", "cook", "cooking", "stove", "oven", "microwave",
        "fridge", "refrigerator", "utensils", "dish", "plates",
        "coffee", "tea", "breakfast", "snacks"
    },

    "parking": {
        "parking", "park", "garage", "street parking",
        "lot", "free parking", "paid parking",
        "spot", "space", "valet"
    },

    "checkin": {
        "check-in", "checkin", "check out", "checkout",
        "easy", "smooth", "instructions", "self check-in",
        "lockbox", "late check-in", "early check-in",
        "delay", "waiting"
    },

    "accuracy": {
        "accurate", "exactly as described", "as described",
        "photos", "pictures", "misleading",
        "not as expected", "listing", "description"
    },

    "amenities": {
        "pool", "gym", "elevator", "laundry", "washer", "dryer",
        "tv", "netflix", "ac", "air conditioning", "heating",
        "balcony", "view", "rooftop", "workspace", "desk"
    },

    "suitability": {
        "family", "kids", "children", "baby", "couple",
        "friends", "group", "solo", "business trip",
        "vacation", "staycation"
    },

    "maintenance": {
        "broken", "not working", "issue", "problem", "fix",
        "repair", "maintenance", "malfunction"
    },

    "view": {
        "view", "scenery", "window", "sunlight", "bright",
        "dark", "natural light", "city view", "beautiful",
        "aesthetic", "design", "decor", "stylish"
    },

    "convenience": {
        "convenient", "accessible", "easy", "nearby",
        "close to", "quick", "minutes away"
    },

    "fees": {
        "fee", "fees", "hidden fee", "extra charge",
        "cleaning fee", "service fee", "deposit",
        "refund", "cancellation", "policy"
    }
}


def load_reviews(
    csv_path: str,
    text_column: str,
    sample_size: int = 100000,
    random_state: int = 42
) -> pd.DataFrame:
    """Load the CSV, clean it, and randomly sample rows if needed."""
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' was not found in the CSV. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].notna()]
    df = df[df[text_column] != ""]
    df = df[df[text_column].str.lower() != "nan"]

    if df.empty:
        raise ValueError("No valid review text was found after cleaning.")

    original_count = len(df)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
        print(f"Sampled {sample_size} reviews out of {original_count} cleaned reviews.")
    else:
        print(f"Using all {original_count} cleaned reviews because it is below the sample size.")

    return df.reset_index(drop=True)


def make_embeddings(reviews: list[str], model_name: str):
    """Create embeddings for the review text."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(reviews, show_progress_bar=True)
    return embeddings


def fit_topic_model(reviews: list[str], embeddings, min_topic_size: int):
    """Fit BERTopic using precomputed embeddings."""
    topic_model = BERTopic(min_topic_size=min_topic_size, verbose=True)
    topics, probs = topic_model.fit_transform(reviews, embeddings)
    return topic_model, topics, probs


def normalize_token(token: str) -> str:
    """Lowercase and keep alphabetic characters for matching."""
    return re.sub(r"[^a-z]+", "", token.lower())


def stem_token(token: str) -> str:
    """Tiny stemmer to improve rule matching without extra dependencies."""
    if len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[:-len(suffix)]
    return token


def assign_semantic_label(topic_words: list[str]) -> tuple[str, float]:
    """
    Assign one semantic label to a topic using rule-based vocab overlap.
    Returns (label, score).
    """
    if not topic_words:
        return "other", 0.0

    normalized = [normalize_token(w) for w in topic_words]
    normalized = [w for w in normalized if w]
    stems = [stem_token(w) for w in normalized]

    best_label = "other"
    best_score = 0.0

    for label, terms in CANDIDATE_LABEL_TERMS.items():
        term_norm = {normalize_token(t) for t in terms}
        term_stem = {stem_token(t) for t in term_norm}
        score = 0.0

        # Weighted overlap: earlier topic words contribute more.
        for idx, (word_norm, word_stem) in enumerate(zip(normalized, stems)):
            if word_norm in term_norm or word_stem in term_stem:
                score += 1.0 / (idx + 1)

        if score > best_score:
            best_label = label
            best_score = score

    if best_score == 0.0:
        return "other", 0.0
    return best_label, round(best_score, 4)


def save_outputs(
    df: pd.DataFrame,
    topic_model: BERTopic,
    topics,
    output_dir: str
) -> None:
    """Save topic info and document-topic assignments."""
    os.makedirs(output_dir, exist_ok=True)

    results_df = df.copy()
    results_df["topic"] = topics
    results_df.to_csv(Path(output_dir) / "reviews_with_topics.csv", index=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(Path(output_dir) / "topic_summary.csv", index=False)

    topic_keywords = []
    topic_label_rows = []
    for topic_id in topic_info["Topic"].tolist():
        if topic_id == -1:
            keywords = "outlier"
            top_words = []
            semantic_label = "outlier"
            label_score = 0.0
        else:
            words = topic_model.get_topic(topic_id)
            top_words = [word for word, _ in words] if words else []
            keywords = ", ".join(top_words) if top_words else ""
            semantic_label, label_score = assign_semantic_label(top_words)

        topic_keywords.append({"Topic": topic_id, "Keywords": keywords})
        topic_label_rows.append(
            {
                "Topic": topic_id,
                "Top_5_Words": ", ".join(top_words[:5]) if top_words else "",
                "Semantic_Label": semantic_label,
                "Label_Score": label_score
            }
        )

    pd.DataFrame(topic_keywords).to_csv(
        Path(output_dir) / "topic_keywords.csv", index=False
    )

    topic_labels_df = pd.DataFrame(topic_label_rows).merge(
        topic_info[["Topic", "Count"]],
        on="Topic",
        how="left"
    )
    topic_labels_df.to_csv(
        Path(output_dir) / "topic_labels.csv", index=False
    )

    # Top 5 unique semantic labels by total topic count.
    top_5_candidate_labels = (
        topic_labels_df[
            (topic_labels_df["Topic"] != -1)
            & (topic_labels_df["Semantic_Label"] != "outlier")
            & (topic_labels_df["Semantic_Label"] != "other")
        ]
        .groupby("Semantic_Label", as_index=False)["Count"]
        .sum()
        .rename(columns={"Count": "Total_Count"})
        .sort_values("Total_Count", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_5_candidate_labels.to_csv(
        Path(output_dir) / "top_5_candidate_labels.csv", index=False
    )

    print("\nSaved files:")
    print(f"- {Path(output_dir) / 'reviews_with_topics.csv'}")
    print(f"- {Path(output_dir) / 'topic_summary.csv'}")
    print(f"- {Path(output_dir) / 'topic_keywords.csv'}")
    print(f"- {Path(output_dir) / 'topic_labels.csv'}")
    print(f"- {Path(output_dir) / 'top_5_candidate_labels.csv'}")

    print("\nTop topics preview:")
    print(topic_info.head(10).to_string(index=False))


def main():
    print("Loading review data...")
    df = load_reviews(
        INPUT_CSV,
        TEXT_COLUMN,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    reviews = df[TEXT_COLUMN].tolist()
    print(f"Loaded {len(reviews)} reviews for BERTopic.")

    print("\nCreating embeddings...")
    embeddings = make_embeddings(reviews, EMBEDDING_MODEL)

    print("\nFitting BERTopic...")
    topic_model, topics, probs = fit_topic_model(
        reviews,
        embeddings,
        min_topic_size=MIN_TOPIC_SIZE
    )

    print("\nSaving outputs...")
    save_outputs(df, topic_model, topics, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
