"""
Frequency scoring for Airbnb / hotel reviews.

This script scores review text and adds five negation-aware
frequency scores:
    freq_location, freq_transport, freq_space,
    freq_amenities, freq_host

Each score = signed keyword hits per 100 words.
  +1 per positive mention  (e.g. "clean", "cozy")
  -1 per negated mention   (e.g. "not clean", "wasn't cozy")

Usage
-----
    python score_reviews.py

Edit the DATASETS section below if your file paths differ.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# =========================
# Configuration
# =========================
DATASETS = [
    {
        "input_csv": "Airbnb_Data/airbnb_merged.csv",
        "output_csv": "Airbnb_Data/airbnb_merged_scored.csv",
        "text_column": "comments",
    },
    {
        "input_csv": "Hotel_Data/hotel_merged_redone.csv",
        "output_csv": "Hotel_Data/hotel_merged_redone_scored.csv",
        "text_column": "text",
    },
]


# =========================
# Keyword lists
# Extend any list to improve coverage for your dataset.
# =========================
FREQUENCY_KEYWORDS: dict[str, list[str]] = {
    "location": [
        "location", "located", "area", "neighborhood", "neighbourhood",
        "city", "downtown", "central", "nearby", "close to", "distance",
        "convenient", "accessibility", "accessible", "district",
    ],
    "transport": [
        "transport", "transit", "metro", "subway", "station", "stations",
        "train", "bus", "line", "lines", "walk", "walking distance",
        "commute", "airport", "uber", "lyft", "taxi",
    ],
    "space": [
        "space", "spacious", "roomy", "room", "rooms", "studio",
        "apartment", "suite", "layout", "small", "tiny", "cramped",
        "bedroom", "bathroom", "size",
    ],
    "amenities": [
        "amenities", "amenity", "wifi", "wi-fi", "internet", "kitchen",
        "parking", "pool", "gym", "elevator", "laundry", "washer",
        "dryer", "air conditioning", "heating", "tv",
    ],
    "host": [
        "host", "hosts", "hosting", "staff", "service", "receptionist",
        "communication", "communicative", "responsive", "helpful",
        "friendly", "attentive", "welcoming", "hospitality",
        "checkin", "check-in", "check in", "checkout", "check-out",
        "check out", "professional", "prompt",
    ],
}

# Words that flip a keyword hit from +1 to -1.
# Checked within NEGATION_WINDOW tokens before the matched keyword.
NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "neither", "nor",
    "without", "lack", "lacking", "lacks",
    "isn't", "isnt", "wasn't", "wasnt",
    "aren't", "arent", "weren't", "werent",
    "doesn't", "doesnt", "didn't", "didnt",
    "don't", "dont", "can't", "cant",
    "couldn't", "couldnt", "wouldn't", "wouldnt",
    "shouldn't", "shouldnt", "hardly", "barely", "scarcely",
    "nothing", "nowhere", "nobody",
})
NEGATION_WINDOW: int = 3


# =========================
# Internal helpers
# =========================

def _build_pattern(keywords: list[str]) -> re.Pattern:
    sorted_kws = sorted(keywords, key=len, reverse=True)
    parts = [re.escape(kw) for kw in sorted_kws]
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern] = {
    cat: _build_pattern(kws) for cat, kws in FREQUENCY_KEYWORDS.items()
}

_NEGATION_RE: re.Pattern = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(w) for w in NEGATION_WORDS) + r")(?!\w)",
    re.IGNORECASE,
)


def _signed_hits(text: str, pattern: re.Pattern) -> int:
    """
    Count keyword hits in *text*, flipping the sign to -1 when a negation
    word appears within NEGATION_WINDOW tokens before the keyword.
    """
    total = 0
    for match in pattern.finditer(text):
        window = " ".join(text[: match.start()].split()[-NEGATION_WINDOW:])
        total += -1 if _NEGATION_RE.search(window) else 1
    return total


# =========================
# Main scoring function
# =========================

def add_frequency_scores(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Return a copy of *df* with ten new columns:
      freq_<category>_raw  – signed raw hit count
      freq_<category>      – signed hits per 100 words (normalised)

    Scores are per-review and can be negative when negated mentions dominate.
    """
    df = df.copy()
    texts = df[text_column].astype(str)
    word_counts = texts.apply(lambda t: max(len(t.split()), 1))

    for category, pattern in _PATTERNS.items():
        signed = texts.apply(lambda t, p=pattern: _signed_hits(t, p))
        df[f"freq_{category}_raw"] = signed
        df[f"freq_{category}"]     = (signed / word_counts * 100).round(4)

    return df


def topic_score_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by BERTopic topic and compute the mean normalised score for each
    category.  Useful for seeing which topics dominate each dimension.

    Returns a DataFrame indexed by topic id.
    """
    score_cols = [f"freq_{c}" for c in FREQUENCY_KEYWORDS]
    if "topic" not in df.columns:
        print("Warning: no 'topic' column found — skipping topic summary.")
        return pd.DataFrame()
    return df.groupby("topic")[score_cols].mean().round(4)


# =========================
# Entry point
# =========================

def main() -> None:
    for dataset in DATASETS:
        input_path = Path(dataset["input_csv"])
        output_path = Path(dataset["output_csv"])
        text_column = dataset["text_column"]

        if not input_path.exists():
            raise FileNotFoundError(f"Could not find '{input_path}'.")

        print(f"\nLoading {input_path} ...")
        df = pd.read_csv(input_path)

        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in '{input_path}'. "
                f"Available columns: {list(df.columns)}"
            )

        print(
            f"Scoring {len(df):,} reviews across {len(FREQUENCY_KEYWORDS)} categories ..."
        )
        df = add_frequency_scores(df, text_column)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved scored reviews -> {output_path}")

        summary = topic_score_summary(df)
        if not summary.empty:
            summary_name = f"{output_path.stem}_topic_score_summary.csv"
            summary_path = output_path.parent / summary_name
            summary.to_csv(summary_path)
            print(f"Saved topic score summary -> {summary_path}")

        score_cols = [f"freq_{c}" for c in FREQUENCY_KEYWORDS]
        print("Score distribution (per 100 words):")
        print(df[score_cols].describe().round(4).to_string())


if __name__ == "__main__":
    main()
