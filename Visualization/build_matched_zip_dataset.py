from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

import generate_city_comparison_maps as viz


def build_zip_match_rows() -> list[dict[str, object]]:
    hotel_mean, airbnb_mean = viz.aggregate_scores()

    us_geo = json.loads(viz.US_ZIP_GEOJSON.read_text())
    city_zip_features = viz.build_city_zip_features(us_geo)
    rows: list[dict[str, object]] = []

    for city_key, cfg in viz.CITY_CONFIG.items():
        zip_features = city_zip_features[city_key]
        zip_centroids: dict[str, tuple[float, float]] = {}
        zip_bounds: dict[str, tuple[float, float, float, float] | None] = {}

        for feature in zip_features:
            z = viz.clean_zip(feature.get("properties", {}).get("ZCTA5CE10"))
            if not z:
                continue
            zip_centroids[z] = viz.representative_point(feature.get("geometry", {}))
            zip_bounds[z] = viz.geometry_bounds(feature.get("geometry", {}))

        airbnb_geo = json.loads(cfg["airbnb_geojson"].read_text())
        zip_airbnb_acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        zip_neighborhood_mix: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for feature in airbnb_geo.get("features", []):
            neighborhood = viz.normalize_text(feature.get("properties", {}).get("neighbourhood"))
            if not neighborhood:
                continue

            key = (city_key, neighborhood)
            if key not in airbnb_mean:
                continue

            score, count = airbnb_mean[key]
            zip_weights = viz.estimate_zip_overlap_weights(
                feature.get("geometry", {}),
                zip_features,
                zip_centroids,
                zip_bounds,
            )
            for z, weight in zip_weights.items():
                weighted_count = count * weight
                acc = zip_airbnb_acc[z]
                acc[0] += score * weighted_count
                acc[1] += weighted_count
                zip_neighborhood_mix[z][neighborhood] += weighted_count

        zip_ids = sorted(zip_centroids.keys())
        for z in zip_ids:
            hotel_entry = hotel_mean.get((city_key, z))
            airbnb_entry = zip_airbnb_acc.get(z)

            hotel_score = hotel_entry[0] if hotel_entry else None
            hotel_count = int(hotel_entry[1]) if hotel_entry else 0

            airbnb_score = None
            airbnb_count = 0
            if airbnb_entry and airbnb_entry[1] > 0:
                airbnb_score = airbnb_entry[0] / airbnb_entry[1]
                airbnb_count = int(round(airbnb_entry[1]))

            diff = None
            match_type = "no_data"
            if airbnb_score is not None and hotel_score is not None:
                diff = airbnb_score - hotel_score
                match_type = "both"
            elif airbnb_score is not None:
                match_type = "airbnb_only"
            elif hotel_score is not None:
                match_type = "hotel_only"

            neighborhood_weights = zip_neighborhood_mix.get(z, {})
            top_neighborhoods = ",".join(
                name
                for name, _weight in sorted(
                    neighborhood_weights.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
            )

            rows.append(
                {
                    "city": city_key,
                    "city_label": cfg["label"],
                    "postal_code": z,
                    "match_type": match_type,
                    "airbnb_norm_score_z": airbnb_score,
                    "hotel_norm_score_z": hotel_score,
                    "diff_airbnb_minus_hotel_z": diff,
                    "airbnb_row_count": airbnb_count,
                    "hotel_row_count": hotel_count,
                    "top_airbnb_neighborhoods": top_neighborhoods,
                }
            )

    return rows


def main() -> None:
    rows = build_zip_match_rows()
    out_path = viz.OUT_DIR / "hotel_airbnb_zip_matches.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["city", "postal_code"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    both = (df["match_type"] == "both").sum()
    airbnb_only = (df["match_type"] == "airbnb_only").sum()
    hotel_only = (df["match_type"] == "hotel_only").sum()
    no_data = (df["match_type"] == "no_data").sum()
    print(
        "Match coverage:"
        f" both={both}, airbnb_only={airbnb_only}, hotel_only={hotel_only}, no_data={no_data}"
    )


if __name__ == "__main__":
    main()
