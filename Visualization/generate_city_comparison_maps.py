from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PREDICTIONS_CSV = ROOT / "predictions_all.csv"
US_ZIP_GEOJSON = ROOT / "data" / "us_zip_codes_geo.json"
AIRBNB_GEOJSON_DIR = ROOT / "airbnb_geojson"
AIRBNB_SCORED_CSV = ROOT.parent / "Airbnb_Data" / "airbnb_merged_scored.csv"
OUT_DIR = ROOT / "maps"


CITY_CONFIG = {
    "ny": {
        "label": "New York City",
        "hotel_city": "New York City",
        "airbnb_geojson": AIRBNB_GEOJSON_DIR / "ny_nei.geojson",
        "statefp": "36",
        "center_lonlat": (-73.98, 40.75),
        "max_center_distance_deg": 0.8,
    },
    "la": {
        "label": "Los Angeles",
        "hotel_city": "Los Angeles",
        "airbnb_geojson": AIRBNB_GEOJSON_DIR / "la_nei.geojson",
        "statefp": "06",
        "center_lonlat": (-118.25, 34.05),
        "max_center_distance_deg": 1.0,
    },
    "sf": {
        "label": "San Francisco",
        "hotel_city": "San Francisco",
        "airbnb_geojson": AIRBNB_GEOJSON_DIR / "sf_nei.geojson",
        "statefp": "06",
        "center_lonlat": (-122.42, 37.77),
        "max_center_distance_deg": 0.45,
    },
    "sd": {
        "label": "San Diego",
        "hotel_city": "San Diego",
        "airbnb_geojson": AIRBNB_GEOJSON_DIR / "sd_nei.geojson",
        "statefp": "06",
        "center_lonlat": (-117.16, 32.72),
        "max_center_distance_deg": 0.7,
    },
}

NY_BOROUGHS = {"bronx", "brooklyn", "manhattan", "queens", "staten island"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def clean_zip(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 5:
        return None
    return digits[:5]


def city_from_listing_id(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    m = re.match(r"^\s*([a-zA-Z]{2})[_-]?", str(value))
    if not m:
        return None
    prefix = m.group(1).lower()
    if prefix in CITY_CONFIG:
        return prefix
    return None


def point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    x1, y1 = ring[0]
    for i in range(1, n + 1):
        x2, y2 = ring[i % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def geometry_to_polygons(geometry: dict[str, Any] | None) -> list[tuple[list[list[float]], list[list[list[float]]]]]:
    if not geometry:
        return []
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    polygons = []
    if gtype == "Polygon":
        if coords:
            outer = coords[0]
            holes = coords[1:] if len(coords) > 1 else []
            polygons.append((outer, holes))
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                outer = poly[0]
                holes = poly[1:] if len(poly) > 1 else []
                polygons.append((outer, holes))
    return polygons


def polygon_area_and_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    if len(ring) < 3:
        return 0.0, 0.0, 0.0
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area = area2 / 2.0
    if abs(area) < 1e-12:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
    return area, cx / (6.0 * area), cy / (6.0 * area)


def representative_point(geometry: dict[str, Any] | None) -> tuple[float, float]:
    polys = geometry_to_polygons(geometry)
    if not polys:
        return 0.0, 0.0
    best = None
    for outer, _holes in polys:
        area, cx, cy = polygon_area_and_centroid(outer)
        score = abs(area)
        if (best is None) or (score > best[0]):
            best = (score, cx, cy)
    assert best is not None
    return best[1], best[2]


def geometry_contains_point(geometry: dict[str, Any] | None, point: tuple[float, float]) -> bool:
    x, y = point
    for outer, holes in geometry_to_polygons(geometry):
        if point_in_ring(x, y, outer):
            in_hole = any(point_in_ring(x, y, hole) for hole in holes)
            if not in_hole:
                return True
    return False


def geometry_bounds(geometry: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    polys = geometry_to_polygons(geometry)
    if not polys:
        return None
    minx, miny = float("inf"), float("inf")
    maxx, maxy = float("-inf"), float("-inf")
    for outer, holes in polys:
        for ring in [outer, *holes]:
            for x, y in ring:
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)
    if not math.isfinite(minx):
        return None
    return minx, maxx, miny, maxy


def bounds_overlap(
    b1: tuple[float, float, float, float] | None,
    b2: tuple[float, float, float, float] | None,
) -> bool:
    if b1 is None or b2 is None:
        return False
    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2])


def sq_dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def filter_city_zip_features(city_key: str, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cfg = CITY_CONFIG[city_key]
    cx, cy = cfg["center_lonlat"]
    max_d = float(cfg["max_center_distance_deg"])
    out = []
    for f in features:
        geom = f.get("geometry")
        if not geom:
            continue
        px, py = representative_point(geom)
        d = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if d <= max_d:
            out.append(f)
    return out


def neighborhood_bbox_and_points(features: list[dict[str, Any]]) -> tuple[tuple[float, float, float, float], list[tuple[float, float]]]:
    minx, miny = 999.0, 999.0
    maxx, maxy = -999.0, -999.0
    reps: list[tuple[float, float]] = []
    for f in features:
        geom = f.get("geometry")
        if not geom:
            continue
        reps.append(representative_point(geom))
        for outer, holes in geometry_to_polygons(geom):
            for p in outer:
                minx = min(minx, p[0])
                maxx = max(maxx, p[0])
                miny = min(miny, p[1])
                maxy = max(maxy, p[1])
            for h in holes:
                for p in h:
                    minx = min(minx, p[0])
                    maxx = max(maxx, p[0])
                    miny = min(miny, p[1])
                    maxy = max(maxy, p[1])
    return (minx, maxx, miny, maxy), reps


def build_city_zip_features(us_geo: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    city_features: dict[str, list[dict[str, Any]]] = {}
    for city_key, cfg in CITY_CONFIG.items():
        airbnb_geo = json.loads(cfg["airbnb_geojson"].read_text())
        n_features = [f for f in airbnb_geo.get("features", []) if f.get("geometry")]
        n_bbox, n_reps = neighborhood_bbox_and_points(n_features)
        pad = 0.08

        candidates = []
        for f in us_geo.get("features", []):
            props = f.get("properties", {})
            if str(props.get("STATEFP10", "")) != cfg["statefp"]:
                continue
            z = clean_zip(props.get("ZCTA5CE10"))
            if not z:
                continue
            geom = f.get("geometry")
            if not geom:
                continue
            rp = representative_point(geom)
            if not (
                n_bbox[0] - pad <= rp[0] <= n_bbox[1] + pad
                and n_bbox[2] - pad <= rp[1] <= n_bbox[3] + pad
            ):
                continue
            candidates.append(f)

        selected = []
        for zf in candidates:
            zgeom = zf.get("geometry")
            if not zgeom:
                continue
            zrp = representative_point(zgeom)
            keep = False
            for nf in n_features:
                ngeom = nf.get("geometry")
                if not ngeom:
                    continue
                if geometry_contains_point(ngeom, zrp):
                    keep = True
                    break
            if not keep:
                for np in n_reps:
                    if geometry_contains_point(zgeom, np):
                        keep = True
                        break
            if keep:
                selected.append(zf)

        selected = filter_city_zip_features(city_key, selected)
        city_features[city_key] = selected
    return city_features


def build_neighborhood_sets() -> tuple[dict[str, set[str]], dict[str, str]]:
    names_by_city: dict[str, set[str]] = {}
    for key, cfg in CITY_CONFIG.items():
        geo = json.loads(cfg["airbnb_geojson"].read_text())
        names = set()
        for feature in geo.get("features", []):
            names.add(normalize_text(feature.get("properties", {}).get("neighbourhood")))
        names.discard("")
        names_by_city[key] = names

    unique_owner: dict[str, str] = {}
    all_names = set().union(*names_by_city.values())
    for name in all_names:
        owners = [city for city, names in names_by_city.items() if name in names]
        if len(owners) == 1:
            unique_owner[name] = owners[0]
    return names_by_city, unique_owner


def load_listing_neighborhood_map() -> dict[str, str]:
    if not AIRBNB_SCORED_CSV.exists():
        return {}
    out: dict[str, str] = {}
    usecols = ["listing_id", "neighbourhood_cleansed", "neighbourhood"]
    for chunk in pd.read_csv(AIRBNB_SCORED_CSV, usecols=usecols, chunksize=400_000, low_memory=False):
        listing_ids = chunk["listing_id"].astype(str)
        chosen = chunk["neighbourhood_cleansed"].fillna(chunk["neighbourhood"]).map(normalize_text)
        bad = chosen.isin(["", "nan", "neighborhood highlights"])
        chosen.loc[bad] = pd.NA
        valid = pd.DataFrame({"listing_id": listing_ids, "nbh": chosen}).dropna(subset=["listing_id", "nbh"])
        for lid, nbh in zip(valid["listing_id"], valid["nbh"]):
            if lid not in out:
                out[lid] = nbh
    return out


def build_zip_to_city_map() -> dict[str, str]:
    us_geo = json.loads(US_ZIP_GEOJSON.read_text())
    city_zip_features = build_city_zip_features(us_geo)
    zip_to_city: dict[str, str] = {}
    for city_key, features in city_zip_features.items():
        for feature in features:
            z = clean_zip(feature.get("properties", {}).get("ZCTA5CE10"))
            if z:
                zip_to_city[z] = city_key
    return zip_to_city


def prepare_mapping_context() -> dict[str, Any]:
    print("Preparing lookup tables...")
    _names_by_city, unique_neighborhood_owner = build_neighborhood_sets()
    zip_to_city = build_zip_to_city_map()
    listing_neighborhood = load_listing_neighborhood_map()
    return {
        "unique_neighborhood_owner": unique_neighborhood_owner,
        "zip_to_city": zip_to_city,
        "listing_neighborhood": listing_neighborhood,
    }


def map_chunk_records(chunk: pd.DataFrame, ctx: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_neighborhood_owner = ctx["unique_neighborhood_owner"]
    zip_to_city = ctx["zip_to_city"]
    listing_neighborhood = ctx["listing_neighborhood"]

    hotel_scored = pd.DataFrame(columns=["city", "zip", "score"])
    airbnb_scored = pd.DataFrame(columns=["city", "neighborhood", "score"])

    hotels = chunk[chunk["type"] == "H"]
    if not hotels.empty:
        zip_series = hotels["postal_code"].map(clean_zip)
        city_series = zip_series.map(zip_to_city)
        hotel_scored = pd.DataFrame(
            {
                "city": city_series,
                "zip": zip_series,
                "score": hotels["score_0_100"].astype(float),
            }
        ).dropna(subset=["city", "zip", "score"])

    airbnb = chunk[chunk["type"] == "A"]
    if not airbnb.empty:
        listing_id = airbnb.get("listing_id", pd.Series(index=airbnb.index, dtype="object")).astype(str)
        city = listing_id.map(city_from_listing_id)
        neighborhood_norm = airbnb["neighborhood"].map(normalize_text)
        bad_nbh = neighborhood_norm.isin(["", "nan", "neighborhood highlights"])
        if bad_nbh.any() and listing_neighborhood:
            neighborhood_norm.loc[bad_nbh] = listing_id.loc[bad_nbh].map(listing_neighborhood)

        missing_city = city.isna()
        if missing_city.any():
            group = airbnb["Neighborhood group"].map(normalize_text)
            city.loc[missing_city & group.isin(NY_BOROUGHS)] = "ny"
            city.loc[missing_city & (group == "city of los angeles")] = "la"

        missing_city = city.isna()
        if missing_city.any():
            city.loc[missing_city] = neighborhood_norm.loc[missing_city].map(unique_neighborhood_owner)

        airbnb_scored = pd.DataFrame(
            {
                "city": city,
                "neighborhood": neighborhood_norm,
                "score": airbnb["score_0_100"].astype(float),
            }
        ).dropna(subset=["city", "neighborhood", "score"])
        airbnb_scored = airbnb_scored[airbnb_scored["neighborhood"] != ""]

    return hotel_scored, airbnb_scored


def add_normalized_score(df: pd.DataFrame, platform: str, params: dict[tuple[str, str], tuple[float, float]]) -> pd.DataFrame:
    if df.empty:
        return df.assign(norm_score=pd.Series(dtype=float))
    mean_map = {city: params[(platform, city)][0] for city in CITY_CONFIG if (platform, city) in params}
    std_map = {city: params[(platform, city)][1] for city in CITY_CONFIG if (platform, city) in params}
    means = df["city"].map(mean_map)
    stds = df["city"].map(std_map).replace(0, np.nan)
    out = df.copy()
    out["norm_score"] = ((out["score"] - means) / stds).fillna(0.0)
    return out


def aggregate_scores() -> tuple[dict[tuple[str, str], tuple[float, int]], dict[tuple[str, str], tuple[float, int]]]:
    ctx = prepare_mapping_context()

    print("Pass 1/2: computing city+platform normalization stats...")
    stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    usecols = [
        "type",
        "id_x",
        "listing_id",
        "Id_review",
        "postal_code",
        "neighborhood",
        "Neighborhood group",
        "score_0_100",
    ]
    for chunk in pd.read_csv(PREDICTIONS_CSV, usecols=usecols, chunksize=300_000, low_memory=False):
        chunk = chunk.dropna(subset=["score_0_100"])
        hotels, airbnb = map_chunk_records(chunk, ctx)

        if not hotels.empty:
            h = hotels.assign(score_sq=hotels["score"] * hotels["score"]).groupby("city").agg(
                count=("score", "count"),
                score_sum=("score", "sum"),
                score_sq_sum=("score_sq", "sum"),
            )
            for city, row in h.iterrows():
                acc = stats[("H", city)]
                acc[0] += float(row["count"])
                acc[1] += float(row["score_sum"])
                acc[2] += float(row["score_sq_sum"])

        if not airbnb.empty:
            a = airbnb.assign(score_sq=airbnb["score"] * airbnb["score"]).groupby("city").agg(
                count=("score", "count"),
                score_sum=("score", "sum"),
                score_sq_sum=("score_sq", "sum"),
            )
            for city, row in a.iterrows():
                acc = stats[("A", city)]
                acc[0] += float(row["count"])
                acc[1] += float(row["score_sum"])
                acc[2] += float(row["score_sq_sum"])

    params: dict[tuple[str, str], tuple[float, float]] = {}
    for key, (count, score_sum, score_sq_sum) in stats.items():
        if count <= 0:
            continue
        mean = score_sum / count
        var = max((score_sq_sum / count) - (mean * mean), 0.0)
        std = np.sqrt(var)
        if std < 1e-9:
            std = 1.0
        params[key] = (mean, std)

    print("Pass 2/2: aggregating normalized scores by ZIP/neighborhood...")
    hotel_acc: dict[tuple[str, str], list[float | int]] = defaultdict(lambda: [0.0, 0])
    airbnb_acc: dict[tuple[str, str], list[float | int]] = defaultdict(lambda: [0.0, 0])

    for chunk in pd.read_csv(PREDICTIONS_CSV, usecols=usecols, chunksize=300_000, low_memory=False):
        chunk = chunk.dropna(subset=["score_0_100"])
        hotels, airbnb = map_chunk_records(chunk, ctx)

        hotels = add_normalized_score(hotels, "H", params)
        if not hotels.empty:
            h = hotels.groupby(["city", "zip"]).agg(
                norm_sum=("norm_score", "sum"),
                count=("norm_score", "count"),
            )
            for (city, z), row in h.iterrows():
                acc = hotel_acc[(city, z)]
                acc[0] += float(row["norm_sum"])
                acc[1] += int(row["count"])

        airbnb = add_normalized_score(airbnb, "A", params)
        if not airbnb.empty:
            a = airbnb.groupby(["city", "neighborhood"]).agg(
                norm_sum=("norm_score", "sum"),
                count=("norm_score", "count"),
            )
            for (city, nbh), row in a.iterrows():
                key = (city, nbh)
                acc = airbnb_acc[key]
                acc[0] += float(row["norm_sum"])
                acc[1] += int(row["count"])

    hotel_mean = {k: (v[0] / v[1], int(v[1])) for k, v in hotel_acc.items() if v[1] > 0}
    airbnb_mean = {k: (v[0] / v[1], int(v[1])) for k, v in airbnb_acc.items() if v[1] > 0}
    return hotel_mean, airbnb_mean


def find_zip_for_point(
    point: tuple[float, float],
    zip_features: list[dict[str, Any]],
    zip_centroids: dict[str, tuple[float, float]],
) -> str | None:
    for feature in zip_features:
        z = clean_zip(feature.get("properties", {}).get("ZCTA5CE10"))
        if not z:
            continue
        if geometry_contains_point(feature.get("geometry", {}), point):
            return z

    nearest_zip = None
    nearest_dist = None
    for z, c in zip_centroids.items():
        d = sq_dist(point, c)
        if nearest_dist is None or d < nearest_dist:
            nearest_dist = d
            nearest_zip = z
    return nearest_zip


def estimate_zip_overlap_weights(
    geometry: dict[str, Any] | None,
    zip_features: list[dict[str, Any]],
    zip_centroids: dict[str, tuple[float, float]],
    zip_bounds: dict[str, tuple[float, float, float, float] | None],
    samples_per_axis: int = 18,
) -> dict[str, float]:
    bounds = geometry_bounds(geometry)
    if geometry is None or bounds is None:
        return {}

    candidate_features = []
    for feature in zip_features:
        z = clean_zip(feature.get("properties", {}).get("ZCTA5CE10"))
        if not z:
            continue
        if bounds_overlap(bounds, zip_bounds.get(z)):
            candidate_features.append(feature)

    if not candidate_features:
        point = representative_point(geometry)
        z = find_zip_for_point(point, zip_features, zip_centroids)
        return {z: 1.0} if z else {}

    minx, maxx, miny, maxy = bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    inside_total = 0
    counts: dict[str, int] = defaultdict(int)

    for ix in range(samples_per_axis):
        x = minx + ((ix + 0.5) / samples_per_axis) * width
        for iy in range(samples_per_axis):
            y = miny + ((iy + 0.5) / samples_per_axis) * height
            point = (x, y)
            if not geometry_contains_point(geometry, point):
                continue
            inside_total += 1
            matched_zip = None
            for feature in candidate_features:
                z = clean_zip(feature.get("properties", {}).get("ZCTA5CE10"))
                if not z:
                    continue
                if geometry_contains_point(feature.get("geometry", {}), point):
                    matched_zip = z
                    break
            if matched_zip is not None:
                counts[matched_zip] += 1

    if inside_total <= 0 or not counts:
        point = representative_point(geometry)
        z = find_zip_for_point(point, zip_features, zip_centroids)
        return {z: 1.0} if z else {}

    total = sum(counts.values())
    if total <= 0:
        point = representative_point(geometry)
        z = find_zip_for_point(point, zip_features, zip_centroids)
        return {z: 1.0} if z else {}

    return {z: c / total for z, c in counts.items() if c > 0}


def build_city_outputs(hotel_mean: dict, airbnb_mean: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    us_geo = json.loads(US_ZIP_GEOJSON.read_text())
    city_zip_features = build_city_zip_features(us_geo)
    summary_rows = []
    city_payloads: dict[str, dict[str, Any]] = {}

    for city_key, cfg in CITY_CONFIG.items():
        city_label = cfg["label"]
        print(f"Building map for {city_label}...")
        zip_features = city_zip_features[city_key]
        city_zip_geo = {"type": "FeatureCollection", "features": zip_features}

        zip_centroids = {}
        zip_bounds = {}
        for feature in zip_features:
            z = clean_zip(feature.get("properties", {}).get("ZCTA5CE10"))
            if not z:
                continue
            zip_centroids[z] = representative_point(feature.get("geometry", {}))
            zip_bounds[z] = geometry_bounds(feature.get("geometry", {}))

        airbnb_geo = json.loads(cfg["airbnb_geojson"].read_text())
        zip_airbnb_acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

        for feature in airbnb_geo.get("features", []):
            nbh = normalize_text(feature.get("properties", {}).get("neighbourhood"))
            if not nbh:
                continue
            nbh_key = (city_key, nbh)
            if nbh_key not in airbnb_mean:
                continue
            score, count = airbnb_mean[nbh_key]
            zip_weights = estimate_zip_overlap_weights(
                feature.get("geometry", {}),
                zip_features,
                zip_centroids,
                zip_bounds,
            )
            for z, weight in zip_weights.items():
                acc = zip_airbnb_acc[z]
                acc[0] += score * count * weight
                acc[1] += count * weight

        zip_rows = []
        zip_ids = sorted(
            {clean_zip(f.get("properties", {}).get("ZCTA5CE10")) for f in zip_features if clean_zip(f.get("properties", {}).get("ZCTA5CE10"))}
        )
        for z in zip_ids:
            h = hotel_mean.get((city_key, z))
            a_acc = zip_airbnb_acc.get(z)
            a_score = None
            a_count = 0
            if a_acc and a_acc[1] > 0:
                a_score = a_acc[0] / a_acc[1]
                a_count = int(round(a_acc[1]))

            h_score = h[0] if h else None
            h_count = h[1] if h else 0
            diff = None
            winner = "no_data"
            if (a_score is not None) and (h_score is not None):
                diff = a_score - h_score
                if diff > 0:
                    winner = "airbnb"
                elif diff < 0:
                    winner = "hotel"
                else:
                    winner = "tie"
            elif a_score is not None:
                winner = "airbnb_only"
            elif h_score is not None:
                winner = "hotel_only"

            zip_rows.append(
                {
                    "city": city_key,
                    "city_label": city_label,
                    "postal_code": z,
                    "airbnb_norm_score_z": a_score,
                    "hotel_norm_score_z": h_score,
                    "diff_airbnb_minus_hotel_z": diff,
                    "winner": winner,
                    "airbnb_row_count": a_count,
                    "hotel_row_count": h_count,
                }
            )

        scores_by_zip = {}
        for row in zip_rows:
            scores_by_zip[row["postal_code"]] = {
                "airbnb": None if row["airbnb_norm_score_z"] is None else round(row["airbnb_norm_score_z"], 2),
                "hotel": None if row["hotel_norm_score_z"] is None else round(row["hotel_norm_score_z"], 2),
                "diff": None
                if row["diff_airbnb_minus_hotel_z"] is None
                else round(row["diff_airbnb_minus_hotel_z"], 2),
                "winner": row["winner"],
                "airbnb_n": row["airbnb_row_count"],
                "hotel_n": row["hotel_row_count"],
            }

        summary_rows.extend(zip_rows)

        geo_out = OUT_DIR / f"{city_key}_zip_boundaries.geojson"
        geo_out.write_text(json.dumps(city_zip_geo))
        html_out = OUT_DIR / f"{city_key}_postal_comparison_map.html"
        html_out.write_text(render_map_html(city_label, city_key, city_zip_geo, scores_by_zip))
        city_payloads[city_key] = {
            "label": city_label,
            "geo": city_zip_geo,
            "scores": scores_by_zip,
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "postal_comparison_scores.csv", index=False)
    (OUT_DIR / "all_cities_postal_comparison_map.html").write_text(
        render_multi_city_map_html(city_payloads)
    )


def render_map_html(
    city_label: str,
    city_key: str,
    geojson_data: dict[str, Any],
    scores_by_zip: dict[str, dict[str, Any]],
) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{city_label} Postal Comparison Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root {{
      --airbnb: #e91e63;
      --hotel: #1976d2;
      --neutral: #f7f7f7;
      --missing: #d7d7d7;
      --bg: #f3f6fa;
      --ink: #0f172a;
    }}
    body {{
      margin: 0;
      padding: 14px;
      font-family: "Helvetica Neue", Arial, sans-serif;
      background: radial-gradient(circle at 20% 20%, #ffffff, #e9eff7);
      color: var(--ink);
    }}
    h2 {{ margin: 6px 0 4px; }}
    #subtitle {{ margin: 0 0 10px; color: #475569; font-size: 13px; }}
    #card {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 6px 20px rgba(15, 23, 42, 0.12);
      padding: 8px;
      display: inline-block;
    }}
    #map {{
      width: 920px;
      height: 660px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
    }}
    #tooltip {{
      position: absolute;
      opacity: 0;
      pointer-events: none;
      background: #fff;
      border: 1px solid #d0d7e2;
      border-radius: 6px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.18);
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .legend-box {{
      background: rgba(255,255,255,0.94);
      padding: 8px 10px;
      border: 1px solid #d0d7e2;
      border-radius: 6px;
      font-size: 11px;
      line-height: 1.4;
    }}
    .footer-note {{
      margin-top: 8px;
      font-size: 12px;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <h2>{city_label}: Airbnb vs Hotel Predicted Scores by Postal Code</h2>
  <div id="subtitle">Pink = Airbnb stronger or Airbnb-only, Blue = Hotel stronger or Hotel-only, Gray = no data from either side.</div>
  <div id="card"><div id="map"></div></div>
  <div class="footer-note">Source: XGBoost `score_0_100`, normalized within city and platform (z-score), from `model_output/predictions_all.csv`.</div>
  <script>
    const GEO = {json.dumps(geojson_data)};
    const SCORE = {json.dumps(scores_by_zip)};
    const map = L.map("map", {{ zoomControl: true, attributionControl: false }});

    const posPalette = ["#fce4ec", "#f8bbd0", "#f48fb1", "#f06292", "#ec407a", "#e91e63"];
    const negPalette = ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5", "#1e88e5"];
    const diffs = Object.values(SCORE).map(d => d.diff).filter(d => d !== null && Number.isFinite(d));
    const singleSide = Object.values(SCORE)
      .flatMap(d => [d.airbnb, d.hotel])
      .filter(d => d !== null && Number.isFinite(d));
    const maxAbs = Math.max(
      1,
      ...diffs.map(d => Math.abs(d)),
      ...singleSide.map(d => Math.abs(d))
    );
    const posDiffs = diffs.filter(d => d > 0).sort((a, b) => a - b);
    const negDiffs = diffs.filter(d => d < 0).map(d => Math.abs(d)).sort((a, b) => a - b);
    const posSingles = Object.values(SCORE)
      .map(d => d.airbnb)
      .filter(d => d !== null && Number.isFinite(d))
      .map(d => Math.abs(d))
      .sort((a, b) => a - b);
    const negSingles = Object.values(SCORE)
      .map(d => d.hotel)
      .filter(d => d !== null && Number.isFinite(d))
      .map(d => Math.abs(d))
      .sort((a, b) => a - b);

    function quantileBins(values, bins) {{
      if (!values.length) return [1];
      const out = [];
      for (let i = 1; i <= bins; i++) {{
        const idx = Math.floor((i / bins) * (values.length - 1));
        out.push(values[idx]);
      }}
      return out;
    }}

    const posCuts = quantileBins(posDiffs, posPalette.length);
    const negCuts = quantileBins(negDiffs, negPalette.length);
    const posSingleCuts = quantileBins(posSingles, posPalette.length);
    const negSingleCuts = quantileBins(negSingles, negPalette.length);

    function getZip(props) {{
      return String(props.ZCTA5CE10 || props.zip || props.ZIP || props.postal_code || "").padStart(5, "0");
    }}

    function pickBin(value, cuts, palette) {{
      for (let i = 0; i < cuts.length; i++) {{
        if (value <= cuts[i]) return palette[i];
      }}
      return palette[palette.length - 1];
    }}

    function fillForZip(zip) {{
      const row = SCORE[zip];
      if (!row) return "#d7d7d7";
      if (row.diff > 0) return pickBin(row.diff, posCuts, posPalette);
      if (row.diff < 0) return pickBin(Math.abs(row.diff), negCuts, negPalette);
      if (row.diff === 0) return "#f7f7f7";
      if (row.airbnb !== null && row.airbnb !== undefined) return pickBin(Math.abs(row.airbnb), posSingleCuts, posPalette);
      if (row.hotel !== null && row.hotel !== undefined) return pickBin(Math.abs(row.hotel), negSingleCuts, negPalette);
      return "#f7f7f7";
    }}

    const layer = L.geoJSON(GEO, {{
      style: function(feature) {{
        const zip = getZip(feature.properties || {{}});
        return {{
          fillColor: fillForZip(zip),
          fillOpacity: 0.88,
          color: "#334155",
          weight: 1.2,
          opacity: 0.8
        }};
      }},
      onEachFeature: function(feature, lyr) {{
        const zip = getZip(feature.properties || {{}});
        const row = SCORE[zip] || {{}};
        const air = row.airbnb === null || row.airbnb === undefined ? "NA" : row.airbnb.toFixed(2);
        const hot = row.hotel === null || row.hotel === undefined ? "NA" : row.hotel.toFixed(2);
        const dif = row.diff === null || row.diff === undefined ? "NA" : row.diff.toFixed(2);
        const winner = row.winner || "no_data";
        lyr.bindTooltip(
          "<b>ZIP:</b> " + zip + "<br/>" +
          "<b>Winner:</b> " + winner
        );
      }}
    }}).addTo(map);

    map.fitBounds(layer.getBounds(), {{ padding: [10, 10] }});

    const legend = L.control({{ position: "topright" }});
    legend.onAdd = function() {{
      const div = L.DomUtil.create("div", "legend-box");
      div.innerHTML =
        "<div><b>Hotel better</b> to <b>Airbnb better</b></div>" +
        "<div style='margin-top:6px; width:260px; height:12px; background: linear-gradient(to right,#1e88e5,#f7f7f7,#e91e63); border:1px solid #cbd5e1;'></div>" +
        "<div style='display:flex; justify-content:space-between; margin-top:2px;'><span>-" + maxAbs.toFixed(1) + "</span><span>0</span><span>" + maxAbs.toFixed(1) + "</span></div>";
      return div;
    }};
    legend.addTo(map);
  </script>
</body>
</html>
"""


def render_multi_city_map_html(city_payloads: dict[str, dict[str, Any]]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>All Cities Postal Comparison Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root {{
      --airbnb: #e91e63;
      --hotel: #1976d2;
      --neutral: #f7f7f7;
      --missing: #d7d7d7;
      --bg: #f3f6fa;
      --ink: #0f172a;
    }}
    body {{
      margin: 0;
      padding: 14px;
      font-family: "Helvetica Neue", Arial, sans-serif;
      background: radial-gradient(circle at 20% 20%, #ffffff, #e9eff7);
      color: var(--ink);
    }}
    .topbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }}
    h2 {{ margin: 6px 0 4px; }}
    #subtitle {{ margin: 0 0 10px; color: #475569; font-size: 13px; }}
    select {{
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 14px;
      background: #fff;
    }}
    #card {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 6px 20px rgba(15, 23, 42, 0.12);
      padding: 8px;
      display: inline-block;
    }}
    #map {{
      width: 920px;
      height: 660px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
    }}
    .legend-box {{
      background: rgba(255,255,255,0.94);
      padding: 8px 10px;
      border: 1px solid #d0d7e2;
      border-radius: 6px;
      font-size: 11px;
      line-height: 1.4;
    }}
    .footer-note {{
      margin-top: 8px;
      font-size: 12px;
      color: #64748b;
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <h2 id="title">Airbnb vs Hotel Predicted Scores by Postal Code</h2>
    <label for="citySelect">City:</label>
    <select id="citySelect">
      <option value="ny">New York City</option>
      <option value="la">Los Angeles</option>
      <option value="sf">San Francisco</option>
      <option value="sd">San Diego</option>
    </select>
  </div>
  <div id="subtitle">Pink = Airbnb stronger or Airbnb-only, Blue = Hotel stronger or Hotel-only, Gray = no data from either side.</div>
  <div id="card"><div id="map"></div></div>
  <div class="footer-note">Source: XGBoost `score_0_100`, normalized within city and platform (z-score), from `model_output/predictions_all.csv`.</div>
  <script>
    const CITY_DATA = {json.dumps(city_payloads)};
    const title = document.getElementById("title");
    const citySelect = document.getElementById("citySelect");
    const map = L.map("map", {{ zoomControl: true, attributionControl: false }});
    let activeLayer = null;
    let legendControl = null;

    function getZip(props) {{
      return String(props.ZCTA5CE10 || props.zip || props.ZIP || props.postal_code || "").padStart(5, "0");
    }}

    function drawForCity(cityKey) {{
      const city = CITY_DATA[cityKey];
      const geo = city.geo;
      const score = city.scores;

      title.textContent = city.label + ": Airbnb vs Hotel Predicted Scores by Postal Code";

      const posPalette = ["#fce4ec", "#f8bbd0", "#f48fb1", "#f06292", "#ec407a", "#e91e63"];
      const negPalette = ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5", "#1e88e5"];
      const diffs = Object.values(score).map(d => d.diff).filter(d => d !== null && Number.isFinite(d));
      const singleSide = Object.values(score)
        .flatMap(d => [d.airbnb, d.hotel])
        .filter(d => d !== null && Number.isFinite(d));
      const maxAbs = Math.max(
        1,
        ...diffs.map(d => Math.abs(d)),
        ...singleSide.map(d => Math.abs(d))
      );
      const posDiffs = diffs.filter(d => d > 0).sort((a, b) => a - b);
      const negDiffs = diffs.filter(d => d < 0).map(d => Math.abs(d)).sort((a, b) => a - b);
      const posSingles = Object.values(score)
        .map(d => d.airbnb)
        .filter(d => d !== null && Number.isFinite(d))
        .map(d => Math.abs(d))
        .sort((a, b) => a - b);
      const negSingles = Object.values(score)
        .map(d => d.hotel)
        .filter(d => d !== null && Number.isFinite(d))
        .map(d => Math.abs(d))
        .sort((a, b) => a - b);

      function quantileBins(values, bins) {{
        if (!values.length) return [1];
        const out = [];
        for (let i = 1; i <= bins; i++) {{
          const idx = Math.floor((i / bins) * (values.length - 1));
          out.push(values[idx]);
        }}
        return out;
      }}

      const posCuts = quantileBins(posDiffs, posPalette.length);
      const negCuts = quantileBins(negDiffs, negPalette.length);
      const posSingleCuts = quantileBins(posSingles, posPalette.length);
      const negSingleCuts = quantileBins(negSingles, negPalette.length);

      function pickBin(value, cuts, palette) {{
        for (let i = 0; i < cuts.length; i++) {{
          if (value <= cuts[i]) return palette[i];
        }}
        return palette[palette.length - 1];
      }}

      function fillForZip(zip) {{
        const row = score[zip];
        if (!row) return "#d7d7d7";
        if (row.diff > 0) return pickBin(row.diff, posCuts, posPalette);
        if (row.diff < 0) return pickBin(Math.abs(row.diff), negCuts, negPalette);
        if (row.diff === 0) return "#f7f7f7";
        if (row.airbnb !== null && row.airbnb !== undefined) return pickBin(Math.abs(row.airbnb), posSingleCuts, posPalette);
        if (row.hotel !== null && row.hotel !== undefined) return pickBin(Math.abs(row.hotel), negSingleCuts, negPalette);
        return "#d7d7d7";
      }}

      if (activeLayer) map.removeLayer(activeLayer);
      if (legendControl) map.removeControl(legendControl);

      activeLayer = L.geoJSON(geo, {{
        style: function(feature) {{
          const zip = getZip(feature.properties || {{}});
          return {{
            fillColor: fillForZip(zip),
            fillOpacity: 0.88,
            color: "#334155",
            weight: 1.2,
            opacity: 0.8
          }};
        }},
        onEachFeature: function(feature, lyr) {{
          const zip = getZip(feature.properties || {{}});
          const row = score[zip] || {{}};
          const air = row.airbnb === null || row.airbnb === undefined ? "NA" : row.airbnb.toFixed(2);
          const hot = row.hotel === null || row.hotel === undefined ? "NA" : row.hotel.toFixed(2);
          const dif = row.diff === null || row.diff === undefined ? "NA" : row.diff.toFixed(2);
          const winner = row.winner || "no_data";
          lyr.bindTooltip(
            "<b>ZIP:</b> " + zip + "<br/>" +
            "<b>Winner:</b> " + winner
          );
        }}
      }}).addTo(map);

      map.fitBounds(activeLayer.getBounds(), {{ padding: [10, 10] }});

      legendControl = L.control({{ position: "topright" }});
      legendControl.onAdd = function() {{
        const div = L.DomUtil.create("div", "legend-box");
        div.innerHTML =
          "<div><b>Hotel better</b> to <b>Airbnb better</b></div>" +
          "<div style='margin-top:6px; width:260px; height:12px; background: linear-gradient(to right,#1e88e5,#f7f7f7,#e91e63); border:1px solid #cbd5e1;'></div>" +
          "<div style='display:flex; justify-content:space-between; margin-top:2px;'><span>-" + maxAbs.toFixed(1) + "</span><span>0</span><span>" + maxAbs.toFixed(1) + "</span></div>" +
          "<div style='margin-top:6px;'>Gray = no Airbnb and no hotel rows in that ZIP.</div>";
        return div;
      }};
      legendControl.addTo(map);
    }}

    citySelect.addEventListener("change", function() {{
      drawForCity(this.value);
    }});
    drawForCity(citySelect.value);
  </script>
</body>
</html>
"""


def main() -> None:
    if not US_ZIP_GEOJSON.exists():
        raise FileNotFoundError(
            f"Missing ZIP geojson: {US_ZIP_GEOJSON}\n"
            "Download it first to Visualization/data/us_zip_codes_geo.json"
        )

    hotel_mean, airbnb_mean = aggregate_scores()
    build_city_outputs(hotel_mean, airbnb_mean)
    print("\nDone. Wrote maps to:", OUT_DIR)


if __name__ == "__main__":
    main()
