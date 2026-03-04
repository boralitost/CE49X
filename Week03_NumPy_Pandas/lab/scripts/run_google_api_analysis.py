#!/usr/bin/env python3
"""
Reproducible CE49X Lab 3 analysis pipeline.

What this script does:
1) Loads traffic data from istanbul_traffic_week.csv
2) Computes location-level traffic metrics and demand score
3) Collects existing gas stations via Google Places Nearby Search API
4) Computes nearest-station distance with Haversine formula
5) Selects 3 final candidate locations
6) Writes JSON/CSV outputs and interactive HTML visuals

Usage:
    export GOOGLE_MAPS_API_KEY="AIzaSyAXIL3wQnBSJxGDjzVvbBGvCspRKjJcohU"
    python Week03_NumPy_Pandas/lab/scripts/run_google_api_analysis.py
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
from folium.plugins import HeatMap


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "Week03_NumPy_Pandas"
LAB_DIR = DATA_DIR / "lab"
OUTPUT_DIR = LAB_DIR / "outputs"
TRAFFIC_FILE = DATA_DIR / "istanbul_traffic_week.csv"

EARTH_RADIUS_KM = 6371.0


def minmax(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min() + 1e-12)


def haversine_distance_km(
    lat1_deg: np.ndarray, lon1_deg: np.ndarray, lat2_deg: np.ndarray, lon2_deg: np.ndarray
) -> np.ndarray:
    """
    Vectorized Haversine distance (km) between:
      - first set: (lat1_deg, lon1_deg) with shape (N,)
      - second set: (lat2_deg, lon2_deg) with shape (M,)
    Returns matrix shape (N, M).
    """
    lat1 = np.deg2rad(lat1_deg)[:, None]
    lon1 = np.deg2rad(lon1_deg)[:, None]
    lat2 = np.deg2rad(lat2_deg)[None, :]
    lon2 = np.deg2rad(lon2_deg)[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def load_and_score_traffic() -> pd.DataFrame:
    traffic = pd.read_csv(TRAFFIC_FILE)
    traffic["DATE_TIME"] = pd.to_datetime(traffic["DATE_TIME"])
    traffic["date"] = traffic["DATE_TIME"].dt.date
    traffic["hour"] = traffic["DATE_TIME"].dt.hour
    traffic["dow"] = traffic["DATE_TIME"].dt.day_name()

    loc_col = "GEOHASH"
    daily = traffic.groupby([loc_col, "date"], as_index=False)["NUMBER_OF_VEHICLES"].sum()
    mean_daily = (
        daily.groupby(loc_col, as_index=False)["NUMBER_OF_VEHICLES"]
        .mean()
        .rename(columns={"NUMBER_OF_VEHICLES": "mean_daily_vehicle_count"})
    )

    loc = (
        traffic.groupby(loc_col, as_index=False)
        .agg(
            mean_speed=("AVERAGE_SPEED", "mean"),
            peak_hour_vehicle_count=("NUMBER_OF_VEHICLES", "max"),
            total_vehicle_count=("NUMBER_OF_VEHICLES", "sum"),
            lat=("LATITUDE", "mean"),
            lon=("LONGITUDE", "mean"),
            veh_mean=("NUMBER_OF_VEHICLES", "mean"),
            veh_std=("NUMBER_OF_VEHICLES", "std"),
        )
        .merge(mean_daily, on=loc_col, how="left")
        .fillna(0.0)
    )

    loc["cv"] = loc["veh_std"] / (loc["veh_mean"] + 1e-9)
    loc["consistency"] = 1.0 / (1.0 + loc["cv"])

    loc["vol_norm"] = minmax(loc["mean_daily_vehicle_count"])
    loc["speed_norm"] = minmax(loc["mean_speed"])
    loc["cons_norm"] = minmax(loc["consistency"])

    # Demand score: high volume + low speed + consistency
    loc["demand_score"] = (
        0.50 * loc["vol_norm"] + 0.30 * (1.0 - loc["speed_norm"]) + 0.20 * loc["cons_norm"]
    )
    return loc, traffic


def collect_google_places_stations(api_key: str) -> pd.DataFrame:
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # Coverage centers across Istanbul
    centers = [
        (41.0082, 28.9784),
        (41.043, 29.000),
        (41.080, 29.020),
        (41.110, 29.060),
        (41.020, 29.080),
        (41.150, 29.000),
        (41.180, 29.120),
        (41.000, 28.900),
        (41.030, 28.850),
        (41.070, 28.780),
        (41.220, 29.030),
        (40.980, 29.180),
    ]

    rows: list[dict] = []
    for lat, lon in centers:
        params = {
            "location": f"{lat},{lon}",
            "radius": 18000,
            "type": "gas_station",
            "key": api_key,
        }
        page_count = 0
        while page_count < 3:
            resp = requests.get(url, params=params, timeout=40)
            resp.raise_for_status()
            payload = resp.json()
            for place in payload.get("results", []):
                geo = place.get("geometry", {}).get("location", {})
                plat = geo.get("lat")
                plon = geo.get("lng")
                if plat is None or plon is None:
                    continue
                rows.append(
                    {
                        "place_id": place.get("place_id"),
                        "name": place.get("name", "Unknown"),
                        "address": place.get("vicinity", ""),
                        "lat": plat,
                        "lon": plon,
                    }
                )

            token = payload.get("next_page_token")
            if not token:
                break
            time.sleep(2.2)  # token propagation
            params = {"pagetoken": token, "key": api_key}
            page_count += 1

    stations = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["place_id"])
        .drop_duplicates(subset=["lat", "lon"])
        .reset_index(drop=True)
    )
    return stations


def fill_districts_with_nominatim(df: pd.DataFrame) -> pd.DataFrame:
    def reverse_district(lat: float, lon: float) -> str:
        try:
            r = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format": "jsonv2", "lat": lat, "lon": lon, "addressdetails": 1},
                headers={"User-Agent": "CE49X-Lab3/1.0"},
                timeout=25,
            )
            r.raise_for_status()
            ad = r.json().get("address", {})
            return (
                ad.get("town")
                or ad.get("city_district")
                or ad.get("suburb")
                or ad.get("county")
                or ad.get("city")
                or "Unknown"
            )
        except Exception:
            return "Unknown"

    out = df.copy()
    out["district"] = [reverse_district(a, b) for a, b in zip(out["lat"], out["lon"])]
    return out


def select_three_sites(loc: pd.DataFrame) -> pd.DataFrame:
    eligible = loc[loc["demand_score"] >= 0.55].sort_values("underserved_score", ascending=False)
    chosen = []
    for _, row in eligible.iterrows():
        if not chosen:
            chosen.append(row)
        else:
            keep = True
            for prev in chosen:
                d = haversine_distance_km(
                    np.array([row["lat"]]),
                    np.array([row["lon"]]),
                    np.array([prev["lat"]]),
                    np.array([prev["lon"]]),
                )[0, 0]
                if d < 3.0:
                    keep = False
                    break
            if keep:
                chosen.append(row)
        if len(chosen) == 3:
            break
    return pd.DataFrame(chosen).reset_index(drop=True)


def write_visuals(traffic: pd.DataFrame, loc: pd.DataFrame, stations: pd.DataFrame, proposed: pd.DataFrame):
    hourly = traffic.groupby("hour", as_index=False)["NUMBER_OF_VEHICLES"].mean()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day = traffic.groupby("dow", as_index=False)["NUMBER_OF_VEHICLES"].mean()
    by_day["dow"] = pd.Categorical(by_day["dow"], categories=dow_order, ordered=True)
    by_day = by_day.sort_values("dow")

    fig_hour = px.line(
        hourly, x="hour", y="NUMBER_OF_VEHICLES", markers=True, title="Average Traffic by Hour"
    )
    fig_hour.write_html(OUTPUT_DIR / "traffic_by_hour.html", include_plotlyjs="cdn")

    fig_day = px.bar(by_day, x="dow", y="NUMBER_OF_VEHICLES", title="Average Traffic by Day of Week")
    fig_day.write_html(OUTPUT_DIR / "traffic_by_day.html", include_plotlyjs="cdn")

    subset = traffic[traffic["GEOHASH"].isin(proposed["GEOHASH"])]
    by_site_hour = subset.groupby(["GEOHASH", "hour"], as_index=False)["NUMBER_OF_VEHICLES"].mean()
    fig_prop = px.line(
        by_site_hour,
        x="hour",
        y="NUMBER_OF_VEHICLES",
        color="GEOHASH",
        markers=True,
        title="Hourly Traffic at Proposed Locations",
    )
    fig_prop.write_html(OUTPUT_DIR / "proposed_hourly.html", include_plotlyjs="cdn")

    m = folium.Map(location=[41.03, 29.00], zoom_start=10, tiles="cartodbpositron")
    HeatMap(loc[["lat", "lon", "demand_score"]].values.tolist(), radius=9, blur=12).add_to(m)
    for _, s in stations.sample(min(len(stations), 600), random_state=42).iterrows():
        folium.CircleMarker(
            [s["lat"], s["lon"]],
            radius=1.8,
            color="blue",
            fill=True,
            fill_opacity=0.35,
            weight=1,
            popup=folium.Popup(
                f"<b>{s.get('name', 'Gas station')}</b><br>{s.get('address', '')}",
                max_width=250,
            ),
        ).add_to(m)
    for i, r in proposed.iterrows():
        folium.Marker(
            [r["lat"], r["lon"]],
            popup=(
                f"Proposed #{i + 1}<br>District: {r['district']}<br>"
                f"Demand: {r['demand_score']:.3f}<br>"
                f"Nearest station: {r['nearest_station_km']:.2f} km"
            ),
            icon=folium.Icon(color="red", icon="star"),
        ).add_to(m)
    m.save(OUTPUT_DIR / "demand_station_map.html")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY is not set.")

    loc, traffic = load_and_score_traffic()
    stations = collect_google_places_stations(api_key)
    if len(stations) < 200:
        raise RuntimeError(f"Only {len(stations)} stations collected (< 200 required).")

    dist_matrix = haversine_distance_km(
        loc["lat"].to_numpy(),
        loc["lon"].to_numpy(),
        stations["lat"].to_numpy(),
        stations["lon"].to_numpy(),
    )
    loc["nearest_station_km"] = dist_matrix.min(axis=1)
    loc["dist_norm"] = minmax(loc["nearest_station_km"])
    loc["underserved_score"] = 0.80 * loc["demand_score"] + 0.20 * loc["dist_norm"]

    proposed = select_three_sites(loc)
    proposed = fill_districts_with_nominatim(proposed)

    ranked = loc.sort_values("demand_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank_by_demand", np.arange(1, len(ranked) + 1))
    ranked.to_csv(OUTPUT_DIR / "all_location_demand_ranking.csv", index=False)
    stations.to_csv(OUTPUT_DIR / "google_places_stations.csv", index=False)

    # Sensitivity check
    loc["d_alt1"] = 0.4 * loc["vol_norm"] + 0.4 * (1 - loc["speed_norm"]) + 0.2 * loc["cons_norm"]
    loc["d_alt2"] = 0.6 * loc["vol_norm"] + 0.2 * (1 - loc["speed_norm"]) + 0.2 * loc["cons_norm"]
    base_top = set(loc.sort_values("demand_score", ascending=False).head(10)["GEOHASH"])
    alt1_top = set(loc.sort_values("d_alt1", ascending=False).head(10)["GEOHASH"])
    alt2_top = set(loc.sort_values("d_alt2", ascending=False).head(10)["GEOHASH"])

    write_visuals(traffic, loc, stations, proposed)

    result = {
        "traffic_rows": int(len(traffic)),
        "sensor_count": int(traffic["GEOHASH"].nunique()),
        "date_min": str(traffic["DATE_TIME"].min()),
        "date_max": str(traffic["DATE_TIME"].max()),
        "station_count": int(len(stations)),
        "station_source": "Google Places API (Nearby Search)",
        "avg_hourly_vehicles": float(round(traffic["NUMBER_OF_VEHICLES"].mean(), 3)),
        "avg_speed": float(round(traffic["AVERAGE_SPEED"].mean(), 3)),
        "top20": (
            loc.sort_values("total_vehicle_count", ascending=False)
            .head(20)[
                [
                    "GEOHASH",
                    "lat",
                    "lon",
                    "total_vehicle_count",
                    "mean_daily_vehicle_count",
                    "mean_speed",
                    "demand_score",
                    "nearest_station_km",
                ]
            ]
            .round(3)
            .to_dict(orient="records")
        ),
        "proposed": (
            proposed[
                [
                    "GEOHASH",
                    "lat",
                    "lon",
                    "district",
                    "demand_score",
                    "nearest_station_km",
                    "mean_daily_vehicle_count",
                    "mean_speed",
                    "underserved_score",
                ]
            ]
            .round(3)
            .to_dict(orient="records")
        ),
        "sensitivity": {
            "base_vs_alt1_top10_overlap": int(len(base_top & alt1_top)),
            "base_vs_alt2_top10_overlap": int(len(base_top & alt2_top)),
        },
    }
    (OUTPUT_DIR / "analysis_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Done. Stations collected: {len(stations)}")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
