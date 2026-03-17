"""
Fetch gas station locations in Istanbul from OpenStreetMap via the Overpass API.

Queries for amenity=fuel within the Istanbul administrative boundary,
then saves as both CSV and GeoJSON.

Data source: OpenStreetMap contributors, via Overpass API
License: ODbL (https://opendatacommons.org/licenses/odbl/)
"""

import json
import urllib.request
import urllib.parse
import csv
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Query: all fuel stations within the Istanbul admin boundary (relation 223474)
QUERY = """
[out:json][timeout:120];
area["name"="İstanbul"]["admin_level"="4"]->.istanbul;
(
  node["amenity"="fuel"](area.istanbul);
  way["amenity"="fuel"](area.istanbul);
  relation["amenity"="fuel"](area.istanbul);
);
out center tags;
"""

OUTPUT_DIR = Path(__file__).parent


def fetch_gas_stations():
    print("Querying Overpass API for gas stations in Istanbul...")
    encoded = urllib.parse.urlencode({"data": QUERY}).encode("utf-8")
    req = urllib.request.Request(OVERPASS_URL, data=encoded)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    elements = data.get("elements", [])
    print(f"Raw elements returned: {len(elements)}")

    stations = []
    for el in elements:
        # For ways/relations, use the 'center' coordinates
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue

        tags = el.get("tags", {})
        stations.append({
            "osm_id": el["id"],
            "osm_type": el["type"],
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "name": tags.get("name", ""),
            "brand": tags.get("brand", ""),
            "operator": tags.get("operator", ""),
            "addr_street": tags.get("addr:street", ""),
            "addr_district": tags.get("addr:district", ""),
            "opening_hours": tags.get("opening_hours", ""),
        })

    print(f"Stations with valid coordinates: {len(stations)}")
    return stations


def save_csv(stations, path):
    if not stations:
        print("No stations to save.")
        return
    fieldnames = list(stations[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stations)
    print(f"CSV saved: {path} ({len(stations)} rows)")


def save_geojson(stations, path):
    features = []
    for s in stations:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [s["longitude"], s["latitude"]],
            },
            "properties": {k: v for k, v in s.items() if k not in ("latitude", "longitude")},
        })
    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"GeoJSON saved: {path} ({len(features)} features)")


def main():
    stations = fetch_gas_stations()

    save_csv(stations, OUTPUT_DIR / "istanbul_gas_stations.csv")
    save_geojson(stations, OUTPUT_DIR / "istanbul_gas_stations.geojson")

    # Quick summary
    brands = {}
    for s in stations:
        b = s["brand"] or "(unknown)"
        brands[b] = brands.get(b, 0) + 1

    print("\n--- Brand distribution (top 15) ---")
    for brand, count in sorted(brands.items(), key=lambda x: -x[1])[:15]:
        print(f"  {brand}: {count}")


if __name__ == "__main__":
    main()
