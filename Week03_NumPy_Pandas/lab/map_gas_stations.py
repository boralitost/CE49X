"""
Create an interactive Folium map of Istanbul gas stations from the CSV data.
"""

import csv
import folium
from folium.plugins import MarkerCluster
from pathlib import Path

LAB_DIR = Path(__file__).parent
CSV_PATH = LAB_DIR / "istanbul_gas_stations.csv"
OUTPUT_PATH = LAB_DIR / "istanbul_gas_stations_map.html"

# Brand -> marker color mapping
BRAND_COLORS = {
    "Opet": "#ff9f1c",
    "Shell": "#ff3a3a",
    "Petrol Ofisi": "#4cc9f0",
    "BP": "#80ed99",
    "TotalEnergies": "#c77dff",
    "Aytemiz": "#72efdd",
    "Türkiye Petrolleri": "#ff6b6b",
    "Alpet": "#48bfe3",
    "Kadoil": "#b5e48c",
    "Lukoil": "#f9c74f",
}
DEFAULT_COLOR = "#adb5bd"


def load_stations(path):
    stations = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stations.append({
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "name": row["name"] or "Unknown",
                "brand": row["brand"] or "Unknown",
                "operator": row["operator"],
                "addr_street": row["addr_street"],
                "addr_district": row["addr_district"],
            })
    return stations


def build_map(stations):
    # Center on Istanbul
    m = folium.Map(location=[41.015, 29.01], zoom_start=11, tiles="CartoDB dark_matter")

    cluster = MarkerCluster(name="Gas Stations").add_to(m)

    for s in stations:
        color = BRAND_COLORS.get(s["brand"], DEFAULT_COLOR)

        popup_lines = [f"<b>{s['name']}</b>"]
        if s["brand"] != "Unknown":
            popup_lines.append(f"Brand: {s['brand']}")
        if s["operator"]:
            popup_lines.append(f"Operator: {s['operator']}")
        if s["addr_street"]:
            popup_lines.append(f"Street: {s['addr_street']}")
        if s["addr_district"]:
            popup_lines.append(f"District: {s['addr_district']}")
        popup_lines.append(f"Coords: {s['lat']:.4f}, {s['lon']:.4f}")
        popup_html = (
            '<div style="background:#1a1a2e;color:#e0e0e0;padding:6px 8px;'
            'border-radius:4px;font-size:12px;">'
            + "<br>".join(popup_lines)
            + "</div>"
        )

        folium.CircleMarker(
            location=[s["lat"], s["lon"]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{s['name']} ({s['brand']})",
        ).add_to(cluster)

    # Legend
    brand_counts = {}
    for s in stations:
        b = s["brand"]
        brand_counts[b] = brand_counts.get(b, 0) + 1

    legend_items = []
    for brand, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
        color = BRAND_COLORS.get(brand, DEFAULT_COLOR)
        legend_items.append(
            f'<li><span style="background:{color};width:12px;height:12px;'
            f'display:inline-block;border-radius:50%;margin-right:6px;"></span>'
            f'{brand} ({count})</li>'
        )

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:#1a1a2e;color:#e0e0e0;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 12px rgba(0,0,0,0.5);font-size:13px;
                max-height:400px;overflow-y:auto;border:1px solid #333;">
        <b style="color:#fff;">Istanbul Gas Stations ({len(stations)})</b>
        <ul style="list-style:none;padding:4px 0;margin:0;">
            {''.join(legend_items)}
        </ul>
        <div style="font-size:11px;color:#888;margin-top:6px;">
            Source: OpenStreetMap / Overpass API
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    return m


def main():
    stations = load_stations(CSV_PATH)
    print(f"Loaded {len(stations)} stations")
    m = build_map(stations)
    m.save(str(OUTPUT_PATH))
    print(f"Map saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
