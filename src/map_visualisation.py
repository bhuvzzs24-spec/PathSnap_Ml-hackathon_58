"""Folium map helper for real PathSnap trip geometry.

Pass actual GPS observations, road arcs, reference arc IDs (if available),
and predicted arc IDs from a real trip. No synthetic coordinates are created.
Coordinates are expected as [latitude, longitude].
"""
import folium
from folium import FeatureGroup, LayerControl, PolyLine, CircleMarker

def create_trip_map(gps_points, road_segments=None, reference_segments=None,
                    predicted_segments=None, zoom_start=13):
    layers = []
    fmap = folium.Map(
        location=gps_points[len(gps_points)//2] if gps_points else [20.5937, 78.9629],
        zoom_start=zoom_start,
        control_scale=True
    )
    if gps_points:
        gps = FeatureGroup(name="Original GPS observations", show=True)
        for i, point in enumerate(gps_points):
            CircleMarker(
                point, radius=3, color="#2474B5", fill=True,
                fill_color="#2474B5", fill_opacity=0.8,
                tooltip=f"GPS observation {i+1}"
            ).add_to(gps)
        gps.add_to(fmap)
        layers.append(gps_points)

    def add_lines(name, segments, color, weight, show=True):
        if not segments:
            return
        group = FeatureGroup(name=name, show=show)
        for i, segment in enumerate(segments):
            if segment:
                PolyLine(
                    segment, color=color, weight=weight, opacity=0.9,
                    tooltip=f"{name} {i+1}"
                ).add_to(group)
        group.add_to(fmap)

    add_lines("Road-network geometry", road_segments, "#9A9A9A", 2, False)
    add_lines("Reference route", reference_segments, "#2E8B57", 5, True)
    add_lines("Predicted route", predicted_segments, "#E87522", 5, True)

    coords = list(gps_points or [])
    for collection in (road_segments, reference_segments, predicted_segments):
        for segment in (collection or []):
            coords.extend(segment)
    if coords:
        fmap.fit_bounds([
            [min(p[0] for p in coords), min(p[1] for p in coords)],
            [max(p[0] for p in coords), max(p[1] for p in coords)]
        ])
    LayerControl(collapsed=False).add_to(fmap)
    return fmap
