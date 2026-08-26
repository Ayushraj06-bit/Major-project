"""Tile-grid positions for Indian states and union territories.

An approximate geographic arrangement, one cell per state. Layout data rather
than analysis, which is why it lives in the dashboard package and not beside the
canonical registry.

The repository carries no boundary geometry, and fetching a GeoJSON at render
time would make the dashboard depend on the network. To move to a true
choropleth: place boundaries at ``data/raw/geo/india_states.geojson``, add
geopandas, and swap the tile renderer. The tile grid stays useful either way as a
small-multiple layout.
"""

from __future__ import annotations

#: ISO 3166-2:IN codes, used as tile labels.
#:
#: Real codes rather than derived initials. Taking first letters collides —
#: Andhra Pradesh and Arunachal Pradesh both give "AP", Manipur and Maharashtra
#: both give "MA" — and two tiles carrying the same label is worse than a label
#: nobody recognises.
TILE_CODES: dict[str, str] = {
    "Andaman and Nicobar Islands": "AN",
    "Andhra Pradesh": "AP",
    "Arunachal Pradesh": "AR",
    "Assam": "AS",
    "Bihar": "BR",
    "Chandigarh": "CH",
    "Chhattisgarh": "CT",
    "Dadra and Nagar Haveli and Daman and Diu": "DH",
    "Delhi": "DL",
    "Goa": "GA",
    "Gujarat": "GJ",
    "Haryana": "HR",
    "Himachal Pradesh": "HP",
    "Jammu and Kashmir": "JK",
    "Jharkhand": "JH",
    "Karnataka": "KA",
    "Kerala": "KL",
    "Ladakh": "LA",
    "Lakshadweep": "LD",
    "Madhya Pradesh": "MP",
    "Maharashtra": "MH",
    "Manipur": "MN",
    "Meghalaya": "ML",
    "Mizoram": "MZ",
    "Nagaland": "NL",
    "Odisha": "OR",
    "Puducherry": "PY",
    "Punjab": "PB",
    "Rajasthan": "RJ",
    "Sikkim": "SK",
    "Tamil Nadu": "TN",
    "Telangana": "TG",
    "Tripura": "TR",
    "Uttar Pradesh": "UP",
    "Uttarakhand": "UK",
    "West Bengal": "WB",
}

#: (row, column) per canonical state name, roughly north-west to south-east.
TILE_POSITIONS: dict[str, tuple[int, int]] = {
    "Jammu and Kashmir": (0, 2),
    "Ladakh": (0, 3),
    "Himachal Pradesh": (1, 3),
    "Punjab": (1, 2),
    "Chandigarh": (1, 4),
    "Uttarakhand": (2, 4),
    "Haryana": (2, 3),
    "Delhi": (2, 2),
    "Rajasthan": (3, 1),
    "Uttar Pradesh": (3, 3),
    "Sikkim": (3, 6),
    "Arunachal Pradesh": (3, 8),
    "Gujarat": (4, 1),
    "Madhya Pradesh": (4, 2),
    "Bihar": (4, 4),
    "West Bengal": (4, 5),
    "Assam": (4, 7),
    "Nagaland": (4, 8),
    "Dadra and Nagar Haveli and Daman and Diu": (5, 0),
    "Maharashtra": (5, 1),
    "Chhattisgarh": (5, 3),
    "Jharkhand": (5, 4),
    "Meghalaya": (5, 6),
    "Manipur": (5, 8),
    "Goa": (6, 1),
    "Telangana": (6, 2),
    "Odisha": (6, 4),
    "Tripura": (6, 6),
    "Mizoram": (6, 7),
    "Karnataka": (7, 1),
    "Andhra Pradesh": (7, 3),
    "Lakshadweep": (8, 0),
    "Kerala": (8, 1),
    "Tamil Nadu": (8, 2),
    "Puducherry": (8, 3),
    "Andaman and Nicobar Islands": (8, 6),
}
