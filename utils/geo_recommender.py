"""
MediMap AI — GIS Geospatial Recommender
========================================
Maps a predicted disease category to the appropriate medical specialty
and performs a real-time proximity search for nearby specialist clinics
using the Overpass API (OpenStreetMap data) and Geopy for geocoding.

Workflow
--------
1. ``disease_to_specialty(disease_name)``
        → returns the relevant medical specialty string.

2. ``geocode_address(address_string)``
        → returns (latitude, longitude) via Nominatim.

3. ``search_specialists(lat, lon, specialty, radius_m)``
        → queries Overpass API and returns a sorted list of clinics.

4. ``build_folium_map(lat, lon, clinics)``
        → constructs an interactive Folium map centred on the user.

Author : MediMap AI Engineering Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import folium
import requests
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

logger = logging.getLogger(__name__)

# ── Configuration from environment ───────────────────────────────────────────
OVERPASS_URL: str = os.getenv(
    "OVERPASS_API_URL", "https://overpass-api.de/api/interpreter"
)
NOMINATIM_USER_AGENT: str = os.getenv("NOMINATIM_USER_AGENT", "medimap_ai_v1")
DEFAULT_RADIUS_M: int = int(os.getenv("GEO_SEARCH_RADIUS_M", "10000"))
MAX_RESULTS: int = int(os.getenv("GEO_MAX_RESULTS", "10"))

# ── Overpass mirror pool (tried in order on failure) ─────────────────────────
_OVERPASS_MIRRORS: list[str] = [
    os.getenv("OVERPASS_API_URL", "https://overpass-api.de/api/interpreter"),
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
_HEADERS = {"User-Agent": NOMINATIM_USER_AGENT}


# =============================================================================
# 1.  DISEASE → SPECIALTY MAPPING
# =============================================================================

#: Maps disease categories (as returned by the model) to medical specialties.
DISEASE_SPECIALTY_MAP: dict[str, str] = {
    # Respiratory
    "Pneumonia": "Pulmonologist",
    "Tuberculosis": "Pulmonologist",
    "Bronchial Asthma": "Pulmonologist",
    "COPD": "Pulmonologist",
    # Cardiovascular
    "Heart attack": "Cardiologist",
    "Hypertension": "Cardiologist",
    "Arrhythmia": "Cardiologist",
    # Gastrointestinal
    "GERD": "Gastroenterologist",
    "Peptic ulcer diseae": "Gastroenterologist",
    "Peptic ulcer disease": "Gastroenterologist",
    "Chronic cholestasis": "Gastroenterologist",
    "Alcoholic hepatitis": "Gastroenterologist",
    "Hepatitis A": "Gastroenterologist",
    "Hepatitis B": "Gastroenterologist",
    "Hepatitis C": "Gastroenterologist",
    "Hepatitis D": "Gastroenterologist",
    "Hepatitis E": "Gastroenterologist",
    "Jaundice": "Gastroenterologist",
    "Dimorphic hemmorhoids(piles)": "Gastroenterologist",
    # Neurological
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Neurologist",
    "Paralysis (brain hemorrhage)": "Neurologist",
    # Dermatological
    "Fungal infection": "Dermatologist",
    "Psoriasis": "Dermatologist",
    "Acne": "Dermatologist",
    "Impetigo": "Dermatologist",
    "Chickenpox": "Dermatologist",
    # Musculoskeletal
    "Arthritis": "Rheumatologist",
    "Osteoarthritis": "Rheumatologist",
    # Endocrine
    "Diabetes": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    # Urological
    "Urinary tract infection": "Urologist",
    "Kidney stones": "Urologist",
    "Chronic kidney disease": "Nephrologist",
    # General / Infectious
    "Common Cold": "General Practitioner",
    "Dengue": "General Practitioner",
    "Malaria": "General Practitioner",
    "Typhoid": "General Practitioner",
    "AIDS": "Infectious Disease Specialist",
    "Allergy": "Allergist",
    "Drug Reaction": "Allergist",
}

#: Overpass amenity tags aligned to each specialty keyword.
SPECIALTY_OSM_TAGS: dict[str, list[str]] = {
    "Pulmonologist": ["hospital", "clinic", "doctors", "healthcare"],
    "Cardiologist": ["hospital", "clinic", "doctors", "healthcare"],
    "Gastroenterologist": ["hospital", "clinic", "doctors", "healthcare"],
    "Neurologist": ["hospital", "clinic", "doctors", "healthcare"],
    "Dermatologist": ["clinic", "doctors", "healthcare"],
    "Rheumatologist": ["hospital", "clinic", "doctors", "healthcare"],
    "Endocrinologist": ["clinic", "hospital", "doctors", "healthcare"],
    "Urologist": ["hospital", "clinic", "doctors", "healthcare"],
    "Nephrologist": ["hospital", "clinic", "doctors", "healthcare"],
    "General Practitioner": ["clinic", "doctors", "hospital", "healthcare", "pharmacy"],
    "Infectious Disease Specialist": ["hospital", "clinic", "healthcare"],
    "Allergist": ["clinic", "doctors", "hospital", "healthcare"],
}


def disease_to_specialty(disease_name: str) -> str:
    """
    Map a predicted disease label to the corresponding medical specialty.

    Falls back to ``"General Practitioner"`` for unmapped diseases.

    Parameters
    ----------
    disease_name : str
        Disease label as returned by the fusion model.

    Returns
    -------
    str
        Medical specialty name.
    """
    specialty = DISEASE_SPECIALTY_MAP.get(disease_name.strip())
    if specialty is None:
        logger.warning(
            "Disease '%s' not in mapping; defaulting to General Practitioner.",
            disease_name,
        )
        return "General Practitioner"
    return specialty


# =============================================================================
# 2.  GEOCODING  (multi-engine: Plus Code → Geoapify → Photon → Nominatim)
# =============================================================================

# ── Geoapify (free tier: 3,000 req/day, no credit card) ────────────────────
# Sign up free at https://myprojects.geoapify.com/
# Paste your key between the quotes below:
import os
GEOAPIFY_API_KEY: str = os.environ.get("GEOAPIFY_API_KEY", "7834eafe994240c4a16857ad91e357b9")

# ── Plus Code detection ─────────────────────────────────────────────────────
# A Plus Code ALWAYS has the form: 4-8 chars + '+' + 2-3 chars
# MUST be the very first token (before any space) and cannot look like
# a normal word/number. We require at least 4 chars before the '+'.
import re
_PLUS_CODE_RE = re.compile(
    r"^([23456789CFGHJMPQRVWX]{4,8}\+[23456789CFGHJMPQRVWX]{2,3})(\s|$)",
    re.IGNORECASE,
)

import re as _re


def _is_plus_code(text: str) -> bool:
    """Return True only if the text starts with a valid-looking Plus Code."""
    return bool(_PLUS_CODE_RE.match(text.strip()))


def _decode_plus_code(code: str) -> Optional[tuple[float, float]]:
    """
    Decode a Google Plus Code to (lat, lon).
    Handles full codes (offline) and short codes (needs city reference).
    """
    try:
        import openlocationcode.openlocationcode as olc

        parts = code.strip().split()
        pure_code = parts[0]

        if olc.isValid(pure_code) and olc.isFull(pure_code):
            coord = olc.decode(pure_code)
            lat = (coord.latitudeLo + coord.latitudeHi) / 2
            lon = (coord.longitudeLo + coord.longitudeHi) / 2
            logger.info("Plus Code (full) '%s' → (%.6f, %.6f)", pure_code, lat, lon)
            return lat, lon

        if olc.isValid(pure_code) and not olc.isFull(pure_code):
            # Short code needs a reference city from the trailing text
            place_hint = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
            if not place_hint:
                logger.warning("Short Plus Code '%s' needs a city name after it.", pure_code)
                return None

            ref = _geocode_photon(place_hint) or _geocode_nominatim(place_hint, retries=2)
            if ref is None:
                logger.warning("Could not geocode reference '%s' for Plus Code.", place_hint)
                return None

            ref_lat, ref_lon = ref
            full_code = olc.recoverNearest(pure_code, ref_lat, ref_lon)
            coord = olc.decode(full_code)
            lat = (coord.latitudeLo + coord.latitudeHi) / 2
            lon = (coord.longitudeLo + coord.longitudeHi) / 2
            logger.info(
                "Plus Code (short) '%s' + ref → (%.6f, %.6f)", pure_code, lat, lon
            )
            return lat, lon

        logger.warning("'%s' is not a valid Plus Code.", pure_code)
        return None
    except ImportError:
        logger.warning("openlocationcode not installed.")
        return None
    except Exception as exc:
        logger.warning("Plus Code decode error: %s", exc)
        return None


def _geocode_geoapify(address: str) -> Optional[tuple[float, float]]:
    """
    Geocode via Geoapify — free tier 3,000 req/day, excellent Africa coverage.
    Skipped if no API key is set.
    """
    if not GEOAPIFY_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.geoapify.com/v1/geocode/search",
            params={"text": address, "format": "json", "limit": 1,
                    "apiKey": GEOAPIFY_API_KEY},
            timeout=8,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            lat, lon = results[0]["lat"], results[0]["lon"]
            logger.info("Geoapify: '%s' → (%.6f, %.6f)", address, lat, lon)
            return lat, lon
        logger.warning("Geoapify: no results for '%s'.", address)
        return None
    except Exception as exc:
        logger.warning("Geoapify failed: %s", exc)
        return None


def _geocode_photon(address: str) -> Optional[tuple[float, float]]:
    """
    Geocode via Photon (by Komoot) — completely FREE, no API key, no signup.
    - Built on OpenStreetMap data with typo tolerance
    - Good Africa/Nigeria coverage
    - Public API: https://photon.komoot.io
    """
    try:
        r = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": address, "limit": 1, "lang": "en"},
            headers={"User-Agent": "MediMap_AI/1.0"},
            timeout=8,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
        if features:
            coords = features[0]["geometry"]["coordinates"]  # [lon, lat]
            lon, lat = coords[0], coords[1]
            logger.info("Photon: '%s' → (%.6f, %.6f)", address, lat, lon)
            return lat, lon
        logger.warning("Photon: no results for '%s'.", address)
        return None
    except Exception as exc:
        logger.warning("Photon failed: %s", exc)
        return None


def _geocode_nominatim(address: str, retries: int = 3) -> Optional[tuple[float, float]]:
    """Final fallback: Nominatim (OpenStreetMap)."""
    geocoder = Nominatim(user_agent=NOMINATIM_USER_AGENT, timeout=10)
    for attempt in range(1, retries + 1):
        try:
            location = geocoder.geocode(address)
            if location:
                logger.info(
                    "Nominatim: '%s' → (%.6f, %.6f)",
                    address, location.latitude, location.longitude,
                )
                return location.latitude, location.longitude
            logger.warning("Nominatim: no results for '%s'.", address)
            return None
        except GeocoderTimedOut:
            logger.warning("Nominatim timed out (attempt %d/%d).", attempt, retries)
            time.sleep(1.5 * attempt)
        except GeocoderServiceError as exc:
            logger.error("Nominatim service error: %s", exc)
            return None
    return None


def geocode_address(address: str, retries: int = 3) -> Optional[tuple[float, float]]:
    """
    Convert an address, Plus Code, city, or landmark to (latitude, longitude).

    Resolution chain (first success wins):
    1. Google Plus Code  — offline decode, perfectly precise
    2. Geoapify          — free API key, best Nigeria/Africa accuracy
    3. Photon (Komoot)   — keyless, no signup, typo-tolerant, good Africa data
    4. Nominatim (OSM)   — final fallback

    Parameters
    ----------
    address : str
        Any of: city name, street address, landmark, or Google Plus Code.
    """
    address = address.strip()
    if not address:
        return None

    # 1. Plus Code (must genuinely look like one, not just any address)
    if _is_plus_code(address):
        result = _decode_plus_code(address)
        if result:
            return result
        # If Plus Code decode failed, fall through to text geocoders
        # (treat remaining text after the code as a plain address)
        plain_addr = " ".join(address.split()[1:]).strip()
        address = plain_addr if plain_addr else address

    # 2. Geoapify (best accuracy when key is available)
    result = _geocode_geoapify(address)
    if result:
        return result

    # 3. Photon — keyless fallback with typo tolerance
    result = _geocode_photon(address)
    if result:
        return result

    # 4. Nominatim — last resort
    return _geocode_nominatim(address, retries=retries)

# =============================================================================
# 3.  CLINIC DATA STRUCTURE
# =============================================================================

@dataclass(order=True)
class Clinic:
    """
    Represents a single specialist clinic returned by the Overpass query.

    Attributes
    ----------
    distance_km : float
        Crow-fly distance from the user's location (used for sorting).
    name : str
        Clinic / hospital name from OSM tags.
    lat : float
        Latitude of the clinic.
    lon : float
        Longitude of the clinic.
    address : str
        Reconstructed postal address from OSM tags.
    specialty : str
        Medical specialty associated with this result.
    phone : str
        Phone number if available in OSM tags.
    website : str
        Website URL if available in OSM tags.
    osm_id : int
        OpenStreetMap node ID.
    """

    distance_km: float = field(compare=True)
    name: str = field(compare=False)
    lat: float = field(compare=False)
    lon: float = field(compare=False)
    address: str = field(compare=False, default="")
    specialty: str = field(compare=False, default="")
    phone: str = field(compare=False, default="")
    website: str = field(compare=False, default="")
    osm_id: int = field(compare=False, default=0)

    def to_dict(self) -> dict:
        """Serialise to plain dictionary for Streamlit display."""
        return {
            "Name": self.name,
            "Specialty": self.specialty,
            "Distance (km)": round(self.distance_km, 2),
            "Address": self.address,
            "Phone": self.phone,
            "Website": self.website,
        }


# =============================================================================
# 4.  OVERPASS API QUERY
# =============================================================================

def _build_overpass_query(
    lat: float,
    lon: float,
    radius_m: int,
    amenity_tags: list[str],
) -> str:
    """
    Build an Overpass QL query string that searches for health-related
    amenities within a radius.

    Parameters
    ----------
    lat, lon : float
        Centre of the search circle.
    radius_m : int
        Search radius in metres.
    amenity_tags : list[str]
        OSM amenity values to include (e.g. ``["hospital", "clinic"]``).

    Returns
    -------
    str
        Overpass QL query string.
    """
    union_parts = "\n  ".join(
        f'node["amenity"="{tag}"](around:{radius_m},{lat},{lon});\n'
        f'  way["amenity"="{tag}"](around:{radius_m},{lat},{lon});\n'
        f'  node["healthcare"="{tag}"](around:{radius_m},{lat},{lon});\n'
        f'  way["healthcare"="{tag}"](around:{radius_m},{lat},{lon});'
        for tag in amenity_tags
    )
    return f"""
[out:json][timeout:30];
(
  {union_parts}
  node["healthcare"](around:{radius_m},{lat},{lon});
  way["healthcare"](around:{radius_m},{lat},{lon});
);
out center tags;
""".strip()


def search_specialists(
    lat: float,
    lon: float,
    specialty: str,
    radius_m: int = DEFAULT_RADIUS_M,
    max_results: int = MAX_RESULTS,
) -> list[Clinic]:
    """
    Query the Overpass API for specialist clinics near a given location.

    Parameters
    ----------
    lat : float
        User's latitude.
    lon : float
        User's longitude.
    specialty : str
        Medical specialty string (from ``disease_to_specialty``).
    radius_m : int
        Search radius in metres.
    max_results : int
        Maximum number of clinics to return.

    Returns
    -------
    list[Clinic]
        Sorted list of nearest clinics (ascending distance).

    Raises
    ------
    RuntimeError
        If the Overpass API request fails.
    """
    amenity_tags = SPECIALTY_OSM_TAGS.get(specialty, ["hospital", "clinic"])
    query = _build_overpass_query(lat, lon, radius_m, amenity_tags)

    logger.info(
        "Querying Overpass API — specialty='%s', radius=%dm, centre=(%.4f,%.4f)",
        specialty, radius_m, lat, lon,
    )

    last_exc: Exception = RuntimeError("No Overpass mirrors available.")
    for mirror_url in _OVERPASS_MIRRORS:
        try:
            logger.info("Trying Overpass mirror: %s", mirror_url)
            response = requests.post(
                mirror_url,
                data={"data": query},
                headers=_HEADERS,
                timeout=40,
            )
            response.raise_for_status()
            break  # success — exit retry loop
        except requests.RequestException as exc:
            logger.warning("Mirror %s failed: %s", mirror_url, exc)
            last_exc = exc
    else:
        raise RuntimeError(
            f"All Overpass mirrors failed. Last error: {last_exc}"
        ) from last_exc

    data = response.json()
    elements = data.get("elements", [])
    logger.info("Overpass returned %d raw elements.", len(elements))

    clinics: list[Clinic] = []
    user_coords = (lat, lon)

    for elem in elements:
        tags = elem.get("tags", {})
        name = tags.get("name", "Unnamed Clinic")
        if not name or name == "Unnamed Clinic":
            continue  # Skip nameless nodes

        # Resolve lat/lon for ways (use centre)
        node_lat = elem.get("lat") or elem.get("center", {}).get("lat")
        node_lon = elem.get("lon") or elem.get("center", {}).get("lon")
        if node_lat is None or node_lon is None:
            continue

        # Build address string from OSM tags
        addr_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
            tags.get("addr:postcode", ""),
        ]
        address = ", ".join(p for p in addr_parts if p)

        distance_km = geodesic(user_coords, (node_lat, node_lon)).km

        clinics.append(
            Clinic(
                distance_km=distance_km,
                name=name,
                lat=node_lat,
                lon=node_lon,
                address=address or tags.get("addr:full", "Address not available"),
                specialty=specialty,
                phone=tags.get("phone", tags.get("contact:phone", "")),
                website=tags.get("website", tags.get("contact:website", "")),
                osm_id=elem.get("id", 0),
            )
        )

    clinics.sort()  # Sort by distance_km (dataclass field ordering)
    result = clinics[:max_results]
    
    # ── Fallback Logic: If no specialists found, try General Hospitals ──
    if not result and specialty != "General Practitioner":
        logger.info(
            "No '%s' clinics found within %dm. Falling back to General Practitioner...",
            specialty, radius_m
        )
        return search_specialists(
            lat=lat,
            lon=lon,
            specialty="General Practitioner",
            radius_m=radius_m,
            max_results=max_results,
        )

    logger.info("Returning %d clinics after filtering.", len(result))
    return result


# =============================================================================
# 5.  FOLIUM MAP BUILDER
# =============================================================================

def build_folium_map(
    user_lat: float,
    user_lon: float,
    clinics: list[Clinic],
    specialty: str = "",
    disease: str = "",
    zoom_start: int = 13,
    map_tiles: str = "CartoDB Positron",
) -> folium.Map:
    """
    Build an interactive Folium map centred on the user's location with
    markers for each recommended clinic.

    Parameters
    ----------
    user_lat, user_lon : float
        User's geospatial coordinates.
    clinics : list[Clinic]
        List of specialist clinics to plot.
    specialty : str
        Medical specialty label (displayed in the map title popup).
    disease : str
        Predicted disease label (used in popups).
    zoom_start : int
        Initial zoom level for the map.

    Returns
    -------
    folium.Map
        Fully configured Folium map object.
    """
    fmap = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=zoom_start,
        tiles=map_tiles,
        control_scale=True,
    )

    # ── User marker ──────────────────────────────────────────────────────────
    folium.Marker(
        location=[user_lat, user_lon],
        tooltip="<b>Your Location</b>",
        popup=folium.Popup(
            f"<b>Your Location</b><br>Predicted disease: {disease}<br>"
            f"Seeking: {specialty}",
            max_width=250,
        ),
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(fmap)

    # ── Search radius circle ──────────────────────────────────────────────────
    folium.Circle(
        location=[user_lat, user_lon],
        radius=DEFAULT_RADIUS_M,
        color="#3b7dd8",
        fill=True,
        fill_opacity=0.07,
        tooltip=f"Search radius: {DEFAULT_RADIUS_M / 1000:.1f} km",
    ).add_to(fmap)

    # ── Clinic markers ────────────────────────────────────────────────────────
    colour_cycle = [
        "#1a73e8", "#0f9d58", "#f4b400", "#db4437",
        "#673ab7", "#00897b", "#e91e63", "#ff6d00",
    ]
    
    import html
    
    def _sanitize(text: str) -> str:
        """Sanitize strings for Folium Javascript template injection."""
        if not text:
            return ""
        # Folium wraps HTML in backticks (`...`) in JS. 
        # A stray backtick in the name will break the JS string!
        clean_text = text.replace("`", "'")
        return html.escape(clean_text)

    for i, clinic in enumerate(clinics):
        colour = colour_cycle[i % len(colour_cycle)]
        
        safe_name = _sanitize(clinic.name)
        safe_spec = _sanitize(clinic.specialty)
        safe_addr = _sanitize(clinic.address)
        safe_phone = _sanitize(clinic.phone)
        safe_web = _sanitize(clinic.website)
        
        popup_html = (
            f"<div style='font-family:sans-serif;min-width:180px'>"
            f"<b>{safe_name}</b><br>"
            f"<span style='color:#555'>{safe_spec}</span><br>"
            f"📍 {safe_addr}<br>"
            f"📏 {clinic.distance_km:.2f} km away"
            f"{'<br>📞 ' + safe_phone if safe_phone else ''}"
            f"{'<br>🌐 <a href=' + safe_web + ' target=_blank>Website</a>' if safe_web else ''}"
            f"</div>"
        )
        
        folium.Marker(
            location=[clinic.lat, clinic.lon],
            tooltip=f"{i + 1}. {safe_name} ({clinic.distance_km:.2f} km)",
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.DivIcon(
                html=(
                    f'<div style="background:{colour};color:#fff;border-radius:50%;'
                    f'width:28px;height:28px;display:flex;align-items:center;'
                    f'justify-content:center;font-weight:bold;font-size:13px;'
                    f'box-shadow:0 2px 6px rgba(0,0,0,.35)">'
                    f"{i + 1}</div>"
                ),
                icon_size=(28, 28),
                icon_anchor=(14, 14),
            ),
        ).add_to(fmap)

    # ── Fit map bounds to all markers ─────────────────────────────────────────
    if clinics:
        all_lats = [user_lat] + [c.lat for c in clinics]
        all_lons = [user_lon] + [c.lon for c in clinics]
        fmap.fit_bounds(
            [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]],
            padding=(30, 30),
        )

    return fmap


# =============================================================================
# 6.  HIGH-LEVEL PIPELINE ENTRY-POINT
# =============================================================================

def recommend_specialists(
    disease: str,
    user_address: str = "",
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
    radius_km: float = DEFAULT_RADIUS_M / 1000,
    max_results: int = MAX_RESULTS,
    map_tiles: str = "CartoDB Positron",
) -> dict:
    """
    Full GIS recommendation pipeline.

    Accepts either a free-text address or raw coordinates.  Returns a
    dict with a sorted clinic list, ready-to-render Folium map, and the
    resolved medical specialty.

    Parameters
    ----------
    disease : str
        Disease label from the fusion model.
    user_address : str
        Human-readable address (used if lat/lon not provided).
    user_lat, user_lon : float, optional
        Raw coordinates (take precedence over address).
    radius_km : float
        Overpass search radius in kilometres.
    max_results : int
        Maximum number of clinics to return.
    map_tiles : str
        Folium tile layer name (e.g. "CartoDB Positron").

    Returns
    -------
    dict
        Keys: ``clinics`` (list[Clinic]), ``map`` (folium.Map),
        ``specialty`` (str).

    Raises
    ------
    ValueError
        If neither coordinates nor a valid address is provided.
    """
    specialty = disease_to_specialty(disease)
    radius_m  = int(radius_km * 1000)

    # Resolve coordinates
    if user_lat is None or user_lon is None:
        if not user_address:
            raise ValueError(
                "Either (user_lat, user_lon) or user_address must be provided."
            )
        coords = geocode_address(user_address)
        if coords is None:
            raise ValueError(
                f"Could not geocode address: '{user_address}'. "
                "Try a more specific location string."
            )
        user_lat, user_lon = coords

    clinics = search_specialists(user_lat, user_lon, specialty, radius_m, max_results)
    fmap    = build_folium_map(
        user_lat, user_lon, clinics,
        specialty=specialty,
        disease=disease,
        map_tiles=map_tiles,
    )
    return {"clinics": clinics, "map": fmap, "specialty": specialty}



# =============================================================================
# QUICK CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    disease = sys.argv[1] if len(sys.argv) > 1 else "Pneumonia"
    address = sys.argv[2] if len(sys.argv) > 2 else "London, UK"

    print(f"Disease   : {disease}")
    print(f"Specialty : {disease_to_specialty(disease)}")
    print(f"Geocoding : {address}")

    clinics, fmap, spec = recommend_specialists(disease, user_address=address)
    print(f"\nFound {len(clinics)} {spec} clinics:")
    for idx, c in enumerate(clinics, 1):
        print(f"  {idx}. {c.name:40s} | {c.distance_km:.2f} km | {c.address}")

    out = "data/processed/test_map.html"
    fmap.save(out)
    print(f"\nMap saved → {out}")
