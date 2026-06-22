"""
Live geocoding test — runs all three engines against Nigerian addresses.
"""
from utils.geo_recommender import (
    geocode_address,
    _geocode_geoapify,
    _geocode_photon,
    _geocode_nominatim,
    _decode_plus_code,
    _is_plus_code,
)

tests = [
    "Abuja, Nigeria",
    "Ahmadu Bello University, Zaria, Kaduna, Nigeria",
    "National Hospital Abuja",
    "5M23+GR3 Samaru Campus, Ahmadu Bello University, Zaria 810211, Kaduna",
    "6 Ahmadu Bello Way, Victoria Island, Lagos",
    "Limi Childrens Hospital, Abuja",
]

print("=" * 70)
print(f"{'ADDRESS':<45}  {'ENGINE':>10}  {'LAT':>10}  {'LON':>11}")
print("=" * 70)

for addr in tests:
    is_pc = _is_plus_code(addr)
    if is_pc:
        r = _decode_plus_code(addr)
        engine = "PlusCode"
    else:
        r = _geocode_geoapify(addr)
        engine = "Geoapify" if r else ""
        if not r:
            r = _geocode_photon(addr)
            engine = "Photon" if r else ""
        if not r:
            r = _geocode_nominatim(addr, retries=1)
            engine = "Nominatim" if r else "FAILED"
    
    if r:
        print(f"{addr[:44]:<45}  {engine:>10}  {r[0]:>10.5f}  {r[1]:>11.5f}")
    else:
        print(f"{addr[:44]:<45}  {'FAILED':>10}  {'—':>10}  {'—':>11}")

print("=" * 70)
