from utils.geo_recommender import _decode_plus_code, _PLUS_CODE_RE

# Test the ABU Zaria Plus Code
test_addr = "5M23+GR3 Samaru Campus, Community Market, Ahmadu Bello University, Zaria 810211, Kaduna"

match = _PLUS_CODE_RE.match(test_addr)
print("Plus code detected:", bool(match))

result = _decode_plus_code(test_addr)
print("Decoded:", result)
