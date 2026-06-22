import requests
query = '[out:json][timeout:30];(node["healthcare"](around:20000,9.064,7.489);node["amenity"~"hospital|clinic|doctors|pharmacy"](around:20000,9.064,7.489);way["amenity"~"hospital|clinic|doctors|pharmacy"](around:20000,9.064,7.489););out center;'
h = {'User-Agent': 'MediMap_AI/1.0'}
res = requests.post('https://overpass-api.de/api/interpreter', data={'data': query}, headers=h)
if res.status_code == 200:
    print('Found:', len(res.json().get('elements', [])))
else:
    print('Error:', res.status_code)
