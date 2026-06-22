import folium
import html
m = folium.Map([9.0, 7.0])
folium.Marker([9.0, 7.0], tooltip="Limi Children's Hospital", popup=folium.Popup("<b>Limi Children's Hospital</b>", max_width=300)).add_to(m)
out = m._repr_html_()
if "Children's" in out:
    print("Found exact unescaped quote!")
else:
    print("Escaped properly")
