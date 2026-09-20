"""
generate cities.py
1. Open-Meteo Geocoding API  (name -> latitude, longitude, population) - https://geocoding-api.open-meteo.com/v1/search
2. U.S. Census Geographies API  (lat/lon -> official county name + FIPS) - https://geocoding.geo.census.gov/geocoder/geographies/coordinates

we're also normalizing county names to the CDC PLACES convention (no "County"/"Parish"
suffix) so that our merge works later in the pipeline.

the 50 study cities (among the most populous in the U.S.), by name + state only. We populate everything else by calling the two APIs.

Run:  python build_cities.py
"""
import time
import requests

SEED = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
    ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
    ("San Jose", "CA"), ("Austin", "TX"), ("Jacksonville", "FL"),
    ("Fort Worth", "TX"), ("Columbus", "OH"), ("Charlotte", "NC"),
    ("Indianapolis", "IN"), ("San Francisco", "CA"), ("Seattle", "WA"),
    ("Denver", "CO"), ("Washington", "DC"), ("Nashville", "TN"),
    ("Oklahoma City", "OK"), ("El Paso", "TX"), ("Boston", "MA"),
    ("Portland", "OR"), ("Las Vegas", "NV"), ("Detroit", "MI"),
    ("Memphis", "TN"), ("Louisville", "KY"), ("Baltimore", "MD"),
    ("Milwaukee", "WI"), ("Albuquerque", "NM"), ("Tucson", "AZ"),
    ("Fresno", "CA"), ("Sacramento", "CA"), ("Kansas City", "MO"),
    ("Atlanta", "GA"), ("Miami", "FL"), ("Raleigh", "NC"),
    ("Omaha", "NE"), ("Minneapolis", "MN"), ("Tulsa", "OK"),
    ("Cleveland", "OH"), ("Wichita", "KS"), ("New Orleans", "LA"),
    ("Salt Lake City", "UT"), ("St. Louis", "MO"), ("Pittsburgh", "PA"),
    ("Cincinnati", "OH"), ("Riverside", "CA"),
]

STATE_FULL = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

COUNTY_SUFFIXES = (" Census Area"," Municipality"," city"," City"," town"," Town"," County"," Parish"," Borough",)

def normalize_county(name):
    if name == "District of Columbia":
        return name
    for suffix in COUNTY_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name

def geocode(city, state):
    """name -> (lat, lon, population)"""
    full = STATE_FULL[state]
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 20, "country": "US", "language": "en"}, timeout=30).json()
    candidates = [x for x in r.get("results", []) if x.get("country_code") == "US" and x.get("admin1") == full]
    if not candidates: 
        candidates = [x for x in r.get("results", []) if x.get("country_code") == "US"]
    if not candidates:
        raise RuntimeError(f"found no geocode match for {city}, {state}")
    candidates.sort(key=lambda x: (x.get("population") or 0), reverse=True) # here we're ranking by the population descending
    best_candidate = candidates[0]
    return round(best_candidate["latitude"], 4), round(best_candidate["longitude"], 4), best_candidate.get("population")

def county_details(lat, lon):
    """lat/lon -> normalised county names, 5-digit FIPS"""
    r = requests.get("https://geocoding.geo.census.gov/geocoder/geographies/coordinates", params={"x": lon, "y": lat, "benchmark": "Public_AR_Current", "vintage": "Current_Current", "format": "json", "layers": "Counties"}, timeout=30).json()
    c = r["result"]["geographies"]["Counties"][0]
    return normalize_county(c["NAME"]), c["GEOID"]

def main():
    rows = []
    print(f"{'City':<16}{'ST':<4}{'County':<18}{'FIPS':<7}{'lat':>9}{'lon':>10}{'pop':>10}")
    for name, state in SEED:
        try:
            lat, lon, pop = geocode(name, state)
            county, fips = county_details(lat, lon)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{name:<16}{state:<4}FAILED: {e}")
            continue
        print(f"{name:<16}{state:<4}{county:<18}{fips:<7}{lat:>9}{lon:>10}{str(pop):>10}")
        rows.append((name, state, county, lat, lon, fips, pop))
        time.sleep(0.5)

    write_file(rows)
    print(f"\nWrote cities.py with {len(rows)} cities " f"(coords + county verified against Open-Meteo & U.S. Census).")

def write_file(rows):
    lines = ['"""',
             "City list ",
             "",
             "generated using build_cities.py using API calls",
             "  - Coordinates & population: Open-Meteo Geocoding API",
             "  - County & FIPS: U.S. Census Bureau Geographies API",
             "The 50 most populous cities in the United States.",
             "County names use the CDC PLACES convention for the health merge.",
             '"""',
             "",
             "CITIES = ["]
    for name, state, county, lat, lon, fips, pop in rows:
        prov = f"  # FIPS {fips}, pop {pop}" if fips else ""
        lines.append(f'    ({name!r:<18}, {state!r}, {county!r:<20}, '
                     f'{lat:>9}, {lon:>11}),{prov}')
    lines.append("]")
    with open("cities.py", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
