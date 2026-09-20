"""
download county-level respiratory-health prevalence
- 2025 release (primary):  https://data.cdc.gov/resource/swc5-untb.json
- 2024 release (fallback): https://data.cdc.gov/resource/fu4u-a9bh.json

- CASTHMA - Current asthma among adults
- COPD - Chronic obstructive pulmonary disease among adults

- uses 2025 as primary and fills states missing that year from 2024.

Output: data/raw/cdc_places_health_raw.csv
"""

import os
import requests
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)
RELEASES = {2025: "swc5-untb", 2024: "fu4u-a9bh"}

def fetch_release_detail(resource_id):
    def measure(mid, col):
        params = {"measureid": mid, "datavaluetypeid": "CrdPrv", "$select": "stateabbr,locationname,locationid,data_value,totalpopulation", "$limit": 6000,}
        df = pd.DataFrame(requests.get(f"https://data.cdc.gov/resource/{resource_id}.json", params=params, timeout=60).json())
        df["data_value"] = pd.to_numeric(df["data_value"], errors="coerce")
        return df.rename(columns={"data_value": col})

    asthma = measure("CASTHMA", "asthma_prev")
    copd = measure("COPD", "copd_prev")
    out = asthma[["stateabbr", "locationname", "locationid", "totalpopulation", "asthma_prev"]].merge(copd[["locationid", "copd_prev"]], on="locationid", how="outer")
    return out.rename(columns={"stateabbr": "state", "locationname": "county", "locationid": "county_fips", "totalpopulation": "county_population"})


def main():
    primary = fetch_release_detail(RELEASES[2025])
    primary["health_data_year"] = 2025
    fallback = fetch_release_detail(RELEASES[2024])
    fallback["health_data_year"] = 2024
    missing_states = set(fallback.state) - set(primary.state)
    fill = fallback[fallback.state.isin(missing_states)]
    combined = pd.concat([primary, fill], ignore_index=True)
    out = os.path.join(RAW_DIR, "cdc_places_health_raw.csv")
    combined.to_csv(out, index=False)
    print("saved:", os.path.abspath(out))
    print("counties:", len(combined), "| states:", combined.state.nunique())
    print("states filled from 2024:", sorted(missing_states))
    print("asthma_prev range:", combined.asthma_prev.min(), "-", combined.asthma_prev.max())
    print("copd_prev range:", combined.copd_prev.min(), "-", combined.copd_prev.max())


if __name__ == "__main__":
    main()
