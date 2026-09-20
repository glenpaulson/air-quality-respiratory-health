"""
clean & merge the raw air-quality/weather data with CDC health data, and
derive the labels and features.

inputs:
    air_quality_weather_raw.csv - daily pollutants + weather, 50 cities
    cdc_places_health_raw.csv - county asthma/COPD prevalence
output:: air_quality_health_clean.csv - one analysis-ready table

cleaning / prep steps:
1. type fixes and date parsing
2. duplicate + missing-value checks
3. range / outlier sanity checks
4. independent-city county resolution (Baltimore, St. Louis)
5. merge health data by (state, county)
6. feature engineering: month, season
7. label derivation: EPA AQI category (6-class) + simplified 3-class
8. discretization example: PM2.5 tier
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
CLEAN = os.path.join(HERE, "..", "data", "clean")
os.makedirs(CLEAN, exist_ok=True)

# EPA AQI category breakpoints (upper bound -> label)
AQI_BINS = [0, 50, 100, 150, 200, 300, 1000]
AQI_LABELS = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
INDEP_CITY_FIPS = {("MD", "Baltimore"): 24510, ("MO", "St. Louis"): 29510}
COUNTY_RENAME = {"Baltimore City": "Baltimore", "St. Louis City": "St. Louis"}

def log(step, msg):
    print(f"[{step}] {msg}")

def main():
    air_qual = pd.read_csv(os.path.join(RAW, "air_quality_weather_raw.csv"))
    health = pd.read_csv(os.path.join(RAW, "cdc_places_health_raw.csv"))
    log("load", f"air/weather rows={len(air_qual)}, health counties={len(health)}")

    # 1. types
    air_qual["date"] = pd.to_datetime(air_qual["date"])
    log("types", f"date dtype -> {air_qual['date'].dtype}")

    # 2. duplicates + missing
    dups = air_qual.duplicated(subset=["city", "date"]).sum()
    air_qual = air_qual.drop_duplicates(subset=["city", "date"])
    na_before = int(air_qual.isna().sum().sum())
    log("duplicates", f"removed {dups} duplicate city-date rows")
    log("missing", f"missing cells in air/weather: {na_before}")

    # 3. range sanity checks
    checks = {
        "pm2_5": (0, 1000), "pm10": (0, 2000), "ozone": (0, 500),
        "us_aqi_max": (0, 700), "temperature_2m_mean": (-60, 60),
    }
    for col, (lo, hi) in checks.items():
        bad = int(((air_qual[col] < lo) | (air_qual[col] > hi)).sum())
        log("range", f"{col}: {bad} values outside [{lo},{hi}]")

    # 4. resolve independent cities in health data
    def keep_row(r):
        key = (r["state"], r["county"])
        if key in INDEP_CITY_FIPS:
            return int(r["county_fips"]) == INDEP_CITY_FIPS[key]
        return True
    health = health[health.apply(keep_row, axis=1)].copy()
    air_qual["county"] = air_qual["county"].replace(COUNTY_RENAME)
    log("indep-city", "resolved Baltimore & St. Louis to city FIPS")

    # 5. merge health by state, county
    hcols = ["state", "county", "asthma_prev", "copd_prev", "county_population", "health_data_year"]
    merged = air_qual.merge(health[hcols], on=["state", "county"], how="left")
    no_health = sorted(merged[merged.asthma_prev.isna()].city.unique())
    log("merge", f"cities without health match: {no_health or 'none'}")

    # 6. feature engineering
    merged["month"] = merged["date"].dt.month
    season = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
    merged["season"] = merged["month"].map(season)

    # 7. labels
    merged["aqi_category"] = pd.cut(merged["us_aqi_max"], bins=AQI_BINS, labels=AQI_LABELS, right=True, include_lowest=True)
    merged["aqi_class3"] = np.where(merged.us_aqi_max <= 50, "Good", np.where(merged.us_aqi_max <= 100, "Moderate", "Unhealthy"))

    # 8. discretization: PM2.5 tier (WHO-inspired)
    merged["pm25_tier"] = pd.cut(
        merged["pm2_5"], bins=[0, 12, 35.4, 55.4, 1000],
        labels=["Low", "Moderate", "High", "Very High"], include_lowest=True)

    out = os.path.join(CLEAN, "air_quality_health_clean.csv")
    merged.to_csv(out, index=False)
    log("save", os.path.abspath(out))
    print("\nfinal shape:", merged.shape)
    print("columns:", list(merged.columns))
    print("\naqi_category counts:\n", merged.aqi_category.value_counts())
    print("\naqi_class3 counts:\n", merged.aqi_class3.value_counts())
    print("\nremaining missing cells:", int(merged.isna().sum().sum()))
    print(merged.isna().sum()[merged.isna().sum() > 0].to_string() or "  none")

if __name__ == "__main__":
    main()
