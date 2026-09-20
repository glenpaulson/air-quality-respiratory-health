"""
Collect one year of daily air-quality and weather data for around 50 U.S. cities.

Primary API (no key required): Open-Meteo
  - Air Quality (historical, hourly):https://air-quality-api.open-meteo.com/v1/air-quality
  - Weather (historical archive, daily):https://archive-api.open-meteo.com/v1/archive

- hourly pollutants are averaged to daily; `us_aqi` is reduced to a daily maximum (`us_aqi_max`) and mean (`us_aqi_mean`);
- air + weather are merged on `date`.
- Outputs: `data/raw/air_quality_weather_raw.csv`: 18,300 rows, (50 cities x 366 days), 19 columns. 
"""

import os
import time
import datetime as dt
import pandas as pd
import requests

from cities import CITIES

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

END_DATE = dt.date(2026, 9, 15) 
START_DATE = END_DATE - dt.timedelta(days=365) 

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
POLLUTANTS = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "us_aqi",]

def fetch_air_quality(lat, lon):
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(POLLUTANTS), "start_date": START_DATE.isoformat(), "end_date": END_DATE.isoformat(), "timezone": "auto",}
    r = requests.get(AIR_QUALITY_URL, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    agg = {p: "mean" for p in POLLUTANTS if p != "us_aqi"}
    # the air quality index max for the day is generalyy reported for that whole day
    agg["us_aqi"] = "max"
    daily = df.groupby("date").agg(agg).reset_index()
    daily = daily.rename(columns={"us_aqi": "us_aqi_max"})
    daily["us_aqi_mean"] = df.groupby("date")["us_aqi"].mean().values
    return daily

def fetch_weather(lat, lon):
    daily_vars = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max",]
    params = {"latitude": lat, "longitude": lon, "daily": ",".join(daily_vars), "start_date": START_DATE.isoformat(), "end_date": END_DATE.isoformat(), "timezone": "auto",}
    r = requests.get(WEATHER_URL, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(d).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def main():
    frames = []
    failures = []
    for i, (name, state, county, lat, lon) in enumerate(CITIES, 1):
        tag = f"{name}, {state}"
        try:
            air_qua = fetch_air_quality(lat, lon)
            weat = fetch_weather(lat, lon)
            merged = air_qua.merge(weat, on="date", how="inner")
            merged.insert(0, "city", name)
            merged.insert(1, "state", state)
            merged.insert(2, "county", county)
            merged.insert(3, "latitude", lat)
            merged.insert(4, "longitude", lon)
            frames.append(merged)
            print(f"[{i:2d}/{len(CITIES)}] {tag:<24} rows={len(merged)}")
        except Exception as e: # noqa: BLE001 
            failures.append((tag, str(e)))
            print(f"[{i:2d}/{len(CITIES)}] {tag:<24} FAILED: {e}")
        time.sleep(1.0) 

    if not frames:
        raise SystemExit("no data was collected")

    full = pd.concat(frames, ignore_index=True)
    out = os.path.join(RAW_DIR, "air_quality_weather_raw.csv")
    full.to_csv(out, index=False)
    print("\nsaved:", os.path.abspath(out))
    print("total rows:", len(full), "| cities:", full["city"].nunique())
    print("columns:", list(full.columns))
    if failures:
        print("\nFailures:")
        for t, e in failures:
            print("  ", t, "->", e)


if __name__ == "__main__":
    main()
