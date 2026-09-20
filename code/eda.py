"""
EDA - generate 12 figures. 
each figure has a title, labeled axes, and a colour scheme;
a 2-sentence caption for each is written to images/eda/captions.md.
output: images/eda/*.png  +  images/eda/captions.md
"""
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "data", "clean", "air_quality_health_clean.csv")
IMG = os.path.join(HERE, "..", "images", "eda")
os.makedirs(IMG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,})
PRIMARY = "#0e7c7b"       # teal - primary single-series colour
CORAL = "#ef6f53"         # coral - accent / trend lines
SEQ = "mako"              # sequential gradient (maps, rankings, heatmaps)
DIVERGING = "coolwarm"    # correlation heatmap
# AQI category colours kept as the official EPA standard (domain accuracy)
AQI_COLORS = {"Good": "#3bb143", "Moderate": "#efc804", "Unhealthy for Sensitive Groups": "#ff7e00", "Unhealthy": "#d62728", "Very Unhealthy": "#8f3f97", "Hazardous": "#7e0023",}
CLASS3_COLORS = {"Good": "#3bb143", "Moderate": "#efc804", "Unhealthy": "#e82a2a"}

df = pd.read_csv(CLEAN, parse_dates=["date"])
captions = {}

def save(fig, fname, caption):
    fig.tight_layout()
    path = os.path.join(IMG, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    captions[fname] = caption
    print("saved", fname)

# 1. AQI value distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df.us_aqi_max, bins=50, color=PRIMARY, edgecolor="white")
for b, lbl, c in [(50, "Good", "#3bb143"), (100, "Moderate", "#efc804"), (150, "USG", "#ff7e00")]:
    ax.axvline(b, color=c, ls="--", lw=1.5)
ax.set(title="Distribution of Daily Peak Air Quality Index (AQI)",
       xlabel="Daily maximum US AQI", ylabel="Number of city-days")
save(fig, "01_aqi_distribution.png",
     "Most city-days fall in the Good-to-Moderate range, but a long right tail "
     "is reaching into the Unhealthy territory. Those high-AQI days are the rare, "
     "high-impact events this project most wants to understand and predict.")

# 2. AQI category balance 
fig, ax = plt.subplots(figsize=(9, 5))
vc = df.aqi_category.value_counts().reindex(AQI_COLORS.keys()).dropna()
ax.bar(range(len(vc)), vc.values, color=[AQI_COLORS[k] for k in vc.index], edgecolor="white")
ax.set_xticks(range(len(vc)))
ax.set_xticklabels([textwrap.fill(k, 12) for k in vc.index], fontsize=9)
ax.set(title="Air-Quality Category Frequency (EPA Classes)", xlabel="AQI category", ylabel="Number of city-days")
save(fig, "02_category_balance.png",
     "The 6 EPA categories are highly imbalanced, dominated by Moderate days. "
     "This class imbalance is an important modelling consideration for the "
     "later part of the project.")

# 3. National daily PM2.5 time series
daily = df.groupby("date")[["pm2_5", "us_aqi_max"]].mean()
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(daily.index, daily.pm2_5, color=PRIMARY, lw=1.3)
ax.fill_between(daily.index, daily.pm2_5, color=PRIMARY, alpha=0.15)
ax.set(title="National Average Daily PM2.5 Across 50 Cities", xlabel="Date", ylabel="PM2.5 (µg/m³)")
save(fig, "03_pm25_timeseries.png",
     "Averaged across all cities, fine-particle pollution rises and falls with "
     "the seasons and spikes sharply during wildfire and stagnation events. "
     "These patterns motivate the time-aware features used later.")

# 4. Seasonal PM2.5 boxplot 
order = ["Winter", "Spring", "Summer", "Fall"]
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="season", y="pm2_5", order=order, hue="season", palette=SEQ, legend=False, ax=ax, showfliers=False)
ax.set(title="Seasonal Variation in PM2.5", xlabel="Season", ylabel="PM2.5 (µg/m³)")
save(fig, "04_seasonal_pm25.png",
     "Particle pollution is typically highest in the cooler months, when "
     "temperature inversions trap pollutants near the ground. Summer shows a "
     "wider spread driven by wildfire smoke in western cities.")

# 5. Correlation heatmap
corr_cols = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "us_aqi_max", "temperature_2m_mean", "wind_speed_10m_max", "precipitation_sum"]
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap=DIVERGING, center=0, square=True, cbar_kws={"label": "Pearson r"}, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation Between Pollutants and Weather")
save(fig, "05_correlation_heatmap.png",
     "Pollutants correlate strongly with one another while weather variables "
     "such as wind and temperature relate to them more weakly. These "
     "relationships guide feature selection and the PCA analysis later.")

# 6. City ranking by mean AQI
city_aqi = df.groupby("city").us_aqi_max.mean().sort_values()
fig, ax = plt.subplots(figsize=(9, 11))
colors = sns.color_palette(SEQ, as_cmap=True)(np.linspace(0.15, 0.9, len(city_aqi)))
ax.barh(city_aqi.index, city_aqi.values, color=colors)
ax.set(title="Cities Ranked by Average Daily Peak AQI", xlabel="Mean daily peak AQI", ylabel="")
ax.tick_params(axis="y", labelsize=8)
save(fig, "06_city_ranking.png",
     "Average air quality varies widely across the 50 cities, with the most "
     "polluted cities showing roughly double the mean AQI of the cleanest. "
     "This spread is what clustering will explore.")

# 7. Geographic map
geo = df.groupby(["city", "latitude", "longitude"]).us_aqi_max.mean().reset_index()
fig, ax = plt.subplots(figsize=(11, 6.5))
sc = ax.scatter(geo.longitude, geo.latitude, c=geo.us_aqi_max, s=140, cmap=SEQ, edgecolor="black", linewidth=0.5)
for _, r in geo.iterrows():
    ax.annotate(r.city, (r.longitude, r.latitude), fontsize=6, xytext=(3, 3), textcoords="offset points")
plt.colorbar(sc, ax=ax, label="Mean daily peak AQI")
ax.set(title="Geographic Distribution of Air Quality (50 U.S. Cities)", xlabel="Longitude", ylabel="Latitude")
save(fig, "07_geographic_map.png",
     "Plotting each city by longitude and latitude reveals regional air-quality "
     "patterns across the country. Warmer-coloured markers highlight clusters of "
     "poorer air quality, notably in parts of the West and industrial Midwest.")

# 8. Ozone vs temperature 
fig, ax = plt.subplots(figsize=(9, 6))
samp = df.sample(4000, random_state=1)
sc = ax.scatter(samp.temperature_2m_mean, samp.ozone, c=samp.us_aqi_max, cmap="viridis", s=12, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Daily peak AQI")
ax.set(title="Ground-Level Ozone vs Temperature", xlabel="Mean daily temperature (°C)", ylabel="Ozone (µg/m³)")
save(fig, "08_ozone_vs_temp.png",
     "Ozone concentrations tend to climb with temperature, reflecting the "
     "sunlight-driven chemistry that is responsbile for it. This is why hot summer days often "
     "carry elevated ozone-based air-quality warnings.")

# 9. PM2.5 vs asthma (city level)
city = df.groupby("city").agg( pm2_5=("pm2_5", "mean"), us_aqi=("us_aqi_max", "mean"), asthma=("asthma_prev", "first"), copd=("copd_prev", "first")).reset_index()
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(city.pm2_5, city.asthma, color=PRIMARY, s=60, edgecolor="white")
m, b = np.polyfit(city.pm2_5, city.asthma, 1)
xs = np.linspace(city.pm2_5.min(), city.pm2_5.max(), 50)
ax.plot(xs, m * xs + b, color=CORAL, ls="--", lw=2)
r = city.pm2_5.corr(city.asthma)
ax.set(title=f"City PM2.5 vs Adult Asthma Prevalence (r = {r:.2f})", xlabel="Mean PM2.5 (µg/m³)", ylabel="Adult asthma prevalence (%)")
save(fig, "09_pm25_vs_asthma.png",
     "Each point is a city, comparing its average fine-particle pollution with "
     "its adult asthma rate. The fitted line summarises the overall direction of "
     "the pollution-health relationship at the city level.")

# 10. AQI vs COPD 
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(city.us_aqi, city.copd, color=PRIMARY, s=60, edgecolor="white")
m, b = np.polyfit(city.us_aqi, city.copd, 1)
xs = np.linspace(city.us_aqi.min(), city.us_aqi.max(), 50)
ax.plot(xs, m * xs + b, color=CORAL, ls="--", lw=2)
r = city.us_aqi.corr(city.copd)
ax.set(title=f"City Mean AQI vs Adult COPD Prevalence (r = {r:.2f})", xlabel="Mean daily peak AQI", ylabel="Adult COPD prevalence (%)")
save(fig, "10_aqi_vs_copd.png",
     "This view compares each city's typical air quality with its rate of "
     "chronic obstructive pulmonary disease. It offers an early look at whether "
     "long-term air quality tracks with chronic respiratory disease burden.")

# 11. City - month AQI heatmap 
pivot = df.pivot_table(index="city", columns="month", values="us_aqi_max", aggfunc="mean")
pivot = pivot.loc[df.groupby("city").us_aqi_max.mean().sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(10, 11))
sns.heatmap(pivot, cmap=SEQ, cbar_kws={"label": "Mean daily peak AQI"}, ax=ax, linewidths=0.3, linecolor="white")
ax.set(title="Monthly Air-Quality Heatmap by City", xlabel="Month", ylabel="")
ax.tick_params(axis="y", labelsize=8)
save(fig, "11_city_month_heatmap.png",
     "This heatmap shows how each city's average AQI shifts month to month, with "
     "cities sorted from worst to best overall. Horizontal bands of colour reveal "
     "which cities have persistent problems versus sharp seasonal episodes.")

# 12. Wind speed vs PM2.5
fig, ax = plt.subplots(figsize=(9, 6))
samp = df.sample(4000, random_state=2)
ax.scatter(samp.wind_speed_10m_max, samp.pm2_5, color=PRIMARY, s=12, alpha=0.4)
ax.set(title="Wind Speed vs PM2.5", xlabel="Maximum daily wind speed (km/h)", ylabel="PM2.5 (µg/m³)")
ax.yaxis.set_major_locator(mticker.MaxNLocator(8))
save(fig, "12_wind_vs_pm25.png",
     "Higher wind speeds are associated with lower fine-particle concentrations, "
     "as wind disperses pollution. The densest, highest-PM2.5 readings occur on "
     "calm, low-wind days when pollutants accumulate.")


with open(os.path.join(IMG, "captions.md"), "w", encoding="utf-8") as f:
    f.write("# EDA Figure Captions\n\n")
    for fn in sorted(captions):
        f.write(f"## {fn}\n{captions[fn]}\n\n")
print("\nAll figures + captions.md written to", os.path.abspath(IMG))
print("total figures:", len(captions))
