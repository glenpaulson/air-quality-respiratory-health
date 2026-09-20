import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
CLEAN = os.path.join(HERE, "..", "data", "clean")
IMG = os.path.join(HERE, "..", "images", "snapshots")
os.makedirs(IMG, exist_ok=True)

def table_png(df, title, fname, maxcols=9, cols=None):
    df = (df[cols] if cols else df.iloc[:, :maxcols]).iloc[:6].copy()
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].round(1)
    fig, ax = plt.subplots(figsize=(min(1.5 * len(df.columns), 14), 2.2))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=12, loc="left", pad=10)
    t = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.5)
    for (r, _), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0e7c7b")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#e6f2f1")
    fig.tight_layout()
    path = os.path.join(IMG, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.abspath(path))


air_qual = pd.read_csv(os.path.join(RAW, "air_quality_weather_raw.csv"))
health = pd.read_csv(os.path.join(RAW, "cdc_places_health_raw.csv"))
clean = pd.read_csv(os.path.join(CLEAN, "air_quality_health_clean.csv"))

table_png(air_qual, "RAW - Air Quality & Weather (Open-Meteo API)", "raw_air_weather.png")
table_png(health, "RAW - County Health (CDC PLACES)", "raw_health.png", maxcols=6)
table_png(clean, "CLEANED & MERGED - key columns (9 of 28, incl. health + labels)",
          "clean_merged.png",
          cols=["city", "date", "pm2_5", "us_aqi_max", "asthma_prev",
                "copd_prev", "season", "aqi_category", "pm25_tier"])
