import { useState } from "react";
import { Database, Filter, CheckCircle2, BarChart2, X } from "lucide-react";
import { asset } from "../lib/assets";

const REPO = "https://github.com/glenpaulson/air-quality-respiratory-health/blob/main";

interface ActiveImage {
  src: string;
  caption: string;
}

const COLLECTION = [
  "Selected the Open-Meteo Air Quality API to satisfy the dynamic (API) data collection requirement, gathering hourly pollutants for 50 U.S. cities over a full year.",
  "Added the Open-Meteo Historical Weather archive for daily temperature, precipitation and wind at the same locations and dates.",
  "Downloaded county adult asthma and COPD prevalence from CDC PLACES as the second, static data source.",
  "Built the 50-city list (coordinates, county, FIPS, population) from the Open-Meteo Geocoding and U.S. Census Geographies APIs.",
];

const CLEANING = [
  { k: "Types & duplicates", v: "Parsed dates into a calendar type and confirmed exactly one record per city per day (zero duplicates)." },
  { k: "Missing values", v: "Verified completeness, zero null or missing values across the collected air-quality and weather data." },
  { k: "Range checks", v: "Ran physical range checks on every measurement to catch impossible or extreme values (none found)." },
  { k: "Place resolution", v: "Resolved the two independent cities (Baltimore, St. Louis) to their correct county FIPS before the health merge." },
  { k: "Merge", v: "Joined county asthma and COPD prevalence to every city; PA and KY were filled from the 2024 CDC release." },
  { k: "Features & labels", v: "Added month and season, and derived aqi_category, aqi_class3 and a discretized pm25_tier from the raw measurements." },
];

const APIS = [
  {
    name: "Open-Meteo Air Quality API",
    doc: "https://open-meteo.com/en/docs/air-quality-api",
    desc: "Hourly pollutant readings for each city, averaged to a daily value.",
    url: "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=39.7392&longitude=-104.9903&hourly=pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,us_aqi&start_date=2025-09-15&end_date=2026-09-15",
  },
  {
    name: "Open-Meteo Historical Weather API",
    doc: "https://open-meteo.com/en/docs/historical-weather-api",
    desc: "Daily temperature, precipitation and wind for each city over the same year.",
    url: "https://archive-api.open-meteo.com/v1/archive?latitude=39.7392&longitude=-104.9903&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,wind_speed_10m_max&start_date=2025-09-15&end_date=2026-09-15",
  },
  {
    name: "CDC PLACES (Socrata Open Data API)",
    doc: "https://data.cdc.gov/resource/swc5-untb",
    desc: "County adult asthma (CASTHMA) and COPD crude prevalence. Dataset id swc5-untb (2025), with fu4u-a9bh (2024) as fallback.",
    url: "https://data.cdc.gov/resource/swc5-untb.json?measureid=CASTHMA&datavaluetypeid=CrdPrv&$limit=6000",
  },
];

const SOURCES = [
  ["Open-Meteo Air Quality", "API", "Daily PM2.5, PM10, O₃, NO₂, SO₂, CO and US AQI for 50 cities", "18,300 city-days", "CC BY 4.0"],
  ["Open-Meteo Historical Weather", "API", "Daily temperature, precipitation and wind", "merged in", "CC BY 4.0"],
  ["CDC PLACES (County Data)", "Download / API", "County adult asthma & COPD crude prevalence", "3,145 counties", "Public domain"],
  ["Open-Meteo Geocoding", "API", "City coordinates & population", "50 cities", "CC BY 4.0"],
  ["U.S. Census Geographies", "API", "County name & FIPS per city", "50 cities", "Public domain"],
];

const CODE = [
  ["build_cities.py", "Builds the 50-city list (coordinates, county, FIPS, population) from the geocoding APIs", "requests"],
  ["collect_data.py", "Pulls a year of daily air quality + weather for every city", "requests, pandas"],
  ["fetch_health.py", "Downloads county asthma & COPD prevalence from CDC PLACES", "requests, pandas"],
  ["clean_data.py", "Cleans, merges the health data, and derives the labels", "pandas, numpy"],
  ["make_snapshots.py", "Renders the raw/clean sample table images", "pandas, matplotlib"],
  ["eda.py", "Generates the twelve exploratory figures", "pandas, matplotlib, seaborn"],
];

const PIPELINE = [
  ["Air quality + weather", "18,300", "18,300", "0 duplicates; 0 missing", "Range checks passed"],
  ["County health (CDC)", "3,145", "3,145", "Independent-city duplicate rows", "PA & KY filled from 2024"],
  ["Merged clean table", "18,300", "18,300", "Nothing, all rows kept", "month, season & 3 labels added"],
];

const FIGURES: { fn: string; title: string; desc: string; wide?: boolean }[] = [
  { fn: "01_aqi_distribution.png", title: "Distribution of Daily Peak AQI", desc: "Most city-days fall in the Good-to-Moderate range, with a long right tail into Unhealthy territory, the rare, high-impact days the project most wants to predict." },
  { fn: "02_category_balance.png", title: "AQI Category Frequency", desc: "The six EPA categories are highly imbalanced, dominated by Moderate days, an important consideration for later classification." },
  { fn: "03_pm25_timeseries.png", title: "National Daily PM2.5", desc: "Averaged across all cities, fine-particle pollution rises and falls with the seasons and spikes during wildfire events.", wide: true },
  { fn: "04_seasonal_pm25.png", title: "Seasonal Variation in PM2.5", desc: "Particle pollution is typically highest in cooler months when inversions trap pollutants; summer shows a wider, wildfire-driven spread." },
  { fn: "05_correlation_heatmap.png", title: "Pollutant & Weather Correlation", desc: "Pollutants correlate strongly with one another while weather variables relate more weakly, guiding feature selection and PCA." },
  { fn: "06_city_ranking.png", title: "Cities Ranked by Mean AQI", desc: "Average air quality varies widely; the most polluted cities show roughly double the mean AQI of the cleanest." },
  { fn: "07_geographic_map.png", title: "Geographic Distribution", desc: "Plotting cities by longitude and latitude reveals regional patterns, with clusters of poorer air quality in parts of the West and Midwest.", wide: true },
  { fn: "08_ozone_vs_temp.png", title: "Ozone vs Temperature", desc: "Ozone climbs with temperature, reflecting the sunlight-driven chemistry that forms it, why hot days carry ozone warnings." },
  { fn: "09_pm25_vs_asthma.png", title: "PM2.5 vs Adult Asthma", desc: "Each point is a city; the fitted line summarises a weak city-level relationship shaped by many other factors." },
  { fn: "10_aqi_vs_copd.png", title: "AQI vs Adult COPD", desc: "Compares each city's typical air quality with its COPD rate, an early look at whether long-term air quality tracks chronic disease burden." },
  { fn: "11_city_month_heatmap.png", title: "Monthly AQI by City", desc: "Shows how each city's average AQI shifts month to month, revealing persistent problems versus sharp seasonal episodes.", wide: true },
  { fn: "12_wind_vs_pm25.png", title: "Wind Speed vs PM2.5", desc: "Higher winds disperse pollution; the densest PM2.5 readings occur on calm, low-wind days." },
];

const DataPrepEDA = () => {
  const [active, setActive] = useState<ActiveImage | null>(null);

  return (
    <div>
      <div className="bg-[#EBEBEB] min-h-screen pt-32 pb-20 px-6 text-[#1a1a1a]">
        <div className="max-w-7xl mx-auto">
          <div className="mb-16">
            <h1 className="text-4xl md:text-5xl font-medium tracking-tight mb-6">
              Data Preparation &amp; Exploratory Analysis
            </h1>
            <p className="text-xl text-gray-600 leading-relaxed max-w-3xl">
              How the data was gathered, cleaned, and explored. 50 of the most populous U.S. cities,
              a full year of daily readings, joined with county respiratory-health data into one
              table of 18,300 records.
            </p>
          </div>

          {/* Collection + cleaning cards */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-20">
            <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
              <div className="flex items-center gap-2 mb-6">
                <Database className="text-gray-400" size={24} />
                <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500">Data Collection</h3>
              </div>
              <ul className="space-y-3">
                {COLLECTION.map((c, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <CheckCircle2 className="text-blue-500 shrink-0 mt-0.5" size={18} />
                    <p className="text-gray-600 text-sm leading-relaxed">{c}</p>
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
              <div className="flex items-center gap-2 mb-6">
                <Filter className="text-gray-400" size={24} />
                <h3 className="text-sm font-bold uppercase tracking-widest text-gray-500">Cleaning &amp; Preprocessing</h3>
              </div>
              <ul className="space-y-3">
                {CLEANING.map((c) => (
                  <li key={c.k} className="flex items-start gap-3">
                    <CheckCircle2 className="text-blue-500 shrink-0 mt-0.5" size={18} />
                    <p className="text-gray-600 text-sm leading-relaxed"><strong>{c.k}:</strong> {c.v}</p>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Data sources table */}
          <h2 className="text-3xl font-medium mb-6">Data Sources</h2>
          <div className="overflow-x-auto mb-16 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3">Source</th><th className="px-4 py-3">Type</th>
                  <th className="px-4 py-3">What it gives</th><th className="px-4 py-3">Records</th>
                  <th className="px-4 py-3">Licence</th>
                </tr>
              </thead>
              <tbody>
                {SOURCES.map((r) => (
                  <tr key={r[0]} className="border-b border-gray-100 last:border-0">
                    <td className="px-4 py-3 font-medium text-gray-900">{r[0]}</td>
                    <td className="px-4 py-3">{r[1]}</td><td className="px-4 py-3">{r[2]}</td>
                    <td className="px-4 py-3">{r[3]}</td><td className="px-4 py-3">{r[4]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* APIs used */}
          <h2 className="text-3xl font-medium mb-6">APIs Used</h2>
          <div className="space-y-8 mb-16">
            {APIS.map((a) => (
              <div key={a.name}>
                <h3 className="text-xl font-bold text-gray-900">{a.name}</h3>
                <p className="text-sm text-gray-500 mt-1">
                  Documentation:{" "}
                  <a href={a.doc} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{a.doc}</a>
                </p>
                <p className="text-gray-600 my-2">{a.desc}</p>
                <pre className="bg-gray-100 border border-gray-200 text-gray-700 text-xs md:text-sm rounded-lg p-4 overflow-x-auto">
                  <span className="text-blue-600 font-bold">GET </span>{a.url}
                </pre>
              </div>
            ))}
          </div>

          {/* Raw data access */}
          <h2 className="text-3xl font-medium mb-6">Raw Data Access</h2>
          <div className="overflow-x-auto mb-4 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr><th className="px-4 py-3">Source</th><th className="px-4 py-3">Where it came from</th><th className="px-4 py-3">How to get it again</th></tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100"><td className="px-4 py-3">Air quality &amp; weather</td><td className="px-4 py-3">Open-Meteo APIs</td><td className="px-4 py-3 font-mono text-xs"><a href={`${REPO}/code/collect_data.py`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">collect_data.py</a></td></tr>
                <tr className="border-b border-gray-100"><td className="px-4 py-3">County health</td><td className="px-4 py-3">CDC PLACES (data.cdc.gov)</td><td className="px-4 py-3 font-mono text-xs"><a href={`${REPO}/code/fetch_health.py`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">fetch_health.py</a></td></tr>
                <tr><td className="px-4 py-3">City list (coords, county, FIPS)</td><td className="px-4 py-3">Open-Meteo Geocoding + U.S. Census</td><td className="px-4 py-3 font-mono text-xs"><a href={`${REPO}/code/build_cities.py`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">build_cities.py</a></td></tr>
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mb-16">
            Raw files:{" "}
            <a href={`${REPO}/data/raw/air_quality_weather_raw.csv`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">air_quality_weather_raw.csv</a>,{" "}
            <a href={`${REPO}/data/raw/cdc_places_health_raw.csv`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">cdc_places_health_raw.csv</a>{" "}
            &nbsp;|&nbsp; Cleaned:{" "}
            <a href={`${REPO}/data/clean/air_quality_health_clean.csv`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">air_quality_health_clean.csv</a>
          </p>

          {/* Code table */}
          <h2 className="text-3xl font-medium mb-6">Code</h2>
          <div className="overflow-x-auto mb-16 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr><th className="px-4 py-3">Script</th><th className="px-4 py-3">What it does</th><th className="px-4 py-3">Main packages</th></tr>
              </thead>
              <tbody>
                {CODE.map((r) => (
                  <tr key={r[0]} className="border-b border-gray-100 last:border-0">
                    <td className="px-4 py-3 font-mono text-xs">
                      <a href={`${REPO}/code/${r[0]}`} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{r[0]}</a>
                    </td>
                    <td className="px-4 py-3">{r[1]}</td><td className="px-4 py-3">{r[2]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-sm text-gray-500 mb-16 -mt-12">Written in Python 3.13.</p>

          {/* Raw & cleaned snapshots */}
          <h2 className="text-3xl font-medium mb-6">Raw &amp; Cleaned Data</h2>
          <p className="text-gray-600 mb-6 max-w-3xl">
            Small samples of the data before and after cleaning (full datasets are linked, not pasted
            here). The cleaned table shows key columns of 28, including the joined health figures and
            the derived labels.
          </p>
          <div className="space-y-4 mb-20">
            {[
              { fn: "raw_air_weather.png", cap: "Raw air quality & weather (Open-Meteo API)" },
              { fn: "raw_health.png", cap: "Raw county health (CDC PLACES)" },
              { fn: "clean_merged.png", cap: "Cleaned & merged, key columns incl. health + labels" },
            ].map((s) => (
              <img
                key={s.fn}
                src={asset(`images/snapshots/${s.fn}`)}
                alt={s.cap}
                onClick={() => setActive({ src: asset(`images/snapshots/${s.fn}`), caption: s.cap })}
                className="w-full rounded-xl border border-gray-200 shadow-sm cursor-zoom-in"
              />
            ))}
          </div>

          {/* Cleaning pipeline */}
          <h2 className="text-3xl font-medium mb-6">Cleaning Pipeline</h2>
          <div className="overflow-x-auto mb-6 bg-white rounded-2xl border border-gray-200 shadow-sm">
            <table className="w-full text-sm text-left text-gray-600">
              <thead className="text-xs text-gray-900 uppercase bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3">Stage</th><th className="px-4 py-3">Rows in</th><th className="px-4 py-3">Rows out</th>
                  <th className="px-4 py-3">Removed</th><th className="px-4 py-3">Added / flagged</th>
                </tr>
              </thead>
              <tbody>
                {PIPELINE.map((r) => (
                  <tr key={r[0]} className="border-b border-gray-100 last:border-0">
                    <td className="px-4 py-3 font-medium text-gray-900">{r[0]}</td>
                    <td className="px-4 py-3">{r[1]}</td>
                    <td className="px-4 py-3 text-green-600 font-bold">{r[2]}</td>
                    <td className="px-4 py-3">{r[3]}</td><td className="px-4 py-3">{r[4]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-gray-600 max-w-4xl mb-20 leading-relaxed">
            Because the Open-Meteo readings arrive as validated model output, no air-quality or
            weather rows had to be removed, the duplicate, missing-value, and range checks all
            came back clean, which the checks prove rather than assume. No values were imputed or
            invented; the only rows dropped were duplicate county records for the two independent
            cities, resolved by their FIPS code.
          </p>

          {/* Figures */}
          <div className="flex items-center gap-3 mb-8 border-b border-gray-300 pb-6">
            <BarChart2 className="text-gray-500" size={28} />
            <h2 className="text-3xl font-medium text-gray-900">Exploratory Data Visualizations</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FIGURES.map((f) => {
              const src = asset(`images/eda/${f.fn}`);
              return (
                <div key={f.fn} className={`bg-white rounded-xl overflow-hidden border border-gray-200 shadow-sm flex flex-col ${f.wide ? "md:col-span-2" : ""}`}>
                  <div
                    className="h-48 bg-gray-100 flex items-center justify-center cursor-zoom-in group relative overflow-hidden"
                    onClick={() => setActive({ src, caption: `${f.title}, ${f.desc}` })}
                  >
                    <img src={src} alt={f.title} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                    <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <span className="bg-white/90 text-black px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide shadow-sm">View</span>
                    </div>
                  </div>
                  <div className="p-5">
                    <h4 className="font-bold text-gray-900 mb-2">{f.title}</h4>
                    <p className="text-gray-500 text-sm leading-relaxed">{f.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Modal */}
      {active && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4 backdrop-blur-sm cursor-zoom-out"
          onClick={() => setActive(null)}
        >
          <button className="absolute top-6 right-6 text-white/70 hover:text-white p-2 rounded-full bg-white/10 hover:bg-white/20 transition-all" onClick={() => setActive(null)}>
            <X size={32} />
          </button>
          <div className="flex flex-col items-center max-w-5xl w-full" onClick={(e) => e.stopPropagation()}>
            <img src={active.src} alt="" className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-2xl bg-white" />
            <p className="text-white text-center mt-6 text-base max-w-2xl font-light">{active.caption}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataPrepEDA;
