# EDA Figure Captions

## 01_aqi_distribution.png
Most city-days fall in the Good-to-Moderate range, but a long right tail is reaching into the Unhealthy territory. Those high-AQI days are the rare, high-impact events this project most wants to understand and predict.

## 02_category_balance.png
The 6 EPA categories are highly imbalanced, dominated by Moderate days. This class imbalance is an important modelling consideration for the later part of the project.

## 03_pm25_timeseries.png
Averaged across all cities, fine-particle pollution rises and falls with the seasons and spikes sharply during wildfire and stagnation events. These patterns motivate the time-aware features used later.

## 04_seasonal_pm25.png
Particle pollution is typically highest in the cooler months, when temperature inversions trap pollutants near the ground. Summer shows a wider spread driven by wildfire smoke in western cities.

## 05_correlation_heatmap.png
Pollutants correlate strongly with one another while weather variables such as wind and temperature relate to them more weakly. These relationships guide feature selection and the PCA analysis later.

## 06_city_ranking.png
Average air quality varies widely across the 50 cities, with the most polluted cities showing roughly double the mean AQI of the cleanest. This spread is what clustering will explore.

## 07_geographic_map.png
Plotting each city by longitude and latitude reveals regional air-quality patterns across the country. Warmer-coloured markers highlight clusters of poorer air quality, notably in parts of the West and industrial Midwest.

## 08_ozone_vs_temp.png
Ozone concentrations tend to climb with temperature, reflecting the sunlight-driven chemistry that is responsbile for it. This is why hot summer days often carry elevated ozone-based air-quality warnings.

## 09_pm25_vs_asthma.png
Each point is a city, comparing its average fine-particle pollution with its adult asthma rate. The fitted line summarises the overall direction of the pollution-health relationship at the city level.

## 10_aqi_vs_copd.png
This view compares each city's typical air quality with its rate of chronic obstructive pulmonary disease. It offers an early look at whether long-term air quality tracks with chronic respiratory disease burden.

## 11_city_month_heatmap.png
This heatmap shows how each city's average AQI shifts month to month, with cities sorted from worst to best overall. Horizontal bands of colour reveal which cities have persistent problems versus sharp seasonal episodes.

## 12_wind_vs_pm25.png
Higher wind speeds are associated with lower fine-particle concentrations, as wind disperses pollution. The densest, highest-PM2.5 readings occur on calm, low-wind days when pollutants accumulate.

