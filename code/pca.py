"""
PCA: Principal Component Analysis on the daily air and weather measurements
  1. load the cleaned data and keep only the 9 numeric measurement columns
  2. log the long tailed pollutants, then centre and standardise every column
  3. fit PCA with every component kept, so we can look at the full scree first
  4. choose how many components to keep using the two rules we learned from class: the scree elbow and the smallest k that clears a 90 percent variance target
  5. read the loadings and give the first components a name in plain language
  6. draw the 2D and 3D projections, coloured by labels that were held out of the fit
  7. repeat the same PCA on the 50 city table
  8. draw the explanatory figures for overview
output:  images/pca/*.png + data/clean/pca_*.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_FOLDER = os.path.join(THIS_FOLDER, "..", "data", "clean")
CLEAN_DATA_FILE = os.path.join(CLEAN_DATA_FOLDER, "air_quality_health_clean.csv")
IMAGE_FOLDER = os.path.join(THIS_FOLDER, "..", "images", "pca")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
TEAL = "#0e7c7b"
CORAL = "#ef6f53"
NAVY = "#2b4162"
GOLD = "#e8b004"
CLUSTER_COLORS = ["#0e7c7b", "#ef6f53", "#2b4162", "#e8b004", "#7b5aa6", "#4a9b5e"]
AQI_COLORS = {"Good": "#3bb143", "Moderate": "#efc804", "Unhealthy for Sensitive Groups": "#ff7e00", "Unhealthy": "#d62728", "Very Unhealthy": "#8f3f97", "Hazardous": "#7e0023"}
SEASON_COLORS = {"Winter": "#4c72b0", "Spring": "#55a868", "Summer": "#c44e52", "Fall": "#dd8452"}
RANDOM_SEED = 85

def save_figure(figure, file_name):
    figure.tight_layout()
    figure.savefig(os.path.join(IMAGE_FOLDER, file_name), bbox_inches="tight")
    plt.close(figure)
    print("saved", file_name)

def save_table_image(table, title, file_name, note=None, number_of_rows=6, figure_width=None):
    """take the first few rows of a dataframe and save them as a picture"""
    small_table = table.head(number_of_rows).copy()
    # rounding the decimals and turn dates into plain text for more readability on the website
    for column_name in small_table.columns:
        if small_table[column_name].dtype == "float64":
            small_table[column_name] = small_table[column_name].round(2)
        elif str(small_table[column_name].dtype).startswith("datetime"):
            small_table[column_name] = small_table[column_name].dt.strftime("%Y-%m-%d")

    if figure_width is None:
        figure_width = min(1.75 * len(small_table.columns), 19)

    figure, ax = plt.subplots(figsize=(figure_width, 2.4))
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=12, loc="left", pad=10)
    drawn_table = ax.table(cellText=small_table.values, colLabels=small_table.columns, cellLoc="center", loc="center")
    drawn_table.auto_set_font_size(False)
    drawn_table.set_fontsize(8)
    drawn_table.scale(1, 1.5)

    for position, cell in drawn_table.get_celld().items():
        row_number = position[0]
        if row_number == 0:
            cell.set_facecolor(TEAL)
            cell.set_text_props(color="white", fontweight="bold")
        elif row_number % 2 == 0:
            cell.set_facecolor("#e6f2f1")

    if note is not None:
        ax.annotate(note, xy=(0, -0.10), xycoords="axes fraction", fontsize=8, color="#555")

    save_figure(figure, file_name)

def find_the_elbow(eigenvalues):
    """
    we measure how far each component drops to the next one, and the elbow is the last
    component whose drop was bigger than the average drop.
    """
    drops = eigenvalues[:-1] - eigenvalues[1:]
    components_with_a_big_drop = np.where(drops > drops.mean())[0]
    return int(components_with_a_big_drop[-1]) + 2

def components_needed_for(variance_ratio, target):
    """the smallest number of components whose running total helps us to clear the target"""
    return int(np.argmax(np.cumsum(variance_ratio) >= target) + 1)

# ----------------------------------------------------------------------------------
# step 1: we keep only the numeric measurements, because PCA cannot read text or labels

daily_data = pd.read_csv(CLEAN_DATA_FILE, parse_dates=["date"])
FEATURE_COLUMNS = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
SHORT_NAMES = {"pm2_5": "PM2.5", "pm10": "PM10", "ozone": "O3", "nitrogen_dioxide": "NO2", "sulphur_dioxide": "SO2", "carbon_monoxide": "CO", "temperature_2m_mean": "temp", "precipitation_sum": "precip", "wind_speed_10m_max": "wind"}
SHORT_FEATURE_NAMES = [SHORT_NAMES[name] for name in FEATURE_COLUMNS]

print("{:,} rows, {} numeric features going into PCA".format(len(daily_data), len(FEATURE_COLUMNS)))

# missing values will break the fit, so we need to check and say which, we have none
missing_values = daily_data[FEATURE_COLUMNS].isna().sum().sum()
print("missing values in the 9 measurement columns: {}, so nothing is dropped or imputed".format(missing_values))

save_table_image(daily_data[["city", "date"] + FEATURE_COLUMNS[:6] + ["aqi_category"]],
                 "Before: the cleaned daily table, text columns and labels still present",
                 "01_before_raw.png",
                 note="PCA needs a purely numeric matrix, so city, date and aqi_category have to come out first")

# ----------------------------------------------------------------------------------
# step 2: log the skewed pollutants, then centre and standardise. We know that PCA maximises variance and we want to avoid any column overpowering the others

SKEWED_COLUMNS = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "precipitation_sum"]
raw_table = daily_data[FEATURE_COLUMNS].copy()
logged_table = raw_table.copy()
logged_table[SKEWED_COLUMNS] = np.log1p(logged_table[SKEWED_COLUMNS]) # log1p = log(1 + x)

scaled_data = StandardScaler().fit_transform(logged_table.values) # centres and scales in one call
scaled_table = pd.DataFrame(scaled_data, columns=SHORT_FEATURE_NAMES).round(3)

save_table_image(raw_table, "After (step 1/2): numeric matrix only, 9 measurement columns",
                 "02_after_numeric.png",
                 note="still on raw scales, carbon monoxide runs in the hundreds while wind sits near 15")
save_table_image(scaled_table, "After (step 2/2): long tails logged, then every column centred and standardised",
                 "03_after_standardised.png",
                 note="mean 0 and standard deviation 1 everywhere. 18,300 rows x 9 columns is the PCA input")

scaled_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "pca_input_standardised.csv"), index=False)

# ----------------------------------------------------------------------------------
# step 3: fit PCA keeping every component, to look at the scree plot

pca_model = PCA(random_state=RANDOM_SEED)
component_scores = pca_model.fit_transform(scaled_data)
eigenvalues = pca_model.explained_variance_ # how much spread sits along each direction
variance_ratio = pca_model.explained_variance_ratio_ # the same thing as a share of the total
running_total = np.cumsum(variance_ratio)
component_names = ["PC{}".format(number + 1) for number in range(len(eigenvalues))]

variance_summary = pd.DataFrame({"component": component_names,
                                 "eigenvalue": eigenvalues.round(3),
                                 "variance_explained_percent": (variance_ratio * 100).round(2),
                                 "cumulative_percent": (running_total * 100).round(2)})
print(variance_summary.to_string(index=False))
variance_summary.to_csv(os.path.join(CLEAN_DATA_FOLDER, "pca_explained_variance.csv"), index=False)

# ----------------------------------------------------------------------------------
# step 4: how many components do we keep

elbow_components = find_the_elbow(eigenvalues)
target_components = components_needed_for(variance_ratio, 0.90)

print("\nrule 1, the elbow in the scree plot: {} components, keeping {:.1f}% of the variance".format(elbow_components, running_total[elbow_components - 1] * 100))
print("rule 2, the smallest k clearing 90 percent: {} components, keeping {:.1f}%".format(target_components, running_total[target_components - 1] * 100))
print("the two rules disagree, so we keep {} for every picture and for the reduced dataset,".format(elbow_components))
print("and we report {} as the number a later model would need if it must not lose the rare bad days.".format(target_components))

KEPT_COMPONENTS = elbow_components

figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

ax = axes[0]
ax.bar(component_names, eigenvalues, color=TEAL, edgecolor="white")
ax.plot(component_names, eigenvalues, "o-", color=CORAL, lw=2)
ax.axvline(elbow_components - 1, color=NAVY, ls="--", lw=1.8, label="the elbow is at {} components".format(elbow_components))
for position in range(len(eigenvalues)):
    ax.annotate("{:.2f}".format(eigenvalues[position]), (position, eigenvalues[position]), ha="center", va="bottom", fontsize=8)
ax.set(title="scree plot: how much variance each component carries", xlabel="principal component", ylabel="eigenvalue")
ax.legend()

ax = axes[1]
ax.bar(component_names, variance_ratio * 100, color="#cfe3e2", edgecolor="white", label="this component on its own")
ax.plot(component_names, running_total * 100, "o-", color=TEAL, lw=2.5, label="running total")
ax.axhline(90, color=CORAL, ls="--", lw=1.8, label="90 percent target")
ax.axvline(target_components - 1, color=NAVY, ls=":", lw=1.8, label="{} components clear it".format(target_components))
for position in range(len(running_total)):
    ax.annotate("{:.0f}%".format(running_total[position] * 100), (position, running_total[position] * 100), ha="center", va="bottom", fontsize=8)
ax.set(title="cumulative variance retained", xlabel="principal component", ylabel="percent of total variance kept", ylim=(0, 108))
ax.legend(loc="lower right", fontsize=9)

figure.suptitle("Choosing how many principal components to keep", fontweight="bold", y=1.03)
save_figure(figure, "05_scree_and_cumulative.png")

# ----------------------------------------------------------------------------------
# step 5: loadings are the eigenvector entries, one per original column

loadings = pd.DataFrame(pca_model.components_[:4].T, index=SHORT_FEATURE_NAMES, columns=component_names[:4])
loadings.round(3).to_csv(os.path.join(CLEAN_DATA_FOLDER, "pca_loadings.csv"))
print("\nloadings, the eigenvector entries of the first four components:")
print(loadings.round(3).to_string())

# read each component back in the language of the data rather than of eigenvectors
for component in component_names[:3]:
    sorted_loadings = loadings[component].sort_values()
    print("{}: loads up on {} and down on {}".format(component, ", ".join(sorted_loadings.tail(3).index[::-1]), ", ".join(sorted_loadings.head(2).index)))

figure, ax = plt.subplots(figsize=(7.6, 5))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=1, linecolor="white", cbar_kws={"label": "loading, the weight in the eigenvector"}, ax=ax)
ax.set(title="loadings: how each original measurement feeds into each component", xlabel="", ylabel="original measurement")
save_figure(figure, "06_loadings_heatmap.png")

# ----------------------------------------------------------------------------------
# step 6: the projection. the labels below were held out of the fit, so colouring by them afterwards is a good check to see whether the components found something real

projection = pd.DataFrame(component_scores[:, :KEPT_COMPONENTS], columns=component_names[:KEPT_COMPONENTS])
projection["aqi_category"] = daily_data.aqi_category.values
projection["season"] = daily_data.season.values

reduced_dataset = pd.concat([daily_data[["city", "date"]].reset_index(drop=True), projection.round(4)], axis=1)
reduced_dataset.to_csv(os.path.join(CLEAN_DATA_FOLDER, "pca_scores_3components.csv"), index=False)
save_table_image(reduced_dataset, "The reduced dataset: 9 measurement columns replaced by {} component scores".format(KEPT_COMPONENTS),
                 "07b_reduced_table.png",
                 note="aqi_category and season are carried along only to colour the plots, PCA never sees them")

categories_present = [name for name in AQI_COLORS if name in projection.aqi_category.unique()]
chart_sample = projection.sample(n=6000, random_state=RANDOM_SEED)

figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
for panel_number in range(2):
    ax = axes[panel_number]
    if panel_number == 0:
        column_name = "aqi_category"
        colours = AQI_COLORS
        keys = categories_present
        ax.set_title("coloured by the EPA air quality category")
    else:
        column_name = "season"
        colours = SEASON_COLORS
        keys = ["Winter", "Spring", "Summer", "Fall"]
        ax.set_title("coloured by season")

    for key in keys:
        is_this_key = chart_sample[column_name] == key
        ax.scatter(chart_sample.PC1[is_this_key], chart_sample.PC2[is_this_key], s=8, alpha=0.5, color=colours[key], label="{} (n={})".format(key, is_this_key.sum()))

    ax.set(xlabel="PC1 ({:.1f}% of variance)".format(variance_ratio[0] * 100), ylabel="PC2 ({:.1f}%)".format(variance_ratio[1] * 100))
    ax.legend(fontsize=8, markerscale=2.2, loc="best")

figure.suptitle("18,300 city days squeezed from 9 dimensions down to 2, with 6,000 plotted so the dots stay readable", fontweight="bold", fontsize=14, y=1.02)
save_figure(figure, "07_projection_2d.png")

# ----------------------------------------------------------------------------------
# step 7: the same points in 3D, which is what the third component buys us

sample_3d = projection.sample(n=4000, random_state=RANDOM_SEED)
figure = plt.figure(figsize=(9.5, 7.5))
ax = figure.add_subplot(111, projection="3d")
for key in categories_present:
    is_this_key = sample_3d.aqi_category == key
    ax.scatter(sample_3d.PC1[is_this_key], sample_3d.PC2[is_this_key], sample_3d.PC3[is_this_key], s=9, alpha=0.5, color=AQI_COLORS[key], label=key)
ax.set(xlabel="PC1 ({:.0f}%)".format(variance_ratio[0] * 100), ylabel="PC2 ({:.0f}%)".format(variance_ratio[1] * 100), zlabel="PC3 ({:.0f}%)".format(variance_ratio[2] * 100))
ax.set_title("the first three components together keep {:.0f}% of the information".format(running_total[2] * 100), fontweight="bold")
ax.view_init(elev=18, azim=-58)
ax.legend(fontsize=8, markerscale=2.2, loc="upper left")
save_figure(figure, "08_projection_3d.png")

# ----------------------------------------------------------------------------------
# step 8: the biplot puts the rows and the original columns on the same picture.

biplot_sample = projection.sample(n=2500, random_state=RANDOM_SEED)
ARROW_SCALE = 6.5

figure, ax = plt.subplots(figsize=(9, 7.4))
ax.scatter(biplot_sample.PC1, biplot_sample.PC2, s=8, alpha=0.25, color="#9aa0a6", zorder=2)
for position in range(len(SHORT_FEATURE_NAMES)):
    arrow_x = pca_model.components_[0, position] * ARROW_SCALE
    arrow_y = pca_model.components_[1, position] * ARROW_SCALE
    ax.arrow(0, 0, arrow_x, arrow_y, color=CORAL, lw=2.2, head_width=0.16, zorder=4, length_includes_head=True)
    ax.annotate(SHORT_FEATURE_NAMES[position], (arrow_x * 1.14, arrow_y * 1.14), color=NAVY, fontweight="bold", ha="center", fontsize=11, zorder=5)
ax.axhline(0, color="#ccc", lw=1)
ax.axvline(0, color="#ccc", lw=1)
ax.set(title="biplot: the arrows show where each original measurement points", xlabel="PC1 ({:.1f}%)".format(variance_ratio[0] * 100), ylabel="PC2 ({:.1f}%)".format(variance_ratio[1] * 100))
ax.annotate("arrows close together are measurements that rise and fall together.\narrows pointing opposite ways are measurements that trade off.", xy=(0.015, 0.015), xycoords="axes fraction", fontsize=9, color="#333")
save_figure(figure, "09_biplot.png")

# ----------------------------------------------------------------------------------
# step 9: the same method on the 50 city table

city_table_file = os.path.join(CLEAN_DATA_FOLDER, "clustering_city_standardised.csv")
if os.path.exists(city_table_file):
    city_table = pd.read_csv(city_table_file, index_col=0)
    city_pca_model = PCA(random_state=RANDOM_SEED).fit(city_table.values)
    city_points = city_pca_model.transform(city_table.values)
    city_variance_ratio = city_pca_model.explained_variance_ratio_
    print("\ncity level PCA: PC1 {:.1f}%, PC2 {:.1f}%, together {:.1f}%".format(city_variance_ratio[0] * 100, city_variance_ratio[1] * 100, city_variance_ratio[:2].sum() * 100))

    figure, ax = plt.subplots(figsize=(10.5, 7.4))
    ax.scatter(city_points[:, 0], city_points[:, 1], s=110, color=TEAL, edgecolor="white", linewidth=1.2, zorder=3)
    for city_number in range(len(city_table)):
        ax.annotate(city_table.index[city_number], (city_points[city_number, 0], city_points[city_number, 1]), fontsize=7, alpha=0.8, xytext=(5, 4), textcoords="offset points")

    CITY_ARROW_SCALE = 3.2
    for position in range(len(city_table.columns)):
        arrow_x = city_pca_model.components_[0, position] * CITY_ARROW_SCALE
        arrow_y = city_pca_model.components_[1, position] * CITY_ARROW_SCALE
        ax.arrow(0, 0, arrow_x, arrow_y, color=CORAL, lw=2, head_width=0.11, zorder=4, length_includes_head=True)
        ax.annotate(city_table.columns[position], (arrow_x * 1.15, arrow_y * 1.15), color=NAVY, fontweight="bold", fontsize=10, ha="center", zorder=5)

    ax.set(title="the 50 cities in PCA space, the same 2D map the Clustering tab is plotting on", xlabel="PC1 ({:.1f}%)".format(city_variance_ratio[0] * 100), ylabel="PC2 ({:.1f}%)".format(city_variance_ratio[1] * 100))
    save_figure(figure, "10_city_pca_biplot.png")

# ----------------------------------------------------------------------------------
# step 10: rebuild the 9 columns from k components and measure how far the rebuilt
# version sits from the real one. for PCA the reconstruction error comes out exactly equal to 1 minus the cumulative variance, so
# it is the same number as the curve above, that is it tell us what we lost instead of what we kept.

reconstruction_errors = []
for number_of_components in range(1, len(FEATURE_COLUMNS) + 1):
    small_model = PCA(n_components=number_of_components, random_state=RANDOM_SEED).fit(scaled_data)
    rebuilt = small_model.inverse_transform(small_model.transform(scaled_data))
    reconstruction_errors.append(np.mean((scaled_data - rebuilt) ** 2))

figure, ax = plt.subplots(figsize=(8, 4.8))
component_counts = list(range(1, len(FEATURE_COLUMNS) + 1))
ax.plot(component_counts, reconstruction_errors, "o-", color=TEAL, lw=2.5)
ax.axvline(KEPT_COMPONENTS, color=NAVY, ls="--", lw=1.8, label="{} components, the elbow".format(KEPT_COMPONENTS))
ax.axvline(target_components, color=CORAL, ls=":", lw=1.8, label="{} components, the 90 percent target".format(target_components))
for position in range(len(component_counts)):
    ax.annotate("{:.2f}".format(reconstruction_errors[position]), (component_counts[position], reconstruction_errors[position]), ha="center", va="bottom", fontsize=8)
ax.set(title="information actually lost when components are dropped", xlabel="number of components kept", ylabel="reconstruction error, mean squared")
ax.legend()
save_figure(figure, "11_reconstruction_error.png")
print("\nreconstruction error with {} components: {:.4f}, with {} components: {:.4f}, where 0 is perfect".format(KEPT_COMPONENTS, reconstruction_errors[KEPT_COMPONENTS - 1], target_components, reconstruction_errors[target_components - 1]))

# show that the error really is just the leftover variance, so that we are not claiming it as new evidence
leftover_variance = 1 - running_total
print("checking the error is the same number as the variance we dropped:")
for number_of_components in [KEPT_COMPONENTS, target_components]:
    print("  {} components: error {:.4f}, variance not kept {:.4f}".format(number_of_components, reconstruction_errors[number_of_components - 1], leftover_variance[number_of_components - 1]))

# ----------------------------------------------------------------------------------
# diagrams for the overview explanations

random_generator = np.random.default_rng(3)
example_cloud = np.dot(random_generator.normal(0, 1, (160, 2)), np.array([[1.75, 0.55], [0.70, 1.35]]))
example_cloud = example_cloud - example_cloud.mean(axis=0)
example_pca = PCA().fit(example_cloud)
TEXT_BOX = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ddd", alpha=0.92)

figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))

# the eigenvectors drawn on the cloud they came from
ax = axes[0]
ax.scatter(example_cloud[:, 0], example_cloud[:, 1], s=26, color="#9aa0a6", alpha=0.75, zorder=2)
arrow_colours = [CORAL, NAVY]
for component_number in range(2):
    direction = example_pca.components_[component_number] * np.sqrt(example_pca.explained_variance_[component_number]) * 2.4
    ax.arrow(0, 0, direction[0], direction[1], color=arrow_colours[component_number], lw=3.2, head_width=0.26, zorder=4, length_includes_head=True)
    if component_number == 0:
        label_nudge = -0.55
    else:
        label_nudge = 0.45
    ax.annotate("PC{}  ({:.0f}% of the spread)".format(component_number + 1, example_pca.explained_variance_ratio_[component_number] * 100), (direction[0] * 1.05, direction[1] * 1.05 + label_nudge), color=arrow_colours[component_number], fontweight="bold", fontsize=11, ha="center", bbox=TEXT_BOX)
ax.set(title="eigenvectors point along the directions of greatest spread", xlabel="feature 1", ylabel="feature 2", aspect="equal")
ax.annotate("eigenvector = the direction.\neigenvalue = how much spread\nlies along that direction.", xy=(0.02, 0.03), xycoords="axes fraction", fontsize=10, color="#333", bbox=TEXT_BOX)

# the same cloud rotated onto the components, then flattened onto PC1
ax = axes[1]
rotated_cloud = example_pca.transform(example_cloud)
ax.scatter(rotated_cloud[:, 0], rotated_cloud[:, 1], s=26, color="#9aa0a6", alpha=0.6, zorder=2, label="after rotating onto PC1 and PC2")
ax.scatter(rotated_cloud[:, 0], np.zeros(len(rotated_cloud)), s=26, color=CORAL, alpha=0.8, zorder=3, label="after dropping PC2, so 2D becomes 1D")
for row_number in range(0, len(rotated_cloud), 6):
    ax.plot([rotated_cloud[row_number, 0], rotated_cloud[row_number, 0]], [rotated_cloud[row_number, 1], 0], color="#c8c8c8", lw=0.7, zorder=1)
ax.axhline(0, color=CORAL, lw=1.6)
ax.set(title="dimensionality reduction means keeping PC1 and throwing PC2 away", xlabel="PC1", ylabel="PC2", aspect="equal")
ax.legend(fontsize=9, loc="upper left")
ax.annotate("the grey lines are what gets lost.\nshort lines mean little was lost.", xy=(0.02, 0.03), xycoords="axes fraction", fontsize=10, color="#333", bbox=TEXT_BOX)
save_figure(figure, "00_concept_eigenvectors.png")

# why reducing dimensions matters at all
figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))

column_counts = np.arange(1, 31)
axes[0].plot(column_counts, 2.0 ** column_counts, color=CORAL, lw=2.5)
axes[0].set(title="the curse of dimensionality", xlabel="number of columns, or dimensions", ylabel="corners of the data space", yscale="log")
axes[0].annotate("each extra column doubles the space\nthe same rows have to fill", xy=(0.04, 0.82), xycoords="axes fraction", fontsize=9.5, color="#333")

distance_generator = np.random.default_rng(1)
dimension_colours = [TEAL, GOLD, CORAL]
for position, number_of_dimensions in enumerate([2, 10, 50]):
    random_points = distance_generator.normal(0, 1, (400, number_of_dimensions))
    distances = np.sqrt(((random_points[:200] - random_points[200:]) ** 2).sum(axis=1))
    distances = distances / distances.mean()
    axes[1].hist(distances, bins=30, alpha=0.55, color=dimension_colours[position], label="{} dimensions, spread {:.2f}".format(number_of_dimensions, distances.std()))
axes[1].set(title="distances stop being meaningful", xlabel="distance divided by the average distance", ylabel="number of pairs")
axes[1].legend(fontsize=9)

axes[2].plot(column_counts[1:], column_counts[1:] * (column_counts[1:] - 1) / 2, color=NAVY, lw=2.5)
axes[2].set(title="too many charts to look at", xlabel="number of columns", ylabel="scatter plots needed to see every pair")
axes[2].annotate("9 columns already needs 36 charts.\nPCA replaces them with 2 or 3.", xy=(0.04, 0.82), xycoords="axes fraction", fontsize=9.5, color="#333")

figure.suptitle("3 reasons that high dimensional data is hard, and why PCA helps us", fontweight="bold", fontsize=14, y=1.03)
save_figure(figure, "00_concept_challenges.png")

print("\ndone, images in images/pca/, tables in data/clean/pca_*.csv")
