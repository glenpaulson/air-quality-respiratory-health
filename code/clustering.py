"""
CLUSTERING - unsupervised grouping of the 50 study cities we did in part 1
  1. load the cleaned data and build ONE row per city, a yearly air + weather profile
  2. standardise that table so that every column will sit on the same scale 
  3. run k means for k = 2, 3, 4 and score every k with the Silhouette method
  4. run hierarchical clustering using COSINE distance, and draw the dendrogram
  5. compare the k means and hclust labels against each other
  6. profile the winning clusters and map them
  7. draw the explanatory diagrams
output:  images/clustering/*.png + data/clean/clustering_*.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, set_link_color_palette
from scipy.spatial.distance import pdist, squareform

# file locations
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_FOLDER = os.path.join(THIS_FOLDER, "..", "data", "clean")
CLEAN_DATA_FILE = os.path.join(CLEAN_DATA_FOLDER, "air_quality_health_clean.csv")
IMAGE_FOLDER = os.path.join(THIS_FOLDER, "..", "images", "clustering")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
TEAL = "#0e7c7b"
CORAL = "#ef6f53"
NAVY = "#2b4162"
CLUSTER_COLORS = ["#0e7c7b", "#ef6f53", "#2b4162", "#e8b004", "#7b5aa6", "#4a9b5e"]
RANDOM_SEED = 76

def save_figure(figure, file_name):
    figure.tight_layout()
    figure.savefig(os.path.join(IMAGE_FOLDER, file_name), bbox_inches="tight")
    plt.close(figure)
    print("saved", file_name)

def save_table_image(table, title, file_name, note=None, number_of_rows=6, figure_width=None):
    """take the first few rows of a dataframe and save the"""
    small_table = table.head(number_of_rows).copy()
    # round the decimals and turn dates into plain text for more readability on the website
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

def percentile_95(column):
    """the value that 95 percent of the days fall below, so we can see how bad the worst days get"""
    return column.quantile(0.95)

def find_best_k(k_values, scores):
    """return the k that got the highest score"""
    return k_values[int(np.argmax(scores))]

# ----------------------------------------------------------------------------------
# step 1: we need to build one unlabelled numeric row per city

daily_data = pd.read_csv(CLEAN_DATA_FILE, parse_dates=["date"])
print("loaded {:,} city days, {} cities".format(len(daily_data), daily_data.city.nunique()))

# before picture: raw daily rows, still carrying text columns and labels
BEFORE_COLUMNS = ["city", "state", "date", "pm2_5", "ozone", "us_aqi_max", "temperature_2m_mean", "wind_speed_10m_max", "aqi_category"]
save_table_image(daily_data[BEFORE_COLUMNS],
                 "Before: daily rows, text columns and labels still present (9 of 28 columns)",
                 "01_before_daily_rows.png",
                 note="clustering cannot use city / state / date / aqi_category: they are text or labels and not measurements")

# one row per city: the yearly mean of each measurement, plus the 95th percentile of PM2.5 so that we can capture how bad the worst days get
city_table = daily_data.groupby("city").agg(
    pm2_5=("pm2_5", "mean"),
    pm25_p95=("pm2_5", percentile_95),
    pm10=("pm10", "mean"),
    ozone=("ozone", "mean"),
    no2=("nitrogen_dioxide", "mean"),
    so2=("sulphur_dioxide", "mean"),
    co=("carbon_monoxide", "mean"),
    aqi=("us_aqi_max", "mean"),
    temp=("temperature_2m_mean", "mean"),
    precip=("precipitation_sum", "mean"),
    wind=("wind_speed_10m_max", "mean"),
).round(3)

FEATURE_NAMES = list(city_table.columns)
number_of_cities = len(city_table)
print("city matrix: {} rows x {} features: {}".format(city_table.shape[0], city_table.shape[1], FEATURE_NAMES))

# this is held out and is never given to the clustering algorithms, so that we can check
# them afterwards and see if the discovered groups mean anything
held_out_info = daily_data.groupby("city").agg(
    state=("state", "first"),
    lat=("latitude", "first"),
    lon=("longitude", "first"),
    asthma=("asthma_prev", "first"),
    copd=("copd_prev", "first"),
)

# ----------------------------------------------------------------------------------
# step 2: here we need to do two transformations before we can start calculating the distances
# 1. since some cities have very long right tails, we will take the log
# 2. take the z scores

SKEWED_COLUMNS = ["pm2_5", "pm25_p95", "pm10", "no2", "so2", "co"]
print("skew before log: {}".format(city_table[SKEWED_COLUMNS].skew().round(2).to_dict()))

logged_table = city_table.copy()
logged_table[SKEWED_COLUMNS] = np.log1p(logged_table[SKEWED_COLUMNS]) # log1p = log(1 + x)
print("skew after log: {}".format(logged_table[SKEWED_COLUMNS].skew().round(2).to_dict()))

scaler = StandardScaler()
scaled_data = scaler.fit_transform(logged_table.values) # mean 0, std 1 for each column
scaled_table = pd.DataFrame(scaled_data, index=city_table.index, columns=FEATURE_NAMES).round(3)

save_table_image(city_table.reset_index(),
                 "After (step 1/3): one row per city, 11 raw numeric features",
                 "02_after_city_means.png",
                 note="text and labels are gone, but the columns have different scales, for e.g. CO ~ 200, wind ~ 15")
save_table_image(logged_table.reset_index().round(3),
                 "After (step 2/3): log transforming the 6 long tailed pollutant columns",
                 "02b_after_log.png",
                 note="skew for PM2.5 drops from 1.72 to 0.54, this way no extremes can dominate distance")
save_table_image(scaled_table.reset_index(),
                 "After (step 3/3): standardised to z scores",
                 "03_after_standardised.png",
                 note="every column is now standardized to mean 0 and std 1. This matrix is what we use for k means and hclust")

city_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "clustering_city_profiles.csv"))
scaled_table.to_csv(os.path.join(CLEAN_DATA_FOLDER, "clustering_city_standardised.csv"))

# ----------------------------------------------------------------------------------
# Step 3: how many clusters? elbow (wcss) + silhouette

# only 50 cities, so anything past k = 8 starts producing single city "clusters" so we will cap
K_VALUES_TO_TRY = list(range(2, 9))
wcss_scores = []
silhouette_kmeans = []
silhouette_hclust = []

cosine_distances = squareform(pdist(scaled_data, metric="cosine"))      # cosine distance matrix

for k in K_VALUES_TO_TRY:
    kmeans_model = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_SEED).fit(scaled_data)
    wcss_scores.append(kmeans_model.inertia_)
    silhouette_kmeans.append(silhouette_score(scaled_data, kmeans_model.labels_))

    hclust_model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="complete").fit(scaled_data)
    silhouette_hclust.append(silhouette_score(cosine_distances, hclust_model.labels_, metric="precomputed"))

best_k_kmeans = find_best_k(K_VALUES_TO_TRY, silhouette_kmeans)
best_k_hclustering = find_best_k(K_VALUES_TO_TRY, silhouette_hclust)
print("best k by silhouette: k means: {} ({:.3f}), hclust/cosine: {} ({:.3f})".format(best_k_kmeans, max(silhouette_kmeans), best_k_hclustering, max(silhouette_hclust)))

figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].plot(K_VALUES_TO_TRY, wcss_scores, "o-", color=TEAL, lw=2)
axes[0].set(title="elbow method: wcss", xlabel="number of clusters k", ylabel="wcss")
axes[1].plot(K_VALUES_TO_TRY, silhouette_kmeans, "o-", color=TEAL, lw=2, label="k means: euclidean")
axes[1].plot(K_VALUES_TO_TRY, silhouette_hclust, "s--", color=CORAL, lw=2, label="hclust: cosine")
axes[1].axvline(best_k_kmeans, color=TEAL, ls=":", alpha=0.7)
axes[1].set(title="silhouette method", xlabel="number of clusters k", ylabel="avg silhouette score")
axes[1].legend()
figure.suptitle("choosing the k for the 50 city air quality profiles", fontweight="bold", y=1.03)
save_figure(figure, "04_choosing_k.png")

# silhouette plot for the optimal k: one bar for every city
best_kmeans_model = KMeans(n_clusters=best_k_kmeans, n_init=25, random_state=RANDOM_SEED).fit(scaled_data)
city_silhouette_scores = silhouette_samples(scaled_data, best_kmeans_model.labels_)

figure, ax = plt.subplots(figsize=(8, 6))
bar_position = 0
for cluster_number in range(best_k_kmeans):
    is_in_this_cluster = best_kmeans_model.labels_ == cluster_number
    scores_in_this_cluster = np.sort(city_silhouette_scores[is_in_this_cluster])
    bar_slots = np.arange(bar_position, bar_position + len(scores_in_this_cluster))

    ax.barh(bar_slots, scores_in_this_cluster, height=1.0, color=CLUSTER_COLORS[cluster_number], edgecolor="white", linewidth=0.4)
    ax.text(-0.04, bar_position + len(scores_in_this_cluster) / 2, "C{}".format(cluster_number + 1), va="center", ha="right", fontweight="bold")
    bar_position = bar_position + len(scores_in_this_cluster) + 3

average_silhouette = city_silhouette_scores.mean()
ax.axvline(average_silhouette, color=CORAL, ls="--", lw=2, label="average = {:.3f}".format(average_silhouette))
ax.set(title="silhouette plot, kmeans with k = {}".format(best_k_kmeans),
       xlabel="silhouette score per city: 1 is well placed, 0 means on a border, below 0 is probably misplaced",
       yticks=[])
ax.legend(loc="lower right")
save_figure(figure, "05_silhouette_detail.png")

# ----------------------------------------------------------------------------------
# Step 4: k means at 3 different values of k, drawn in 2D with PCA
# we cannot draw 11 columns on a flat page, so PCA bring them down to 2

pca_model = PCA(n_components=2, random_state=RANDOM_SEED)
city_points_2d = pca_model.fit_transform(scaled_data)
variance_percent = pca_model.explained_variance_ratio_ * 100
print("the 2d view is keeping {:.1f}% of the variance (PC1 {:.1f}%, PC2 {:.1f}%)".format(variance_percent.sum(), variance_percent[0], variance_percent[1]))

K_VALUES_TO_COMPARE = [2, 3, 4]
kmeans_labels = {} # we will save the labels for each k so we can reuse them later

figure, axes = plt.subplots(1, 3, figsize=(17, 5.4))
for panel_number in range(len(K_VALUES_TO_COMPARE)):
    k = K_VALUES_TO_COMPARE[panel_number]
    ax = axes[panel_number]

    kmeans_model = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_SEED).fit(scaled_data)
    kmeans_labels[k] = kmeans_model.labels_

    # draw the cities, giving one color for each cluster
    for cluster_number in range(k):
        is_in_this_cluster = kmeans_model.labels_ == cluster_number
        ax.scatter(city_points_2d[is_in_this_cluster, 0], city_points_2d[is_in_this_cluster, 1],
                   s=90, color=CLUSTER_COLORS[cluster_number], edgecolor="white", linewidth=1.2,
                   label="cluster {} (n={})".format(cluster_number + 1, is_in_this_cluster.sum()), zorder=3)

    # the centroids are in 11 dimensions too, so they need squashing to 2 as well
    centroids_2d = pca_model.transform(kmeans_model.cluster_centers_)
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", s=240, c="black", zorder=4, label="centroid")

    # keeping the city name next to each dot
    for city_number in range(number_of_cities):
        city_name = city_table.index[city_number]
        ax.annotate(city_name, (city_points_2d[city_number, 0], city_points_2d[city_number, 1]),
                    fontsize=5.5, alpha=0.65, xytext=(3, 3), textcoords="offset points")

    if k == 2:
        y_label = "PC2:{:.0f}%".format(variance_percent[1])
    else:
        y_label = ""
    ax.set(title="k = {}: silhouette {:.3f}".format(k, silhouette_score(scaled_data, kmeans_model.labels_)),
           xlabel="PC1: {:.0f}% of variance".format(variance_percent[0]), ylabel=y_label)
    ax.legend(fontsize=7, loc="best")

figure.suptitle("k means on our 50 city air quality profiles, with 3 different values of k", fontweight="bold", fontsize=15, y=1.02)
save_figure(figure, "06_kmeans_k2_k3_k4.png")

# ----------------------------------------------------------------------------------
# Step 5: hierarchical clustering, using cosine distance

tree = linkage(scaled_data, method="complete", metric="cosine")

# the way we were taught to read a dendrogram: cut it in a TALL GAP. the gap between two
# consecutive merge heights is a band where the cut can sit without crossing anything, so
# the tallest band near the top of the tree is the most stable number of clusters.
merge_heights = tree[:, 2]
print("\nreading the dendrogram, gaps between the top merge heights:")

gap_for_k = {}
for step_from_top in range(1, 6):
    top_of_gap = merge_heights[-step_from_top]
    bottom_of_gap = merge_heights[-step_from_top - 1]
    gap = top_of_gap - bottom_of_gap
    clusters_if_we_cut_here = step_from_top + 1
    gap_for_k[clusters_if_we_cut_here] = gap
    print("  height {:.3f} down to {:.3f}: gap {:.3f} gives {} clusters if we cut here".format(top_of_gap, bottom_of_gap, gap, clusters_if_we_cut_here))

# find which number of clusters has the tallest gap
k_from_tall_gap = max(gap_for_k, key=gap_for_k.get)
print("the tallest gap is {:.3f}, so the dendrogram suggests k = {}".format(gap_for_k[k_from_tall_gap], k_from_tall_gap))

# two cuts are drawn, because the two rules we were taught disagree on this data:
# silhouette: the rule for k means, says k=2, the tall gap: the rule for trees says k=3
silhouette_cut_height = (merge_heights[-best_k_hclustering] + merge_heights[-(best_k_hclustering - 1)]) / 2
tall_gap_cut_height = (merge_heights[-k_from_tall_gap] + merge_heights[-(k_from_tall_gap - 1)]) / 2

figure, ax = plt.subplots(figsize=(14, 6))
dendrogram(tree, labels=city_table.index.tolist(), leaf_rotation=90, leaf_font_size=8,
           color_threshold=tall_gap_cut_height, above_threshold_color="#9aa0a6", ax=ax)
ax.axhspan(merge_heights[-k_from_tall_gap], merge_heights[-(k_from_tall_gap - 1)], color=TEAL, alpha=0.10, zorder=0)
ax.axhline(tall_gap_cut_height, color=TEAL, ls="--", lw=2,
           label="tall gap cut: gap {:.3f} gives {} clusters".format(gap_for_k[k_from_tall_gap], k_from_tall_gap))
ax.axhline(silhouette_cut_height, color=CORAL, ls=":", lw=2,
           label="silhouette cut: gap {:.3f} gives {} clusters".format(gap_for_k[best_k_hclustering], best_k_hclustering))
ax.set(title="hierarchical clustering of the 50 cities: cosine distance, complete linkage",
       ylabel="cosine distance at which the groups are merging")
ax.legend(loc="center left")
ax.grid(False)
save_figure(figure, "07_dendrogram_cosine.png")

# cut the tree at each k for doing comparison against the k means
hclust_labels = {}
for k in K_VALUES_TO_COMPARE:
    hclust_labels[k] = fcluster(tree, k, criterion="maxclust") - 1

# ----------------------------------------------------------------------------------
# Step 6: are the methods agreeing or not?
# the cluster numbers are arbitrary, so k means "cluster 1" and hclust "cluster 1" are not
# exactly the same group. to count agreement we try every way of matching one method's labels onto the other's and keep the best

def count_cities_grouped_the_same(labels_one, labels_two, number_of_clusters):
    """how many of the 50 cities land in the same group under both methods"""
    counts = pd.crosstab(labels_one, labels_two).values
    totals = []
    for matching in permutations(range(number_of_clusters)):
        totals.append(sum(counts[row][matching[row]] for row in range(number_of_clusters)))
    return max(totals)

for k in K_VALUES_TO_COMPARE:
    cities_agreeing = count_cities_grouped_the_same(kmeans_labels[k], hclust_labels[k], k)
    print("at k={}: {} of {} cities land in the same group under both our methods".format(k, cities_agreeing, number_of_cities))

# look at k = 2 in detail, because that is where the two methods agree the most
K_TO_COMPARE_IN_DETAIL = 2
kmeans_column = pd.Series(kmeans_labels[K_TO_COMPARE_IN_DETAIL] + 1, index=city_table.index, name="k means cluster")
hclust_column = pd.Series(hclust_labels[K_TO_COMPARE_IN_DETAIL] + 1, index=city_table.index, name="hclust: cosine cluster")
comparison_table = pd.crosstab(kmeans_column, hclust_column)
cities_agreeing = count_cities_grouped_the_same(kmeans_labels[K_TO_COMPARE_IN_DETAIL], hclust_labels[K_TO_COMPARE_IN_DETAIL], K_TO_COMPARE_IN_DETAIL)
print(comparison_table)

# work out which way round the two methods' cluster numbers line up best
def total_on_the_diagonal(matching):
    return sum(comparison_table.values[row][matching[row]] for row in range(K_TO_COMPARE_IN_DETAIL))

best_matching = max(permutations(range(K_TO_COMPARE_IN_DETAIL)), key=total_on_the_diagonal)

# name the cities that are disagreed by both our methods, basically are the borderline ones
disagreeing_cities = []
for city_number in range(number_of_cities):
    kmeans_cluster = kmeans_labels[K_TO_COMPARE_IN_DETAIL][city_number]
    hclust_cluster = hclust_labels[K_TO_COMPARE_IN_DETAIL][city_number]
    if best_matching[kmeans_cluster] != hclust_cluster:
        disagreeing_cities.append(city_table.index[city_number])
print("the {} cities the two methods disagree about: {}".format(len(disagreeing_cities), ", ".join(disagreeing_cities)))

figure, ax = plt.subplots(figsize=(6.4, 5))
sns.heatmap(comparison_table, annot=True, fmt="d", cmap="mako", cbar_kws={"label": "number of cities"},
            linewidths=1.5, linecolor="white", ax=ax)
ax.set(title="k means vs hierarchical at k = {}\n{} of {} cities grouped the same way".format(K_TO_COMPARE_IN_DETAIL, cities_agreeing, number_of_cities))
save_figure(figure, "09_kmeans_vs_hclust.png")

# ----------------------------------------------------------------------------------
# Step 7: profile heat map, map, and the health cross check, helps us explain what is each cluster
# the silhouette prefers k=2, but k=3 is the version worth describing on the website:
# it splits the cleaner half along a climate line, which k=2 hides.

FINAL_K = 3
final_labels = kmeans_labels[FINAL_K]

# save every city with all of its cluster labels, so that w can check the results
city_results = held_out_info.copy()
city_results["kmeans_k2"] = kmeans_labels[2] + 1
city_results["kmeans_k3"] = kmeans_labels[3] + 1
city_results["kmeans_k4"] = kmeans_labels[4] + 1
city_results["hclust_cosine_k3"] = hclust_labels[3] + 1
city_results = city_results.join(city_table)
city_results.to_csv(os.path.join(CLEAN_DATA_FOLDER, "clustering_city_clusters.csv"))

# average z score of every feature inside every cluster
profile_table = scaled_table.copy()
profile_table["cluster"] = final_labels + 1
cluster_averages = profile_table.groupby("cluster")[FEATURE_NAMES].mean()

figure, ax = plt.subplots(figsize=(11, 3.4))
sns.heatmap(cluster_averages, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            cbar_kws={"label": "standard deviations from the 50 city average"},
            linewidths=1, linecolor="white", ax=ax)
ax.set(title="What makes each cluster different: k means, k = {}".format(FINAL_K), ylabel="cluster", xlabel="")
save_figure(figure, "10_cluster_profiles.png")

# put the clusters on a map, using longitude and latitude that the model never saw
figure, ax = plt.subplots(figsize=(11, 6.2))
for cluster_number in range(FINAL_K):
    is_in_this_cluster = final_labels == cluster_number
    ax.scatter(held_out_info.lon[is_in_this_cluster], held_out_info.lat[is_in_this_cluster],
               s=120, color=CLUSTER_COLORS[cluster_number], edgecolor="white", linewidth=1.2,
               label="cluster {}".format(cluster_number + 1), zorder=3)

for city_number in range(number_of_cities):
    city_name = city_table.index[city_number]
    ax.annotate(city_name, (held_out_info.lon.iloc[city_number], held_out_info.lat.iloc[city_number]),
                fontsize=6, alpha=0.7, xytext=(4, 3), textcoords="offset points")

ax.set(title="Where the clusters sit on the map: k means, k = {}".format(FINAL_K), xlabel="longitude", ylabel="latitude")
ax.legend()
save_figure(figure, "11_cluster_map.png")

# the health cross check: we never gave asthma and copd to the clustering
health_table = city_results.copy()
health_table["cluster"] = final_labels + 1

HEALTH_COLUMNS = ["asthma", "copd"]
HEALTH_LABELS = ["Adult asthma prevalence in %", "Adult COPD prevalence in %"]

np.random.seed(RANDOM_SEED) # to keep the results generated consistent

figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for panel_number in range(len(HEALTH_COLUMNS)):
    ax = axes[panel_number]
    column_name = HEALTH_COLUMNS[panel_number]
    axis_label = HEALTH_LABELS[panel_number]

    sns.boxplot(data=health_table, x="cluster", y=column_name, hue="cluster", legend=False,
                palette=CLUSTER_COLORS[:FINAL_K], ax=ax, width=0.6)
    sns.stripplot(data=health_table, x="cluster", y=column_name, color="black", size=4, alpha=0.6, ax=ax)
    ax.set(xlabel="cluster", ylabel=axis_label, title=axis_label.split(" (")[0])

figure.suptitle("we held out the helath data during clustering, and here we're cross checking the same", fontweight="bold", y=1.02)
save_figure(figure, "12_cluster_vs_health.png")

# cluster summaries
for cluster_number in range(FINAL_K):
    is_in_this_cluster = final_labels == cluster_number
    member_cities = city_table.index[is_in_this_cluster]
    sorted_features = cluster_averages.loc[cluster_number + 1].sort_values()

    if len(member_cities) > 8:
        trailing_dots = " ..."
    else:
        trailing_dots = ""

    highest_three = list(sorted_features.tail(3).index)
    highest_three.reverse()
    lowest_three = list(sorted_features.head(3).index)

    print("\ncluster {} ({} cities): {}{}".format(cluster_number + 1, len(member_cities), ", ".join(member_cities[:8]), trailing_dots))
    print("highest: {} | lowest: {}".format(", ".join(highest_three), ", ".join(lowest_three)))
    print("mean asthma {:.2f}%  mean COPD {:.2f}%".format(health_table.asthma[is_in_this_cluster].mean(), health_table.copd[is_in_this_cluster].mean()))

# ----------------------------------------------------------------------------------
# Step 8: clustering individual city days instead of whole cities.
# a yearly average hides every bad day. clustering single days helps us see if there are
# recurring kinds of day, and do they match the EPA categories?

DAY_FEATURES = ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
DAY_SKEWED_COLUMNS = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "precipitation_sum"]

day_sample = daily_data.sample(n=3000, random_state=RANDOM_SEED).copy() # 3,000 of the 18,300 rows
day_features_table = day_sample[DAY_FEATURES].copy()
day_features_table[DAY_SKEWED_COLUMNS] = np.log1p(day_features_table[DAY_SKEWED_COLUMNS])
scaled_days = StandardScaler().fit_transform(day_features_table.values)

silhouette_days = []
for k in K_VALUES_TO_TRY:
    day_kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(scaled_days)
    silhouette_days.append(silhouette_score(scaled_days, day_kmeans_model.labels_))

best_k_days = find_best_k(K_VALUES_TO_TRY, silhouette_days)
print("\nday level: best k by silhouette = {} ({:.3f})".format(best_k_days, max(silhouette_days)))

# we also run hierarchical clustering on the days, so we can apply the same tall gap rule
# we used for the cities, and see whether it backs up the number of clusters we pick
day_tree = linkage(scaled_days, method="complete", metric="cosine")
day_cosine_distances = squareform(pdist(scaled_days, metric="cosine"))

silhouette_days_hclust = []
for k in K_VALUES_TO_TRY:
    day_hclust_labels = fcluster(day_tree, k, criterion="maxclust")
    silhouette_days_hclust.append(silhouette_score(day_cosine_distances, day_hclust_labels, metric="precomputed"))

best_k_days_hclust = find_best_k(K_VALUES_TO_TRY, silhouette_days_hclust)
print("day level: best k for hclust by silhouette = {} ({:.3f})".format(best_k_days_hclust, max(silhouette_days_hclust)))

day_merge_heights = day_tree[:, 2]
day_gap_for_k = {}
print("the tall gap test on the day tree:")
for step_from_top in range(1, 7):
    gap = day_merge_heights[-step_from_top] - day_merge_heights[-step_from_top - 1]
    day_gap_for_k[step_from_top + 1] = gap
    print("  height {:.3f} down to {:.3f}: gap {:.3f} gives {} clusters if we cut here".format(day_merge_heights[-step_from_top], day_merge_heights[-step_from_top - 1], gap, step_from_top + 1))

k_from_day_tall_gap = max(day_gap_for_k, key=day_gap_for_k.get)
print("the tallest gap on the day tree is only {:.3f} (the city tree had {:.3f}),".format(max(day_gap_for_k.values()), max(gap_for_k.values())))
print("so with 3,000 points every top merge happens at almost the same height and this rule stops being useful.")

# since neither silhouette nor the tall gap settles it, we judge the groups against the EPA
# category instead, which the model never saw. a good k should produce a group of days that
# are almost all safe to breathe.
WORSE_THAN_MODERATE = ["Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]

def bad_day_rate_of_cleanest_cluster(cluster_labels, categories):
    """in whichever cluster is cleanest, what percent of its days were worse than moderate"""
    checks = pd.DataFrame({"cluster": cluster_labels, "category": list(categories)})
    checks["is_a_bad_day"] = checks.category.isin(WORSE_THAN_MODERATE)
    rate_per_cluster = checks.groupby("cluster").is_a_bad_day.mean()
    return rate_per_cluster.min() * 100

cleanest_cluster_bad_rates = []
for k in K_VALUES_TO_COMPARE:
    day_kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(scaled_days)
    rate = bad_day_rate_of_cleanest_cluster(day_kmeans_model.labels_, day_sample.aqi_category)
    cleanest_cluster_bad_rates.append(rate)
    print("k={}: the cleanest day cluster is {:.1f}% bad days".format(k, rate))

day_pca_model = PCA(n_components=2, random_state=RANDOM_SEED)
day_points_2d = day_pca_model.fit_transform(scaled_days)
day_variance_percent = day_pca_model.explained_variance_ratio_ * 100

figure, axes = plt.subplots(1, 4, figsize=(20, 4.8))

# the silhouette curve for days
axes[0].plot(K_VALUES_TO_TRY, silhouette_days, "o-", color=TEAL, lw=2)
axes[0].axvline(best_k_days, color=CORAL, ls="--", lw=1.5)
axes[0].set(title="Silhouette, 3,000 city days", xlabel="number of clusters k", ylabel="average silhouette score")

# k means at k = 2, 3 and 4
day_labels = {}
for panel_number in range(len(K_VALUES_TO_COMPARE)):
    k = K_VALUES_TO_COMPARE[panel_number]
    ax = axes[panel_number + 1]

    day_kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(scaled_days)
    day_labels[k] = day_kmeans_model.labels_

    for cluster_number in range(k):
        is_in_this_cluster = day_kmeans_model.labels_ == cluster_number
        ax.scatter(day_points_2d[is_in_this_cluster, 0], day_points_2d[is_in_this_cluster, 1],
                   s=9, alpha=0.55, color=CLUSTER_COLORS[cluster_number],
                   label="cluster {} (n={})".format(cluster_number + 1, is_in_this_cluster.sum()))

    centroids_2d = day_pca_model.transform(day_kmeans_model.cluster_centers_)
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", s=200, c="black", zorder=4)
    ax.set(title="k = {}   (silhouette {:.3f})".format(k, silhouette_score(scaled_days, day_kmeans_model.labels_)),
           xlabel="PC1 ({:.0f}%)".format(day_variance_percent[0]),
           ylabel="PC2 ({:.0f}%)".format(day_variance_percent[1]))
    ax.legend(fontsize=7, markerscale=2)

figure.suptitle("clustering individual city days, three values of k", fontweight="bold", fontsize=15, y=1.03)
save_figure(figure, "13_days_kmeans.png")

# figure 1 shows us both silhouettes agreeing on 2, and figure 2 shows the tall gap rule breaking
# down on 3000 points, and figure 3 shows the test that actually decides it
figure, axes = plt.subplots(1, 3, figsize=(17, 4.6))

axes[0].plot(K_VALUES_TO_TRY, silhouette_days, "o-", color=TEAL, lw=2, label="k means: euclidean")
axes[0].plot(K_VALUES_TO_TRY, silhouette_days_hclust, "s--", color=CORAL, lw=2, label="hclust: cosine")
axes[0].axvline(best_k_days, color=NAVY, ls=":", lw=1.8)
axes[0].set(title="1. both silhouettes prefer k = 2", xlabel="number of clusters k", ylabel="average silhouette score")
axes[0].legend(fontsize=9)

city_gap_values = [gap_for_k[k] for k in sorted(gap_for_k)]
day_gap_values = [day_gap_for_k[k] for k in sorted(day_gap_for_k)[:len(city_gap_values)]]
bar_slots = np.arange(len(city_gap_values))
axes[1].bar(bar_slots - 0.2, city_gap_values, width=0.4, color=TEAL, edgecolor="white", label="50 cities")
axes[1].bar(bar_slots + 0.2, day_gap_values, width=0.4, color=CORAL, edgecolor="white", label="3,000 days")
axes[1].set_xticks(bar_slots)
axes[1].set_xticklabels([str(k) for k in sorted(gap_for_k)])
axes[1].set(title="2. the tall gap rule stops working on days", xlabel="clusters we would get by cutting there", ylabel="height of the gap")
axes[1].legend(fontsize=9)

bar_slots = np.arange(len(K_VALUES_TO_COMPARE))
bars = axes[2].bar(bar_slots, cleanest_cluster_bad_rates, width=0.55, color=[CORAL, TEAL, CORAL], edgecolor="white")
for bar_number in range(len(bars)):
    axes[2].annotate("{:.1f}%".format(cleanest_cluster_bad_rates[bar_number]), (bar_slots[bar_number], cleanest_cluster_bad_rates[bar_number]), ha="center", va="bottom", fontweight="bold")
axes[2].set_xticks(bar_slots)
axes[2].set_xticklabels(["k = {}".format(k) for k in K_VALUES_TO_COMPARE])
axes[2].set(title="3. so we judge the groups on the held out EPA label", xlabel="", ylabel="percent bad days in the cleanest cluster")

figure.suptitle("why the day clustering settles on 3 groups", fontweight="bold", fontsize=15, y=1.04)
save_figure(figure, "13b_days_choosing_k.png")

# compare the day clusters against the EPA category that our model never saw
# we go with 3 day clusters even though both silhouettes said 2. the tall gap rule could not
# settle it either, so the deciding test was the held out EPA label: at k = 3 the cleanest
# cluster is only 0.6% bad days, against 4.8% at k = 2, because k = 2 lumps the calm clean
# days in with the hot ozone days instead of separating them.
DAY_K = 3
day_sample["cluster"] = day_labels[DAY_K] + 1
day_vs_category_table = pd.crosstab(day_sample.cluster, day_sample.aqi_category)

# put the EPA categories in their proper order
CATEGORY_ORDER = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
columns_in_order = [name for name in CATEGORY_ORDER if name in day_vs_category_table.columns]
day_vs_category_table = day_vs_category_table[columns_in_order]

print("\nday clusters vs EPA AQI category, clusters never saw the category:")
print(day_vs_category_table)

scaled_days_table = pd.DataFrame(scaled_days, columns=DAY_FEATURES)
scaled_days_table["cluster"] = day_labels[DAY_K] + 1
day_cluster_averages = scaled_days_table.groupby("cluster")[DAY_FEATURES].mean()

figure, axes = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

sns.heatmap(day_vs_category_table, annot=True, fmt="d", cmap="mako", linewidths=1, linecolor="white",
            cbar_kws={"label": "number of days"}, ax=axes[0])
axes[0].set(title="day clusters vs the EPA AQI category they were never shown",
            xlabel="EPA AQI category", ylabel="cluster")

# shorten the label
shortened_labels = []
for tick in axes[0].get_xticklabels():
    shortened_labels.append(tick.get_text().replace(" for Sensitive Groups", "\nfor Sens."))
axes[0].set_xticklabels(shortened_labels, fontsize=8)

sns.heatmap(day_cluster_averages, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=1, linecolor="white", cbar_kws={"label": "std devs from average"}, ax=axes[1])
axes[1].set(title="what each day cluster looks like for k = {}".format(DAY_K), xlabel="", ylabel="cluster")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right", fontsize=8)
save_figure(figure, "14_days_meaning.png")

day_output_columns = ["city", "date"] + DAY_FEATURES + ["aqi_category", "cluster"]
day_sample[day_output_columns].to_csv(os.path.join(CLEAN_DATA_FOLDER, "clustering_day_clusters.csv"), index=False)

# ----------------------------------------------------------------------------------
# Step 9: example diagrams for the overview

# (a) three distance measures on the same two points
point_a = np.array([1.0, 4.0])
point_b = np.array([5.0, 1.0])

euclidean_distance = np.linalg.norm(point_a - point_b)
manhattan_distance = np.abs(point_a - point_b).sum()
cosine_similarity = np.dot(point_a, point_b) / (np.linalg.norm(point_a) * np.linalg.norm(point_b))
angle_in_degrees = np.degrees(np.arccos(cosine_similarity))

figure, ax = plt.subplots(figsize=(7.4, 5.4))
ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color=TEAL, lw=3,
        label="Euclidean = {:.1f}  (straight line)".format(euclidean_distance))
ax.plot([point_a[0], point_b[0], point_b[0]], [point_a[1], point_a[1], point_b[1]], color=CORAL, lw=3, ls="--",
        label="Manhattan = {:.0f}  (along the grid)".format(manhattan_distance))

# draw an arrow from the origin to each point, because cosine is about the angle between them
ax.annotate("", xy=point_a, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.2))
ax.annotate("", xy=point_b, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=NAVY, lw=2.2))
ax.plot([], [], color=NAVY, lw=2.2,
        label="Cosine distance = {:.2f}  (angle = {:.0f} deg)".format(1 - cosine_similarity, angle_in_degrees))

ax.scatter(point_a[0], point_a[1], s=130, color=NAVY, zorder=4)
ax.annotate("A (1, 4)", point_a, xytext=(8, 8), textcoords="offset points", fontweight="bold")
ax.scatter(point_b[0], point_b[1], s=130, color=NAVY, zorder=4)
ax.annotate("B (5, 1)", point_b, xytext=(8, 8), textcoords="offset points", fontweight="bold")

ax.annotate("Euclidean and Manhattan measure position.\n"
            "Cosine measures direction, so two cities with\n"
            "the same pollution mix at different levels\n"
            "still count as similar.",
            xy=(1.45, 0.35), fontsize=9.5, color="#333")
ax.set(title="3 ways to measure how far apart the 2 rows are",
       xlim=(-0.3, 6.6), ylim=(-0.3, 5.4), xlabel="feature 1", ylabel="feature 2")
ax.legend(loc="upper right", fontsize=9)
save_figure(figure, "00_concept_distance.png")

# (b) partitional vs hierarchical, using one tiny 9 point dataset for both panels
random_generator = np.random.default_rng(7)
example_points = np.vstack([random_generator.normal([2, 4], 0.4, (3, 2)),random_generator.normal([5, 4.4], 0.4, (3, 2)),random_generator.normal([3.6, 1.6], 0.4, (3, 2))])

example_names = ["p{}".format(point_number + 1) for point_number in range(len(example_points))]
example_kmeans = KMeans(n_clusters=3, n_init=25, random_state=RANDOM_SEED).fit(example_points)
figure, axes = plt.subplots(1, 2, figsize=(13, 5))

# k means, every point joined to its centroid
ax = axes[0]
for cluster_number in range(3):
    is_in_this_cluster = example_kmeans.labels_ == cluster_number
    ax.scatter(example_points[is_in_this_cluster, 0], example_points[is_in_this_cluster, 1],
               s=110, color=CLUSTER_COLORS[cluster_number], edgecolor="white", linewidth=1.2, zorder=3)

    centre_x = example_kmeans.cluster_centers_[cluster_number][0]
    centre_y = example_kmeans.cluster_centers_[cluster_number][1]
    ax.scatter(centre_x, centre_y, marker="X", s=240, color="black", zorder=4)

    # draw a line from every point in this cluster to its centroid
    for point_number in range(len(example_points)):
        if is_in_this_cluster[point_number]:
            ax.plot([example_points[point_number, 0], centre_x],
                    [example_points[point_number, 1], centre_y],
                    color=CLUSTER_COLORS[cluster_number], lw=1, alpha=0.5, zorder=2)

for point_number in range(len(example_names)):
    if point_number % 2 == 1:
        label_offset = 5
    else:
        label_offset = -11
    ax.annotate(example_names[point_number], example_points[point_number],
                xytext=(8, label_offset), textcoords="offset points", fontsize=9)

ax.scatter([], [], marker="X", s=160, color="black", label="centroid")
ax.set(title="Partitional: k means\nchoose k up front, each point joins its nearest centroid",
       xlim=(0.8, 6.2), ylim=(0.4, 5.4), xlabel="feature 1", ylabel="feature 2")
ax.legend(loc="lower right")

# right panel: the same 9 points as a tree
ax = axes[1]
example_tree = linkage(example_points, method="complete", metric="euclidean")
example_cut_height = example_tree[-2, 2] * 0.95

# colour the branches 
set_link_color_palette([CLUSTER_COLORS[1], CLUSTER_COLORS[2], CLUSTER_COLORS[0]])
dendrogram(example_tree, labels=example_names, ax=ax, color_threshold=example_cut_height, above_threshold_color="#9aa0a6")
set_link_color_palette(None)

ax.axhline(example_cut_height, color=CORAL, ls="--", lw=1.8, label="cut here to get 3 groups")
ax.set(title="Hierarchical: hclust\nno k needed, merge the two closest groups over and over",
       ylabel="distance at which the merge happened", xlabel="the same 9 points")
ax.legend()
ax.grid(False)
save_figure(figure, "00_concept_partition_vs_hierarchical.png")

# (c) what clustering is actually trying to do
example_groups = [random_generator.normal([2, 4], 0.45, (9, 2)),random_generator.normal([5.4, 4.2], 0.45, (9, 2)),random_generator.normal([3.7, 1.4], 0.45, (9, 2))]

figure, ax = plt.subplots(figsize=(8, 5.6))
for group_number in range(len(example_groups)):
    group_points = example_groups[group_number]
    ax.scatter(group_points[:, 0], group_points[:, 1], s=75, color=CLUSTER_COLORS[group_number], edgecolor="white", zorder=3)

    centre_x = group_points[:, 0].mean()
    centre_y = group_points[:, 1].mean()
    ax.add_patch(plt.Circle((centre_x, centre_y), 1.05, fill=False, ls="--", lw=1.8,
                            color=CLUSTER_COLORS[group_number], zorder=2))

# an arrow between two cluster centres
first_centre = example_groups[0].mean(axis=0)
second_centre = example_groups[1].mean(axis=0)
ax.annotate("", xy=second_centre, xytext=first_centre, arrowprops=dict(arrowstyle="<->", color=CORAL, lw=2.5))
ax.annotate("between clusters: as large as possible", xy=((first_centre[0] + second_centre[0]) / 2, 5.95),
            ha="center", fontsize=11, fontweight="bold", color=CORAL)

# an arrow inside one cluster
third_group = example_groups[2]
ax.annotate("", xy=third_group[0], xytext=third_group[1], arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2.5))
ax.annotate("within a cluster: as small as possible", xy=(third_group[:, 0].mean(), -0.15),
            ha="center", fontsize=11, fontweight="bold", color=NAVY)

ax.set(title="What every clustering algorithm is trying to do", xlim=(0, 7.4), ylim=(-0.6, 6.4),
       xticks=[], yticks=[], xlabel="feature 1", ylabel="feature 2")
ax.grid(False)
save_figure(figure, "00_concept_goal.png")

print("\ndone, images in images/clustering/, tables in data/clean/clustering_*.csv")
