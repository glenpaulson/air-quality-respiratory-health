"""
linear regression for the AQI number 
logistic regression for a yes or no
  1. identical feature table and identical 80 / 20 split from supervised_prep.py
  2. two versions: 
      linear regression: one with the daily AQI as a number
      logistic regression: one with labels Good against Not Good
  3. settings with 5 fold cross validation inside the training set
  4. fit the chosen models once on the whole training set and score them once on the test set
  5. linear: RMSE, MAE, R squared, predicted against actual and the coefficients
  6. logistic: confusion matrix, accuracy and the coefficients
  7. compare logistic regression against NB refitted on the same two class label, and
     a three class logistic regression against the Part 3 NB and tree results
  8. explanatory diagrams for the overview
output: images/regression/*.png + data/clean/reg_*.csv

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score)
import supervised_prep as prep

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA_FOLDER = os.path.join(THIS_FOLDER, "..", "data", "clean")
IMAGE_FOLDER = os.path.join(THIS_FOLDER, "..", "images", "regression")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
TEAL = "#0e7c7b"
CORAL = "#ef6f53"
NAVY = "#2b4162"
GOLD = "#e8b004"
PURPLE = "#7b5aa6"
CLASS_COLORS = {"Good": "#3bb143", "Moderate": "#efc804", "Unhealthy": "#d62728", "Not Good": "#ef6f53"}
TEXT_BOX = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#ddd", alpha=0.95)
RANDOM_SEED = prep.RANDOM_SEED
CLASS_ORDER = prep.CLASS_ORDER
LABEL = prep.LABEL_COLUMN

TARGET = "us_aqi_max"
BINARY_LABEL = "aqi_class2"
BINARY_ORDER = ["Good", "Not Good"]

NUMBER_OF_FOLDS = 5

def save_figure(figure, file_name, use_tight_layout=True):
    if use_tight_layout:
        figure.tight_layout()
    figure.savefig(os.path.join(IMAGE_FOLDER, file_name), bbox_inches="tight")
    plt.close(figure)
    print("saved", file_name)

def save_table_image(table, title, file_name, note=None, number_of_rows=6, figure_width=None):
    """take the first few rows of a dataframe and save them as a picture"""
    small_table = table.head(number_of_rows).copy()
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

def draw_confusion_matrix(ax, true_labels, predicted_labels, title, colour_map, class_order):
    """confusion matrix with the counts written in, true labels down, predictions across"""
    matrix = confusion_matrix(true_labels, predicted_labels, labels=class_order)
    sns.heatmap(matrix, annot=True, fmt=",d", cmap=colour_map, cbar=False, linewidths=1.2,
                linecolor="white", xticklabels=class_order, yticklabels=class_order, ax=ax,
                annot_kws={"fontsize": 13, "fontweight": "bold"})
    ax.set(title=title, xlabel="what the model predicted", ylabel="what the day actually was")
    ax.grid(False)
    return matrix

def sigmoid(score):
    return 1 / (1 + np.exp(-score))

# ----------------------------------------------------------------------------------
# step 1: the same starting point as naive_bayes.py and decision_trees.py. 
# the feature table from supervised_prep does not carry the AQI number only the class made from it.

daily_data, feature_table, train_rows, test_rows = prep.get_everything()
train_rows[TARGET] = daily_data[TARGET].values[train_rows.row_id.values]
test_rows[TARGET] = daily_data[TARGET].values[test_rows.row_id.values]
# Good = daily AQI <= 50 and Not Good = Moderate and Unhealthy
train_rows[BINARY_LABEL] = np.where(train_rows[LABEL] == "Good", "Good", "Not Good")
test_rows[BINARY_LABEL] = np.where(test_rows[LABEL] == "Good", "Good", "Not Good")
# sanity check
assert (train_rows[TARGET] <= 50).eq(train_rows[BINARY_LABEL] == "Good").all()

print("\ndaily AQI in the training set: mean {:.1f}, sd {:.1f}, from {} to {}".format(
    train_rows[TARGET].mean(), train_rows[TARGET].std(), train_rows[TARGET].min(), train_rows[TARGET].max()))
print("share of Not Good days: training {:.2%}, testing {:.2%}".format(
    (train_rows[BINARY_LABEL] == "Not Good").mean(), (test_rows[BINARY_LABEL] == "Not Good").mean()))

save_table_image(daily_data[["city", "date", "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "season", "us_aqi_max", "aqi_class3"]],
                 "Before: the cleaned daily table from Module 1, 28 columns wide",
                 "01_before_raw.png",
                 note="us_aqi_max becomes the target for linear regression. the six pollutant columns it is calculated from are still left out to avoid target leakage")

# ----------------------------------------------------------------------------------
# step 2: the regression matrix. regression multiplies every column by a coefficient and adds
# them up so text has to become numbers. season becomes 0 / 1 flags  but only three of them 
# winter is left out and becomes the baseline that the other three seasons are measured against.
# to avoid the dummy variable problem

SEASON_FLAGS = ["is_Spring", "is_Summer", "is_Fall"]
WEATHER_COLUMNS = ["temp_mean", "temp_range", "precip", "wind"]
PLACE_COLUMNS = ["latitude", "longitude"]
ALL_COLUMNS = WEATHER_COLUMNS + PLACE_COLUMNS + SEASON_FLAGS

def build_regression_matrix(rows):
    """the 6 numeric columns as they are, plus 3 season flags with Winter as the baseline"""
    matrix = rows[WEATHER_COLUMNS + PLACE_COLUMNS].reset_index(drop=True).copy()
    for flag in SEASON_FLAGS:
        matrix[flag] = (rows["season"].values == flag.replace("is_", "")).astype(int)
    return matrix

X_train = build_regression_matrix(train_rows)
X_test = build_regression_matrix(test_rows)
y_train_number = train_rows[TARGET].values.astype(float)
y_test_number = test_rows[TARGET].values.astype(float)
y_train_class = train_rows[BINARY_LABEL].values
y_test_class = test_rows[BINARY_LABEL].values

print("\nregression input: {} x {}".format(X_train.shape[0], X_train.shape[1]))
print("features: " + ", ".join(ALL_COLUMNS))

readable_columns = train_rows[["city", "date"]].reset_index(drop=True)
save_table_image(pd.concat([readable_columns, X_train, pd.Series(y_train_number.astype(int), name=TARGET)], axis=1),
                 "After, for linear regression: nine numeric columns and the daily AQI as the target",
                 "02_after_linear.png",
                 note="season is spread into three 0 / 1 flags. Winter is the day with all three at 0 so every season coefficient is measured against winter")

# logistic regression has the same nine columns but standardised, the scaler is fitted on the training rows only
preview_scaler = StandardScaler().fit(X_train)
X_train_standardised_preview = pd.DataFrame(preview_scaler.transform(X_train), columns=ALL_COLUMNS)
save_table_image(pd.concat([readable_columns, X_train_standardised_preview, pd.Series(y_train_class, name=BINARY_LABEL)], axis=1),
                 "After, for logistic regression: the same nine columns standardised and a two class label",
                 "03_after_logistic.png",
                 note="label is Good (AQI <= 50) or Not Good (AQI > 50)")

# save the two halves before scaling
pd.concat([readable_columns, X_train, pd.Series(y_train_number.astype(int), name=TARGET), pd.Series(y_train_class, name=BINARY_LABEL)], axis=1).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_train.csv"), index=False)
pd.concat([test_rows[["city", "date"]].reset_index(drop=True), X_test, pd.Series(y_test_number.astype(int), name=TARGET), pd.Series(y_test_class, name=BINARY_LABEL)], axis=1).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_test.csv"), index=False)

# ----------------------------------------------------------------------------------
# step 3: the split picture check that the AQI number and the new two class label also came out balanced

binary_balance = pd.DataFrame({
    "class": BINARY_ORDER,
    "train_percent": [round(100 * (y_train_class == name).mean(), 2) for name in BINARY_ORDER],
    "test_percent": [round(100 * (y_test_class == name).mean(), 2) for name in BINARY_ORDER],
})
print("\ntwo class balance in each half:")
print(binary_balance.to_string(index=False))

figure, axes = plt.subplots(1, 3, figsize=(18, 4.9), gridspec_kw={"width_ratios": [1.1, 1.2, 0.9]})

ax = axes[0]
ax.barh(["training set", "testing set"], [len(train_rows), len(test_rows)], color=[TEAL, CORAL], height=0.5)
for position, count in enumerate([len(train_rows), len(test_rows)]):
    fits_inside = count > 0.5 * len(train_rows)
    ax.annotate("{:,} days  ({:.0f}%)".format(count, 100 * count / len(feature_table)), (count, position), va="center",
                ha="right" if fits_inside else "left", color="white" if fits_inside else "#222", fontweight="bold",
                xytext=(-10 if fits_inside else 8, 0), textcoords="offset points")
ax.set(title="the 80 / 20 split, {:,} city days".format(len(feature_table)), xlabel="number of days")
ax.annotate("the same rows as the NB and\nDecision Trees tabs. 0 days in both halves", xy=(0.03, 0.08), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
bins = np.arange(20, 215, 5)
ax.hist(y_train_number, bins=bins, density=True, color=TEAL, alpha=0.55, label="training set, mean {:.1f}".format(y_train_number.mean()))
ax.hist(y_test_number, bins=bins, density=True, histtype="step", color=CORAL, lw=2.2, label="testing set, mean {:.1f}".format(y_test_number.mean()))
ax.axvline(50, color=NAVY, ls="--", lw=1.6)
ax.annotate("dashed line at 50: the boundary\nbetween Good and Not Good", xy=(50, ax.get_ylim()[1] * 0.62), xytext=(105, ax.get_ylim()[1] * 0.55),
            fontsize=9, color=NAVY, arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2), bbox=TEXT_BOX)
ax.set(title="linear regression target: daily AQI in each half", xlabel="daily AQI us_aqi_max", ylabel="share of days")
ax.legend(fontsize=9, loc="upper right")

ax = axes[2]
bar_positions = np.arange(len(BINARY_ORDER))
ax.bar(bar_positions - 0.19, binary_balance.train_percent, width=0.38, color=TEAL, label="training set", edgecolor="white")
ax.bar(bar_positions + 0.19, binary_balance.test_percent, width=0.38, color=CORAL, label="testing set", edgecolor="white")
for position in range(len(BINARY_ORDER)):
    ax.annotate("{:.1f}%".format(binary_balance.train_percent[position]), (position - 0.19, binary_balance.train_percent[position]), ha="center", va="bottom", fontsize=9)
    ax.annotate("{:.1f}%".format(binary_balance.test_percent[position]), (position + 0.19, binary_balance.test_percent[position]), ha="center", va="bottom", fontsize=9)
ax.set(title="logistic regression label in each half", xlabel="two class label", ylabel="percent of that half", xticks=bar_positions, xticklabels=BINARY_ORDER, ylim=(0, 85))
ax.legend(fontsize=9, loc="upper left")

figure.suptitle("Training set and testing set: disjoint and matched on both targets", fontweight="bold", y=1.03)
save_figure(figure, "04_train_test_split.png")

# ----------------------------------------------------------------------------------
# step 4: linear regression 5 fold cross validation

regression_folds = KFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_SEED)

LINEAR_CANDIDATES = [
    {"name": "weather only", "columns": WEATHER_COLUMNS, "curved": False},
    {"name": "weather + season", "columns": WEATHER_COLUMNS + SEASON_FLAGS, "curved": False},
    {"name": "weather + place", "columns": WEATHER_COLUMNS + PLACE_COLUMNS, "curved": False},
    {"name": "weather + place + season", "columns": ALL_COLUMNS, "curved": False},
    {"name": "all nine + squares and products", "columns": ALL_COLUMNS, "curved": True},
]

def make_linear_model(curved):
    if curved:
        return make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
    return LinearRegression()

linear_cv_rows = []
for candidate in LINEAR_CANDIDATES:
    scores = cross_validate(make_linear_model(candidate["curved"]), X_train[candidate["columns"]], y_train_number,
                            cv=regression_folds, scoring=["r2", "neg_root_mean_squared_error"])
    number_of_terms = len(candidate["columns"])
    if candidate["curved"]:
        number_of_terms = PolynomialFeatures(degree=2, include_bias=False).fit(X_train[candidate["columns"]]).n_output_features_
    linear_cv_rows.append({
        "model": candidate["name"],
        "terms": number_of_terms,
        "cv_r2_mean": scores["test_r2"].mean(),
        "cv_r2_sd": scores["test_r2"].std(),
        "cv_rmse_mean": -scores["test_neg_root_mean_squared_error"].mean(),
        "cv_rmse_sd": scores["test_neg_root_mean_squared_error"].std(),
    })

linear_cv = pd.DataFrame(linear_cv_rows)
print("\nlinear regression 5 fold cross validation on the training set:")
print(linear_cv.round(4).to_string(index=False))
linear_cv.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_linear_cv.csv"), index=False)
best_overall_name = linear_cv.loc[linear_cv.cv_r2_mean.idxmax(), "model"]
plain_cv = linear_cv[~linear_cv.model.str.contains("squares")]
best_plain_name = plain_cv.loc[plain_cv.cv_r2_mean.idxmax(), "model"]
print("best overall by cross validation: {}".format(best_overall_name))
print("best readable model by cross validation: {}".format(best_plain_name))
best_overall = [item for item in LINEAR_CANDIDATES if item["name"] == best_overall_name][0]
best_plain = [item for item in LINEAR_CANDIDATES if item["name"] == best_plain_name][0]

figure, axes = plt.subplots(1, 2, figsize=(15, 5))
short_names = ["weather\nonly", "weather\n+ season", "weather\n+ place", "weather\n+ place\n+ season", "all nine\n+ squares\n+ products"]

ax = axes[0]
bar_colours = [CORAL if name == best_overall_name else (TEAL if name == best_plain_name else "#c3d4d3") for name in linear_cv.model]
bars = ax.bar(short_names, linear_cv.cv_r2_mean, yerr=linear_cv.cv_r2_sd, color=bar_colours, edgecolor="white", capsize=5, width=0.6)
for bar, value in zip(bars, linear_cv.cv_r2_mean):
    ax.annotate("{:.3f}".format(value), (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontweight="bold", fontsize=9.5, xytext=(0, 6), textcoords="offset points")
ax.set(title="cross validated R squared higher is better", ylabel="R squared mean of 5 folds", ylim=(0, 0.33))
ax.tick_params(axis="x", labelsize=9)
ax.annotate("teal: best readable model the one read below\norange: best overall\nthe whiskers are the spread across the five folds", xy=(0.03, 0.80), xycoords="axes fraction", fontsize=9, color="#333", bbox=TEXT_BOX)

ax = axes[1]
bars = ax.bar(short_names, linear_cv.cv_rmse_mean, yerr=linear_cv.cv_rmse_sd, color=bar_colours, edgecolor="white", capsize=5, width=0.6)
for bar, value in zip(bars, linear_cv.cv_rmse_mean):
    ax.annotate("{:.2f}".format(value), (bar.get_x() + bar.get_width() / 2, value), ha="center", va="top", fontweight="bold", fontsize=9.5, color="white", xytext=(0, -8), textcoords="offset points")
ax.axhline(y_train_number.std(), color="#999", ls=":", lw=1.8, label="always guess the training mean ({:.2f})".format(y_train_number.std()))
ax.set(title="cross validated RMSE lower is better", ylabel="RMSE in AQI points mean of 5 folds", ylim=(0, 25))
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=9, loc="lower right")

figure.suptitle("Choosing the linear model with 5 fold cross validation on the training set only", fontweight="bold", y=1.02)
save_figure(figure, "05_linear_cv.png")

# ----------------------------------------------------------------------------------
# step 5: fitting both chosen linear models on the whole training set and scoring them once on the test set

plain_model = make_linear_model(False).fit(X_train[best_plain["columns"]], y_train_number)
plain_predictions = plain_model.predict(X_test[best_plain["columns"]])
curved_model = make_linear_model(best_overall["curved"]).fit(X_train[best_overall["columns"]], y_train_number)
curved_predictions = curved_model.predict(X_test[best_overall["columns"]])
mean_guess = np.full(len(y_test_number), y_train_number.mean())

def regression_scores(name, predictions):
    return {
        "model": name,
        "test_rmse": mean_squared_error(y_test_number, predictions) ** 0.5,
        "test_mae": mean_absolute_error(y_test_number, predictions),
        "test_r2": r2_score(y_test_number, predictions),
    }

linear_results = pd.DataFrame([
    regression_scores("always guess the training mean", mean_guess),
    regression_scores("plain linear: " + best_plain_name, plain_predictions),
    regression_scores("curved linear: " + best_overall_name, curved_predictions),
])
print("\nlinear regression on the held out test set:")
print(linear_results.round(4).to_string(index=False))
linear_results.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_linear_results.csv"), index=False)

pd.DataFrame({"city": test_rows.city.values, "date": test_rows.date.dt.strftime("%Y-%m-%d").values,
              "actual_aqi": y_test_number.astype(int), "plain_prediction": plain_predictions.round(2),
              "curved_prediction": curved_predictions.round(2)}).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_linear_test_predictions.csv"), index=False)

plain_r2 = r2_score(y_test_number, plain_predictions)
curved_r2 = r2_score(y_test_number, curved_predictions)

print("spread of the actual test AQI: sd {:.1f}. spread of the plain predictions: sd {:.1f}".format(y_test_number.std(), plain_predictions.std()))
for low, high in [(0, 50), (51, 100), (101, 500)]:
    in_band = (y_test_number >= low) & (y_test_number <= high)
    print("  actual AQI {}-{}: {} days, mean plain prediction {:.1f}".format(low, high, in_band.sum(), plain_predictions[in_band].mean()))

figure, axes = plt.subplots(1, 3, figsize=(19, 5.8))
for position, (name, predictions, colour) in enumerate([("plain linear, nine terms", plain_predictions, TEAL), ("curved linear, {} terms".format(int(linear_cv.loc[linear_cv.model == best_overall_name, "terms"].iloc[0])), curved_predictions, PURPLE)]):
    ax = axes[position]
    ax.axhspan(0, 50, color="#3bb143", alpha=0.07)
    ax.axhspan(50, 100, color="#efc804", alpha=0.08)
    ax.axhspan(100, 220, color="#d62728", alpha=0.06)
    ax.scatter(y_test_number, predictions, s=7, alpha=0.30, color=colour, edgecolor="none")
    ax.plot([20, 210], [20, 210], color=NAVY, ls="--", lw=1.6, label="a perfect prediction")
    ax.set(title=name, xlabel="actual daily AQI on the test day", ylabel="predicted daily AQI", xlim=(15, 215), ylim=(15, 215))
    ax.legend(fontsize=9, loc="upper left")
    ax.annotate("RMSE {:.1f}   MAE {:.1f}   R squared {:.3f}".format(mean_squared_error(y_test_number, predictions) ** 0.5, mean_absolute_error(y_test_number, predictions), r2_score(y_test_number, predictions)),
                xy=(0.35, 0.06), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[2]
plain_residuals = y_test_number - plain_predictions
ax.hist(plain_residuals, bins=np.arange(-60, 150, 4), color=TEAL, alpha=0.75, edgecolor="white")
ax.axvline(0, color=NAVY, ls="--", lw=1.6)
ax.set(title="plain model: how far off it was", xlabel="actual minus predicted AQI points", ylabel="number of test days")
ax.annotate("most days land within about 15 points\nbut the long tail on the right is the\nbad days which it badly under calls.\nthe worst day it missed by {:.0f} points.".format(plain_residuals.max()),
            xy=(0.40, 0.62), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

figure.suptitle("Predicted against actual on the {:,} held out test days: the cloud is flat so the extremes are missed".format(len(y_test_number)), fontweight="bold", y=1.02)
save_figure(figure, "06_linear_predicted_vs_actual.png")

plain_coefficients = pd.DataFrame({
    "feature": best_plain["columns"],
    "coefficient_raw_units": plain_model.coef_,
    "training_sd": X_train[best_plain["columns"]].std().values,
})

plain_coefficients["effect_of_one_sd"] = np.where(plain_coefficients.feature.str.startswith("is_"),
                                                 plain_coefficients.coefficient_raw_units,
                                                 plain_coefficients.coefficient_raw_units * plain_coefficients.training_sd)
print("\nplain linear model, intercept {:.2f}".format(plain_model.intercept_))
print(plain_coefficients.round(4).to_string(index=False))
plain_coefficients.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_linear_coefficients.csv"), index=False)

UNITS = {"temp_mean": "per 1 C", "temp_range": "per 1 C of swing", "precip": "per 1 mm of rain", "wind": "per 1 km/h",
         "latitude": "per 1 degree north", "longitude": "per 1 degree east", "is_Spring": "spring vs winter",
         "is_Summer": "summer vs winter", "is_Fall": "fall vs winter"}

figure, ax = plt.subplots(figsize=(11, 5.8))
ordered = plain_coefficients.sort_values("effect_of_one_sd")
bar_colours = [CORAL if value > 0 else TEAL for value in ordered.effect_of_one_sd]
ax.barh(ordered.feature, ordered.effect_of_one_sd, color=bar_colours, edgecolor="white")
for position, (name, effect, raw) in enumerate(zip(ordered.feature, ordered.effect_of_one_sd, ordered.coefficient_raw_units)):
    ax.annotate("{:+.2f} AQI {}".format(raw, UNITS[name]), (effect, position), va="center",
                ha="left" if effect > 0 else "right", fontsize=8.5, xytext=(4 if effect > 0 else -4, 0), textcoords="offset points")
ax.axvline(0, color="#555", lw=1)
ax.set(title="what each column does to the predicted AQI, holding the others fixed", xlabel="change in predicted AQI for a typical change in that column (one sd, or a switch in the season)",
       xlim=(min(ordered.effect_of_one_sd) * 1.9, max(ordered.effect_of_one_sd) * 2.2))
ax.annotate("the label on each bar is the coefficient itself\nin the column's own units. the bar length rescales\nit to a typical day to day change so the columns\ncan be compared. intercept: {:.1f}".format(plain_model.intercept_),
            xy=(0.62, 0.04), xycoords="axes fraction", fontsize=9, color="#333", bbox=TEXT_BOX)
save_figure(figure, "07_linear_coefficients.png")

# ----------------------------------------------------------------------------------
# step 6: logistic regression: choosing C and the class weighting with 5 fold CV.
# C is the inverse of the penalty strength: a small C pulls every coefficient towards 0 and a big C
# lets them be whatever fits the training rows best. stratified folds helps to keep the Good / Not Good mix
# the same in every fold.

classification_folds = StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_SEED)
C_VALUES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
CLASS_WEIGHTS = [None, "balanced"]

def make_logistic_model(C, class_weight):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=class_weight, max_iter=5000))

logistic_cv_rows = []
for class_weight in CLASS_WEIGHTS:
    for C in C_VALUES:
        scores = cross_validate(make_logistic_model(C, class_weight), X_train, y_train_class, cv=classification_folds,
                                scoring=["accuracy", "balanced_accuracy"])
        logistic_cv_rows.append({"class_weight": "balanced" if class_weight else "none", "C": C,
                                 "cv_accuracy_mean": scores["test_accuracy"].mean(), "cv_accuracy_sd": scores["test_accuracy"].std(),
                                 "cv_balanced_accuracy_mean": scores["test_balanced_accuracy"].mean()})

logistic_cv = pd.DataFrame(logistic_cv_rows)
print("\nlogistic regression with 5 fold cross validation on the training set:")
print(logistic_cv.round(4).to_string(index=False))
logistic_cv.round(4).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_logistic_cv.csv"), index=False)

best_row = logistic_cv.loc[logistic_cv.cv_accuracy_mean.idxmax()]
best_C = float(best_row.C)
best_class_weight = None if best_row.class_weight == "none" else "balanced"
print("chosen by cross validated accuracy: C = {}, class_weight = {}".format(best_C, best_row.class_weight))

# a second model at the same C with the classes balanced
best_balanced_row = logistic_cv.loc[logistic_cv.cv_balanced_accuracy_mean.idxmax()]
balanced_C = float(best_balanced_row.C)
balanced_class_weight = None if best_balanced_row.class_weight == "none" else "balanced"
print("chosen by cross validated balanced accuracy: C = {}, class_weight = {}".format(balanced_C, best_balanced_row.class_weight))

majority_binary_accuracy = (y_test_class == "Not Good").mean()

figure, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
for class_weight, colour, marker in [("none", TEAL, "o"), ("balanced", CORAL, "s")]:
    rows = logistic_cv[logistic_cv.class_weight == class_weight]
    ax.errorbar(rows.C, rows.cv_accuracy_mean * 100, yerr=rows.cv_accuracy_sd * 100, color=colour, marker=marker, lw=2.4, capsize=4,
                label="class_weight = {}".format(class_weight))
ax.axhline((y_train_class == "Not Good").mean() * 100, color="#999", ls=":", lw=1.8, label="always answer Not Good ({:.1f}%)".format((y_train_class == "Not Good").mean() * 100))
ax.axvline(best_C, color=NAVY, ls="--", lw=1.6)
ax.set(xscale="log", title="cross validated accuracy against C", xlabel="C the inverse of the penalty in log scale", ylabel="accuracy mean of 5 folds in percent", ylim=(50, 76))
ax.legend(fontsize=9, loc="lower right")
ax.annotate("chosen: C = {:g} no class weighting\n{:.2f}% across the five folds".format(best_C, best_row.cv_accuracy_mean * 100), xy=(0.03, 0.30), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

ax = axes[1]
for class_weight, colour, marker in [("none", TEAL, "o"), ("balanced", CORAL, "s")]:
    rows = logistic_cv[logistic_cv.class_weight == class_weight]
    ax.plot(rows.C, rows.cv_balanced_accuracy_mean * 100, color=colour, marker=marker, lw=2.4, label="class_weight = {}".format(class_weight))
ax.axhline(50, color="#999", ls=":", lw=1.8, label="always answer Not Good (50%)")
ax.set(xscale="log", title="cross validated balanced accuracy against C", xlabel="C the inverse of the penalty in log scale", ylabel="mean of the two recalls in percent", ylim=(45, 70))
ax.legend(fontsize=9, loc="lower right")
ax.annotate("balanced accuracy is the average of how many Good\ndays it finds and how many Not Good days it finds\nso answering Not Good every day only scores 50.\nby this measure the balanced weighting wins.", xy=(0.20, 0.24), xycoords="axes fraction", fontsize=9, color="#333", bbox=TEXT_BOX)

figure.suptitle("Choosing the logistic regression with 5 fold cross validation on the training set only", fontweight="bold", y=1.02)
save_figure(figure, "08_logistic_cv.png")

# fit the two chosen logistic models on the whole training set and score each one once on the test set
logistic_model = make_logistic_model(best_C, best_class_weight).fit(X_train, y_train_class)
logistic_predictions = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test_class, logistic_predictions)

balanced_logistic_model = make_logistic_model(balanced_C, balanced_class_weight).fit(X_train, y_train_class)
balanced_logistic_predictions = balanced_logistic_model.predict(X_test)
balanced_logistic_accuracy = accuracy_score(y_test_class, balanced_logistic_predictions)

print("\nalways answer Not Good on the test set: {:.4f}".format(majority_binary_accuracy))
print("logistic regression chosen for accuracy: {:.4f}".format(logistic_accuracy))
print(classification_report(y_test_class, logistic_predictions, labels=BINARY_ORDER, zero_division=0))
print("logistic regression chosen for balanced accuracy: {:.4f}".format(balanced_logistic_accuracy))
print(classification_report(y_test_class, balanced_logistic_predictions, labels=BINARY_ORDER, zero_division=0))

figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
logistic_matrix = draw_confusion_matrix(axes[0], y_test_class, logistic_predictions,
                                        "chosen for accuracy C = {:g}\naccuracy {:.1f}%".format(best_C, logistic_accuracy * 100), "Blues", BINARY_ORDER)
balanced_logistic_matrix = draw_confusion_matrix(axes[1], y_test_class, balanced_logistic_predictions,
                                                 "classes balanced C = {:g}\naccuracy {:.1f}%".format(balanced_C, balanced_logistic_accuracy * 100), "Oranges", BINARY_ORDER)
figure.suptitle("Logistic regression on the {:,} held out test days".format(len(y_test_class)), fontweight="bold", y=1.04)
save_figure(figure, "09_logistic_confusion.png")

pd.DataFrame(logistic_matrix, index=BINARY_ORDER, columns=BINARY_ORDER).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_confusion_logistic.csv"))
pd.DataFrame(balanced_logistic_matrix, index=BINARY_ORDER, columns=BINARY_ORDER).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_confusion_logistic_balanced.csv"))

# the coefficients. the model learned them on standardised columns
fitted_scaler = logistic_model.named_steps["standardscaler"]
fitted_logistic = logistic_model.named_steps["logisticregression"]
positive_class = fitted_logistic.classes_[1]
print("the positive class the one the probability is for is: {}".format(positive_class))

logistic_coefficients = pd.DataFrame({
    "feature": ALL_COLUMNS,
    "coefficient_per_sd": fitted_logistic.coef_[0],
    "training_sd": fitted_scaler.scale_,
})
logistic_coefficients["coefficient_raw_units"] = logistic_coefficients.coefficient_per_sd / logistic_coefficients.training_sd
logistic_coefficients["odds_ratio_per_sd"] = np.exp(logistic_coefficients.coefficient_per_sd)
logistic_coefficients["odds_ratio_raw_unit"] = np.exp(logistic_coefficients.coefficient_raw_units)
print("\nlogistic regression coefficients intercept {:.4f} on the standardised scale".format(fitted_logistic.intercept_[0]))
print(logistic_coefficients.round(4).to_string(index=False))
logistic_coefficients.round(5).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_logistic_coefficients.csv"), index=False)

wind_raw = float(logistic_coefficients.loc[logistic_coefficients.feature == "wind", "coefficient_raw_units"].iloc[0])
print("wind: every extra 10 km/h multiplies the odds of a Not Good day by {:.3f}".format(np.exp(10 * wind_raw)))

# numeric columns: one standard deviation more. season flags: that season instead of winter
logistic_coefficients["odds_ratio_shown"] = np.where(logistic_coefficients.feature.str.startswith("is_"),
                                                     logistic_coefficients.odds_ratio_raw_unit, logistic_coefficients.odds_ratio_per_sd)
figure, ax = plt.subplots(figsize=(11, 5.8))
ordered = logistic_coefficients.sort_values("odds_ratio_shown")
bar_colours = [CORAL if value > 1 else TEAL for value in ordered.odds_ratio_shown]
ax.barh(ordered.feature, ordered.odds_ratio_shown - 1, left=1, color=bar_colours, edgecolor="white")
for position, (name, ratio) in enumerate(zip(ordered.feature, ordered.odds_ratio_shown)):
    ax.annotate("x {:.2f}".format(ratio), (ratio, position), va="center", ha="left" if ratio > 1 else "right", fontsize=9,
                xytext=(4 if ratio > 1 else -4, 0), textcoords="offset points")
ax.axvline(1, color="#555", lw=1.2)
ax.set(xscale="log", title="what each column does to the odds of a Not Good day", xlabel="odds ratio in log scale 1 = no effect: one standard deviation more of a number or that season instead of winter",
       xlim=(min(ordered.odds_ratio_shown) * 0.80, max(ordered.odds_ratio_shown) * 1.25))
ax.set_xticks([0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0])
ax.set_xticklabels(["0.3", "0.4", "0.5", "0.7", "1", "1.5", "2"])
ax.minorticks_off()
ax.annotate("orange raises the odds teal lowers them.\nx 0.70 means the odds are cut by 30 percent,\nx 1.50 means they go up by half.",
            xy=(0.66, 0.04), xycoords="axes fraction", fontsize=9, color="#333", bbox=TEXT_BOX)
save_figure(figure, "10_logistic_odds_ratios.png")

# ----------------------------------------------------------------------------------
# step 7: against NB. the Part 3 Naive Bayes answered a three class questio. each numeric column cut into four ranges learned on the training rows only,
# rain given its own dry bin, every day turned into seven words and one hot encoded.

def learn_bin_edges(training_values):
    cuts = np.unique(np.quantile(training_values, [0.25, 0.50, 0.75]))
    return np.concatenate([[-np.inf], cuts, [np.inf]])

def learn_rain_edges(training_values):
    wet_days = training_values[training_values > 0]
    cuts = np.unique(np.quantile(wet_days, [1 / 3, 2 / 3]))
    return np.concatenate([[-np.inf, 0.0], cuts, [np.inf]])

nb_bin_edges = {}
for column_name in prep.NUMERIC_FEATURES:
    if column_name == "precip":
        nb_bin_edges[column_name] = learn_rain_edges(train_rows[column_name].values)
    else:
        nb_bin_edges[column_name] = learn_bin_edges(train_rows[column_name].values)

def naive_bayes_matrix(rows):
    """the seven word description of each day one hot encoded in a fixed column order"""
    pieces = []
    for column_name in prep.NUMERIC_FEATURES:
        bin_number = pd.cut(rows[column_name], bins=nb_bin_edges[column_name], labels=False, include_lowest=True)
        pieces.append(pd.get_dummies(bin_number.astype(int), prefix=column_name).reindex(
            columns=["{}_{}".format(column_name, number) for number in range(len(nb_bin_edges[column_name]) - 1)], fill_value=0))
    pieces.append(pd.get_dummies(rows["season"], prefix="season").reindex(columns=["season_" + name for name in ["Winter", "Spring", "Summer", "Fall"]], fill_value=0))
    return pd.concat(pieces, axis=1).astype(int).values

nb_train_matrix = naive_bayes_matrix(train_rows)
nb_test_matrix = naive_bayes_matrix(test_rows)
binary_multinomial = MultinomialNB(alpha=1.0).fit(nb_train_matrix, y_train_class)
binary_bernoulli = BernoulliNB(alpha=1.0, binarize=None).fit(nb_train_matrix, y_train_class)
binary_multinomial_predictions = binary_multinomial.predict(nb_test_matrix)
binary_bernoulli_predictions = binary_bernoulli.predict(nb_test_matrix)

def recall_of(true_labels, predicted_labels, class_name):
    is_class = true_labels == class_name
    return (predicted_labels[is_class] == class_name).mean()

binary_comparison = pd.DataFrame([
    {"model": "always answer Not Good", "predictions": np.full(len(y_test_class), "Not Good")},
    {"model": "Multinomial NB (alpha=1)", "predictions": binary_multinomial_predictions},
    {"model": "Bernoulli NB (alpha=1)", "predictions": binary_bernoulli_predictions},
    {"model": "Logistic regression, C = {:g}".format(best_C), "predictions": logistic_predictions},
    {"model": "Logistic regression, balanced, C = {:g}".format(balanced_C), "predictions": balanced_logistic_predictions},
])
binary_comparison["accuracy"] = [accuracy_score(y_test_class, predictions) for predictions in binary_comparison.predictions]
binary_comparison["recall_Good"] = [recall_of(y_test_class, predictions, "Good") for predictions in binary_comparison.predictions]
binary_comparison["recall_Not_Good"] = [recall_of(y_test_class, predictions, "Not Good") for predictions in binary_comparison.predictions]
binary_comparison = binary_comparison.drop(columns="predictions")
print("\nGood against Not Good, every model on the same test set:")
print(binary_comparison.round(4).to_string(index=False))

# the three class version to sit next to the actual Part 3 numbers. 
three_class_cv = []
for C in C_VALUES:
    scores = cross_validate(make_logistic_model(C, None), X_train, train_rows[LABEL].values, cv=classification_folds, scoring="accuracy")
    three_class_cv.append({"C": C, "cv_accuracy_mean": scores["test_score"].mean()})
three_class_cv = pd.DataFrame(three_class_cv)
three_class_C = float(three_class_cv.loc[three_class_cv.cv_accuracy_mean.idxmax(), "C"])
three_class_model = make_logistic_model(three_class_C, None).fit(X_train, train_rows[LABEL].values)
three_class_predictions = three_class_model.predict(X_test)
three_class_accuracy = accuracy_score(test_rows[LABEL].values, three_class_predictions)
three_class_matrix = confusion_matrix(test_rows[LABEL].values, three_class_predictions, labels=CLASS_ORDER)
print("\nthree class logistic regression, C = {:g} chosen by cross validation: test accuracy {:.4f}".format(three_class_C, three_class_accuracy))
print(classification_report(test_rows[LABEL].values, three_class_predictions, labels=CLASS_ORDER, zero_division=0))
pd.DataFrame(three_class_matrix, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(os.path.join(CLEAN_DATA_FOLDER, "reg_confusion_logistic_3class.csv"))

three_class_comparison = [{"model": "always answer Moderate", "accuracy": prep.majority_class_accuracy(test_rows)}]
earlier_results_file = os.path.join(CLEAN_DATA_FOLDER, "dt_vs_nb_comparison.csv")
if os.path.exists(earlier_results_file):
    for _, row in pd.read_csv(earlier_results_file).iterrows():
        if row.model in ("Multinomial NB (alpha=1)", "Bernoulli NB (alpha=1)", "Tree 4: tuned depth, GINI"):
            three_class_comparison.append({"model": row.model, "accuracy": row.accuracy})
three_class_comparison.append({"model": "Logistic regression, 3 class", "accuracy": three_class_accuracy})
three_class_comparison = pd.DataFrame(three_class_comparison)
print(three_class_comparison.round(4).to_string(index=False))

pd.concat([binary_comparison.assign(question="Good vs Not Good"),
           three_class_comparison.assign(question="Good / Moderate / Unhealthy")]).round(4).to_csv(
    os.path.join(CLEAN_DATA_FOLDER, "reg_vs_nb_comparison.csv"), index=False)

figure, axes = plt.subplots(1, 3, figsize=(20, 5.6), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})

ax = axes[0]
short_names = ["always\nNot Good", "Multinomial\nNB", "Bernoulli\nNB", "Logistic\nC = {:g}".format(best_C), "Logistic\nbalanced"]
bar_colours = ["#9aa0a6", PURPLE, PURPLE, TEAL, CORAL]
bars = ax.bar(short_names, binary_comparison.accuracy * 100, color=bar_colours, edgecolor="white", width=0.6)
for bar, value in zip(bars, binary_comparison.accuracy):
    ax.annotate("{:.1f}".format(value * 100), (bar.get_x() + bar.get_width() / 2, value * 100), ha="center", va="bottom", fontweight="bold", fontsize=9.5)
ax.axhline(majority_binary_accuracy * 100, color="#999", ls=":", lw=1.6)
ax.set(title="Good against Not Good: test accuracy", ylabel="percent of test days classified correctly", ylim=(0, 85))
ax.tick_params(axis="x", labelsize=9)

ax = axes[1]
bar_positions = np.arange(len(binary_comparison))
ax.bar(bar_positions - 0.19, binary_comparison.recall_Good * 100, width=0.38, color="#3bb143", edgecolor="white", label="Good days found")
ax.bar(bar_positions + 0.19, binary_comparison.recall_Not_Good * 100, width=0.38, color=CORAL, edgecolor="white", label="Not Good days found")
for position in range(len(binary_comparison)):
    ax.annotate("{:.0f}".format(binary_comparison.recall_Good[position] * 100), (position - 0.19, binary_comparison.recall_Good[position] * 100), ha="center", va="bottom", fontsize=8.5)
    ax.annotate("{:.0f}".format(binary_comparison.recall_Not_Good[position] * 100), (position + 0.19, binary_comparison.recall_Not_Good[position] * 100), ha="center", va="bottom", fontsize=8.5)
ax.set(title="recall of each class", ylabel="percent of that class found", xticks=bar_positions, xticklabels=short_names, ylim=(0, 115))
ax.tick_params(axis="x", labelsize=9)
ax.legend(fontsize=9, loc="upper right")

ax = axes[2]
three_short = {"always answer Moderate": "always\nModerate", "Multinomial NB (alpha=1)": "Multinomial\nNB", "Bernoulli NB (alpha=1)": "Bernoulli\nNB",
               "Tree 4: tuned depth, GINI": "Tree 4\n(Part 3)", "Logistic regression, 3 class": "Logistic\n3 class"}
three_colours = {"always answer Moderate": "#9aa0a6", "Multinomial NB (alpha=1)": PURPLE, "Bernoulli NB (alpha=1)": PURPLE,
                 "Tree 4: tuned depth, GINI": NAVY, "Logistic regression, 3 class": TEAL}
bars = ax.bar([three_short[name] for name in three_class_comparison.model], three_class_comparison.accuracy * 100,
              color=[three_colours[name] for name in three_class_comparison.model], edgecolor="white", width=0.6)
for bar, value in zip(bars, three_class_comparison.accuracy):
    ax.annotate("{:.1f}".format(value * 100), (bar.get_x() + bar.get_width() / 2, value * 100), ha="center", va="bottom", fontweight="bold", fontsize=9.5)
ax.set(title="the original three class question", ylabel="percent of test days classified correctly", ylim=(0, 85))
ax.tick_params(axis="x", labelsize=9)

figure.suptitle("Logistic regression against NB on identical training and testing rows", fontweight="bold", y=1.02)
save_figure(figure, "11_model_comparison.png")

# ----------------------------------------------------------------------------------
# diagrams

# diagram 1: linear and logistic side by side on real data, one feature each
sample = train_rows.sample(n=2500, random_state=RANDOM_SEED)
one_feature_linear = LinearRegression().fit(train_rows[["wind"]], y_train_number)
one_feature_logistic = LogisticRegression(max_iter=5000).fit(train_rows[["wind"]], y_train_class)
straight_line_on_labels = LinearRegression().fit(train_rows[["wind"]], (y_train_class == "Not Good").astype(int))

figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
wind_grid = np.linspace(0, train_rows.wind.quantile(0.995) + 5, 300).reshape(-1, 1)
wind_grid_frame = pd.DataFrame(wind_grid, columns=["wind"])

ax = axes[0]
ax.scatter(sample.wind, sample[TARGET], s=9, alpha=0.35, color=TEAL, edgecolor="none", label="training days (2,500 shown)")
ax.plot(wind_grid, one_feature_linear.predict(wind_grid_frame), color=CORAL, lw=3, label="fitted line")
ax.set(title="linear regression predicts a number", xlabel="maximum wind speed, km/h", ylabel="daily AQI", ylim=(15, 180))
ax.legend(fontsize=9, loc="upper right")
ax.annotate("AQI = {:.1f} {:+.2f} x wind\nevery extra km/h of wind lowers the\npredicted AQI by {:.2f} points".format(
    one_feature_linear.intercept_, one_feature_linear.coef_[0], -one_feature_linear.coef_[0]), xy=(0.45, 0.62), xycoords="axes fraction", fontsize=10, color="#333", bbox=TEXT_BOX)

ax = axes[1]
jitter = np.random.default_rng(RANDOM_SEED).uniform(-0.04, 0.04, len(sample))
is_not_good = (sample[BINARY_LABEL] == "Not Good").astype(int).values
ax.scatter(sample.wind, is_not_good + jitter, s=9, alpha=0.30, c=np.where(is_not_good == 1, CORAL, "#3bb143"), edgecolor="none")
# drawn out past the windiest real day on purpose, to show where each one ends up
long_wind_grid = pd.DataFrame(np.linspace(0, 80, 300), columns=["wind"])
ax.plot(long_wind_grid.wind, one_feature_logistic.predict_proba(long_wind_grid)[:, 1], color=NAVY, lw=3, label="logistic regression: a probability")
ax.plot(long_wind_grid.wind, straight_line_on_labels.predict(long_wind_grid), color="#999", lw=2, ls="--", label="a straight line forced onto the 0s and 1s")
ax.axvspan(train_rows.wind.max(), 80, color="#eee", alpha=0.6)
ax.axhline(0, color="#555", lw=0.8)
ax.axhline(0.5, color=GOLD, lw=1.6, ls=":")
ax.set(title="logistic regression predicts the chance of a class", xlabel="maximum wind speed, km/h", ylabel="P(Not Good)", ylim=(-0.45, 1.15), xlim=(0, 80))
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(fontsize=9, loc="lower left")
ax.annotate("top row: Not Good days (1)\nbottom row: Good days (0)\nthe grey area is windier than any real\nday. the straight line drops below 0 there,\na probability that cannot exist. the S curve\nflattens out and never leaves 0 to 1", xy=(0.55, 0.50), xycoords="axes fraction", fontsize=9.5, color="#333", bbox=TEXT_BOX)

figure.suptitle("Same column two questions: how bad will the air be and how likely is it to be Not Good", fontweight="bold", y=1.02)
save_figure(figure, "00_concept_linear_vs_logistic.png")

# diagram 2: the sigmoid, and one real test day pushed all the way through the fitted model
test_scores = logistic_model.decision_function(X_test)
test_probabilities = logistic_model.predict_proba(X_test)[:, 1]

# the example day: a real test day that the model is fairly sure is Not Good, and gets right
distance_from_080 = np.abs(test_probabilities - 0.80) + np.where(logistic_predictions == y_test_class, 0, 10)
example_position = int(np.argmin(distance_from_080))
example_raw = X_test.iloc[example_position]
example_standardised = (example_raw.values - fitted_scaler.mean_) / fitted_scaler.scale_
example_contributions = fitted_logistic.coef_[0] * example_standardised
example_score = fitted_logistic.intercept_[0] + example_contributions.sum()
example_probability = sigmoid(example_score)
# the hand calculation has to give the same score and probability the fitted model gives
assert np.isclose(example_score, test_scores[example_position]), "hand score does not match the model"
assert np.isclose(example_probability, test_probabilities[example_position]), "hand probability does not match the model"
print("\nworked example: {} on {}, score {:.3f}, probability {:.3f}, actually {}".format(
    test_rows.city.iloc[example_position], test_rows.date.iloc[example_position].strftime("%Y-%m-%d"),
    example_score, example_probability, y_test_class[example_position]))

figure, axes = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1, 1.1]})

ax = axes[0]
score_grid = np.linspace(-6, 6, 400)
ax.plot(score_grid, sigmoid(score_grid), color=NAVY, lw=3)
ax.axhline(0.5, color=GOLD, ls="--", lw=1.6)
ax.axvline(0, color="#999", ls=":", lw=1.4)
ax.fill_between(score_grid, 0, 1, where=score_grid >= 0, color=CORAL, alpha=0.07)
ax.fill_between(score_grid, 0, 1, where=score_grid < 0, color="#3bb143", alpha=0.07)
ax.scatter(np.clip(test_scores, -6, 6), sigmoid(np.clip(test_scores, -6, 6)), s=6, color=TEAL, alpha=0.25, zorder=3, label="the {:,} test days".format(len(test_scores)))
ax.scatter([example_score], [example_probability], s=120, color=CORAL, edgecolor="black", zorder=4, label="the worked example on the right")
ax.set(title="the sigmoid turns any score into a probability", xlabel="z, the linear score  b0 + b1 x1 + ... + b9 x9", ylabel="P(Not Good) = 1 / (1 + e^-z)", ylim=(-0.03, 1.03))
ax.text(-5.7, 0.92, "predicted Good\n(probability below 0.5)", fontsize=9.5, color="#2a7a2f")
ax.text(1.0, 0.08, "predicted Not Good\n(probability 0.5 or more)", fontsize=9.5, color=CORAL)
ax.legend(fontsize=9, loc="center left")

ax = axes[1]
ax.axis("off")
ax.set_title("one real test day, worked through by hand", fontweight="bold", loc="left")
worked_rows = [["column", "value", "standardised", "x  coefficient", "= adds to z"]]
for name, raw, standardised, coefficient, contribution in zip(ALL_COLUMNS, example_raw.values, example_standardised, fitted_logistic.coef_[0], example_contributions):
    worked_rows.append([name, "{:.2f}".format(raw) if not name.startswith("is_") else "{:.0f}".format(raw), "{:+.2f}".format(standardised), "{:+.3f}".format(coefficient), "{:+.3f}".format(contribution)])
worked_rows.append(["intercept b0", "", "", "", "{:+.3f}".format(fitted_logistic.intercept_[0])])
worked_rows.append(["z, the sum", "", "", "", "{:+.3f}".format(example_score)])
drawn = ax.table(cellText=worked_rows[1:], colLabels=worked_rows[0], cellLoc="center", loc="upper center", colWidths=[0.24, 0.16, 0.20, 0.20, 0.20])
drawn.auto_set_font_size(False)
drawn.set_fontsize(8.5)
drawn.scale(1, 1.45)
for position, cell in drawn.get_celld().items():
    if position[0] == 0:
        cell.set_facecolor(TEAL)
        cell.set_text_props(color="white", fontweight="bold")
    elif position[0] == len(worked_rows) - 1:
        cell.set_facecolor("#fdf0e6")
        cell.set_text_props(fontweight="bold")
    elif position[0] % 2 == 0:
        cell.set_facecolor("#e6f2f1")
ax.annotate("{}, {}\nP(Not Good) = 1 / (1 + e^-({:.3f})) = {:.3f}\n{:.3f} is at least 0.5, so the model says Not Good. the day was actually {} (AQI {:.0f})".format(
    test_rows.city.iloc[example_position], test_rows.date.iloc[example_position].strftime("%d %B %Y"),
    example_score, example_probability, example_probability, y_test_class[example_position], y_test_number[example_position]),
    xy=(0.0, -0.02), xycoords="axes fraction", fontsize=10, color="#222", bbox=TEXT_BOX, va="top")

figure.suptitle("From a linear score to a probability to a class", fontweight="bold", y=1.0)
save_figure(figure, "00_concept_sigmoid.png")

print("\ndone, images in images/regression/, tables in data/clean/reg_*.csv")
